"""Individual Exercise 1: MLESAC and Preemptive RANSAC."""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset: float


@dataclass
class PlaneResult:
    model: Plane
    inliers: np.ndarray
    runtime: float
    evaluations: int
    used_points: int

    @property
    def inlier_fraction(self) -> float:
        return float(np.mean(self.inliers))


# -----------------------------------------------------------------------------
# Loading and plane geometry
# -----------------------------------------------------------------------------


def find_key(data: dict, prefix: str) -> str:
    keys = [key for key in data.keys() if key.startswith(prefix)]
    if not keys:
        raise KeyError(f"No variable starting with '{prefix}' was found.")
    return sorted(keys)[0]


def load_exercise_data(path: Path):
    data = loadmat(path)
    distance = np.asarray(data[find_key(data, "distances")])
    cloud = np.asarray(data[find_key(data, "cloud")], dtype=np.float64)

    if cloud.ndim != 3 or cloud.shape[2] != 3:
        raise ValueError(f"Unexpected point-cloud shape: {cloud.shape}")

    valid_mask = np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] != 0)
    points = cloud[valid_mask]
    return distance, cloud, valid_mask, points


def normalize_hypotheses(normals: np.ndarray, offsets: np.ndarray):
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-10
    normals = normals[valid] / lengths[valid, None]
    offsets = offsets[valid] / lengths[valid]

    flip = offsets < 0
    normals[flip] *= -1
    offsets[flip] *= -1
    return normals, offsets


def generate_hypotheses(points: np.ndarray, count: int, rng: np.random.Generator):
    normals_list = []
    offsets_list = []
    collected = 0

    while collected < count:
        number = max(64, 2 * (count - collected))
        indices = rng.integers(0, len(points), size=(number, 3))
        different = (
            (indices[:, 0] != indices[:, 1])
            & (indices[:, 0] != indices[:, 2])
            & (indices[:, 1] != indices[:, 2])
        )
        samples = points[indices[different]]
        if len(samples) == 0:
            continue

        v1 = samples[:, 1] - samples[:, 0]
        v2 = samples[:, 2] - samples[:, 0]
        normals = np.cross(v1, v2)
        offsets = np.einsum("ij,ij->i", normals, samples[:, 0])
        normals, offsets = normalize_hypotheses(normals, offsets)

        take = min(count - collected, len(offsets))
        if take > 0:
            normals_list.append(normals[:take])
            offsets_list.append(offsets[:take])
            collected += take

    return np.concatenate(normals_list), np.concatenate(offsets_list)


def fit_plane(points: np.ndarray) -> Plane:
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    offset = float(np.dot(normal, center))

    if offset < 0:
        normal = -normal
        offset = -offset

    return Plane(normal, offset)


def distances_to_plane(points: np.ndarray, plane: Plane) -> np.ndarray:
    return np.abs(points @ plane.normal - plane.offset)


def refine_plane(points: np.ndarray, plane: Plane, epsilon: float):
    inliers = distances_to_plane(points, plane) < epsilon
    if np.count_nonzero(inliers) >= 3:
        plane = fit_plane(points[inliers])
        inliers = distances_to_plane(points, plane) < epsilon
    return plane, inliers


def plane_angle(plane_a: Plane, plane_b: Plane) -> float:
    cosine = np.clip(abs(np.dot(plane_a.normal, plane_b.normal)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def plane_offset_difference(plane_a: Plane, plane_b: Plane) -> float:
    sign = 1.0 if np.dot(plane_a.normal, plane_b.normal) >= 0 else -1.0
    return float(abs(plane_a.offset - sign * plane_b.offset))


# -----------------------------------------------------------------------------
# RANSAC and MLESAC
# -----------------------------------------------------------------------------


def model_costs(residuals: np.ndarray, epsilon: float, gamma: float, method: str):
    if method == "ransac":
        return np.count_nonzero(residuals >= epsilon, axis=1).astype(float)
    if method == "mlesac":
        return np.where(residuals < epsilon, residuals, gamma).sum(axis=1)
    raise ValueError("method must be 'ransac' or 'mlesac'")


def score_hypotheses(points, normals, offsets, epsilon, gamma, method):
    scores = np.empty(len(normals), dtype=float)
    evaluations = 0
    chunk_size = 32

    for start in range(0, len(normals), chunk_size):
        stop = min(start + chunk_size, len(normals))
        residuals = np.abs(normals[start:stop] @ points.T - offsets[start:stop, None])
        scores[start:stop] = model_costs(residuals, epsilon, gamma, method)
        evaluations += (stop - start) * len(points)

    return scores, evaluations


def estimate_plane(points, epsilon, hypotheses, seed, method, gamma, selection_points=None):
    if gamma <= epsilon and method == "mlesac":
        raise ValueError("gamma must be larger than epsilon")

    if selection_points is None:
        selection_points = points

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    normals, offsets = generate_hypotheses(points, hypotheses, rng)
    scores, evaluations = score_hypotheses(
        selection_points, normals, offsets, epsilon, gamma, method
    )

    best = int(np.argmin(scores))
    plane = Plane(normals[best], float(offsets[best]))
    plane, inliers = refine_plane(points, plane, epsilon)

    return PlaneResult(
        model=plane,
        inliers=inliers,
        runtime=time.perf_counter() - start,
        evaluations=evaluations,
        used_points=len(selection_points),
    )


# -----------------------------------------------------------------------------
# Preemptive RANSAC
# -----------------------------------------------------------------------------


def preemptive_ransac(points, epsilon, gamma, M, B, seed):
    if M < 1 or B < 1:
        raise ValueError("M and B must be positive")
    if gamma <= epsilon:
        raise ValueError("gamma must be larger than epsilon")

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    normals, offsets = generate_hypotheses(points, M, rng)

    active = np.arange(M)
    scores = np.zeros(M, dtype=float)
    point_order = rng.permutation(len(points))
    used_points = 0
    evaluations = 0

    while len(active) > 1 and used_points < len(points):
        block_indices = point_order[used_points : used_points + B]
        block = points[block_indices]

        residuals = np.abs(normals[active] @ block.T - offsets[active, None])
        scores[active] += model_costs(residuals, epsilon, gamma, "mlesac")
        evaluations += len(active) * len(block)
        used_points += len(block)

        completed_blocks = used_points // B
        keep_number = int(math.floor(M * 2 ** (-completed_blocks)))
        keep_number = max(1, min(len(active), keep_number))

        order = np.argsort(scores[active], kind="stable")
        active = active[order[:keep_number]]

    best = int(active[np.argmin(scores[active])])
    plane = Plane(normals[best], float(offsets[best]))
    plane, inliers = refine_plane(points, plane, epsilon)

    return PlaneResult(
        model=plane,
        inliers=inliers,
        runtime=time.perf_counter() - start,
        evaluations=evaluations,
        used_points=used_points,
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def sample_points(points: np.ndarray, maximum: int, seed: int):
    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=maximum, replace=False)
    return points[indices]


def median_inlier_residual(result: PlaneResult, points: np.ndarray) -> float:
    residuals = distances_to_plane(points, result.model)
    return float(np.median(residuals[result.inliers]))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_names: tuple[str, ...], value_names: tuple[str, ...]):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_names)
        groups.setdefault(key, []).append(row)

    result = []
    for key in sorted(groups):
        group = groups[key]
        row = {name: value for name, value in zip(group_names, key)}
        row["runs"] = len(group)
        for name in value_names:
            values = np.asarray([float(item[name]) for item in group])
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values))
        result.append(row)
    return result


def build_reference(points, epsilon, gamma):
    selection = sample_points(points, 50000, 2026)
    return estimate_plane(
        points,
        epsilon=epsilon,
        hypotheses=4096,
        seed=2026,
        method="mlesac",
        gamma=gamma,
        selection_points=selection,
    )


def evaluate_epsilon(points, reference, output_dir):
    epsilon_values = [0.004, 0.006, 0.008, 0.010, 0.012]
    seeds = [0, 1, 2, 3, 4]
    selection = sample_points(points, 30000, 12345)
    gamma = 0.020
    rows = []

    for epsilon in epsilon_values:
        for seed in seeds:
            for method in ["ransac", "mlesac"]:
                result = estimate_plane(
                    points,
                    epsilon=epsilon,
                    hypotheses=256,
                    seed=seed,
                    method=method,
                    gamma=gamma,
                    selection_points=selection,
                )
                rows.append(
                    {
                        "method": method.upper(),
                        "epsilon_m": epsilon,
                        "inlier_fraction": result.inlier_fraction,
                        "median_residual_mm": 1000 * median_inlier_residual(result, points),
                        "angle_to_reference_deg": plane_angle(result.model, reference.model),
                        "offset_to_reference_mm": 1000
                        * plane_offset_difference(result.model, reference.model),
                        "runtime_s": result.runtime,
                    }
                )

    summary = summarize(
        rows,
        ("method", "epsilon_m"),
        (
            "inlier_fraction",
            "median_residual_mm",
            "angle_to_reference_deg",
            "offset_to_reference_mm",
            "runtime_s",
        ),
    )
    write_csv(output_dir / "mlesac_epsilon_summary.csv", summary)
    plot_epsilon(summary, output_dir / "mlesac_epsilon_comparison.png")
    return summary


def plot_epsilon(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for method in ["RANSAC", "MLESAC"]:
        subset = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["epsilon_m"]),
        )
        x = [1000 * float(row["epsilon_m"]) for row in subset]
        angle = [float(row["angle_to_reference_deg_mean"]) for row in subset]
        angle_std = [float(row["angle_to_reference_deg_std"]) for row in subset]
        offset = [float(row["offset_to_reference_mm_mean"]) for row in subset]
        offset_std = [float(row["offset_to_reference_mm_std"]) for row in subset]

        axes[0].errorbar(x, angle, yerr=angle_std, marker="o", capsize=3, label=method)
        axes[1].errorbar(x, offset, yerr=offset_std, marker="o", capsize=3, label=method)

    axes[0].set_title("Plane orientation")
    axes[0].set_xlabel("epsilon [mm]")
    axes[0].set_ylabel("angle to reference [degree]")
    axes[1].set_title("Plane position")
    axes[1].set_xlabel("epsilon [mm]")
    axes[1].set_ylabel("offset to reference [mm]")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_preemptive(points, reference, output_dir):
    M_values = [64, 256, 1024]
    B_values = [64, 256, 1024]
    seeds = [0, 1, 2, 3, 4]
    epsilon = 0.008
    gamma = 0.020
    rows = []

    for M in M_values:
        for B in B_values:
            for seed in seeds:
                result = preemptive_ransac(points, epsilon, gamma, M, B, seed)
                rows.append(
                    {
                        "M": M,
                        "B": B,
                        "inlier_fraction": result.inlier_fraction,
                        "median_residual_mm": 1000 * median_inlier_residual(result, points),
                        "angle_to_reference_deg": plane_angle(result.model, reference.model),
                        "offset_to_reference_mm": 1000
                        * plane_offset_difference(result.model, reference.model),
                        "runtime_s": result.runtime,
                        "model_point_evaluations": result.evaluations,
                    }
                )

    summary = summarize(
        rows,
        ("M", "B"),
        (
            "inlier_fraction",
            "median_residual_mm",
            "angle_to_reference_deg",
            "offset_to_reference_mm",
            "runtime_s",
            "model_point_evaluations",
        ),
    )
    write_csv(output_dir / "preemptive_parameter_summary.csv", summary)
    plot_preemptive(summary, output_dir / "preemptive_parameter_comparison.png")
    return summary


def plot_preemptive(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for B in [64, 256, 1024]:
        subset = sorted([row for row in rows if int(row["B"]) == B], key=lambda row: int(row["M"]))
        M = [int(row["M"]) for row in subset]
        angle = [float(row["angle_to_reference_deg_mean"]) for row in subset]
        angle_std = [float(row["angle_to_reference_deg_std"]) for row in subset]
        evaluations = [float(row["model_point_evaluations_mean"]) for row in subset]

        axes[0].errorbar(M, angle, yerr=angle_std, marker="o", capsize=3, label=f"B={B}")
        axes[1].plot(M, evaluations, marker="o", label=f"B={B}")

    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Plane accuracy")
    axes[0].set_xlabel("number of hypotheses M")
    axes[0].set_ylabel("angle to reference [degree]")

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_title("Computation")
    axes[1].set_xlabel("number of hypotheses M")
    axes[1].set_ylabel("model-point evaluations")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def visualize_three_M(distance, valid_mask, points, output_dir):
    M_values = [64, 256, 1024]
    B = 256
    epsilon = 0.008
    gamma = 0.020
    results = []

    for M in M_values:
        result = preemptive_ransac(points, epsilon, gamma, M, B, seed=0)
        mask = np.zeros(valid_mask.shape, dtype=bool)
        mask[valid_mask] = result.inliers
        results.append((M, result, mask))

        fig, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(distance, cmap="gray")
        overlay = np.ma.masked_where(~mask, mask)
        axis.imshow(overlay, cmap="autumn", alpha=0.70)
        axis.set_title(f"Estimated floor plane: M={M}, B={B}")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"floor_plane_M{M}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for axis, (M, result, mask) in zip(axes, results):
        axis.imshow(distance, cmap="gray")
        overlay = np.ma.masked_where(~mask, mask)
        axis.imshow(overlay, cmap="autumn", alpha=0.70)
        axis.set_title(f"M={M}, B={B}\n{result.inlier_fraction:.1%} inliers")
        axis.axis("off")

    fig.suptitle("Estimated floor plane for three values of M", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_dir / "three_M_comparison.png", dpi=180)
    plt.close(fig)


def choose_input_file(data_dir: Path, requested_file: str | None):
    if requested_file:
        path = Path(requested_file)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    preferred = data_dir / "example4kinect.mat"
    if preferred.exists():
        return preferred

    files = sorted(data_dir.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    return files[-1]


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = choose_input_file(Path(args.data_dir), args.input_file)
    distance, _, valid_mask, points = load_exercise_data(input_file)

    print(f"Evaluation file: {input_file.name}")
    print(f"Valid points: {len(points)}")

    reference = build_reference(points, epsilon=0.008, gamma=0.020)
    evaluate_epsilon(points, reference, output_dir)
    evaluate_preemptive(points, reference, output_dir)
    visualize_three_M(distance, valid_mask, points, output_dir)

    print(f"Results written to: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Individual Exercise 1")
    parser.add_argument("--data-dir", default="data", help="Folder containing the .mat files")
    parser.add_argument("--input-file", default=None, help="Optional .mat file name")
    parser.add_argument("--output-dir", default="results", help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
