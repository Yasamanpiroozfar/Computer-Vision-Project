"""Individual Exercise 2: HDR reconstruction from a JPG exposure stack."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


CHANNEL_NAMES = ("R", "G", "B")
EPS = 1e-8


def read_exposure_time(path: Path) -> float:
    """Read exposure time in seconds from JPG EXIF data."""

    with Image.open(path) as image:
        exif = image.getexif()
        exif_ifd = exif.get_ifd(34665)
        exposure_tag = 33434
        if exposure_tag not in exif_ifd:
            raise ValueError(f"No exposure time found in {path.name}")
        return float(exif_ifd[exposure_tag])


def find_jpg_stack(input_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Find JPG files and order them from longest to shortest exposure."""

    paths = sorted(input_dir.glob("*.JPG")) + sorted(input_dir.glob("*.jpg"))
    paths = sorted(set(paths))
    if len(paths) < 3:
        raise FileNotFoundError(f"At least three JPG files are required in {input_dir}")

    pairs = [(path, read_exposure_time(path)) for path in paths]
    pairs.sort(key=lambda item: item[1], reverse=True)
    ordered_paths = [item[0] for item in pairs]
    exposure_times = np.asarray([item[1] for item in pairs], dtype=np.float64)
    return ordered_paths, exposure_times


def load_small_stack(paths: list[Path], step: int = 6) -> np.ndarray:
    """Load a regularly subsampled stack for response estimation."""

    images = []
    expected_shape = None
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        rgb = rgb[::step, ::step]
        if expected_shape is None:
            expected_shape = rgb.shape
        elif rgb.shape != expected_shape:
            raise ValueError("All JPG images must have the same size.")
        images.append(rgb)
    return np.stack(images, axis=0)


def collect_response_samples(
    image_stack: np.ndarray,
    exposure_times: np.ndarray,
    channel: int,
    seed: int = 2026,
    samples_per_pair: int = 18000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect corresponding, non-dark and non-saturated JPG values."""

    rng = np.random.default_rng(seed + channel)
    long_values = []
    short_values = []
    target_log_ratios = []

    for index in range(len(exposure_times) - 1):
        longer = image_stack[index, :, :, channel].reshape(-1)
        shorter = image_stack[index + 1, :, :, channel].reshape(-1)

        valid = (
            (longer >= 12)
            & (longer <= 245)
            & (shorter >= 8)
            & (shorter <= 240)
            & (longer >= shorter + 2)
        )
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            continue
        if candidates.size > samples_per_pair:
            candidates = rng.choice(candidates, size=samples_per_pair, replace=False)

        long_values.append(longer[candidates].astype(np.float64))
        short_values.append(shorter[candidates].astype(np.float64))
        ratio = np.log(exposure_times[index] / exposure_times[index + 1])
        target_log_ratios.append(
            np.full(candidates.size, ratio, dtype=np.float64)
        )

    if not long_values:
        raise RuntimeError("No valid JPG pixel pairs were found for response estimation.")

    return (
        np.concatenate(long_values),
        np.concatenate(short_values),
        np.concatenate(target_log_ratios),
    )


def estimate_inverse_response(
    image_stack: np.ndarray,
    exposure_times: np.ndarray,
    channel: int,
    iterations: int = 80,
) -> tuple[np.ndarray, float]:
    """Estimate a monotonic inverse response curve from adjacent exposures."""

    longer, shorter, target_log_ratio = collect_response_samples(
        image_stack, exposure_times, channel, samples_per_pair=30000
    )
    longer = longer.astype(np.int16)
    shorter = shorter.astype(np.int16)

    values = np.arange(256, dtype=np.float64)
    log_curve = 2.2 * np.log(np.clip(values / 128.0, 1e-4, None))
    log_curve[0] = log_curve[1] - 5.0

    long_indices = [np.flatnonzero(longer == value) for value in range(256)]
    short_indices = [np.flatnonzero(shorter == value) for value in range(256)]

    for _ in range(iterations):
        updated = log_curve.copy()

        for value in range(1, 256):
            proposals = []

            indices = long_indices[value]
            if indices.size:
                proposals.append(
                    log_curve[shorter[indices]] + target_log_ratio[indices]
                )

            indices = short_indices[value]
            if indices.size:
                proposals.append(
                    log_curve[longer[indices]] - target_log_ratio[indices]
                )

            if proposals:
                updated[value] = float(np.median(np.concatenate(proposals)))

        smoothed = updated.copy()
        for value in range(2, 254):
            smoothed[value] = float(np.median(updated[value - 2 : value + 3]))

        smoothed = np.maximum.accumulate(
            smoothed + 1e-5 * np.arange(256, dtype=np.float64)
        )
        smoothed -= smoothed[128]
        log_curve = 0.35 * log_curve + 0.65 * smoothed

    # Extrapolate unreliable values near 0 and 255 using the local slope.
    low_slope = float(np.median(np.diff(log_curve[8:31])))
    high_slope = float(np.median(np.diff(log_curve[220:246])))
    low_slope = max(low_slope, 1e-4)
    high_slope = max(high_slope, 1e-4)

    for value in range(7, -1, -1):
        log_curve[value] = log_curve[value + 1] - low_slope
    for value in range(246, 256):
        log_curve[value] = log_curve[value - 1] + high_slope

    log_curve = np.maximum.accumulate(log_curve)
    inverse = np.exp(log_curve - log_curve[-1])
    inverse[0] = 0.0

    predicted_ratios = inverse[longer] / (inverse[shorter] + EPS)
    expected_ratios = np.exp(target_log_ratio)
    median_relative_error = float(
        np.median(np.abs(predicted_ratios - expected_ratios) / expected_ratios)
    )

    return inverse.astype(np.float32), median_relative_error

def estimate_all_curves(
    image_stack: np.ndarray, exposure_times: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """Estimate one inverse response curve for each RGB channel."""

    curves = []
    diagnostics = []
    for channel, name in enumerate(CHANNEL_NAMES):
        curve, median_relative_error = estimate_inverse_response(
            image_stack, exposure_times, channel
        )
        curves.append(curve)
        diagnostics.append(
            {
                "channel": name,
                "median_relative_pair_error": median_relative_error,
            }
        )
    return np.stack(curves, axis=0), diagnostics

def save_response_plot(curves: np.ndarray, output_path: Path) -> None:
    """Save the estimated inverse and forward JPG response curves."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    jpg_values = np.arange(256)
    relative_exposure = np.linspace(0.0, 1.0, 500)

    for channel, name in enumerate(CHANNEL_NAMES):
        inverse = curves[channel]
        axes[0].plot(jpg_values, inverse, label=name)

        unique_inverse, unique_indices = np.unique(inverse, return_index=True)
        forward = np.interp(
            relative_exposure,
            unique_inverse,
            jpg_values[unique_indices],
            left=0.0,
            right=255.0,
        )
        axes[1].plot(relative_exposure, forward, label=name)

    axes[0].set_title("Estimated inverse response")
    axes[0].set_xlabel("JPG value")
    axes[0].set_ylabel("Relative linear exposure")
    axes[1].set_title("Estimated camera response")
    axes[1].set_xlabel("Relative linear exposure")
    axes[1].set_ylabel("JPG value")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def measure_pair_ratios(
    image_stack: np.ndarray,
    exposure_times: np.ndarray,
    curves: np.ndarray,
) -> list[dict]:
    """Compare expected exposure ratios with ratios after linearization."""

    rows = []
    for index in range(len(exposure_times) - 1):
        expected = float(exposure_times[index] / exposure_times[index + 1])
        row = {
            "long_image_index": index,
            "short_image_index": index + 1,
            "expected_ratio": expected,
        }

        for channel, name in enumerate(CHANNEL_NAMES):
            longer = image_stack[index, :, :, channel].reshape(-1)
            shorter = image_stack[index + 1, :, :, channel].reshape(-1)
            valid = (
                (longer >= 12)
                & (longer <= 245)
                & (shorter >= 8)
                & (shorter <= 240)
                & (longer >= shorter + 2)
            )
            ratios = curves[channel, longer[valid]] / (
                curves[channel, shorter[valid]] + EPS
            )
            row[f"measured_{name}"] = float(np.median(ratios))
        rows.append(row)
    return rows


def save_linearity_plot(rows: list[dict], output_path: Path) -> None:
    """Plot expected ratios and measured ratios after linearization."""

    expected = np.asarray([row["expected_ratio"] for row in rows])
    positions = np.arange(len(rows))

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(positions, expected, "k--", marker="o", label="Expected ratio")
    for name in CHANNEL_NAMES:
        measured = [row[f"measured_{name}"] for row in rows]
        axis.plot(positions, measured, marker="o", label=f"Measured {name}")

    axis.set_xlabel("Adjacent exposure pair (long to short)")
    axis.set_ylabel("Exposure ratio")
    axis.set_title("Linearity check after inverting the JPG response")
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{i}-{i+1}" for i in positions])
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_ratio_csv(rows: list[dict], output_path: Path) -> None:
    """Save the response validation values."""

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def linearize_rgb(image: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Apply the three inverse-response lookup tables to an RGB JPG image."""

    output = np.empty(image.shape, dtype=np.float32)
    for channel in range(3):
        output[:, :, channel] = curves[channel, image[:, :, channel]]
    return output


def combine_hdr(
    paths: list[Path],
    exposure_times: np.ndarray,
    curves: np.ndarray,
    threshold_fraction: float = 0.8,
) -> np.ndarray:
    """Combine the linearized exposure stack using threshold replacement."""

    with Image.open(paths[0]) as image:
        first_jpg = np.asarray(image.convert("RGB"), dtype=np.uint8)

    hdr = linearize_rgb(first_jpg, curves)
    base_max = float(np.max(hdr))
    base_exposure = float(exposure_times[0])
    previous_scale = 1.0

    for path, exposure_time in zip(paths[1:], exposure_times[1:]):
        with Image.open(path) as image:
            jpg = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if jpg.shape != first_jpg.shape:
            raise ValueError("All JPG images must have the same size.")

        scale = base_exposure / float(exposure_time)
        scaled = linearize_rgb(jpg, curves) * scale
        threshold = threshold_fraction * base_max * previous_scale
        replace = hdr > threshold
        hdr[replace] = scaled[replace]

        print(
            f"  {path.name}: scale={scale:.4g}, "
            f"threshold={threshold:.4g}, replaced={replace.mean() * 100:.2f}%"
        )
        previous_scale = scale

    return hdr


def gray_world_balance(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply gray-world white balance to an HDR RGB image."""

    means = rgb.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    gains = gray / (means + EPS)
    balanced = rgb * gains[None, None, :]
    return balanced.astype(np.float32), gains.astype(np.float32)


def tone_map_log(hdr: np.ndarray) -> np.ndarray:
    """Apply logarithmic compression and min-max normalization."""

    compressed = np.log1p(np.maximum(hdr, 0.0))
    minimum = float(np.min(compressed))
    maximum = float(np.max(compressed))
    result = (compressed - minimum) / (maximum - minimum + EPS)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def tone_map_log_enhanced(
    hdr: np.ndarray,
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
    display_gamma: float = 0.85,
) -> np.ndarray:
    """Apply log compression, percentile normalization and display gamma."""

    compressed = np.log1p(np.maximum(hdr, 0.0))
    low = float(np.percentile(compressed, low_percentile))
    high = float(np.percentile(compressed, high_percentile))
    result = np.clip((compressed - low) / (high - low + EPS), 0.0, 1.0)
    result = np.power(result, display_gamma)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def save_rgb(image: np.ndarray, output_path: Path, quality: int = 99) -> None:
    """Save a floating-point RGB image in [0, 1] as a high-quality JPG."""

    image8 = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image8, mode="RGB").save(
        output_path, quality=quality, subsampling=0
    )


def save_exposure_preview(
    paths: list[Path], exposure_times: np.ndarray, output_path: Path
) -> None:
    """Create a contact sheet of the input exposure sequence."""

    width, height = 360, 270
    columns = 3
    rows = int(np.ceil(len(paths) / columns))
    sheet = Image.new("RGB", (columns * width, rows * height), "white")

    for index, (path, exposure) in enumerate(zip(paths, exposure_times)):
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((width, height - 30), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (width, height), "white")
        cell.paste(thumb, ((width - thumb.width) // 2, 0))
        draw = ImageDraw.Draw(cell)
        draw.text((5, height - 25), f"{path.name}   t={exposure:g} s", fill="black")
        sheet.paste(cell, ((index % columns) * width, (index // columns) * height))

    sheet.save(output_path, quality=92, subsampling=0)


def save_comparison(
    longest_path: Path,
    middle_path: Path,
    baseline_result: np.ndarray,
    enhanced_result: np.ndarray,
    output_path: Path,
) -> None:
    """Save source exposures next to the baseline and enhanced HDR results."""

    with Image.open(longest_path) as image:
        longest = image.convert("RGB")
    with Image.open(middle_path) as image:
        middle = image.convert("RGB")

    def to_pil(array: np.ndarray) -> Image.Image:
        return Image.fromarray(
            np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="RGB",
        )

    images = [
        longest,
        middle,
        to_pil(baseline_result),
        to_pil(enhanced_result),
    ]
    titles = [
        "Longest JPG exposure",
        "Middle JPG exposure",
        "HDR log baseline",
        "HDR enhanced log",
    ]

    target_width = 600
    scale = target_width / longest.width
    target_height = int(longest.height * scale)
    images = [
        image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        for image in images
    ]

    canvas = Image.new("RGB", (len(images) * target_width, target_height + 45), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(zip(images, titles)):
        x = index * target_width
        canvas.paste(image, (x, 45))
        draw.text((x + 10, 12), title, fill="black")
    canvas.save(output_path, quality=95, subsampling=0)


def run(input_dir: Path, output_dir: Path) -> None:
    """Run response estimation, validation and HDR construction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths, exposure_times = find_jpg_stack(input_dir)

    print("Exposure sequence:")
    for path, exposure in zip(paths, exposure_times):
        print(f"  {path.name}: {exposure:g} s")

    small_stack = load_small_stack(paths, step=6)
    curves, diagnostics = estimate_all_curves(small_stack, exposure_times)

    print("Response-estimation diagnostics:")
    for row in diagnostics:
        print(
            f"  {row['channel']}: median relative pair error="
            f"{100.0 * row['median_relative_pair_error']:.2f}%"
        )

    save_response_plot(curves, output_dir / "estimated_camera_response.png")
    validation_rows = measure_pair_ratios(small_stack, exposure_times, curves)
    save_linearity_plot(validation_rows, output_dir / "linearization_check.png")
    write_ratio_csv(validation_rows, output_dir / "linearization_ratios.csv")
    save_exposure_preview(paths, exposure_times, output_dir / "exposure_stack_preview.jpg")

    print("Combining full-resolution exposure stack")
    hdr = combine_hdr(paths, exposure_times, curves)
    hdr_wb, gains = gray_world_balance(hdr)
    print(f"Gray-world gains [R, G, B]: {gains.tolist()}")

    baseline = tone_map_log(hdr_wb)
    save_rgb(baseline, output_dir / "hdr_from_jpg.jpg")

    enhanced = tone_map_log_enhanced(hdr_wb)
    save_rgb(enhanced, output_dir / "hdr_from_jpg_enhanced.jpg")

    save_comparison(
        paths[0],
        paths[len(paths) // 2],
        baseline,
        enhanced,
        output_dir / "hdr_comparison.jpg",
    )

    print(f"Results written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Individual Exercise 2: HDR from JPG")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/hdr-jpg"),
        help="Directory containing the bracketed JPG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for result images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input_dir, arguments.output_dir)
