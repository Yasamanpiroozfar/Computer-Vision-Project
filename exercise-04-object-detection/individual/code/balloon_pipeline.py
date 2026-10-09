"""Exercise 5.2: Selective Search + ResNet18 + SVM balloon detector.

1. Generate Selective Search proposals and save them as JSON.
2. Label proposals as positive/negative using IoU thresholds tp/tn.
3. Optionally save proposal crops.
4. Extract 512-D pretrained ResNet18 embeddings from proposal crops.
5. Train a scikit-learn SVM classifier.
6. Run inference with confidence threshold + NMS.
7. Keep the assignment-required detection evaluation: COCO-style mAP and MABO.

Typical full run from the project root:

    python code/balloon_pipeline.py --mode all --resnet-weights default

"""
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform as skt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from selective_search import selective_search

Box = List[int]  # [x, y, w, h]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Shared helpers

def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image(img_path: Path | str) -> np.ndarray:
    img = skimage.io.imread(str(img_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    return img


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def clip_box(box: Sequence[float], width: int, height: int) -> Box:
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def safe_crop(img: np.ndarray, box: Sequence[float]) -> np.ndarray:
    height, width = img.shape[:2]
    x, y, w, h = clip_box(box, width, height)
    crop = img[y : y + h, x : x + w]
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return crop


def load_coco(json_path: Path | str) -> Tuple[Dict[int, dict], Dict[int, List[Box]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {int(im["id"]): im for im in coco.get("images", [])}
    gts: Dict[int, List[Box]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        gts.setdefault(img_id, []).append([int(round(v)) for v in ann["bbox"]])
    return images, gts


def prepare_dataset(project_root: Path) -> Path:
    dataset_root = project_root / "data" / "balloon_dataset"
    if (dataset_root / "train" / "_annotations.coco.json").exists():
        return dataset_root
    zip_path = project_root / "data" / "balloon_dataset.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root} or {zip_path}")
    ensure_dir(dataset_root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_root)
    return dataset_root


# Task 5.2.1 — proposal generation, single-configuration style

def run_selective_search_single_config(
    img_path: Path | str,
    scale: float = 200.0,
    sigma: float = 0.9,
    min_size: int = 50,
    max_props: int = 1500,
    min_side: int = 5,
) -> List[Box]:
    """Generate proposals using one Selective Search setting
    """
    image = read_image(img_path)
    height, width = image.shape[:2]
    _, regions = selective_search(image, scale=scale, sigma=sigma, min_size=min_size)

    seen = set()
    boxes: List[Box] = []
    for region in regions:
        x, y, w, h = region["rect"]
        if w < min_side or h < min_side:
            continue
        box = clip_box([x, y, w, h], width, height)
        key = tuple(box)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(box)
        if len(boxes) >= max_props:
            break
    return boxes


def process_split_proposals(
    data_root: Path,
    split: str,
    out_dir: Path,
    scale: float,
    sigma: float,
    min_size: int,
    max_props: int,
) -> Path:
    ann_path = data_root / split / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    images, _ = load_coco(ann_path)
    results, failures = [], []
    start = time.time()

    for i, (img_id, meta) in enumerate(images.items(), 1):
        img_path = data_root / split / meta["file_name"]
        try:
            proposals = run_selective_search_single_config(
                img_path,
                scale=scale,
                sigma=sigma,
                min_size=min_size,
                max_props=max_props,
            )
            results.append(
                {
                    "image_id": img_id,
                    "file_name": meta["file_name"],
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "proposals": proposals,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"image_id": img_id, "file_name": meta["file_name"], "error": repr(exc)})
        if i % 5 == 0 or i == len(images):
            print(f"[{split}] processed {i}/{len(images)}")

    ensure_dir(out_dir)
    out_path = out_dir / f"proposals_{split}.json"
    avg_props = sum(len(r["proposals"]) for r in results) / max(1, len(results))
    payload = {
        "split": split,
        "dataset_root": str(data_root),
        "params": {"scale": scale, "sigma": sigma, "min_size": min_size, "max_props": max_props},
        "results": results,
        "failures": failures,
        "runtime_sec": time.time() - start,
        "avg_proposals": avg_props,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[{split}] saved {out_path} | avg proposals={avg_props:.1f} | failures={len(failures)}")
    return out_path


# Task 5.2.2 — IoU-labeled samples and optional crop saving

def build_samples_for_split(
    data_root: Path,
    split: str,
    proposals_json: Path,
    samples_dir: Path,
    tp: float = 0.75,
    tn: float = 0.25,
    save_crops: bool = True,
    crop_size: int = 128,
    max_neg_per_image: int | None = None,
    add_gt_positives: bool = True,
) -> Path:
    with open(proposals_json, "r", encoding="utf-8") as f:
        proposal_payload = json.load(f)
    if proposal_payload["split"] != split:
        raise ValueError(f"Split mismatch in {proposals_json}")

    ann_path = data_root / split / "_annotations.coco.json"
    _, gt_by_image = load_coco(ann_path)

    split_samples_dir = samples_dir / split
    pos_dir = split_samples_dir / "pos"
    neg_dir = split_samples_dir / "neg"
    if save_crops:
        ensure_dir(pos_dir)
        ensure_dir(neg_dir)

    samples = []
    pos_count = neg_count = ignored_count = 0

    for item in proposal_payload["results"]:
        img_id = int(item["image_id"])
        file_name = item["file_name"]
        img_path = data_root / split / file_name
        image = read_image(img_path)
        gts = gt_by_image.get(img_id, [])

        # use GT boxes as guaranteed positives.
        candidate_boxes = []
        if add_gt_positives:
            candidate_boxes.extend((gt, 1, 1.0, "gt") for gt in gts)
        candidate_boxes.extend((box, None, None, "proposal") for box in item["proposals"])

        neg_kept = 0
        for idx, (box, forced_label, forced_iou, source) in enumerate(candidate_boxes):
            if forced_label is not None:
                label = forced_label
                max_iou = forced_iou
            else:
                max_iou = max((iou_xywh(box, gt) for gt in gts), default=0.0)
                if max_iou >= tp:
                    label = 1
                elif max_iou <= tn:
                    label = 0
                else:
                    ignored_count += 1
                    continue

            if label == 0 and max_neg_per_image is not None and neg_kept >= max_neg_per_image:
                continue
            if label == 0:
                neg_kept += 1

            entry = {
                "image_id": img_id,
                "file_name": file_name,
                "bbox": [int(v) for v in box],
                "iou": float(max_iou),
                "label": int(label),
                "source": source,
                "crop_path": None,
            }
            if save_crops:
                crop = safe_crop(image, box)
                if crop_size is not None:
                    crop = skt.resize(
                        crop,
                        (crop_size, crop_size),
                        anti_aliasing=True,
                        preserve_range=True,
                    ).astype(np.uint8)
                target_dir = pos_dir if label == 1 else neg_dir
                out_path = target_dir / f"{img_id}_{idx}_{source}_{int(round(max_iou * 100))}.jpg"
                skimage.io.imsave(out_path.as_posix(), crop, check_contrast=False)
                entry["crop_path"] = str(out_path)

            samples.append(entry)
            pos_count += int(label == 1)
            neg_count += int(label == 0)

    ensure_dir(split_samples_dir)
    meta_path = split_samples_dir / f"samples_{split}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "tp": tp,
                "tn": tn,
                "save_crops": save_crops,
                "crop_size": crop_size,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "ignored_count": ignored_count,
                "samples": samples,
            },
            f,
            indent=2,
        )
    print(f"[{split}] positives={pos_count} negatives={neg_count} ignored={ignored_count} -> {meta_path}")
    return meta_path


# Task 5.2.2 — ResNet18 features

def build_resnet18(device: str = "cpu", weights_mode: str = "default"):
    import torch
    import torchvision.models as models
    from torch import nn

    if weights_mode == "default":
        weights = models.ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        model = models.resnet18(weights=weights)
    elif weights_mode == "none":
        weights = None
        transform = models.ResNet18_Weights.DEFAULT.transforms()
        model = models.resnet18(weights=None)
    else:
        raise ValueError("weights_mode must be 'default' or 'none'")

    model.fc = nn.Identity()
    model.eval().to(device)
    return model, transform


def load_crop_for_sample(sample: dict, data_root: Path, split: str) -> np.ndarray:
    crop_path = sample.get("crop_path")
    if crop_path and Path(crop_path).exists():
        return read_image(crop_path)
    image = read_image(data_root / split / sample["file_name"])
    return safe_crop(image, sample["bbox"])


def extract_features_resnet18(
    samples_json: Path,
    features_dir: Path,
    data_root: Path,
    split: str,
    batch_size: int = 64,
    device: str = "cpu",
    weights_mode: str = "default",
) -> Path:
    import torch
    from PIL import Image

    with open(samples_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = meta["samples"]

    model, transform = build_resnet18(device=device, weights_mode=weights_mode)
    feats: List[np.ndarray] = []
    labels: List[int] = []
    batch_tensors, batch_labels = [], []

    for i, sample in enumerate(samples, 1):
        crop = load_crop_for_sample(sample, data_root, split)
        pil = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
        batch_tensors.append(transform(pil).unsqueeze(0))
        batch_labels.append(int(sample["label"]))

        if len(batch_tensors) == batch_size or i == len(samples):
            x = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                z = model(x).detach().cpu().numpy().astype(np.float32)
            feats.append(z)
            labels.extend(batch_labels)
            batch_tensors, batch_labels = [], []

        if i % 200 == 0 or i == len(samples):
            print(f"[{split}] ResNet18 features {i}/{len(samples)}")

    X = np.concatenate(feats, axis=0).astype(np.float32) if feats else np.empty((0, 512), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    ensure_dir(features_dir)
    out_path = features_dir / f"features_{split}_resnet18.npz"
    np.savez_compressed(out_path, X=X, y=y, extractor="resnet18", weights=weights_mode)
    print(f"[{split}] features {X.shape} -> {out_path}")
    return out_path


# Task 5.2.3 — train SVM and validate on crop classification

def load_features(features_dir: Path, split: str, extractor: str = "resnet18") -> Tuple[np.ndarray, np.ndarray]:
    path = features_dir / f"features_{split}_{extractor}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing features: {path}")
    data = np.load(path)
    return data["X"], data["y"]


def train_svm(
    features_dir: Path,
    model_out: Path,
    valid_split: str = "valid",
    kernel: str = "linear",
    C: float = 1.0,
    use_pca: bool = False,
    pca_components: int = 128,
    probability: bool = False,
) -> Tuple[Path, Path]:
    X_train, y_train = load_features(features_dir, "train")
    X_valid, y_valid = load_features(features_dir, valid_split)

    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_SEED)))
    if kernel == "linear" and not probability:
        clf = LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_SEED, dual=False, max_iter=50000)
    else:
        clf = SVC(C=C, kernel=kernel, class_weight="balanced", probability=probability, random_state=RANDOM_SEED)
    steps.append(("svm", clf))
    pipe = Pipeline(steps)

    print(f"Training SVM on {X_train.shape}; valid={X_valid.shape}; kernel={kernel}; PCA={use_pca}")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_valid, y_pred).tolist()
    if probability:
        scores = pipe.predict_proba(X_valid)[:, 1]
    else:
        scores = pipe.decision_function(X_valid)
    ap = float(average_precision_score(y_valid, scores))

    print("\nValidation classification report:")
    print(classification_report(y_valid, y_pred, digits=4, zero_division=0))
    print(f"Accuracy={acc:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} AP={ap:.4f}")

    ensure_dir(model_out.parent)
    joblib.dump(pipe, model_out)
    metrics_path = model_out.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "resnet18_svm_detection_pipeline",
                "train_shape": list(X_train.shape),
                "valid_shape": list(X_valid.shape),
                "kernel": kernel,
                "C": C,
                "use_pca": use_pca,
                "pca_components": pca_components,
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "classification_AP": ap,
                "confusion_matrix": cm,
            },
            f,
            indent=2,
        )
    print(f"Saved model: {model_out}\nSaved validation metrics: {metrics_path}")
    return model_out, metrics_path


# Task 5.2.4/5.2.5 — inference, NMS, mAP, MABO

def predict_scores(pipe, X: np.ndarray) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1].astype(np.float32)
        except Exception:
            pass
    dec = pipe.decision_function(X).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-dec))).astype(np.float32)


def nms_xywh(boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_thresh: float = 0.3) -> List[int]:
    if not boxes:
        return []
    boxes_np = np.asarray(boxes, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)
    x1, y1 = boxes_np[:, 0], boxes_np[:, 1]
    x2, y2 = boxes_np[:, 0] + boxes_np[:, 2], boxes_np[:, 1] + boxes_np[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores_np.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        remaining = np.where(overlap <= iou_thresh)[0]
        order = order[remaining + 1]
    return keep


def extract_proposal_features_for_image(
    img: np.ndarray,
    proposals: Sequence[Box],
    device: str = "cpu",
    weights_mode: str = "default",
    batch_size: int = 64,
) -> np.ndarray:
    import torch
    from PIL import Image

    model, transform = build_resnet18(device=device, weights_mode=weights_mode)
    feats: List[np.ndarray] = []
    batch = []
    for i, box in enumerate(proposals, 1):
        crop = safe_crop(img, box)
        pil = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
        batch.append(transform(pil).unsqueeze(0))
        if len(batch) == batch_size or i == len(proposals):
            x = torch.cat(batch, dim=0).to(device)
            with torch.no_grad():
                feats.append(model(x).detach().cpu().numpy().astype(np.float32))
            batch = []
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 512), dtype=np.float32)


def score_proposals(
    image: np.ndarray,
    proposals: Sequence[Box],
    model_path: Path,
    device: str = "cpu",
    weights_mode: str = "default",
    batch_size: int = 64,
) -> np.ndarray:
    pipe = joblib.load(model_path)
    X = extract_proposal_features_for_image(image, proposals, device=device, weights_mode=weights_mode, batch_size=batch_size)
    return predict_scores(pipe, X)


def average_precision_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_at_iou(detections: List[dict], gt_by_image: Dict[str, List[Box]], threshold: float) -> float:
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    matched = {name: np.zeros(len(boxes), dtype=bool) for name, boxes in gt_by_image.items()}
    tp = np.zeros(len(detections), dtype=np.float64)
    fp = np.zeros(len(detections), dtype=np.float64)

    for i, det in enumerate(detections):
        gts = gt_by_image.get(det["image"], [])
        if not gts:
            fp[i] = 1
            continue
        ious = np.asarray([iou_xywh(det["bbox"], gt) for gt in gts])
        best_idx = int(ious.argmax()) if len(ious) else -1
        if best_idx >= 0 and ious[best_idx] >= threshold and not matched[det["image"]][best_idx]:
            tp[i] = 1
            matched[det["image"]][best_idx] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(1, total_gt)
    precision = cum_tp / np.maximum(1, cum_tp + cum_fp)
    return average_precision_from_pr(recall, precision)


def compute_mabo(proposal_by_image: Dict[str, List[Box]], gt_by_image: Dict[str, List[Box]]) -> float:
    best = []
    for image_name, gt_boxes in gt_by_image.items():
        props = proposal_by_image.get(image_name, [])
        for gt in gt_boxes:
            best.append(max((iou_xywh(prop, gt) for prop in props), default=0.0))
    return float(np.mean(best)) if best else 0.0


def compute_proposal_recall(proposal_by_image: Dict[str, List[Box]], gt_by_image: Dict[str, List[Box]]) -> dict:
    out = {}
    for threshold in [0.50, 0.75]:
        hits = []
        for image_name, gt_boxes in gt_by_image.items():
            props = proposal_by_image.get(image_name, [])
            for gt in gt_boxes:
                hits.append(max((iou_xywh(prop, gt) for prop in props), default=0.0) >= threshold)
        out[str(threshold)] = float(np.mean(hits)) if hits else 0.0
    return out


def load_proposals_json(proposals_json: Path) -> Dict[str, List[Box]]:
    with open(proposals_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {item["file_name"]: item["proposals"] for item in payload["results"]}


def draw_detections(image: np.ndarray, detections: Sequence[dict], out_path: Path, max_draw: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for det in list(detections)[:max_draw]:
        x, y, w, h = det["bbox"]
        score = det["score"]
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(x, max(0, y - 4), f"balloon {score:.2f}", color="red", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def evaluate_detection(
    data_root: Path,
    proposals_dir: Path,
    model_path: Path,
    results_dir: Path,
    scale: float = 200.0,
    sigma: float = 0.9,
    min_size: int = 50,
    max_props: int = 1500,
    score_thr: float = 0.5,
    nms_iou: float = 0.3,
    top_k: int = 200,
    device: str = "cpu",
    weights_mode: str = "default",
    batch_size: int = 64,
) -> dict:
    test_prop = proposals_dir / "proposals_test.json"
    if not test_prop.exists():
        process_split_proposals(data_root, "test", proposals_dir, scale, sigma, min_size, max_props)
    proposal_by_image = load_proposals_json(test_prop)

    images, gt_by_id = load_coco(data_root / "test" / "_annotations.coco.json")
    gt_by_image: Dict[str, List[Box]] = {}
    all_detections: List[dict] = []
    vis_dir = results_dir / "visualizations"
    ensure_dir(vis_dir)

    for img_id, meta in images.items():
        image_name = meta["file_name"]
        img = read_image(data_root / "test" / image_name)
        gt_by_image[image_name] = gt_by_id.get(img_id, [])
        props = proposal_by_image.get(image_name, [])
        if not props:
            continue
        scores = score_proposals(img, props, model_path, device=device, weights_mode=weights_mode, batch_size=batch_size)
        order = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        boxes_top = [props[int(i)] for i in order]
        scores_top = [float(scores[int(i)]) for i in order]
        filtered = [(box, score) for box, score in zip(boxes_top, scores_top) if score >= score_thr]
        if filtered:
            boxes_f, scores_f = zip(*filtered)
            keep = nms_xywh(list(boxes_f), list(scores_f), iou_thresh=nms_iou)
            final_dets = [{"image": image_name, "bbox": list(boxes_f[i]), "score": float(scores_f[i])} for i in keep]
        else:
            final_dets = []
        all_detections.extend(final_dets)
        draw_detections(img, final_dets, vis_dir / f"{Path(image_name).stem}_detections.jpg")
        print(f"[eval] {image_name}: props={len(props)} final={len(final_dets)}")

    thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]
    ap_by_thr = {str(t): ap_at_iou(all_detections, gt_by_image, t) for t in thresholds}
    metrics = {
        "method": "resnet18_svm_detection_pipeline",
        "classification_model": str(model_path),
        "score_threshold": score_thr,
        "nms_iou": nms_iou,
        "top_k": top_k,
        "coco_style_mAP_0.50_0.95": float(np.mean(list(ap_by_thr.values()))),
        "AP_by_IoU_threshold": ap_by_thr,
        "MABO_test_proposals": compute_mabo(proposal_by_image, gt_by_image),
        "proposal_recall": compute_proposal_recall(proposal_by_image, gt_by_image),
        "num_test_detections_after_threshold_nms": len(all_detections),
        "num_test_ground_truth_boxes": sum(len(v) for v in gt_by_image.values()),
    }
    ensure_dir(results_dir)
    with open(results_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def infer_single_image(
    image_path: Path,
    model_path: Path,
    out_dir: Path,
    scale: float = 200.0,
    sigma: float = 0.9,
    min_size: int = 50,
    max_props: int = 1500,
    score_thr: float = 0.5,
    nms_iou: float = 0.3,
    top_k: int = 200,
    device: str = "cpu",
    weights_mode: str = "default",
    batch_size: int = 64,
) -> Path:
    image = read_image(image_path)
    props = run_selective_search_single_config(
        image_path, scale=scale, sigma=sigma, min_size=min_size, max_props=max_props
    )
    if props:
        scores = score_proposals(image, props, model_path, device=device, weights_mode=weights_mode, batch_size=batch_size)
        order = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        boxes_top = [props[int(i)] for i in order]
        scores_top = [float(scores[int(i)]) for i in order]
        filtered = [(box, score) for box, score in zip(boxes_top, scores_top) if score >= score_thr]
        if filtered:
            boxes_f, scores_f = zip(*filtered)
            keep = nms_xywh(list(boxes_f), list(scores_f), iou_thresh=nms_iou)
            detections = [{"image": image_path.name, "bbox": list(boxes_f[i]), "score": float(scores_f[i])} for i in keep]
        else:
            detections = []
    else:
        detections = []
    ensure_dir(out_dir)
    out_image = out_dir / f"{image_path.stem}_detections.jpg"
    out_json = out_dir / f"{image_path.stem}_detections.json"
    draw_detections(image, detections, out_image)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
    print(f"Saved inference image: {out_image}")
    print(f"Saved inference JSON: {out_json}")
    return out_image


# CLI orchestration

def run_all(args: argparse.Namespace, project_root: Path) -> None:
    data_root = Path(args.data_root) if args.data_root else prepare_dataset(project_root)
    results_dir = project_root / "results" / "balloon_pipeline"
    proposals_dir = results_dir / "proposals"
    samples_dir = results_dir / "samples"
    features_dir = results_dir / "features"
    model_out = results_dir / "models" / "svm_resnet18.joblib"

    for split in ["train", "valid", "test"]:
        process_split_proposals(data_root, split, proposals_dir, args.scale, args.sigma, args.min_size, args.max_props)

    for split in ["train", "valid"]:
        build_samples_for_split(
            data_root,
            split,
            proposals_dir / f"proposals_{split}.json",
            samples_dir,
            tp=args.tp,
            tn=args.tn,
            save_crops=True,
            crop_size=args.crop_size,
            max_neg_per_image=args.max_neg_per_image,
            add_gt_positives=True,
        )

    for split in ["train", "valid"]:
        extract_features_resnet18(
            samples_dir / split / f"samples_{split}.json",
            features_dir,
            data_root,
            split,
            batch_size=args.batch_size,
            device=args.device,
            weights_mode=args.resnet_weights,
        )

    train_svm(
        features_dir,
        model_out,
        kernel=args.kernel,
        C=args.C,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        probability=args.probability,
    )

    evaluate_detection(
        data_root,
        proposals_dir,
        model_out,
        results_dir,
        scale=args.scale,
        sigma=args.sigma,
        min_size=args.min_size,
        max_props=args.max_props,
        score_thr=args.score_thr,
        nms_iou=args.nms_iou,
        top_k=args.top_k,
        device=args.device,
        weights_mode=args.resnet_weights,
        batch_size=args.batch_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection-pipeline 5.2 balloon detector: Selective Search + ResNet18 + SVM")
    parser.add_argument("--mode", choices=["all", "proposals", "samples", "features", "train", "eval", "infer"], default="all")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])

    parser.add_argument("--scale", type=float, default=200.0)
    parser.add_argument("--sigma", type=float, default=0.9)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--max-props", type=int, default=1500)

    parser.add_argument("--tp", type=float, default=0.75, help="IoU threshold for positives")
    parser.add_argument("--tn", type=float, default=0.25, help="IoU threshold for negatives")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-neg-per-image", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resnet-weights", choices=["default", "none"], default="default")

    parser.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--probability", action="store_true")
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--pca-components", type=int, default=128)

    parser.add_argument("--score-thr", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=200)

    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--proposals-dir", type=Path, default=None)
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None, help="Image path for --mode infer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root) if args.data_root else prepare_dataset(project_root)

    if args.mode == "all":
        run_all(args, project_root)
        return

    if args.mode == "proposals":
        out_dir = args.out_dir or (project_root / "results" / "balloon_pipeline" / "proposals")
        for split in args.splits:
            process_split_proposals(data_root, split, out_dir, args.scale, args.sigma, args.min_size, args.max_props)

    elif args.mode == "samples":
        proposals_dir = args.proposals_dir or (project_root / "results" / "balloon_pipeline" / "proposals")
        samples_dir = args.samples_dir or (project_root / "results" / "balloon_pipeline" / "samples")
        for split in args.splits:
            build_samples_for_split(
                data_root,
                split,
                proposals_dir / f"proposals_{split}.json",
                samples_dir,
                tp=args.tp,
                tn=args.tn,
                save_crops=True,
                crop_size=args.crop_size,
                max_neg_per_image=args.max_neg_per_image,
                add_gt_positives=True,
            )

    elif args.mode == "features":
        samples_dir = args.samples_dir or (project_root / "results" / "balloon_pipeline" / "samples")
        features_dir = args.features_dir or (project_root / "results" / "balloon_pipeline" / "features")
        for split in args.splits:
            extract_features_resnet18(
                samples_dir / split / f"samples_{split}.json",
                features_dir,
                data_root,
                split,
                batch_size=args.batch_size,
                device=args.device,
                weights_mode=args.resnet_weights,
            )

    elif args.mode == "train":
        features_dir = args.features_dir or (project_root / "results" / "balloon_pipeline" / "features")
        model_out = args.model_out or (project_root / "results" / "balloon_pipeline" / "models" / "svm_resnet18.joblib")
        train_svm(features_dir, model_out, kernel=args.kernel, C=args.C, use_pca=args.use_pca, pca_components=args.pca_components, probability=args.probability)

    elif args.mode == "eval":
        proposals_dir = args.proposals_dir or (project_root / "results" / "balloon_pipeline" / "proposals")
        model_path = args.model or (project_root / "results" / "balloon_pipeline" / "models" / "svm_resnet18.joblib")
        results_dir = args.out_dir or (project_root / "results" / "balloon_pipeline")
        evaluate_detection(
            data_root,
            proposals_dir,
            model_path,
            results_dir,
            scale=args.scale,
            sigma=args.sigma,
            min_size=args.min_size,
            max_props=args.max_props,
            score_thr=args.score_thr,
            nms_iou=args.nms_iou,
            top_k=args.top_k,
            device=args.device,
            weights_mode=args.resnet_weights,
            batch_size=args.batch_size,
        )

    elif args.mode == "infer":
        if args.image is None:
            raise SystemExit("--image is required for --mode infer")
        model_path = args.model or (project_root / "results" / "balloon_pipeline" / "models" / "svm_resnet18.joblib")
        out_dir = args.out_dir or (project_root / "results" / "balloon_pipeline" / "inference")
        infer_single_image(
            args.image,
            model_path,
            out_dir,
            scale=args.scale,
            sigma=args.sigma,
            min_size=args.min_size,
            max_props=args.max_props,
            score_thr=args.score_thr,
            nms_iou=args.nms_iou,
            top_k=args.top_k,
            device=args.device,
            weights_mode=args.resnet_weights,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
