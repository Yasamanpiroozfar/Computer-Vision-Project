"""One-command runner for Exercise 5.2.

This script runs the full detection pipeline in the safer step-by-step order:
1) Selective Search proposals for train/valid/test
2) Positive/negative sample generation for train/valid
3) ResNet18 feature extraction for train/valid
4) SVM training
5) Test-set evaluation with COCO-style mAP and MABO
6) Clean Selective Search preview images for the test set


Run from the project root:
    python code\pipeline.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(description: str, args: list[str]) -> None:
    """Run one pipeline command and stop immediately if it fails."""
    print("\n" + "=" * 80)
    print(description)
    print("COMMAND:", " ".join(args))
    print("=" * 80)
    subprocess.run(args, check=True)


def reset_clean_results(results_dir: Path) -> None:
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "selective_search_test").mkdir(exist_ok=True)
    (results_dir / "visualizations").mkdir(exist_ok=True)


def draw_selective_search_previews(project_root: Path, proposals_json: Path, out_dir: Path) -> None:
    """Create Selective Search preview images from the proposal JSON used by the classifier."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import skimage.io

    data_root = project_root / "data" / "balloon_dataset"
    if not (data_root / "test").exists():
        data_root = project_root / "data" / "balloon_dataset" / "test"

    with open(proposals_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = out_dir / "proposal_counts_test.csv"
    with open(counts_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "num_proposals", "num_drawn"])

        for item in payload.get("results", []):
            image_name = item["file_name"]
            img_path = project_root / "data" / "balloon_dataset" / "test" / image_name
            if not img_path.exists():
                img_path = next((project_root / "data").rglob(image_name))
            image = skimage.io.imread(str(img_path))
            boxes = item.get("proposals", [])
            drawn = boxes  

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image)
            for x, y, w, h in drawn:
                rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=0.7, alpha=0.85)
                ax.add_patch(rect)
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            out_path = out_dir / f"{Path(image_name).stem}_selective_search.jpg"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            writer.writerow([image_name, len(boxes), len(drawn)])
            print(f"[selective preview] {image_name}: proposals used by classifier={len(boxes)} drawn={len(drawn)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full 5.2 detection pipeline with one command.")
    parser.add_argument("--tp", type=float, default=0.75, help="IoU threshold for positive samples")
    parser.add_argument("--tn", type=float, default=0.25, help="IoU threshold for negative samples")
    parser.add_argument("--C", type=float, default=0.01, help="SVM regularization parameter")
    parser.add_argument("--resnet-weights", choices=["default", "none"], default="default")
    parser.add_argument("--feature-batch-size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Batch size for final evaluation")
    parser.add_argument("--score-thr", type=float, default=0.4, help="Detection score threshold")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="NMS IoU threshold")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; the pipeline overwrites visible result previews.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    balloon_script = project_root / "code" / "balloon_pipeline.py"
    py = sys.executable

    results_dir = project_root / "results" / "balloon_pipeline"
    work_dir = project_root / "work" / "balloon_pipeline"
    proposals_dir = work_dir / "proposals"
    samples_dir = work_dir / "samples"
    features_dir = work_dir / "features"
    models_dir = work_dir / "models"
    model_path = models_dir / "svm_resnet18.joblib"

    reset_clean_results(results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        "Step 1/6: Generate Selective Search proposals for train, valid and test",
        [py, str(balloon_script), "--mode", "proposals", "--splits", "train", "valid", "test", "--out-dir", str(proposals_dir)],
    )

    run_step(
        "Step 2/6: Build positive/negative samples from IoU thresholds",
        [
            py,
            str(balloon_script),
            "--mode",
            "samples",
            "--splits",
            "train",
            "valid",
            "--proposals-dir",
            str(proposals_dir),
            "--samples-dir",
            str(samples_dir),
            "--tp",
            str(args.tp),
            "--tn",
            str(args.tn),
        ],
    )

    run_step(
        "Step 3/6: Extract pretrained ResNet18 features",
        [
            py,
            str(balloon_script),
            "--mode",
            "features",
            "--splits",
            "train",
            "valid",
            "--samples-dir",
            str(samples_dir),
            "--features-dir",
            str(features_dir),
            "--resnet-weights",
            args.resnet_weights,
            "--batch-size",
            str(args.feature_batch_size),
        ],
    )

    run_step(
        "Step 4/6: Train SVM classifier",
        [
            py,
            str(balloon_script),
            "--mode",
            "train",
            "--features-dir",
            str(features_dir),
            "--model-out",
            str(model_path),
            "--C",
            str(args.C),
        ],
    )

    run_step(
        "Step 5/6: Evaluate detections on the test set",
        [
            py,
            str(balloon_script),
            "--mode",
            "eval",
            "--proposals-dir",
            str(proposals_dir),
            "--model",
            str(model_path),
            "--out-dir",
            str(results_dir),
            "--resnet-weights",
            args.resnet_weights,
            "--score-thr",
            str(args.score_thr),
            "--nms-iou",
            str(args.nms_iou),
            "--batch-size",
            str(args.eval_batch_size),
        ],
    )

    print("\n" + "=" * 80)
    print("Step 6/6: Create Selective Search previews from the proposals used by the classifier")
    print("=" * 80)
    ss_out = results_dir / "selective_search_test"
    draw_selective_search_previews(project_root, proposals_dir / "proposals_test.json", ss_out)
    shutil.copy2(proposals_dir / "proposals_test.json", ss_out / "proposals_test_used_by_classifier.json")

    print("\nDONE. The visible result folder is now clean:")
    print(results_dir)
    print("\nOn Windows PowerShell:")
    print(r"type results\balloon_pipeline\evaluation.json")
    print(r"explorer results\balloon_pipeline\selective_search_test")
    print(r"explorer results\balloon_pipeline\visualizations")
    print("\nIntermediate files are stored here:")
    print(work_dir)


if __name__ == "__main__":
    main()
