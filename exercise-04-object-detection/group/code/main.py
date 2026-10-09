"""Run Exercise 5.1 selective search on all humanities images.

Usage from the project root:
    python code/main.py

"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image, ImageDraw

from proposal_utils import resize_for_search, scale_box, clip_box
from selective_search import selective_search


DATASETS = ["chrisarch", "arthist", "classarch"]
Box = Tuple[int, int, int, int]

# Uijlings et al. report a strong single strategy with HSV, C+T+S+F and k=100.
SEARCH_MAX_DIM = 420
FELZENSZWALB_K = 100
SIGMA = 0.8
MIN_SIZE = 20
COLOUR_SPACE = "hsv"
MIN_REGION_SIZE = 2000
MAX_ASPECT_RATIO = 1.5


def draw_boxes(image, boxes: Sequence[Box], out_path: Path, linewidth: float = 0.7) -> None:
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(image.shape[0] - 1, 0)
    for x, y, w, h in boxes:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=linewidth, clip_on=True)
        rect.set_clip_path(ax.patch)
        ax.add_patch(rect)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_contact_sheet(image_paths: Sequence[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((280, 210))
        canvas = Image.new("RGB", (280, 240), "white")
        canvas.paste(im, ((280 - im.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 215), path.stem, fill="black")
        thumbs.append(canvas)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 280, rows * 240), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 280, (i // cols) * 240))
    sheet.save(out_path)


def filter_regions_moderate(regions, sx: float, sy: float, image_width: int, image_height: int) -> list[Box]:
    boxes: list[Box] = []
    seen_small = set()
    seen_final = set()
    for r in regions:
        x, y, w, h = map(int, r["rect"])
        if w <= 0 or h <= 0:
            continue
        rect = (x, y, w, h)
        if rect in seen_small:
            continue
        seen_small.add(rect)
        if int(r.get("size", 0)) < MIN_REGION_SIZE:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > MAX_ASPECT_RATIO:
            continue
        final_box = clip_box(scale_box(rect, sx, sy), image_width, image_height)
        if final_box in seen_final:
            continue
        seen_final.add(final_box)
        boxes.append(final_box)
    return boxes


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "selective_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("*_proposals.jpg"):
        old.unlink()

    rows = []
    output_images = []
    for dataset in DATASETS:
        for image_path in sorted((root / "data" / dataset).glob("*.jpg")):
            image = skimage.io.imread(image_path)
            search_image, sx, sy = resize_for_search(image, max_dim=SEARCH_MAX_DIM)

            _, regions = selective_search(
                search_image,
                scale=FELZENSZWALB_K,
                sigma=SIGMA,
                min_size=MIN_SIZE,
                color_space=COLOUR_SPACE,
            )

            boxes = filter_regions_moderate(regions, sx, sy, image.shape[1], image.shape[0])

            out_path = out_dir / f"{dataset}_{image_path.stem}_proposals.jpg"
            draw_boxes(image, boxes, out_path)
            output_images.append(out_path)

            rows.append({
                "dataset": dataset,
                "image": image_path.name,
                "felzenszwalb_k": FELZENSZWALB_K,
                "colour_space": COLOUR_SPACE,
                "raw_hierarchical_regions": len(regions),
                "moderate_filtered_proposals": len(boxes),
                "min_region_size": MIN_REGION_SIZE,
                "max_aspect_ratio": MAX_ASPECT_RATIO,
                "output": str(out_path.relative_to(root)),
            })
            print(f"{dataset}/{image_path.name}: {len(regions)} raw regions, {len(boxes)} moderate filtered proposals")

    with open(out_dir / "proposal_counts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_contact_sheet(output_images, out_dir / "contact_sheet_moderate_filter.jpg")


if __name__ == "__main__":
    main()
