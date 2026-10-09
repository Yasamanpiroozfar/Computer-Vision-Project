"""Utilities shared by selective-search visualization and balloon detection."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # x, y, width, height


def box_area(box: Box) -> float:
    return max(0, box[2]) * max(0, box[3])


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    denom = box_area(box_a) + box_area(box_b) - inter
    return float(inter / denom) if denom > 0 else 0.0


def clip_box(box: Box, width: int, height: int) -> Box:
    """Clip a box so that the drawn rectangle stays fully inside the image."""
    x, y, w, h = box
    if width <= 1 or height <= 1:
        return (0, 0, 1, 1)
    x1 = max(0, min(width - 2, int(round(x))))
    y1 = max(0, min(height - 2, int(round(y))))
    x2 = max(x1 + 1, min(width - 1, int(round(x + w))))
    y2 = max(y1 + 1, min(height - 1, int(round(y + h))))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def filter_proposals(
    regions: Iterable[dict],
    image_shape: Sequence[int],
    min_area: int = 200,
    max_area_ratio: float = 0.95,
    min_side: int = 8,
    max_aspect_ratio: float = 6.0,
    max_proposals: int | None = 600,
) -> List[Box]:
    """Filter raw selective-search regions into useful candidate boxes."""
    h, w = int(image_shape[0]), int(image_shape[1])
    image_area = h * w
    boxes: List[Box] = []
    seen = set()

    for r in regions:
        x, y, bw, bh = map(int, r["rect"])
        if bw < min_side or bh < min_side:
            continue
        area = bw * bh
        if area < min_area or area > max_area_ratio * image_area:
            continue
        ar = max(bw / max(1, bh), bh / max(1, bw))
        if ar > max_aspect_ratio:
            continue
        box = clip_box((x, y, bw, bh), w, h)
        if box in seen:
            continue
        seen.add(box)
        boxes.append(box)

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    if max_proposals is not None:
        boxes = boxes[:max_proposals]
    return boxes


def nms(boxes: Sequence[Box], scores: Sequence[float], iou_threshold: float = 0.35) -> List[int]:
    """Non-maximum suppression. Returns kept indices."""
    if len(boxes) == 0:
        return []
    order = np.argsort(np.asarray(scores))[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        remaining = []
        for j in order[1:]:
            if iou(boxes[i], boxes[int(j)]) <= iou_threshold:
                remaining.append(int(j))
        order = np.asarray(remaining, dtype=int)
    return keep


def scale_box(box: Box, sx: float, sy: float) -> Box:
    x, y, w, h = box
    return (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy)))


def resize_for_search(image, max_dim: int = 640):
    """Resize image for faster proposal generation; return resized image and scale factors back to original."""
    import skimage.transform

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0, 1.0
    scale = max_dim / float(longest)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = skimage.transform.resize(
        image, (new_h, new_w), preserve_range=True, anti_aliasing=True
    ).astype(image.dtype)
    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, sx, sy
