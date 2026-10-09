"""
Selective Search implementation for the Visual Recognition for Humanities project.

The implementation follows the simplified Uijlings et al. pipeline:
  1. Felzenszwalb initial over-segmentation.
  2. Region descriptors: colour histogram, texture histogram, size and fill.
  3. Hierarchical region merging by maximum similarity.
  4. Region proposals as bounding boxes around all generated regions.

"""
from __future__ import division

import itertools
from typing import Dict, Iterable, List, Tuple

import numpy as np
import skimage.color
import skimage.feature
import skimage.segmentation
import skimage.util


Region = Dict[str, object]
Box = Tuple[int, int, int, int]


def _as_float_image(im: np.ndarray) -> np.ndarray:
    """Return an RGB image in [0, 1] as float64."""
    if im.dtype.kind in "ui":
        return skimage.util.img_as_float(im)
    return np.clip(im.astype(np.float64), 0.0, 1.0)


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 5.1: Segment smallest regions with the Felzenszwalb algorithm.
    """
    image = _as_float_image(im_orig)
    labels = skimage.segmentation.felzenszwalb(
        image, scale=scale, sigma=sigma, min_size=min_size
    )
    return np.dstack((image, labels.astype(np.float64)))


def sim_colour(r1, r2):
    """Task 5.2: histogram-intersection colour similarity."""
    return float(np.minimum(r1["hist_c"], r2["hist_c"]).sum())


def sim_texture(r1, r2):
    """Task 5.2: histogram-intersection texture similarity."""
    return float(np.minimum(r1["hist_t"], r2["hist_t"]).sum())


def sim_size(r1, r2, imsize):
    """Task 5.2: prefer merging small regions first."""
    return 1.0 - float(r1["size"] + r2["size"]) / float(imsize)


def sim_fill(r1, r2, imsize):
    """Task 5.2: prefer regions that tightly fill their joint bounding box."""
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])
    bbsize = (max_x - min_x + 1) * (max_y - min_y + 1)
    return 1.0 - float(bbsize - r1["size"] - r2["size"]) / float(imsize)


def calc_sim(r1, r2, imsize):
    return (
        sim_colour(r1, r2)
        + sim_texture(r1, r2)
        + sim_size(r1, r2, imsize)
        + sim_fill(r1, r2, imsize)
    )


def calc_colour_hist(img):
    """
    Task 5.2.5.1: HSV colour histogram.
    """
    BINS = 25
    hist_parts = []
    if img.size == 0:
        return np.zeros(BINS * 3, dtype=np.float64)

    # The caller passes HSV pixels. Values are in [0, 1].
    for ch in range(3):
        h, _ = np.histogram(img[:, :, ch].ravel(), bins=BINS, range=(0.0, 1.0))
        hist_parts.append(h.astype(np.float64))
    hist = np.concatenate(hist_parts)
    denom = hist.sum()
    if denom > 0:
        hist /= denom
    return hist


def calc_texture_gradient(img):
    """
    Task 5.2.5.2: texture gradient for the whole image.

    The original paper uses Gaussian derivatives. The exercise suggests LBP, so
    we compute a uniform local-binary-pattern image for each colour channel.
    """
    rgb = _as_float_image(img[:, :, :3])
    ret = np.zeros_like(rgb, dtype=np.float64)
    for ch in range(3):
        channel = skimage.util.img_as_ubyte(rgb[:, :, ch])
        ret[:, :, ch] = skimage.feature.local_binary_pattern(
            channel, P=8, R=1, method="uniform"
        )
    return ret


def calc_texture_hist(img):
    """
    Task 5.2.5.3: texture histogram.

    We concatenate 10-bin histograms for the LBP response in each channel and
    L1-normalize the full vector.
    """
    BINS = 10
    hist_parts = []
    if img.size == 0:
        return np.zeros(BINS * 3, dtype=np.float64)
    for ch in range(3):
        # Uniform LBP with P=8 has values in [0, 9].
        h, _ = np.histogram(img[:, :, ch].ravel(), bins=BINS, range=(0, BINS))
        hist_parts.append(h.astype(np.float64))
    hist = np.concatenate(hist_parts)
    denom = hist.sum()
    if denom > 0:
        hist /= denom
    return hist


def extract_regions(img):
    """
    Task 5.2.5: Generate the region data structure R.

    Each region stores bounding-box coordinates, pixel count, labels, colour
    histogram and texture histogram. Coordinates are inclusive internally.
    """
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    texture_gradient = calc_texture_gradient(img[:, :, :3])
    labels = img[:, :, 3].astype(np.int64)

    R: Dict[float, Region] = {}
    for label in np.unique(labels):
        mask = labels == label
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        region_hsv = hsv[mask].reshape((-1, 1, 3))
        region_texture = texture_gradient[mask].reshape((-1, 1, 3))
        key = float(label)
        R[key] = {
            "min_x": int(xs.min()),
            "min_y": int(ys.min()),
            "max_x": int(xs.max()),
            "max_y": int(ys.max()),
            "size": int(mask.sum()),
            "labels": [key],
            "hist_c": calc_colour_hist(region_hsv),
            "hist_t": calc_texture_hist(region_texture),
        }
    return R


def _boxes_touch_or_overlap(a: Region, b: Region) -> bool:
    """True if two inclusive bounding boxes overlap or touch."""
    return not (
        a["max_x"] < b["min_x"]
        or b["max_x"] < a["min_x"]
        or a["max_y"] < b["min_y"]
        or b["max_y"] < a["min_y"]
    )


def extract_neighbours(regions):
    """
    Task 5.3: Extract neighbouring regions.
    """
    neighbours = []
    keys = list(regions.keys())
    for ai, bi in itertools.combinations(keys, 2):
        if _boxes_touch_or_overlap(regions[ai], regions[bi]):
            neighbours.append(((ai, regions[ai]), (bi, regions[bi])))
    return neighbours


def merge_regions(r1, r2):
    """Task 5.4: Merge two regions and combine their descriptors."""
    new_size = int(r1["size"] + r2["size"])
    if new_size <= 0:
        w1 = w2 = 0.5
    else:
        w1 = float(r1["size"]) / float(new_size)
        w2 = float(r2["size"]) / float(new_size)

    return {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "labels": list(r1["labels"]) + list(r2["labels"]),
        "hist_c": w1 * r1["hist_c"] + w2 * r2["hist_c"],
        "hist_t": w1 * r1["hist_t"] + w2 * r2["hist_t"],
    }


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    """
    Simplified Selective Search for Object Recognition.
    """
    assert image_orig.ndim == 3 and image_orig.shape[2] == 3, (
        "Please use image with three channels."
    )
    imsize = image_orig.shape[0] * image_orig.shape[1]

    image = generate_segments(image_orig, scale, sigma, min_size)
    if image is None:
        return None, []

    R = extract_regions(image)
    neighbours = extract_neighbours(R)

    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions.
    next_label = (max(R.keys()) + 1.0) if R else 0.0
    while S:
        # Highest similarity pair.
        i, j = max(S.items(), key=lambda item: item[1])[0]

        # Task 5.4: Merge corresponding regions.
        t = next_label
        next_label += 1.0
        R[t] = merge_regions(R[i], R[j])

        # Tasks 5.5 and 5.6: mark and remove old similarities involving i or j.
        affected = set()
        keys_to_remove = []
        for (a, b) in list(S.keys()):
            if a in (i, j) or b in (i, j):
                keys_to_remove.append((a, b))
                if a not in (i, j):
                    affected.add(a)
                if b not in (i, j):
                    affected.add(b)
        for key in keys_to_remove:
            del S[key]

        # Task 5.7: calculate similarities between the new region and the
        # regions that were neighbours of either old region.
        for k in affected:
            if k == t or k not in R:
                continue
            if _boxes_touch_or_overlap(R[t], R[k]):
                S[(min(t, k), max(t, k))] = calc_sim(R[t], R[k], imsize)

    # Task 5.8: Generate the final region proposals.
    regions = []
    seen = set()
    for region in R.values():
        x = int(region["min_x"])
        y = int(region["min_y"])
        w = int(region["max_x"] - region["min_x"] + 1)
        h = int(region["max_y"] - region["min_y"] + 1)
        rect = (x, y, w, h)
        if w <= 0 or h <= 0 or rect in seen:
            continue
        seen.add(rect)
        regions.append({
            "rect": rect,
            "size": int(region["size"]),
            "labels": list(region["labels"]),
        })

    return image, regions
