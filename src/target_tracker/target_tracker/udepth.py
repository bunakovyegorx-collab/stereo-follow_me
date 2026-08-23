"""U-depth detector: obstacles from a per-column depth histogram.

Ported from Zhefan-Xu/onboard_detector (uvDetector.cpp). The U-map is a
top-down view: same width as the image, vertical axis = distance from the
camera, cell value = how many pixels of that column sit at that distance.
A vertical surface (a person, a pole) produces a compact bright blob because
many pixels of one column share a depth; the floor smears across every bin.

The original walks the rows with a hand-rolled union-find. Here the same
grouping falls out of ``scipy.ndimage.label`` on the thresholded 2D map.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)

# The original assumes the far side of an object is cut off by occlusion, so
# the depth slab is stretched backwards before the height search.
_FAR_STRETCH = 1.3


@dataclass(frozen=True)
class UBox:
    """One U-depth detection, in the camera optical frame."""

    center: np.ndarray   # (3,) x, y, z in optical metres
    size: np.ndarray     # (3,) width, height, depth-thickness
    u_rect: tuple[int, int, int, int]   # x, y, w, h inside the U-map (debug)


def build_u_map(
    depth: np.ndarray,
    *,
    z_min: float = 1.2,
    z_max: float = 5.5,
    bins: int = 64,
    col_scale: float = 0.5,
    blur: bool = True,
) -> np.ndarray:
    """Return the ``(bins, scaled_width)`` uint16 per-column depth histogram."""
    if depth.ndim != 2:
        raise ValueError('expected a 2D depth image')
    if bins < 4:
        raise ValueError('bins must be >= 4')
    if not 0.0 < col_scale <= 1.0:
        raise ValueError('col_scale must be in (0, 1]')
    if not z_min < z_max:
        raise ValueError('expected z_min < z_max')

    width = max(1, int(round(depth.shape[1] * col_scale)))
    scaled = cv2.resize(
        np.asarray(depth, dtype=np.float32), (width, depth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    valid = np.isfinite(scaled) & (scaled >= z_min) & (scaled < z_max)
    # NaNs would poison the int cast, so bin the finite values only.
    ratio = np.where(valid, (scaled - z_min) / (z_max - z_min) * bins, 0.0)
    bin_index = ratio.astype(np.int32)
    np.clip(bin_index, 0, bins - 1, out=bin_index)

    columns = np.broadcast_to(np.arange(width, dtype=np.int32), scaled.shape)
    flat = bin_index[valid] * width + columns[valid]
    u_map = np.bincount(flat, minlength=bins * width).reshape(bins, width)
    u_map = np.minimum(u_map, 65535).astype(np.uint16)

    if blur:
        # Same intent as the original GaussianBlur(5, 9): rejoin a mass that
        # stereo noise split across neighbouring depth bins.
        u_map = cv2.GaussianBlur(u_map.astype(np.float32), (5, 9), 2.0, 2.0)
        u_map = u_map.astype(np.uint16)
    return u_map


def extract_u_boxes(
    u_map: np.ndarray,
    *,
    min_count: int = 6,
    min_width: int = 3,
    min_area: int = 12,
) -> list[tuple[int, int, int, int]]:
    """Threshold the U-map and return connected blobs as ``(x, y, w, h)``."""
    if u_map.ndim != 2:
        raise ValueError('expected a 2D U-map')

    mask = u_map >= min_count
    labels, count = ndimage.label(mask, structure=_CONNECTIVITY_8)
    if count == 0:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for y_slice, x_slice in ndimage.find_objects(labels):
        x, y = int(x_slice.start), int(y_slice.start)
        w = int(x_slice.stop) - x
        h = int(y_slice.stop) - y
        if w < min_width or w * h < min_area:
            continue
        boxes.append((x, y, w, h))
    return boxes


def u_box_to_3d(
    depth: np.ndarray,
    u_rect: tuple[int, int, int, int],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_min: float = 1.2,
    z_max: float = 5.5,
    bins: int = 64,
    col_scale: float = 0.5,
    min_run: int = 8,
) -> UBox | None:
    """Recover object height from the depth image and build a 3D box.

    The U-map fixes the object's width and depth-thickness but says nothing
    about its height, so this repeats the original ``extract_3Dbox`` pass: in
    the columns the blob covers, find the longest unbroken vertical run of
    pixels whose depth falls inside the blob's slab.
    """
    if depth.ndim != 2:
        raise ValueError('expected a 2D depth image')
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('expected positive focal lengths')
    if not 0.0 < col_scale <= 1.0:
        raise ValueError('col_scale must be in (0, 1]')

    x, y, w, h = u_rect
    height_px, width_px = depth.shape
    bin_size = (z_max - z_min) / bins
    near = z_min + y * bin_size
    far = z_min + (y + h) * bin_size
    far = near + (far - near) * _FAR_STRETCH

    u0 = int(np.clip(round(x / col_scale), 0, width_px - 1))
    u1 = int(np.clip(round((x + w) / col_scale), u0 + 1, width_px))

    strip = depth[:, u0:u1]
    band = np.isfinite(strip) & (strip >= near) & (strip <= far)
    rows = band.any(axis=1)
    run = _longest_run(rows)
    if run is None:
        return None
    v_up, v_down = run
    if (v_down - v_up + 1) < min_run:
        return None

    z = (near + far) * 0.5
    u_center = (u0 + u1) * 0.5
    v_center = (v_up + v_down + 1) * 0.5
    x_c = (u_center - cx) * z / fx
    y_c = (v_center - cy) * z / fy
    box_width = (u1 - u0) * z / fx
    box_height = (v_down - v_up + 1) * z / fy

    return UBox(
        center=np.array([x_c, y_c, z], dtype=np.float32),
        size=np.array(
            [max(box_width, 0.05), max(box_height, 0.05), max(far - near, 0.05)],
            dtype=np.float32,
        ),
        u_rect=(int(x), int(y), int(w), int(h)),
    )


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """Return the inclusive bounds of the longest True run, or None."""
    if not bool(flags.any()):
        return None
    # Pad with False so every run has an explicit rising and falling edge.
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]
    lengths = stops - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(stops[best]) - 1
