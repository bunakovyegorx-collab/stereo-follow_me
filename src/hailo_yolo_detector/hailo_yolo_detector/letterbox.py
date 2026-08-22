"""Pure preprocessing/postprocessing math, testable without ROS or HailoRT."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """Everything needed to map letterboxed-model coords back to source pixels."""

    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int


def letterbox(image_rgb: np.ndarray, target_size: int) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio onto a target_size x target_size canvas,
    padding with 114/114/114 gray (Ultralytics/YOLO convention), centered.
    """
    src_h, src_w = image_rgb.shape[:2]
    scale = min(target_size / src_w, target_size / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    # Guard rounding so top+bottom+new_h == target_size exactly (and same for width).
    bottom = target_size - new_h - top
    right = target_size - new_w - left

    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return canvas, LetterboxTransform(scale, left, top, src_w, src_h)


def undo_letterbox_box(
    x1: float, y1: float, x2: float, y2: float, transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """Map a box in letterboxed-canvas pixel coords back to source-image pixel coords."""
    sx1 = (x1 - transform.pad_x) / transform.scale
    sy1 = (y1 - transform.pad_y) / transform.scale
    sx2 = (x2 - transform.pad_x) / transform.scale
    sy2 = (y2 - transform.pad_y) / transform.scale
    sx1 = max(0.0, min(sx1, transform.src_w))
    sy1 = max(0.0, min(sy1, transform.src_h))
    sx2 = max(0.0, min(sx2, transform.src_w))
    sy2 = max(0.0, min(sy2, transform.src_h))
    return sx1, sy1, sx2, sy2
