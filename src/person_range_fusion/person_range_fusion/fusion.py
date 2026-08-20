"""Pure geometry and filtering helpers for person range fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned image bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DepthEstimate:
    """Robust disparity and axial-depth estimate."""

    disparity: float
    z: float
    valid_pixels: int
    valid_fraction: float
    sigma_z: float


@dataclass(frozen=True)
class RangeCandidate:
    """One valid person range before temporal smoothing."""

    bbox: BBox
    z: float


@dataclass(frozen=True)
class TrackedRange:
    """One candidate associated with a lightweight temporal track."""

    track_id: str
    filtered_z: float


@dataclass
class _Track:
    bbox: BBox
    filtered_z: float
    stamp_ns: int


def bbox_iou(first: BBox, second: BBox) -> float:
    """Return intersection-over-union for two boxes."""
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def estimate_person_depth(
    disparity: np.ndarray,
    bbox: BBox,
    *,
    f: float,
    baseline: float,
    min_disparity: float,
    max_disparity: float,
    min_depth: float,
    max_depth: float,
    roi_width_fraction: float = 0.5,
    roi_y_start: float = 0.2,
    roi_y_end: float = 0.7,
    min_valid_pixels: int = 20,
    min_valid_fraction: float = 0.1,
) -> DepthEstimate | None:
    """Estimate Z from the robust median disparity in a person's torso."""
    if disparity.ndim != 2 or f <= 0.0 or baseline <= 0.0:
        raise ValueError('expected a 2D disparity image and positive stereo geometry')
    if not 0.0 < roi_width_fraction <= 1.0:
        raise ValueError('roi_width_fraction must be in (0, 1]')
    if not 0.0 <= roi_y_start < roi_y_end <= 1.0:
        raise ValueError('expected 0 <= roi_y_start < roi_y_end <= 1')

    height, width = disparity.shape
    center_x = (bbox.x1 + bbox.x2) / 2.0
    half_roi_width = max(0.0, bbox.x2 - bbox.x1) * roi_width_fraction / 2.0
    box_height = max(0.0, bbox.y2 - bbox.y1)
    x1 = max(0, min(width, int(np.floor(center_x - half_roi_width))))
    x2 = max(0, min(width, int(np.ceil(center_x + half_roi_width))))
    y1 = max(0, min(height, int(np.floor(bbox.y1 + box_height * roi_y_start))))
    y2 = max(0, min(height, int(np.ceil(bbox.y1 + box_height * roi_y_end))))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = disparity[y1:y2, x1:x2]
    geometry = f * baseline
    lower_disparity = max(0.0, float(min_disparity))
    valid = np.isfinite(roi) & (roi > lower_disparity)
    if max_disparity > lower_disparity:
        valid &= roi <= float(max_disparity)
    safe = np.where(valid, roi, 1.0)
    depths = geometry / safe
    valid &= (depths >= min_depth) & (depths <= max_depth)
    values = roi[valid].astype(np.float64, copy=False)
    valid_fraction = float(values.size) / float(roi.size)
    if values.size < min_valid_pixels or valid_fraction < min_valid_fraction:
        return None

    initial_median = float(np.median(values))
    mad = float(np.median(np.abs(values - initial_median)))
    tolerance = max(0.5, 2.5 * 1.4826 * mad)
    inliers = values[np.abs(values - initial_median) <= tolerance]
    if inliers.size < min_valid_pixels:
        return None

    median_disparity = float(np.median(inliers))
    if median_disparity <= 0.0:
        return None
    z = geometry / median_disparity
    sigma_disparity = max(0.25, 1.4826 * mad)
    sigma_z = geometry * sigma_disparity / (median_disparity ** 2)
    return DepthEstimate(
        disparity=median_disparity,
        z=z,
        valid_pixels=int(inliers.size),
        valid_fraction=valid_fraction,
        sigma_z=sigma_z,
    )


class RangeTracker:
    """Associate boxes by IoU and smooth Z for a stationary camera."""

    def __init__(
        self,
        *,
        alpha: float = 0.4,
        iou_threshold: float = 0.3,
        timeout_sec: float = 0.7,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError('iou_threshold must be in [0, 1]')
        if timeout_sec <= 0.0:
            raise ValueError('timeout_sec must be positive')
        self._alpha = alpha
        self._iou_threshold = iou_threshold
        self._timeout_ns = int(timeout_sec * 1_000_000_000)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(
        self,
        candidates: list[RangeCandidate],
        stamp_ns: int,
    ) -> list[TrackedRange]:
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if stamp_ns - track.stamp_ns <= self._timeout_ns
        }
        unmatched_tracks = set(self._tracks)
        result: list[TrackedRange] = []
        for candidate in candidates:
            best_id = None
            best_iou = self._iou_threshold
            for track_id in unmatched_tracks:
                overlap = bbox_iou(candidate.bbox, self._tracks[track_id].bbox)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                filtered_z = candidate.z
            else:
                previous = self._tracks[best_id].filtered_z
                filtered_z = self._alpha * candidate.z + (1.0 - self._alpha) * previous
                unmatched_tracks.remove(best_id)
            self._tracks[best_id] = _Track(candidate.bbox, filtered_z, stamp_ns)
            result.append(TrackedRange(f'person-{best_id}', filtered_z))
        return result
