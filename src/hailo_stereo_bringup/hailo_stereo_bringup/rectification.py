"""Rectification and resize helpers for the Hailo StereoNet pilot node.

This node does its own rectification (rather than depending on the shared
production ``stereo_sgbm_light_container``) because that container only
rectifies a mono stream -- StereoNet needs a 3-channel color input. Doing
rectification locally keeps this pilot fully isolated from the production
launch, per docs/HAILO_OFFLOAD_PLAN.md.
"""

from __future__ import annotations

import cv2
import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo


class SideRectifier:
    """Builds and caches undistort/rectify maps from one CameraInfo message.

    image_geometry.PinholeCameraModel.rectify_image() recomputes
    cv2.initUndistortRectifyMap() on every call, which is wasteful for a
    node that rectifies every frame. This wrapper computes the map once
    per CameraInfo update (camera_info is static in practice) and reuses
    it via cv2.remap directly.
    """

    def __init__(self) -> None:
        self._model: PinholeCameraModel | None = None
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None

    def update(self, camera_info: CameraInfo) -> None:
        model = PinholeCameraModel()
        model.from_camera_info(camera_info)
        map_x, map_y = cv2.initUndistortRectifyMap(
            model.intrinsic_matrix(), model.distortion_coeffs(),
            model.rotation_matrix(), model.projection_matrix(),
            model.full_resolution(), cv2.CV_32FC1,
        )
        self._model = model
        self._map_x = map_x
        self._map_y = map_y

    @property
    def ready(self) -> bool:
        return self._model is not None

    def rectify(self, bgr_image: np.ndarray) -> np.ndarray:
        if self._map_x is None or self._map_y is None:
            raise RuntimeError('camera_info not received yet')
        return cv2.remap(bgr_image, self._map_x, self._map_y, cv2.INTER_LINEAR)

    def focal_length_px(self) -> float:
        if self._model is None:
            raise RuntimeError('camera_info not received yet')
        return float(self._model.fx())

    def baseline_m(self) -> float:
        """abs(Tx)/fx from the projection matrix (meaningful on the right camera)."""
        if self._model is None:
            raise RuntimeError('camera_info not received yet')
        return abs(self._model.tx()) / self._model.fx()


def resize_stretch(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Non-aspect-preserving resize -- the MVP resize strategy.

    Known tradeoff: this stretches image content instead of cropping, since
    the current camera profile (320x240, 4:3) is narrower than StereoNet's
    native input (1232x368, ~3.35:1 KITTI-style framing). See
    docs/HAILO_OFFLOAD_PLAN.md -- a crop-based alternative sourced from a
    wider camera profile is a documented follow-up, not a blocker for first
    hardware validation.
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    raise NotImplementedError(
        "resize_strategy='crop' is a documented follow-up, not implemented "
        'yet -- see docs/HAILO_OFFLOAD_PLAN.md'
    )


RESIZE_STRATEGIES = {
    'stretch': resize_stretch,
    'crop': resize_crop,
}
