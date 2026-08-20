import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo

from hailo_stereo_bringup.rectification import RESIZE_STRATEGIES
from hailo_stereo_bringup.rectification import SideRectifier
from hailo_stereo_bringup.rectification import resize_crop
from hailo_stereo_bringup.rectification import resize_stretch


def _identity_camera_info(width: int = 8, height: int = 6) -> CameraInfo:
    info = CameraInfo()
    info.width = width
    info.height = height
    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def test_resize_stretch_changes_dimensions() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    resized = resize_stretch(image, width=1232, height=368)
    assert resized.shape == (368, 1232, 3)


def test_resize_crop_not_implemented_yet() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        resize_crop(image, width=1232, height=368)


def test_resize_strategies_registry_has_both_options() -> None:
    assert set(RESIZE_STRATEGIES) == {'stretch', 'crop'}


def test_side_rectifier_not_ready_before_camera_info() -> None:
    rectifier = SideRectifier()
    assert not rectifier.ready
    with pytest.raises(RuntimeError):
        rectifier.rectify(np.zeros((6, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        rectifier.focal_length_px()
    with pytest.raises(RuntimeError):
        rectifier.baseline_m()


def test_side_rectifier_identity_calibration_preserves_shape_and_content() -> None:
    rectifier = SideRectifier()
    rectifier.update(_identity_camera_info(width=8, height=6))
    assert rectifier.ready

    image = (np.arange(8 * 6 * 3, dtype=np.uint8) % 255).reshape(6, 8, 3)
    rectified = rectifier.rectify(image)

    assert rectified.shape == image.shape
    # Zero distortion + identity R + P==K should map the image very close
    # to itself (allow small interpolation error at the borders).
    assert np.mean(np.abs(rectified.astype(np.int16) - image.astype(np.int16))) < 5.0
    assert rectifier.focal_length_px() == pytest.approx(8.0)


def test_side_rectifier_baseline_from_projection_matrix() -> None:
    rectifier = SideRectifier()
    info = _identity_camera_info(width=8, height=6)
    fx = info.k[0]
    baseline_m = 0.112
    info.p[3] = -fx * baseline_m  # P[0,3] = -fx * Tx convention
    rectifier.update(info)

    assert rectifier.baseline_m() == pytest.approx(baseline_m)
