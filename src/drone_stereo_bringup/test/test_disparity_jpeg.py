import cv2
import numpy as np

from drone_stereo_bringup.disparity_jpeg_node import encode_jpeg
from drone_stereo_bringup.disparity_jpeg_node import DisparityJpeg
from drone_stereo_bringup.disparity_viz_node import disparity_bounds


class FakeDisparity:
    def __init__(self, min_disparity, max_disparity):
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity


def test_disparity_bounds_uses_absolute_upper_bound():
    minimum, maximum = disparity_bounds(FakeDisparity(8.0, 128.0))
    assert (minimum, maximum) == (8.0, 128.0)


def test_disparity_bounds_treats_smaller_max_as_a_span():
    # Publishers that report max_disparity as a width, not a bound.
    minimum, maximum = disparity_bounds(FakeDisparity(16.0, 8.0))
    assert (minimum, maximum) == (16.0, 24.0)


def test_encode_jpeg_round_trips_and_shrinks_the_frame():
    mono8 = np.tile(
        np.linspace(0, 255, 640, dtype=np.uint8), (480, 1),
    )
    payload = encode_jpeg(mono8, quality=60)
    decoded = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    assert decoded.shape == (480, 640, 3)
    # The point of the node: far below the 1.23 MB raw 32FC1 frame.
    assert len(payload) < 100 * 1024


def test_encode_jpeg_without_colormap_stays_single_channel():
    mono8 = np.zeros((16, 16), dtype=np.uint8)
    payload = encode_jpeg(mono8, quality=60, colormap=None)
    decoded = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED,
    )
    assert decoded.ndim == 2


def test_due_throttles_to_the_configured_period():
    node = DisparityJpeg.__new__(DisparityJpeg)
    node._min_period_ns = 200_000_000  # 5 Hz
    node._last_stamp_ns = None

    assert node._due(1_000_000_000) is True
    node._last_stamp_ns = 1_000_000_000
    assert node._due(1_100_000_000) is False
    assert node._due(1_200_000_000) is True


def test_due_passes_everything_when_throttling_is_disabled():
    node = DisparityJpeg.__new__(DisparityJpeg)
    node._min_period_ns = 0
    node._last_stamp_ns = 5
    assert node._due(6) is True
