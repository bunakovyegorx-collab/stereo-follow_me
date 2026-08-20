import numpy as np
import pytest
from std_msgs.msg import Header

from hailo_stereo_bringup.disparity_message import build_disparity_message


def test_build_disparity_message_basic_fields() -> None:
    raw = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    header = Header(frame_id='stereo_left_optical_frame')
    header.stamp.sec = 42

    message = build_disparity_message(
        raw, header, f=500.0, baseline_m=0.112, min_disparity=0.0, max_disparity=255.0,
    )

    assert message.header.frame_id == 'stereo_left_optical_frame'
    assert message.header.stamp.sec == 42
    assert message.f == 500.0
    assert message.t == 0.112
    assert message.min_disparity == 0.0
    assert message.max_disparity == 255.0
    assert message.image.encoding == '32fc1' or message.image.encoding == '32FC1'
    assert message.image.height == 2
    assert message.image.width == 2
    assert message.image.step == 2 * 4
    assert len(message.image.data) == 2 * 2 * 4


def test_build_disparity_message_roundtrips_values() -> None:
    raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    header = Header()

    message = build_disparity_message(
        raw, header, f=100.0, baseline_m=0.1, min_disparity=0.0, max_disparity=255.0,
    )

    decoded = np.frombuffer(bytes(message.image.data), dtype=np.float32)
    assert np.allclose(decoded, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_build_disparity_message_rejects_invalid_bounds() -> None:
    raw = np.zeros((2, 2), dtype=np.float32)
    header = Header()

    with pytest.raises(ValueError):
        build_disparity_message(
            raw, header, f=1.0, baseline_m=0.1, min_disparity=10.0, max_disparity=5.0,
        )


def test_build_disparity_message_rejects_non_2d_after_squeeze() -> None:
    raw = np.zeros((2, 2, 2), dtype=np.float32)
    header = Header()

    with pytest.raises(ValueError):
        build_disparity_message(
            raw, header, f=1.0, baseline_m=0.1, min_disparity=0.0, max_disparity=255.0,
        )


def test_build_disparity_message_sanitizes_nan_and_inf() -> None:
    raw = np.array([[np.nan, np.inf], [-np.inf, 5.0]], dtype=np.float32)
    header = Header()

    message = build_disparity_message(
        raw, header, f=1.0, baseline_m=0.1, min_disparity=0.0, max_disparity=255.0,
    )

    decoded = np.frombuffer(bytes(message.image.data), dtype=np.float32).reshape(2, 2)
    assert decoded[0, 0] == -1.0  # nan -> min_disparity - 1.0
    assert decoded[0, 1] == 255.0  # +inf -> max_disparity
    assert decoded[1, 0] == -1.0  # -inf -> min_disparity - 1.0
    assert decoded[1, 1] == 5.0
