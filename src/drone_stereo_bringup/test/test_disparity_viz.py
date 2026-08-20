import numpy as np
from rclpy.qos import ReliabilityPolicy

from drone_stereo_bringup.disparity_viz_node import disparity_to_mono8
from drone_stereo_bringup.disparity_viz_node import RVIZ_QOS


def test_disparity_to_mono8_scales_valid_values_and_preserves_padding():
    source = np.array([
        [np.nan, 0.0, 16.0, 99.0],
        [8.0, 4.0, 12.0, 99.0],
    ], dtype=np.float32)
    result = disparity_to_mono8(
        source.tobytes(), height=2, width=3, step=16,
        minimum=4.0, maximum=16.0,
    )
    assert result.shape == (2, 3)
    assert result[0].tolist() == [0, 0, 255]
    assert result[1, 0] > result[1, 1] > 0
    assert result[1, 2] > result[1, 0]


def test_disparity_to_mono8_rejects_bad_buffer():
    try:
        disparity_to_mono8(
            b'', height=1, width=1, step=4,
            minimum=0.0, maximum=16.0,
        )
    except ValueError as exc:
        assert 'buffer' in str(exc)
    else:
        raise AssertionError('an empty disparity buffer must be rejected')


def test_disparity_visualization_offers_reliable_qos_for_rviz():
    assert RVIZ_QOS.reliability == ReliabilityPolicy.RELIABLE
