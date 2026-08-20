import numpy as np
from rclpy.qos import ReliabilityPolicy

from drone_stereo_bringup.point_cloud_preview_node import decimate_organized_data
from drone_stereo_bringup.point_cloud_preview_node import decimate_mono8_data
from drone_stereo_bringup.point_cloud_preview_node import PREVIEW_QOS
from drone_stereo_bringup.point_cloud_preview_node import validate_preview_stride


def test_decimate_organized_data_preserves_point_records():
    source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    data, height, width = decimate_organized_data(
        source.tobytes(),
        height=4,
        width=6,
        point_step=3,
        stride=2,
    )
    expected = source[::2, ::2].copy()
    assert height == 2
    assert width == 3
    assert data == expected.tobytes()


def test_decimate_organized_data_rejects_invalid_stride():
    try:
        decimate_organized_data(b'', height=0, width=0, point_step=12, stride=0)
    except ValueError as exc:
        assert 'stride' in str(exc)
    else:
        raise AssertionError('stride=0 must raise ValueError')


def test_decimate_mono8_data_removes_row_padding():
    source = np.arange(4 * 8, dtype=np.uint8).reshape(4, 8)
    data, height, width = decimate_mono8_data(
        source.tobytes(),
        height=4,
        width=6,
        step=8,
        stride=2,
    )
    expected = source[:, :6][::2, ::2].copy()
    assert height == 2
    assert width == 3
    assert data == expected.tobytes()


def test_preview_publishers_offer_reliable_qos():
    assert PREVIEW_QOS.reliability == ReliabilityPolicy.RELIABLE


def test_validate_preview_stride_accepts_realtime_stride():
    assert validate_preview_stride(2) == 2


def test_validate_preview_stride_rejects_zero():
    try:
        validate_preview_stride(0)
    except ValueError as exc:
        assert 'stride' in str(exc)
    else:
        raise AssertionError('stride=0 must raise ValueError')
