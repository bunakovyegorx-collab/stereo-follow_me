from copy import deepcopy

from drone_stereo_bringup.calibration import scale_calibration


def source_calibration():
    return {
        'image_width': 1640,
        'image_height': 1232,
        'camera_name': 'quality_camera',
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [1582.0, 0.0, 1012.0,
                     0.0, 1580.0, 620.0,
                     0.0, 0.0, 1.0],
        },
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {
            'rows': 1,
            'cols': 5,
            'data': [0.1, -0.2, 0.0, 0.0, 0.1],
        },
        'rectification_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0],
        },
        'projection_matrix': {
            'rows': 3,
            'cols': 4,
            'data': [1582.0, 0.0, 1012.0, -114.9,
                     0.0, 1580.0, 620.0, 0.0,
                     0.0, 0.0, 1.0, 0.0],
        },
    }


def test_scale_calibration_scales_intrinsics_and_projection():
    source = source_calibration()
    original = deepcopy(source)
    scaled = scale_calibration(
        source,
        width=640,
        height=480,
        camera_name='realtime_camera',
        source_crop=(0, 0, 1640, 1230),
    )

    sx = 640 / 1640
    sy = 480 / 1230
    assert source == original
    assert scaled['image_width'] == 640
    assert scaled['image_height'] == 480
    assert scaled['camera_name'] == 'realtime_camera'
    assert scaled['camera_matrix']['data'] == [
        1582.0 * sx, 0.0, 1012.0 * sx,
        0.0, 1580.0 * sy, 620.0 * sy,
        0.0, 0.0, 1.0,
    ]
    assert scaled['projection_matrix']['data'] == [
        1582.0 * sx, 0.0, 1012.0 * sx, -114.9 * sx,
        0.0, 1580.0 * sy, 620.0 * sy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    assert scaled['distortion_coefficients'] == original[
        'distortion_coefficients'
    ]
    assert scaled['rectification_matrix'] == original[
        'rectification_matrix'
    ]


def test_scale_calibration_rejects_invalid_dimensions():
    try:
        scale_calibration(
            source_calibration(), width=0, height=480,
            camera_name='realtime_camera',
        )
    except ValueError as exc:
        assert 'dimensions' in str(exc)
    else:
        raise AssertionError('zero target width must raise ValueError')
