from pathlib import Path

import yaml

from drone_stereo_bringup.profiles import calculate_disparity_search
from drone_stereo_bringup.profiles import get_camera_profile
from drone_stereo_bringup.profiles import load_stereo_profile
from drone_stereo_bringup.profiles import load_named_stereo_profile


def test_quality_camera_profile_uses_full_resolution():
    profile = get_camera_profile('quality')
    assert profile['sensor_mode'] == '1640:1232'
    assert profile['width'] == 1640
    assert profile['height'] == 1232
    assert profile['format'] == 'YUYV'
    assert profile['orientation'] == 180
    assert profile['FrameDurationLimits'] == [1000000, 1000000]


def test_realtime_camera_profile_uses_full_field_scaled_output():
    profile = get_camera_profile('realtime')
    assert profile['sensor_mode'] == '1640:1232'
    assert profile['width'] == 640
    assert profile['height'] == 480
    assert profile['format'] == 'YUYV'
    assert profile['orientation'] == 180
    assert profile['FrameDurationLimits'] == [100000, 100000]


def test_live_10fps_camera_profile_uses_320x240_at_24fps():
    profile = get_camera_profile('live_10fps')
    assert profile['sensor_mode'] == '1640:1232'
    assert profile['width'] == 320
    assert profile['height'] == 240
    assert profile['format'] == 'YUYV'
    assert profile['FrameDurationLimits'] == [41667, 41667]


def test_foxglove_live_camera_profile_reuses_fast_24fps_mode():
    profile = get_camera_profile('foxglove_live')
    assert profile['sensor_mode'] == '1640:1232'
    assert profile['width'] == 320
    assert profile['height'] == 240
    assert profile['format'] == 'YUYV'
    assert profile['FrameDurationLimits'] == [41667, 41667]


def test_person_range_camera_profile_uses_320x240_at_12fps():
    profile = get_camera_profile('person_range')
    assert profile['sensor_mode'] == '1640:1232'
    assert profile['width'] == 320
    assert profile['height'] == 240
    assert profile['format'] == 'YUYV'
    assert profile['FrameDurationLimits'] == [83333, 83333]


def test_theoretical_maximum_camera_profile_uses_full_sensor_raw_output():
    profile = get_camera_profile('theoretical_maximum')
    assert profile['sensor_mode'] == '3280:2464'
    assert profile['width'] == 3280
    assert profile['height'] == 2464
    assert profile['format'] == 'SRGGB10_CSI2P'
    assert profile['orientation'] == 180
    assert profile['FrameDurationLimits'] == [33333, 33333]


def test_maximum_sgbm_profile_uses_full_sensor_at_light_frame_rate():
    profile = get_camera_profile('maximum_sgbm')
    assert profile['sensor_mode'] == '3280:2464'
    assert profile['width'] == 3280
    assert profile['height'] == 2464
    assert profile['format'] == 'YUYV'
    assert profile['FrameDurationLimits'] == [1000000, 1000000]


def test_camera_profile_rejects_unknown_name():
    try:
        get_camera_profile('turbo')
    except ValueError as exc:
        assert 'quality' in str(exc)
        assert 'realtime' in str(exc)
        assert 'theoretical_maximum' in str(exc)
    else:
        raise AssertionError('unknown profile must raise ValueError')


def test_disparity_search_covers_expected_full_field_geometry():
    minimum, disparity_range = calculate_disparity_search(101.4)
    assert minimum == 24
    assert disparity_range == 192
    assert disparity_range % 16 == 0
    assert minimum <= 101.4 / 3.0
    assert minimum + disparity_range - 1 >= 101.4 / 0.5


def test_disparity_search_rejects_invalid_geometry():
    for value in (0.0, -1.0):
        try:
            calculate_disparity_search(value)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid fB must raise ValueError')


def test_load_stereo_profile_uses_projection_baseline(tmp_path: Path):
    path = tmp_path / 'right.yaml'
    path.write_text(yaml.safe_dump({
        'image_width': 1640,
        'image_height': 1232,
        'projection_matrix': {
            'rows': 3,
            'cols': 4,
            'data': [1360.0, 0.0, 820.0, -101.4,
                     0.0, 1360.0, 616.0, 0.0,
                     0.0, 0.0, 1.0, 0.0],
        },
    }))
    profile = load_stereo_profile(path, expected_size=(1640, 1232))
    assert profile['min_disparity'] == 24
    assert profile['disparity_range'] == 192
    assert profile['stereo_algorithm'] == 1
    assert profile['sgbm_mode'] == 2
    assert profile['P1'] == 392.0
    assert profile['P2'] == 1568.0
    assert profile['use_color'] is False
    assert profile['avoid_point_cloud_padding'] is True


def test_load_stereo_profile_supports_realtime_calibration(tmp_path: Path):
    path = tmp_path / 'right.yaml'
    path.write_text(yaml.safe_dump({
        'image_width': 640,
        'image_height': 480,
        'projection_matrix': {
            'data': [617.0, 0.0, 320.0, -44.85,
                     0.0, 616.0, 240.0, 0.0,
                     0.0, 0.0, 1.0, 0.0],
        },
    }))
    profile = load_stereo_profile(path, expected_size=(640, 480))
    assert profile['min_disparity'] == 0
    assert profile['disparity_range'] == 96


def test_live_10fps_stereo_profile_applies_performance_overrides(
    tmp_path: Path,
):
    path = tmp_path / 'right.yaml'
    path.write_text(yaml.safe_dump({
        'image_width': 640,
        'image_height': 480,
        'projection_matrix': {
            'data': [617.0, 0.0, 320.0, -44.85,
                     0.0, 616.0, 240.0, 0.0,
                     0.0, 0.0, 1.0, 0.0],
        },
    }))
    profile = load_named_stereo_profile(
        'live_10fps', path, expected_size=(640, 480),
    )
    assert profile['stereo_algorithm'] == 1
    assert profile['sgbm_mode'] == 2
    assert profile['min_disparity'] == 0
    assert profile['disparity_range'] == 48
    assert profile['correlation_window_size'] == 5
    assert profile['P1'] == 200.0
    assert profile['P2'] == 800.0
    assert profile['speckle_size'] == 0
    assert profile['disp12_max_diff'] == 0
    assert profile['queue_size'] == 2


def test_foxglove_live_stereo_profile_reuses_fast_sgbm_overrides(
    tmp_path: Path,
):
    path = tmp_path / 'right.yaml'
    path.write_text(yaml.safe_dump({
        'image_width': 320,
        'image_height': 240,
        'projection_matrix': {
            'data': [308.7, 0.0, 160.0, -22.42,
                     0.0, 308.7, 120.0, 0.0,
                     0.0, 0.0, 1.0, 0.0],
        },
    }))
    profile = load_named_stereo_profile(
        'foxglove_live', path, expected_size=(320, 240),
    )
    assert profile['sgbm_mode'] == 2
    assert profile['disparity_range'] == 48
    assert profile['correlation_window_size'] == 5
    assert profile['queue_size'] == 2


def test_person_range_stereo_profile_reuses_fast_sgbm_overrides(
    tmp_path: Path,
):
    path = tmp_path / 'right.yaml'
    path.write_text(yaml.safe_dump({
        'image_width': 320,
        'image_height': 240,
        'projection_matrix': {
            'data': [308.7, 0.0, 160.0, -22.42,
                     0.0, 308.7, 120.0, 0.0,
                     0.0, 0.0, 1.0, 0.0],
        },
    }))
    profile = load_named_stereo_profile(
        'person_range', path, expected_size=(320, 240),
    )
    assert profile['sgbm_mode'] == 2
    assert profile['disparity_range'] == 48
    assert profile['correlation_window_size'] == 5
    assert profile['queue_size'] == 2


def test_load_stereo_profile_rejects_wrong_resolution(tmp_path: Path):
    path = tmp_path / 'right.yaml'
    path.write_text(yaml.safe_dump({
        'image_width': 1640,
        'image_height': 1232,
        'projection_matrix': {
            'data': [1.0, 0.0, 0.0, -1.0] + [0.0] * 8,
        },
    }))
    try:
        load_stereo_profile(path, expected_size=(640, 480))
    except ValueError as exc:
        assert '640x480' in str(exc)
    else:
        raise AssertionError('wrong-resolution calibration must be rejected')
