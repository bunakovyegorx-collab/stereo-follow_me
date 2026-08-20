import numpy as np

from person_range_fusion.depth_convert import disparity_to_depth_m


def test_depth_matches_stereo_formula() -> None:
    disparity = np.full((4, 4), 20.0, dtype=np.float32)
    depth = disparity_to_depth_m(disparity, f=300.0, baseline=0.075)
    # Z = 300 * 0.075 / 20 = 1.125
    assert depth.dtype == np.float32
    assert np.allclose(depth, 1.125, atol=1e-5)


def test_invalid_disparity_becomes_nan() -> None:
    disparity = np.array(
        [[np.nan, 0.0, -1.0, 10.0]],
        dtype=np.float32,
    )
    depth = disparity_to_depth_m(
        disparity, f=200.0, baseline=0.1, min_disparity=0.0,
    )
    assert np.isnan(depth[0, 0])
    assert np.isnan(depth[0, 1])
    assert np.isnan(depth[0, 2])
    assert abs(float(depth[0, 3]) - 2.0) < 1e-5


def test_rejects_non_2d_or_bad_geometry() -> None:
    try:
        disparity_to_depth_m(np.zeros(3, dtype=np.float32), f=1.0, baseline=1.0)
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        disparity_to_depth_m(np.ones((2, 2), dtype=np.float32), f=0.0, baseline=1.0)
        assert False, 'expected ValueError'
    except ValueError:
        pass
