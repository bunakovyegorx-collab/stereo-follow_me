import numpy as np
from stereo_msgs.msg import DisparityImage

from person_range_fusion.fusion import BBox
from person_range_fusion.fusion import RangeCandidate
from person_range_fusion.fusion import RangeTracker
from person_range_fusion.fusion import bbox_iou
from person_range_fusion.fusion import estimate_person_depth
from person_range_fusion.fusion_node import disparity_array


def test_depth_uses_torso_median_and_rejects_background_outliers() -> None:
    disparity = np.full((100, 100), 4.0, dtype=np.float32)
    disparity[20:70, 35:65] = 20.0
    disparity[30:35, 40:45] = np.nan
    disparity[40:45, 50:55] = 2.0

    estimate = estimate_person_depth(
        disparity, BBox(20, 0, 80, 100),
        f=300.0, baseline=0.075,
        min_disparity=0.0, max_disparity=48.0,
        min_depth=0.5, max_depth=5.0,
    )

    assert estimate is not None
    assert abs(estimate.disparity - 20.0) < 0.01
    assert abs(estimate.z - 1.125) < 0.01
    assert estimate.valid_pixels >= 1000


def test_depth_rejects_sparse_or_out_of_range_values() -> None:
    disparity = np.full((40, 40), np.nan, dtype=np.float32)
    disparity[10:12, 10:12] = 10.0
    assert estimate_person_depth(
        disparity, BBox(0, 0, 40, 40),
        f=300.0, baseline=0.075,
        min_disparity=0.0, max_disparity=48.0,
        min_depth=0.5, max_depth=5.0,
    ) is None


def test_bbox_is_clipped_to_the_image() -> None:
    disparity = np.full((20, 20), 15.0, dtype=np.float32)
    estimate = estimate_person_depth(
        disparity, BBox(-20, -10, 30, 30),
        f=300.0, baseline=0.075,
        min_disparity=0.0, max_disparity=48.0,
        min_depth=0.5, max_depth=5.0,
        min_valid_pixels=5,
    )
    assert estimate is not None
    assert abs(estimate.z - 1.5) < 0.01


def test_bbox_iou() -> None:
    assert bbox_iou(BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)) == 1.0
    assert bbox_iou(BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)) == 0.0


def test_tracker_smooths_matched_person_and_expires_stale_track() -> None:
    tracker = RangeTracker(alpha=0.4, iou_threshold=0.3, timeout_sec=0.7)
    first = tracker.update([RangeCandidate(BBox(0, 0, 10, 20), 2.0)], 0)
    second = tracker.update(
        [RangeCandidate(BBox(1, 0, 11, 20), 3.0)], 300_000_000,
    )
    expired = tracker.update(
        [RangeCandidate(BBox(1, 0, 11, 20), 4.0)], 1_100_000_000,
    )

    assert first[0].track_id == 'person-1'
    assert second[0].track_id == 'person-1'
    assert abs(second[0].filtered_z - 2.4) < 1e-6
    assert expired[0].track_id == 'person-2'


def test_tracker_keeps_multiple_people_separate() -> None:
    tracker = RangeTracker()
    tracked = tracker.update([
        RangeCandidate(BBox(0, 0, 10, 20), 1.0),
        RangeCandidate(BBox(20, 0, 30, 20), 3.0),
    ], 0)
    assert [item.track_id for item in tracked] == ['person-1', 'person-2']
    assert min(item.filtered_z for item in tracked) == 1.0


def test_padded_disparity_message_is_read_without_padding_columns() -> None:
    message = DisparityImage()
    message.image.height = 2
    message.image.width = 2
    message.image.step = 12
    message.image.encoding = '32FC1'
    message.image.data = np.array([
        1.0, 2.0, 99.0,
        3.0, 4.0, 99.0,
    ], dtype=np.float32).tobytes()

    result = disparity_array(message)

    assert result.shape == (2, 2)
    assert np.array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
