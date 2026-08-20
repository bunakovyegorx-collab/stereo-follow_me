from std_msgs.msg import Header
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose

from person_range_fusion.foxglove_viz import build_image_annotations
from person_range_fusion.foxglove_viz import build_scene_update
from person_range_fusion.foxglove_viz import confidence_to_color
from person_range_fusion.fusion import BBox
from person_range_fusion.fusion import DepthEstimate
from person_range_fusion.fusion import TrackedRange
from person_range_fusion.fusion_node import detection_bbox
from person_range_fusion.fusion_node import hypothesis


def _person_detection(x: float, y: float, w: float, h: float, score: float = 0.9) -> Detection2D:
    detection = Detection2D()
    detection.bbox.center.position.x = x
    detection.bbox.center.position.y = y
    detection.bbox.size_x = w
    detection.bbox.size_y = h
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = 'person'
    result.hypothesis.score = score
    detection.results.append(result)
    return detection


def test_confidence_to_color_endpoints() -> None:
    high = confidence_to_color(1.0)
    low = confidence_to_color(0.0)
    assert high.g == 1.0 and high.r == 0.0
    assert low.r == 1.0 and low.g == 0.0


def test_build_image_annotations_valid_and_invalid_depth() -> None:
    header = Header()
    header.stamp.sec = 10
    header.stamp.nanosec = 5
    people = [
        _person_detection(50, 50, 40, 80, 0.91),
        _person_detection(150, 60, 40, 70, 0.70),
    ]
    estimates = [
        DepthEstimate(disparity=20.0, z=1.5, valid_pixels=100, valid_fraction=0.5, sigma_z=0.05),
        None,
    ]
    tracked = {0: TrackedRange(track_id='person-1', filtered_z=1.4)}
    msg = build_image_annotations(
        people, estimates, tracked, header,
        detection_bbox_fn=detection_bbox,
        hypothesis_fn=hypothesis,
    )
    assert len(msg.points) == 2
    assert len(msg.texts) == 2
    assert msg.timestamp.sec == 10
    assert 'Z=1.40m' in msg.texts[0].text
    assert 'depth unavailable' in msg.texts[1].text
    # LINE_LOOP rectangle has 4 corners.
    assert len(msg.points[0].points) == 4


def test_build_scene_update_entities_and_lifetime() -> None:
    output = Detection3DArray()
    # Final fusion products publish in REP-103 (stereo_left_up).
    output.header.frame_id = 'stereo_left_up'
    output.header.stamp.sec = 3
    det = Detection3D()
    det.id = 'person-1'
    # X forward = range, Z up (label offset uses marker height on Z).
    det.bbox.center.position.x = 2.5
    det.bbox.center.position.y = 0.0
    det.bbox.center.position.z = 0.0
    det.bbox.center.orientation.w = 1.0
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = 'person'
    hyp.hypothesis.score = 0.8
    det.results.append(hyp)
    output.detections.append(det)

    scene = build_scene_update(output, lifetime_sec=1.0)
    assert len(scene.entities) == 1
    entity = scene.entities[0]
    assert entity.id == 'person-1'
    assert entity.frame_id == 'stereo_left_up'
    assert entity.lifetime.sec == 1
    # Default: red wireframe only (no solid fill cube).
    assert len(entity.cubes) == 0
    assert len(entity.lines) == 1
    line = entity.lines[0]
    assert line.pose.position.x == 2.5
    assert line.color.r == 1.0 and line.color.g == 0.0 and line.color.b == 0.0
    # 12 edges × 2 points for LINE_LIST.
    assert len(line.points) == 24
    # Label shows camera-origin distance (frame-invariant).
    assert '2.50m' in entity.texts[0].text


def test_build_scene_update_solid_fill_when_not_wireframe() -> None:
    output = Detection3DArray()
    output.header.frame_id = 'stereo_left_up'
    det = Detection3D()
    det.id = 'person-2'
    det.bbox.center.position.x = 1.0
    det.bbox.center.orientation.w = 1.0
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = 'person'
    hyp.hypothesis.score = 1.0
    det.results.append(hyp)
    output.detections.append(det)

    scene = build_scene_update(output, wireframe=False, fill_alpha=0.4)
    entity = scene.entities[0]
    assert len(entity.lines) == 0
    assert len(entity.cubes) == 1
    assert entity.cubes[0].color.a == 0.4
