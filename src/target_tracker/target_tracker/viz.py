"""Turn detector output into Detection3DArray and Foxglove SceneUpdate.

Deliberately light on the wire: a SceneUpdate of wireframe boxes is a few KB,
unlike the point cloud it was derived from. Nothing here publishes pixels.

The 12-edge box primitive is reused from
``person_range_fusion.foxglove_viz.wireframe_cube_lines`` rather than copied.
"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Duration
from foxglove_msgs.msg import Color
from foxglove_msgs.msg import SceneEntity
from foxglove_msgs.msg import SceneUpdate
from foxglove_msgs.msg import TextPrimitive
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox3D
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose

from person_range_fusion.foxglove_viz import wireframe_cube_lines

from target_tracker.clustering import Cluster
from target_tracker.udepth import UBox

# One colour per detector so the two opinions stay tellable apart in the
# 3D panel: clustering is blue, U-depth is yellow.
CLUSTER_BLUE = Color(r=0.2, g=0.5, b=1.0, a=1.0)
UDEPTH_YELLOW = Color(r=1.0, g=0.9, b=0.1, a=1.0)


def clusters_to_detections(
    clusters: list[Cluster], header: Header,
) -> Detection3DArray:
    """Wrap voxel clusters as a Detection3DArray in the header's frame."""
    output = Detection3DArray()
    output.header = header
    for i, cluster in enumerate(clusters):
        output.detections.append(
            _detection(
                header,
                identifier=f'cluster_{i}',
                center=cluster.center,
                size=cluster.size,
                class_id='cluster',
                score=1.0,
            )
        )
    return output


def u_boxes_to_detections(
    boxes: list[UBox], header: Header,
) -> Detection3DArray:
    """Wrap U-depth boxes as a Detection3DArray in the header's frame."""
    output = Detection3DArray()
    output.header = header
    for i, box in enumerate(boxes):
        output.detections.append(
            _detection(
                header,
                identifier=f'udepth_{i}',
                center=box.center,
                size=box.size,
                class_id='udepth',
                score=1.0,
            )
        )
    return output


def build_scene_update(
    detections: Detection3DArray,
    *,
    color: Color,
    lifetime_sec: float = 0.5,
    edge_thickness: float = 0.02,
    label_prefix: str = '',
) -> SceneUpdate:
    """Wireframe cube per detection, sized from its own bounding box.

    ``person_range_fusion.foxglove_viz.build_scene_update`` draws every box at
    one fixed marker size, which is exactly what must not happen here: the
    measured extent of a cluster is the thing being inspected.
    """
    scene = SceneUpdate()
    lifetime = Duration()
    lifetime.sec = int(lifetime_sec)
    lifetime.nanosec = int(round((lifetime_sec - int(lifetime_sec)) * 1e9))

    for detection in detections.detections:
        entity = SceneEntity()
        entity.id = detection.id
        entity.frame_id = detections.header.frame_id
        entity.timestamp = detections.header.stamp
        entity.lifetime = lifetime
        entity.frame_locked = True

        pose = Pose()
        pose.position.x = detection.bbox.center.position.x
        pose.position.y = detection.bbox.center.position.y
        pose.position.z = detection.bbox.center.position.z
        pose.orientation = detection.bbox.center.orientation
        size = Vector3(
            x=detection.bbox.size.x,
            y=detection.bbox.size.y,
            z=detection.bbox.size.z,
        )
        entity.lines.append(
            wireframe_cube_lines(
                pose, size, color=color, thickness=edge_thickness,
            )
        )

        label = TextPrimitive()
        label.pose.position.x = pose.position.x
        label.pose.position.y = pose.position.y
        label.pose.position.z = pose.position.z + float(size.z) * 0.6
        label.pose.orientation.w = 1.0
        label.billboard = True
        label.font_size = 0.12
        label.scale_invariant = False
        label.color = color
        distance = float(np.linalg.norm([
            pose.position.x, pose.position.y, pose.position.z,
        ]))
        label.text = f'{label_prefix}{distance:.2f} m'
        entity.texts.append(label)

        scene.entities.append(entity)
    return scene


def _detection(
    header: Header,
    *,
    identifier: str,
    center: np.ndarray,
    size: np.ndarray,
    class_id: str,
    score: float,
) -> Detection3D:
    """One Detection3D with an axis-aligned box (identity orientation)."""
    detection = Detection3D()
    detection.header = header
    detection.id = identifier

    bbox = BoundingBox3D()
    bbox.center.position.x = float(center[0])
    bbox.center.position.y = float(center[1])
    bbox.center.position.z = float(center[2])
    bbox.center.orientation.w = 1.0
    bbox.size.x = float(size[0])
    bbox.size.y = float(size[1])
    bbox.size.z = float(size[2])
    detection.bbox = bbox

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_id
    hypothesis.hypothesis.score = float(score)
    hypothesis.pose.pose = bbox.center
    detection.results.append(hypothesis)
    return detection
