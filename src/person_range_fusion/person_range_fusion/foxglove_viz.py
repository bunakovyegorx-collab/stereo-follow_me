"""Build lightweight Foxglove annotation and scene messages (no image drawing)."""

from __future__ import annotations

import math
from typing import Any

from builtin_interfaces.msg import Duration
from foxglove_msgs.msg import Color
from foxglove_msgs.msg import CubePrimitive
from foxglove_msgs.msg import ImageAnnotations
from foxglove_msgs.msg import LinePrimitive
from foxglove_msgs.msg import Point2
from foxglove_msgs.msg import PointsAnnotation
from foxglove_msgs.msg import SceneEntity
from foxglove_msgs.msg import SceneUpdate
from foxglove_msgs.msg import TextAnnotation
from foxglove_msgs.msg import TextPrimitive
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection3DArray

from person_range_fusion.fusion import BBox
from person_range_fusion.fusion import DepthEstimate
from person_range_fusion.fusion import TrackedRange


GREEN = Color(r=0.0, g=1.0, b=0.0, a=1.0)
ORANGE = Color(r=1.0, g=0.65, b=0.0, a=1.0)
LABEL_BG = Color(r=0.0, g=0.0, b=0.0, a=0.55)
# Person 3D marker: red wireframe edges (body is empty / no solid faces).
WIREFRAME_RED = Color(r=1.0, g=0.0, b=0.0, a=1.0)

# Cube corner indices → 12 edges as LINE_LIST pairs (local frame, center origin).
_CUBE_EDGE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom (-Z)
    (4, 5), (5, 6), (6, 7), (7, 4),  # top (+Z)
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
)


def confidence_to_color(score: float, *, alpha: float = 0.85) -> Color:
    """Map detection confidence to a green→yellow→red RGB color."""
    clamped = max(0.0, min(1.0, float(score)))
    a = max(0.0, min(1.0, float(alpha)))
    # High confidence → green, low → red.
    return Color(r=1.0 - clamped, g=clamped, b=0.0, a=a)


def wireframe_cube_lines(
    pose: Pose,
    size: Vector3,
    *,
    color: Color = WIREFRAME_RED,
    thickness: float = 0.03,
) -> LinePrimitive:
    """Build a red wireframe box (12 edges) centered at ``pose``."""
    hx = float(size.x) * 0.5
    hy = float(size.y) * 0.5
    hz = float(size.z) * 0.5
    corners = (
        Point(x=-hx, y=-hy, z=-hz),
        Point(x=+hx, y=-hy, z=-hz),
        Point(x=+hx, y=+hy, z=-hz),
        Point(x=-hx, y=+hy, z=-hz),
        Point(x=-hx, y=-hy, z=+hz),
        Point(x=+hx, y=-hy, z=+hz),
        Point(x=+hx, y=+hy, z=+hz),
        Point(x=-hx, y=+hy, z=+hz),
    )
    line = LinePrimitive()
    line.type = LinePrimitive.LINE_LIST
    line.pose = pose
    line.thickness = float(thickness)
    line.scale_invariant = False
    line.color = color
    for i, j in _CUBE_EDGE_PAIRS:
        line.points.append(corners[i])
        line.points.append(corners[j])
    return line


def build_image_annotations(
    people: list[Detection2D],
    estimates: list[DepthEstimate | None],
    tracked_by_index: dict[int, TrackedRange],
    header: Header,
    *,
    detection_bbox_fn: Any,
    hypothesis_fn: Any,
) -> ImageAnnotations:
    """Convert person boxes + depth into Foxglove ImageAnnotations."""
    msg = ImageAnnotations()
    msg.timestamp = header.stamp
    for index, detection in enumerate(people):
        bbox: BBox = detection_bbox_fn(detection)
        estimate = estimates[index]
        class_id, score = hypothesis_fn(detection)
        valid = estimate is not None
        color = GREEN if valid else ORANGE

        corners = PointsAnnotation()
        corners.timestamp = header.stamp
        corners.type = PointsAnnotation.LINE_LOOP
        corners.points = [
            Point2(x=float(bbox.x1), y=float(bbox.y1)),
            Point2(x=float(bbox.x2), y=float(bbox.y1)),
            Point2(x=float(bbox.x2), y=float(bbox.y2)),
            Point2(x=float(bbox.x1), y=float(bbox.y2)),
        ]
        corners.outline_color = color
        corners.thickness = 2.0
        msg.points.append(corners)

        label = TextAnnotation()
        label.timestamp = header.stamp
        label.position = Point2(x=float(bbox.x1), y=max(float(bbox.y1) - 6.0, 0.0))
        if valid:
            filtered = tracked_by_index[index].filtered_z
            label.text = f'{class_id} {score:.2f} | Z={filtered:.2f}m'
        else:
            label.text = f'{class_id} {score:.2f} | depth unavailable'
        label.font_size = 12.0
        label.text_color = color
        label.background_color = LABEL_BG
        msg.texts.append(label)
    return msg


def build_scene_update(
    output: Detection3DArray,
    *,
    marker_size: tuple[float, float, float] = (0.5, 0.5, 1.7),
    lifetime_sec: float = 1.0,
    wireframe: bool = True,
    edge_color: Color | None = None,
    edge_thickness: float = 0.03,
    fill_alpha: float = 0.0,
) -> SceneUpdate:
    """Convert Detection3DArray into a Foxglove SceneUpdate.

    Default: **red wireframe** (12 edges), no solid body so the person stays
    visible. Set ``wireframe=False`` or ``fill_alpha>0`` for a filled cube.
    """
    scene = SceneUpdate()
    lifetime = Duration()
    lifetime.sec = int(lifetime_sec)
    lifetime.nanosec = int(round((lifetime_sec - int(lifetime_sec)) * 1e9))
    size = Vector3(x=marker_size[0], y=marker_size[1], z=marker_size[2])
    edges = edge_color if edge_color is not None else WIREFRAME_RED

    for detection in output.detections:
        entity = SceneEntity()
        entity.id = detection.id
        entity.frame_id = output.header.frame_id
        entity.timestamp = output.header.stamp
        entity.lifetime = lifetime
        entity.frame_locked = True

        score = 0.0
        if detection.results:
            score = float(detection.results[0].hypothesis.score)

        pose = Pose()
        pose.position.x = detection.bbox.center.position.x
        pose.position.y = detection.bbox.center.position.y
        pose.position.z = detection.bbox.center.position.z
        pose.orientation = detection.bbox.center.orientation

        if wireframe:
            entity.lines.append(
                wireframe_cube_lines(
                    pose, size, color=edges, thickness=edge_thickness,
                )
            )
        # Optional translucent fill (off by default — fully transparent body).
        if not wireframe or fill_alpha > 0.0:
            cube = CubePrimitive()
            cube.pose = pose
            cube.size = size
            if wireframe:
                cube.color = Color(r=edges.r, g=edges.g, b=edges.b, a=fill_alpha)
            else:
                cube.color = confidence_to_color(score, alpha=fill_alpha or 0.85)
            entity.cubes.append(cube)

        label = TextPrimitive()
        label.pose.position.x = detection.bbox.center.position.x
        label.pose.position.y = detection.bbox.center.position.y
        # marker_size[2] is height (Z-up) in REP-103 output frame.
        label.pose.position.z = detection.bbox.center.position.z + marker_size[2] * 0.6
        label.pose.orientation.w = 1.0
        label.billboard = True
        label.font_size = 0.15
        label.scale_invariant = False
        label.color = Color(r=1.0, g=1.0, b=1.0, a=1.0)
        # Camera-origin distance is frame-invariant (optical Z ≈ this near center).
        px = detection.bbox.center.position.x
        py = detection.bbox.center.position.y
        pz = detection.bbox.center.position.z
        range_m = math.sqrt(px * px + py * py + pz * pz)
        label.text = f'{detection.id} · {range_m:.2f}m'
        entity.texts.append(label)

        scene.entities.append(entity)
    return scene
