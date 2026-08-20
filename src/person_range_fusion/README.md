# Person range fusion

Stereo (320×240) + YOLO26n NCNN person + metric range, with **host-side Foxglove
offload** so the Pi does not paint frames or build point clouds.

## Design phases (cost model)

| Phase | Pi cost | Role |
|-------|---------|------|
| 1 | **lower** | Replace `debug_image` OpenCV draw with `ImageAnnotations` |
| 2 | fixed ~1 Hz | `pi_diagnostics` (power/temp/RSS), cores 0–2 |
| 3 | tiny | `/stereo/depth` = vectorized `f·B/d` (no colormap) |
| 4 | **zero** | Mac Foxglove 3D depth-map / cloud from depth + camera_info |
| 5 | tiny | `SceneUpdate` fixed-size person cubes from existing 3D detections |
| 6 | network | Bridge whitelist; no `debug_image` |

## Launch — **FULL pipeline only** (canonical runtime)

> **Always** start the **full** stack for normal operation, demos, and hardware
> checks. Do not disable foxglove / rqt / image_view / rqt_graph for day-to-day
> use. Partial/headless modes are for isolated CI or package debugging only.

Preferred entrypoints (same full stack):

```bash
# from workspace root
./run_person_range.sh
```

```bash
source /opt/ros/jazzy/setup.bash
source ~/camera_ws/install/setup.bash
export DISPLAY=:0
ros2 launch person_range_fusion person_range.launch.py
```

One launch starts stereo + YOLO + fusion + **foxglove_bridge** +
**rqt_image_view** + **image_view** + **rqt_graph**:

| Component | Default | Notes |
|-----------|---------|--------|
| `foxglove_bridge` | **on** (`foxglove:=true`) | **required** — `0.0.0.0:8765`; hotspot short: `ws://qqqq:8765` |
| `rqt_image_view` | **on** (`rqt_image_view:=true`) | **required** — `/stereo/left/image_rect` |
| `image_view` | **on** (`image_view:=true`) | **required** — OpenCV fallback if rqt SIP broken |
| `rqt_graph` | **on** (`rqt_graph:=true`) | **required** — graph on Pi display |
| `visualization` | off | optional heavy `debug_image` + legacy debug `image_view` |

Foxglove Studio on Mac/PC: layout `foxglove/person_range_clean.json`.
Project root `README.md` and `AGENTS.md` document the full-pipeline-only rule.

Legacy local debug (still full stack; heavier on Pi):

```bash
./run_person_range.sh visualization:=true
```

Headless flags that drop GUI/bridge are **not** the supported runtime (CI only).

## Topics

| Topic | Type | Notes |
|-------|------|--------|
| `/stereo/depth` | `sensor_msgs/Image` 32FC1 | Metric Z (m); Phase 3 |
| `/person_range/image_annotations` | `foxglove_msgs/ImageAnnotations` | Phase 1 |
| `/person_range/scene` | `foxglove_msgs/SceneUpdate` | Phase 5 |
| `/person_range/detections_3d` | `vision_msgs/Detection3DArray` | Valid stereo people (`stereo_left_up`) |
| `/person_range/nearest_point` | `geometry_msgs/PointStamped` | Nearest by stereo range; pose in `stereo_left_up` |
| `/person_range/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Phase 2 |
| `/person_range/debug_image` | `sensor_msgs/Image` | Only if `visualization:=true` |
| `/tf_static` | TF | `stereo_left_up` → `stereo_left_optical_frame` |

## Frames (optical vs REP-103)

| Data | Frame | Notes |
|------|-------|--------|
| `/stereo/left/image_rect`, `camera_info`, `/stereo/depth`, disparity | `stereo_left_optical_frame` | Optical convention (Z into scene). Do **not** reframe. |
| `/person_range/detections_3d`, `scene`, `nearest_point` | `stereo_left_up` | REP-103: X forward, Y left, Z up |
| Point cloud from depth in Foxglove | optical (via `camera_info`) | Independent of person cubes |

Fusion projects pixels in the optical frame (`X=(u-cx)Z/fx`), then transforms
the finished 3D point into `output_frame` (default `stereo_left_up`) with
`tf2`. `person_marker_size` is therefore natural: **`[0.5, 0.5, 1.7]`** =
width × depth × height (Z = person height).

Foxglove **Fixed frame / followTf**: `stereo_left_up` (data frame, not
view-only). Depth cloud still uses optical + `camera_info`; cubes use
`stereo_left_up` — both share the static TF so they align.

## CPU layout

- **0–2:** SGBM container, depth node, diagnostics (`nice -5`)
- **3:** YOLO (≤3 FPS) + fusion
- **GUI:** rqt_image_view (Pi display); bridge on network threads

## Parameters (fusion)

`enable_image_annotations` (default true), `enable_scene` (true),
`enable_debug_image` (false), `output_frame` (`stereo_left_up`),
`person_marker_size` (`[0.5, 0.5, 1.7]` — REP-103 width/depth/height),
`person_marker_wireframe` (true — red edges only, transparent body),
`person_marker_edge_thickness` (`0.03` m),
`person_marker_fill_alpha` (`0.0` — set `>0` for a translucent fill),
depth ROI/EMA thresholds as before.

## Checks

```bash
ros2 node list | grep -E 'foxglove_bridge|rqt|person_range|stereo'
ss -ltn | grep 8765
ros2 topic hz /stereo/depth                 # ~12 Hz like disparity
ros2 topic hz /person_range/diagnostics     # ~1 Hz
ros2 topic hz /person_range/image_annotations
# debug_image should not appear unless visualization:=true
ros2 topic list | grep debug_image || true
```
