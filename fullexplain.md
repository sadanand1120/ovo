# Full Computation Walkthrough For `build_rgb_map.py`

This document describes what happens in the current OVO RGB-map runtime when you run:

```bash
python build_rgb_map.py \
  --dataset_name ScanNet \
  --scene_name scene0011_00 \
  --slam_module vanilla
```

or, for Replica:

```bash
python build_rgb_map.py \
  --dataset_name Replica \
  --scene_name office0 \
  --slam_module vanilla
```

In this repo, the build path is:

```text
RGB-D dataset frames
+ per-frame or estimated camera poses
+ SAM seed masks
+ dense CLIP features
+ SAM2 mask tracking
-> fused RGB point map with normals
-> per-point CLIP features
-> per-point instance memberships and collapsed instance labels
```

The final scene directory contains:

```text
rgb_map.ply
clip_feats.npy
instance_gid_slots.npy
instance_labels.npy
instance_seed_hits.npy
stats.json
timing.json
```

## High-Level Flow

The runtime is easiest to understand as eight stages.

```mermaid
flowchart LR
    S0[Stage 0: CLI and config] --> S1[Stage 1: dataset and pose backend]
    S1 --> S2[Stage 2: mapper state]
    S2 --> S3[Stage 3: frame loop]
    S3 --> S4[Stage 4: match visible existing points]
    S4 --> S5[Stage 5: seed-frame point birth and CLIP]
    S5 --> S6[Stage 6: SAM/SAM2 instance update]
    S6 --> S7[Stage 7: final downsample]
    S7 --> S8[Stage 8: save artifacts]
```

| Stage | Loop grain | Main output |
| --- | --- | --- |
| Stage 0 | Once | Parsed args and merged runtime config. |
| Stage 1 | Once, frames remain lazy | Dataset object plus pose backend. |
| Stage 2 | Once | `RGBMapper` with map tensors, CLIP extractor, and SAM instance runtime. |
| Stage 3 | Once over all frames | Incrementally updated map and instance state. |
| Stage 4 | Per frame | `point_ids_after`, a pixel image of existing map-point ids visible in this frame. |
| Stage 5 | Seed frames only | New 3D points, colors, normals, and per-point CLIP features. |
| Stage 6 | Every frame | Updated per-point instance gid slots. |
| Stage 7 | Once | Optional deterministic point cap downsample. |
| Stage 8 | Once | PLY, CLIP `.npy`, instance arrays, stats, timing. |

Important global idea:

```text
Only seed frames add new 3D points and compute CLIP features.
Default seed cadence is frame_id % map_every == 0, with map_every=8.

Non-seed frames do not add geometry.
They can still run SAM2 tracking and assign existing visible points to instances.
```

## Stage 0: CLI, Config, And Output Location

Source files:

| File | Role |
| --- | --- |
| `build_rgb_map.py` | Main entrypoint and RGB map fusion loop. |
| `configs/ovo.yaml` | Base runtime config. |
| `configs/replica.yaml` | Replica camera and depth metadata. |
| `configs/scannet.yaml` | ScanNet camera and depth metadata. |
| `map_runtime/scene.py` | Config merge, dataset construction, SLAM backend construction. |

The CLI entrypoint is:

```python
parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
parser.add_argument("--scene_name", required=True)
add_build_args(parser, default_output_root=DEFAULT_RGB_MAP_OUTPUT_ROOT)
```

`add_build_args(...)` adds:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--output_root` | `data/output/rgb_maps` | Root under which `<dataset>/<scene>` is written. |
| `--frame_limit` | `None` | Optional limit on dataset length. |
| `--slam_module` | config value | Pose backend override: `vanilla`, `orbslam`, or `cuvslam`. |
| `--disable_loop_closure` | false | Forces `slam.close_loops=false` for ORB-SLAM. |
| `--config_path` | `configs/ovo.yaml` | Base runtime config. |
| `--map_every` | `8` | Seed-frame cadence for geometry birth, CLIP, and SAM reseeding. |
| `--max_total_points` | `7e6` | Final point cap. |
| `--match_distance_th` | `0.03` | Depth agreement threshold, in meters, for matching existing map points. |

The output path is:

```python
output_dir = Path(output_root) / canonical_dataset_name(dataset_name) / scene_name
```

For example:

```text
data/output/rgb_maps/ScanNet/scene0011_00
```

## Stage 1: Dataset And Pose Backend

`run_scene_build(...)` chooses the compute device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Then it calls:

```python
config, dataset, slam_backbone = load_dataset_and_slam(...)
```

### Config Merge

`map_runtime/scene.py::build_scene_config(...)` does:

```python
config = load_config(config_path)                         # usually configs/ovo.yaml
dataset_cfg = load_config(configs/<dataset>.yaml)
update_recursive(config, dataset_cfg)
config["data"]["scene_name"] = scene_name
config["data"]["input_path"] = data/input/<dataset>/<scene>
```

Base `configs/ovo.yaml` is small:

```yaml
device: cuda
data:
  frame_limit: -1
slam:
  slam_module: vanilla
  fps: 30.0
  use_viewer: False
  close_loops: False
```

Dataset configs add camera geometry and depth scaling.

Replica:

```yaml
H: 680
W: 1200
fx: 600.0
fy: 600.0
cx: 599.5
cy: 339.5
depth_scale: 6553.5
```

ScanNet:

```yaml
H: 480
W: 640
fx: 577.590698
fy: 578.729797
cx: 318.905426
cy: 242.683609
depth_scale: 1000.0
depth_th: 4.0
crop_edge: 12
```

### Dataset Objects

Source file:

```text
map_runtime/datasets.py
```

Both dataset classes return tuples whose first fields are:

```python
(frame_id, color_data, depth_data, c2w, ...)
```

`color_data` is RGB `uint8`.

`depth_data` is `float32` meters.

`c2w` is camera-to-world pose from the dataset.

#### Replica Frame Loading

Replica reads:

```text
data/input/Replica/<scene>/results/frame*.jpg
data/input/Replica/<scene>/results/depth*.png
data/input/Replica/<scene>/traj.txt
```

Per frame:

```python
color = cv2.imread(frame_path)             # BGR
color = cv2.resize(color, (W, H))
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

depth = cv2.imread(depth_path, IMREAD_UNCHANGED)
depth = cv2.resize(depth, (W, H), nearest)
depth = depth.astype(np.float32) / depth_scale
```

#### ScanNet Frame Loading

ScanNet reads:

```text
data/input/ScanNet/<scene>/color/*.jpg
data/input/ScanNet/<scene>/depth/*.png
data/input/ScanNet/<scene>/pose/*.txt
```

Per frame:

```python
color = cv2.imread(color_path)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
lr_color = cv2.resize(color, (W, H))

depth = cv2.imread(depth_path, IMREAD_UNCHANGED)
depth = depth.astype(np.float32) / depth_scale
depth[depth > depth_th] = 0
```

If `crop_edge=12`, ScanNet removes 12 pixels from each border of both color and depth. The effective map image size becomes:

```text
height = 480 - 24 = 456
width  = 640 - 24 = 616
cx     = 318.905426 - 12
cy     = 242.683609 - 12
```

### Pose Backends

Source file:

```text
map_runtime/slam_backends.py
```

The pose backend is created by:

```python
get_slam_backbone(config, dataset, cam_intrinsics)
```

Supported backends:

| Backend | Meaning |
| --- | --- |
| `vanilla` | Uses dataset GT pose directly. Deterministic and best regression check. |
| `orbslam` or prefix `orbslam...` | Runs ORB-SLAM3 RGB-D tracking. |
| `cuvslam` | Runs NVIDIA cuVSLAM RGB-D tracking. |

The common handoff in the frame loop is:

```python
estimated_c2w = get_tracked_pose(slam_backbone, frame_data)
```

`get_tracked_pose(...)` calls:

```python
slam_backbone.track_camera(frame_data)
if not np.any(frame_data[2] > 0):
    return None
return slam_backbone.get_c2w(frame_id)
```

So a frame with no valid depth gets no pose for mapping, even if the backend had one.

#### Vanilla Backend

`VanillaMapper.track_camera(...)` stores the dataset pose if it is finite:

```python
self.estimated_c2ws[frame_id] = torch.from_numpy(c2w).to(device)
```

This is the simplest path:

```text
dataset pose -> map fusion
```

#### ORB-SLAM Backend

`WrapperORBSLAM` builds a temporary ORB-SLAM settings YAML, initializes ORB-SLAM3, and on each frame calls:

```python
self.orbslam.process_image_rgbd(rgb_image, depth_image, frame_id)
```

If tracking state is OK, it converts the ORB trajectory to a camera-to-world matrix and left-multiplies by `world_ref`, the dataset first-frame pose:

```python
self.estimated_c2ws[frame_id] = self.world_ref @ convert_pose(...)
```

That anchors ORB-SLAM's local world to the dataset's first-frame world.

#### cuVSLAM Backend

`WrapperCuVSLAM` converts depth back to uint16 depth units for the tracker, calls:

```python
pose_estimate, slam_pose = self.tracker.track(...)
```

and stores:

```python
c2w = self.world_ref @ _pose_to_matrix(pose, device)
```

Again, `world_ref` anchors to the first dataset pose.

## Stage 2: RGBMapper Construction

After loading the dataset and pose backend, `run_scene_build(...)` creates:

```python
mapper = RGBMapper(
    intrinsics=dataset.intrinsics,
    device=device,
    total_frames=len(dataset),
    map_every=map_every,
    max_total_points=max_total_points,
    match_distance_th=match_distance_th,
)
```

`RGBMapper` owns the live map.

### Core Map State

| Field | Shape | Meaning |
| --- | --- | --- |
| `points` | `(capacity, 3)` float | World-space XYZ point positions. |
| `colors` | `(capacity, 3)` uint8 | Averaged RGB per point. |
| `normals` | `(capacity, 3)` float | Averaged world-space normal per point. |
| `color_sum` | `(capacity, 3)` float | Running color sum for reobserved points. |
| `normal_sum` | `(capacity, 3)` float | Running normal sum for reobserved points. |
| `obs_count` | `(capacity,)` float | Number of observations contributing to color/normal. |
| `n_points` | scalar | Number of live points currently used. |
| `total_points_original` | scalar | Number of born candidate points before final cap. |

The tensors are capacity-managed. When more room is needed, `_ensure_capacity(...)` grows all map buffers together.

### CLIP Feature Store

`RGBMapper` also builds:

```python
self.clip_extractor = DenseCLIPExtractor(device)
self.feature_tmpdir = tempfile.mkdtemp(prefix="rgb_map_feats_")
self.feature_tmp_path = <tmpdir>/clip_feats.bin
self.feature_tmp_file = open(..., "wb")
```

CLIP features are not kept as one giant GPU tensor. They are streamed as float16 rows into a temporary binary file as new points are born. At save time, that binary body is wrapped with a NumPy `.npy` header and copied to `clip_feats.npy`.

### Instance Runtime

`RGBMapper` constructs:

```python
self.instance_runtime = SAMInstanceRuntime(
    config=SAMInstanceRuntimeConfig(),
    device=device,
    total_frames=total_frames,
    map_every=map_every,
    n_points=0,
)
```

The instance runtime never changes geometry. It consumes:

```text
rgb image
is_seed_frame
valid_pose
point_ids_after
optional seed_labels
```

and updates per-point instance memberships.

## Stage 3: Main Frame Loop

The main loop is in `run_scene_build(...)`:

```python
for frame_id in range(len(dataset)):
    frame_data = dataset[frame_id]
    prev_n = mapper.n_points
    estimated_c2w = get_tracked_pose(slam_backbone, frame_data)
    mapper.add_frame(frame_data, c2w_override=estimated_c2w)
    if snapshot_hook is not None:
        snapshot_hook(frame_id, prev_n, mapper.n_points, estimated_c2w)
```

The progress bar displays:

```text
points=<current n_points>
active=<number of gids currently seeded into SAM2>
objs=<number of existing gid buckets>
```

Inside `RGBMapper.add_frame(...)`:

```python
frame_id, image_np, depth_np = frame_data[:3]
is_seed_frame = frame_id % map_every == 0
c2w_np = c2w_override
```

It creates a per-frame image of map point ids:

```python
point_ids_after = full((H, W), -1)
```

Meaning:

| Pixel value | Meaning |
| --- | --- |
| `-1` | No map point is associated with this pixel. |
| `>= 0` | This pixel corresponds to that global map-point id. |

`point_ids_after` is the key bridge between 2D SAM masks and 3D map points.

If the pose is missing, NaN, or Inf:

```text
no geometry matching
no new points
instance runtime still receives the frame and valid_pose=False
```

If the pose is valid, the mapper proceeds to point matching.

## Stage 4: Match Existing 3D Points Into The Current Frame

The mapper first marks all valid depth pixels as possible new-point candidates:

```python
mask = depth > 0
```

If the map already has points, it tries to find which existing points are visible in this frame.

### 4.1 Frustum Culling

`geometry.compute_camera_frustum_corners(...)` builds 8 camera frustum corners from:

```text
image width, image height, min valid depth, max valid depth, intrinsics, c2w
```

The corners are transformed to world space.

Then:

```python
frustum_mask = geometry.compute_frustum_point_ids(
    self.points[:self.n_points],
    frustum_corners,
)
```

This first checks an axis-aligned bounding box around the frustum, then checks frustum planes. The result is a 1D tensor of candidate global point ids.

### 4.2 Depth-Consistent Projection Match

For frustum-candidate points, the code computes:

```python
w2c = invert_rigid_transform(c2w)
matched_ids, matches = geometry.match_3d_points_to_2d_pixels(
    depth,
    w2c,
    self.points[frustum_mask],
    self.cam_intrinsics,
    self.match_distance_th,
)
```

For each candidate world point:

1. Transform world point into current camera coordinates.
2. Project with the camera intrinsics.
3. Round to the nearest pixel.
4. Keep it only if the pixel is inside the image.
5. Compare projected point depth to the measured depth at that pixel.
6. Accept if:

```python
abs(projected_depth - depth[y, x]) < match_distance_th
```

Default threshold:

```text
0.03 meters
```

Accepted matches update the per-frame point-id image:

```python
point_ids_after[y, x] = global_point_id
```

and remove those pixels from the new-point candidate mask:

```python
mask[y, x] = False
```

So matched pixels reobserve old points. Unmatched valid-depth pixels remain candidates for new point birth, but only if this is a seed frame.

## Stage 5: Seed-Frame Geometry Birth, Normals, And CLIP

This stage runs only when:

```python
is_seed_frame == True
```

Default:

```text
frame 0, 8, 16, 24, ...
```

Non-seed frames skip this whole geometry-birth block.

### 5.1 SAM Seed Labels For Text Regions

The first seed-frame operation is:

```python
seed_labels_np = self.instance_runtime.extract_seed_labels(image_np)
tr_labels_full = torch.from_numpy(seed_labels_np).to(device)
```

`extract_seed_labels(...)` runs the SAM automatic mask generator and flattens overlapping masks into one local label map:

```text
seed_labels_np: H x W int32
```

Label meanings:

| Value | Meaning |
| --- | --- |
| `-1` | No selected SAM mask owns this pixel. |
| `0,1,2,...` | Local SAM mask id for this seed frame only. |

The same seed labels are used for two things:

1. Dense CLIP text-region feature extraction.
2. Instance runtime seed-frame gid creation/reuse.

### 5.2 Normal Computation

Normals are computed from the current depth map:

```python
normals_cam, normal_valid = compute_normals_from_depth(x, y, depth, intrinsics)
```

For each inner pixel, the code:

1. Backprojects neighboring pixels into camera-space vertices.
2. Computes finite differences in x and y.
3. Uses a cross product to form a normal.
4. Normalizes it.
5. Flips normals so they face the camera.
6. Marks pixels invalid if any required neighbor depth is invalid.

Border pixels are invalid because the finite-difference stencil needs neighbors.

### 5.3 Reobserved Existing Points Update RGB And Normals

For matched visible existing points with valid normals:

```python
visible_existing = (point_ids_after >= 0) & normal_valid
```

The mapper gathers:

```python
visible_ids
visible_colors
visible_normals
```

The normals are rotated from camera space to world space:

```python
visible_normals = c2w[:3, :3] @ normals_cam
```

Then `_update_observed_points(...)` does running averages:

```python
color_sum.index_add_(point_ids, colors)
normal_sum.index_add_(point_ids, normals)
obs_count.index_add_(point_ids, 1)

colors[unique_ids] = round(color_sum / obs_count)
normals[unique_ids] = normalize(normal_sum / obs_count)
```

Important:

```text
Existing point RGB/normals are updated only on seed frames.
Non-seed frames can contribute instance assignments but not color/normal fusion.
```

### 5.4 New Point Candidate Mask

After existing-point matching and normal validity:

```python
mask = (depth > 0)                  # started as valid depth
mask[matched_existing_pixels] = False
mask = mask & normal_valid
```

Remaining `True` pixels are:

```text
valid depth
+ not already explained by an existing map point
+ have valid local normal
+ on a seed frame
```

Only these pixels become new map points.

### 5.5 Dense CLIP Extraction

If there are new pixels, the mapper computes dense CLIP features once for the full seed image:

```python
dense_clip = self.clip_extractor.extract_dense(full_image, tr_labels_full)
features = dense_clip[y_keep, x_keep].half()
```

`DenseCLIPExtractor` uses:

```text
CLIP_MODEL_NAME = ViT-L-14-336-quickgelu
CLIP_PRETRAINED = openai
CLIP_LOAD_SIZE = 1024
```

Conceptually:

1. Resize image so the shorter side is `1024`.
2. Normalize with CLIP mean/std.
3. Pad to the ViT patch multiple.
4. Run CLIP visual transformer up to the last attention block.
5. Build baseline patch features.
6. Convert SAM label regions to patch weights.
7. Remove overly global patches using `CLIP_GLOBAL_PATCH_THRESHOLD`.
8. For valid SAM regions, aggregate attention value features into region features.
9. Paint region features back to patches.
10. Bilinearly upsample feature grid back to original image size.

The output is:

```text
dense_clip: H x W x clip_feature_dim
```

For `ViT-L-14-336-quickgelu`, the feature dimension is read from the loaded model, not hardcoded in the builder.

Each newly born point receives the feature at its source pixel.

Important:

```text
CLIP features are assigned at point birth.
They are not averaged or updated on later reobservations.
```

### 5.6 Backproject New Points

For every kept pixel `(x, y)` with depth `z`, camera-space coordinates are:

```math
X_c = \frac{(x-c_x)z}{f_x}
```

```math
Y_c = \frac{(y-c_y)z}{f_y}
```

```math
Z_c = z
```

The code forms homogeneous points:

```python
points_cam = [X_c, Y_c, Z_c, 1]
```

and transforms them to world coordinates:

```python
points_world = c2w @ points_cam
```

Normals are also rotated to world space:

```python
normals_world = c2w[:3, :3] @ normals_cam
normals_world = normalize(normals_world)
```

Then the mapper assigns new global point ids:

```python
old_n = self.n_points
new_ids = arange(old_n, old_n + num_new_points)
point_ids_after[y_keep, x_keep] = new_ids
```

and appends:

```python
self._append_points(points, colors, normals, features)
```

`_append_points(...)` writes:

| Buffer | Initial value for new points |
| --- | --- |
| `points` | world XYZ |
| `colors` | source RGB |
| `normals` | world normal |
| `color_sum` | source RGB as float |
| `normal_sum` | world normal |
| `obs_count` | `1` |
| feature temp file | float16 CLIP row |

At this point, `point_ids_after` contains both:

```text
matched old point ids
+ newly born point ids
```

That complete image is passed to the instance runtime.

## Stage 6: SAM/SAM2 Instance Runtime

Source files:

| File | Role |
| --- | --- |
| `map_runtime/sam_instance_runtime.py` | Per-point global instance gid state. |
| `map_runtime/sam_masks.py` | SAM/SAM2 automatic mask generation and local-label flattening. |
| `map_runtime/sam2_tracking.py` | SAM2 video predictor wrapper for mask propagation. |

After geometry handling, every frame calls:

```python
self.instance_runtime.ensure_point_capacity(self.n_points)
self.instance_runtime.process_frame(
    frame_id=frame_id,
    is_seed_frame=is_seed_frame,
    valid_pose=valid_pose,
    rgb=image_np,
    point_ids_after=point_ids_after.cpu().numpy(),
    seed_labels=seed_labels_np,
)
```

The instance runtime consumes the current 2D-to-3D association image. It does not create, delete, or move map points.

### Core Instance State

| Field | Meaning |
| --- | --- |
| `point_gids` | Shape `(n_points, point_gid_slots)`. Each map point can belong to up to `K` global instance gids. Empty slots are `-1`. |
| `buckets` | `gid -> InstanceBucket`, storing support and point counts. |
| `next_gid` | Next global instance id to allocate. |
| `seeded_gids` | Gids currently inserted into the SAM2 tracker. |
| `_gid_scores` | Packed support/point/last-frame score for tie-breaking and collapse. |

Default instance pipeline config:

| Config | Default | Meaning |
| --- | ---: | --- |
| `point_gid_slots` | `10` | Max gids per map point. |
| `reuse_inside_frac_th` | `0.40` | Minimum fraction of mask points already belonging to a candidate gid for reuse. |
| `reuse_outside_frac_th` | `0.10` | Maximum leakage of that gid outside the candidate mask. |
| `min_mask_points` | `1` | Minimum visible map points under a seed mask. |
| `min_track_visible_points` | `1` | Minimum visible points to record seed support. |
| `prune_start_at` | `0` | First frame eligible for pruning. |
| `prune_every_frames` | `64` | Pruning cadence. |
| `prune_min_support_perc` | `0.0` | Minimum detector support ratio. |
| `prune_min_points_perc` | `0.00015` | Minimum map-point fraction. |

### 6.1 Invalid Seed Frame Special Case

If:

```python
valid_pose == False and is_seed_frame == True
```

then the runtime closes the active tracker and does not run SAM seed extraction.

There is no equivalent hard skip for invalid non-seed frames. They can enter the tracker path, but with no visible valid `point_ids_after` they typically assign nothing.

### 6.2 Seed Frame Logic

On seed frames, the runtime uses the SAM label map:

```python
seed_labels = extract_seed_labels(rgb)
seed_pairs = build_label_masks(seed_labels)
```

`build_label_masks(...)` returns:

```text
[(local_mask_id, binary_mask), ...]
```

sorted by descending mask area.

For each local SAM mask:

1. Collect unique visible map point ids under the mask:

```python
point_ids = unique(point_ids_after[mask])
point_ids = point_ids[point_ids >= 0]
```

2. Split those points:

```text
labeled_points    = rows whose point_gids already contain any gid
background_points = rows whose point_gids are all -1
```

3. If labeled points exist, propose a candidate gid:

```text
count each gid under the mask
inside_frac = count(gid) / number of visible points in mask
candidate gid must satisfy inside_frac >= reuse_inside_frac_th
```

Among candidate gids, it picks the one with highest `_gid_scores`, tie-breaking by smaller gid.

### 6.3 Grouped Existing-Gid Reuse

All masks that proposed the same candidate gid are grouped before the final decision.

For each candidate gid group:

1. Union all masks in the group.
2. Union all visible points in the group.
3. Union all background points in the group.
4. Recompute:

```text
inside_frac  = fraction of union visible points that already contain this gid
outside_frac = fraction of all other visible frame points that already contain this gid
```

Reuse succeeds iff:

```text
union has visible points
inside_frac >= reuse_inside_frac_th
outside_frac <= reuse_outside_frac_th
```

On reuse:

```python
_add_gid_to_points(union_background_points, gid, background_only=True)
_record_support(gid, frame_id)
gid_to_mask[gid] = union_mask
```

The gid is added only to background points in the mask. Already-labeled points remain as they are.

### 6.4 New Gid Birth

If a seed mask cannot reuse an existing gid and has at least one visible point:

```python
gid = _create_gid(frame_id)
_add_gid_to_points(point_ids, gid, background_only=False)
_record_support(gid, frame_id)
gid_to_mask[gid] = mask
```

`_create_gid(...)` initializes:

```python
InstanceBucket(
    gid=gid,
    support_frames=0,
    point_count=0,
    last_support_frame=-1,
    birth_frame=frame_id,
)
```

`_record_support(...)` then increments `support_frames` once for that frame and sets `last_support_frame`.

### 6.5 Point Gid Insertion Rule

`_add_gid_to_points(point_ids, gid, background_only)`:

1. Drops invalid ids.
2. Deduplicates point ids.
3. Skips points that already contain the gid.
4. If `background_only=True`, only rows with all slots `-1` are eligible.
5. If `background_only=False`, any row is eligible as long as it has a free slot.
6. Inserts the gid into the first free slot.
7. Counts rows with no free slot as `overflow_points`.
8. Increments `bucket.point_count` by successful insertions.

This means a point can carry overlapping instance memberships:

```text
point_gids[point_id] = [7, 12, -1, -1, ...]
```

The full multi-gid row is the real stored instance state.

### 6.6 SAM2 Tracker Reseed

At the end of a seed frame:

```python
seeded_masks = _seed_tracker(rgb, gid_to_mask)
```

`_seed_tracker(...)`:

1. Keeps the largest `sam2_max_num_objects` masks by area.
2. Creates or restarts `SAM2VideoTracker` on the current RGB frame.
3. Adds those masks as SAM2 object ids.
4. Sets `seeded_gids` to the gids actually accepted by SAM2.

Default tracker config:

```text
model_level = 24
max_num_objects = 16
boundary_masking_width = 0
```

`model_level=24` maps to:

```text
sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml
```

### 6.7 Non-Seed Tracking Logic

On non-seed frames:

```python
tracker_labels, decisions = _step_non_seed(frame)
```

If no tracker is active or `seeded_gids` is empty, nothing happens.

Otherwise:

1. Append current RGB frame to the SAM2 lazy video loader.
2. Run SAM2 single-frame inference for active object ids.
3. For each tracked gid mask, collect visible map point ids from `point_ids_after`.
4. Add that gid to those points:

```python
_add_gid_to_points(point_ids, gid, background_only=False)
```

Important:

```text
Tracker propagation does not call _record_support.
Only seed-frame detector validation increments support_frames.
```

### 6.8 Pruning

Pruning runs after seed or tracker update for the frame.

It is eligible only if:

```text
frame_id >= prune_start_at
prune_every_frames > 0
frame_id % prune_every_frames == 0
```

For each gid:

```text
num_seed_frames = 1 + floor(frame_id / map_every)
support_perc = support_frames / num_seed_frames
points_perc = point_count / current_num_map_points
```

A gid is dropped if:

```text
support_perc < prune_min_support_perc
or
points_perc < prune_min_points_perc
```

Dropping a gid removes it from every `point_gids` row, left-compacts rows, removes the bucket, and removes it from `seeded_gids`.

## Stage 7: Final Downsample / Point Cap

After the frame loop:

```python
mapper.finalize_for_output()
```

If:

```text
n_points <= max_total_points
```

nothing is changed.

If the map is too large, the mapper keeps a deterministic strided subset:

```python
step = total_count / keep_count
keep = floor(arange(keep_count) * step)
```

Then it rewrites all point-aligned arrays:

```text
points
colors
normals
color_sum
normal_sum
obs_count
point_gids
clip feature temp file
```

The instance runtime receives:

```python
self.instance_runtime.keep_point_ids(keep)
```

That filters `point_gids`, recomputes bucket point counts, drops empty gids, and refreshes gid scores.

The CLIP temp feature file is rewritten by memory-mapping the original float16 feature body and streaming only kept rows into a replacement binary file.

Important:

```text
The final point cap is applied only once after all frames.
It is not an online sampling policy.
```

## Stage 8: Save Artifacts

`mapper.save(output_dir, stats)` writes four groups of artifacts.

### 8.1 Point Cloud PLY

```python
pcd.points = points[:n_points]
pcd.colors = colors[:n_points] / 255.0
pcd.normals = normals[:n_points]
o3d.io.write_point_cloud(output_dir / "rgb_map.ply", pcd)
```

The PLY stores:

```text
x, y, z
red, green, blue
nx, ny, nz
```

### 8.2 CLIP Features

The builder writes a valid `.npy` file manually:

```python
np.lib.format.write_array_header_2_0(
    f,
    {
        "descr": dtype(float16),
        "fortran_order": False,
        "shape": (n_points, clip_feature_dim),
    },
)
copy temp binary body into f
```

Final:

```text
clip_feats.npy: float16 array, shape (n_points, clip_feature_dim)
```

Rows are aligned with `rgb_map.ply` vertex order.

### 8.3 Instance Arrays

The builder writes:

```python
np.save(output_dir / "instance_gid_slots.npy", export_point_gids())
np.save(output_dir / "instance_labels.npy", export_collapsed_labels())
np.save(output_dir / "instance_seed_hits.npy", export_support_counts())
```

Meanings:

| File | Shape | Meaning |
| --- | --- | --- |
| `instance_gid_slots.npy` | `(n_points, point_gid_slots)` | Full multi-gid membership rows. |
| `instance_labels.npy` | `(n_points,)` | One collapsed gid per point, for visualization and single-label consumers. |
| `instance_seed_hits.npy` | `(next_gid,)` | Detector support-frame count per gid. |

`instance_labels.npy` uses support-score collapse:

```text
score = (support_frames << 42) | (point_count << 21) | last_support_frame
```

For each point row, the gid with the highest score wins. Empty rows become `-1`.

Important:

```text
The full instance representation is instance_gid_slots.npy.
instance_labels.npy is a collapsed convenience view.
```

### 8.4 Stats And Timing

`build_run_stats(...)` records run-level metadata:

```text
dataset_name
scene_name
n_frames
n_points
device
slam_module
map_every
max_total_points
total_points_original
match_distance_th
CLIP model settings
sam2_seed_frames
```

`mapper.save(...)` extends stats with SAM/SAM2 config and instance runtime counters, then writes:

```text
stats.json
```

Timing is written separately:

```text
timing.json
```

with:

```text
dataset_load_sec
frame_loop_sec
save.write_ply_sec
save.store_clip_sec
save.instance_labels_sec
save.stats_sec
save.save_total_sec
total_sec
```

Finally, `build_rgb_map.py` prints:

```python
print(json.dumps({"timing": timing_summary}, indent=2))
print(output_dir / "rgb_map.ply")
```

## Per-Frame Toy Example

Assume:

```text
frame_id = 16
map_every = 8
is_seed_frame = True
current map has 100,000 points
```

The frame has a valid pose and depth.

### Existing Match

The mapper projects existing points into the frame.

Suppose:

```text
20,000 map points are inside the camera frustum
12,000 pass the projected-depth threshold
```

Then:

```text
12,000 pixels in point_ids_after get existing point ids
those pixels are removed from the new-point candidate mask
```

### Seed Geometry

The frame is a seed frame, so the mapper runs SAM and depth normals.

Suppose:

```text
80,000 depth pixels are valid
12,000 are matched existing pixels
5,000 fail normal validity
```

Then roughly:

```text
63,000 pixels become new 3D points
```

Their ids are:

```text
100000 ... 162999
```

`point_ids_after` now contains:

```text
12,000 old ids
63,000 new ids
all other pixels = -1
```

### Instance Update

SAM local masks are converted into global gids.

If a chair mask overlaps mostly with existing gid `7`, and gid `7` is not leaking much outside the mask, the mask reuses gid `7` and adds gid `7` to new background points inside the mask.

If a table mask has visible points but no good existing gid, it creates a new gid, for example `gid=31`, and inserts `31` into all visible point rows under the mask.

At the end of the seed frame, the largest active gids are reseeded into SAM2. Frames `17` through `23` will use SAM2 propagation rather than fresh SAM automatic masks.

## Important Runtime Invariants

### Point Order Is The Shared Index Space

The row order is shared by:

```text
rgb_map.ply vertices
clip_feats.npy rows
instance_gid_slots.npy rows
instance_labels.npy rows
```

If point `i` is a chair point, then:

```text
PLY vertex i
CLIP feature row i
instance row i
collapsed label i
```

all describe the same 3D point.

### Geometry Birth Is Sparse In Time

Only frames satisfying:

```python
frame_id % map_every == 0
```

create new points.

This means increasing `map_every` reduces:

```text
number of points
number of CLIP extractions
number of SAM automatic-mask extractions
SAM2 reseed frequency
```

but also reduces geometric coverage and detector correction frequency.

### Non-Seed Frames Are Still Useful

Non-seed frames do not add geometry, but they can still:

```text
track active instances with SAM2
assign visible existing points to tracked gids
```

This is why `point_ids_after` is computed for every valid-pose frame, not only seed frames.

### Existing Point Fusion Is Conservative

An existing map point is matched only if:

```text
inside current camera frustum
projects inside image bounds
current measured depth is nonzero
projected depth agrees within match_distance_th
```

Unmatched pixels on seed frames create new points rather than forcing a nearest-neighbor merge.

### CLIP Features Are Birth-Time Features

CLIP features are not fused across views.

The feature for a point is:

```text
dense CLIP feature at the source pixel where the point was born
```

Colors and normals can be averaged across seed-frame reobservations, but CLIP is fixed at birth.

### Instance Labels Are Multi-Membership Internally

The saved collapsed label is not the whole story.

Actual state:

```text
instance_gid_slots.npy: point -> up to K gids
```

Convenience view:

```text
instance_labels.npy: point -> one support-score winner gid
```

This matters because later metrics/debuggers may use the full multi-gid membership rather than the collapsed file.

## Consumer Scripts

### `visualize_rgb_map.py`

Uses output artifacts as:

| Mode | Artifact |
| --- | --- |
| `rgb` | PLY colors. |
| `normals` | PLY normals colorized as `(normal + 1) / 2`. |
| `feat` | `clip_feats.npy` PCA projection. |
| `feature-similarity` | `clip_feats.npy` compared to text CLIP embeddings. |
| `instances` | `instance_labels.npy` colorized after optional min-component filtering. |

### `get_metrics_map.py`

Loads:

```text
rgb_map.ply
clip_feats.npy
instance_labels.npy
instance_gid_slots.npy
instance_seed_hits.npy
stats.json
```

and compares against ScanNet or Replica GT assets for geometry, RGB, normals, semantic/OVO-style features, and instance metrics.

### `topdown_vis.py`

Calls `run_scene_build(...)` directly with a `snapshot_hook`.

The hook records:

```text
frame ids where points were added
point counts after those additions
camera poses
```

After build, it renders incremental top-down videos from the final map.

## Object Glossary

| Object | Meaning |
| --- | --- |
| `dataset` | Lazy RGB-D frame reader for Replica or ScanNet. |
| `slam_backbone` | Pose provider: dataset poses, ORB-SLAM3, or cuVSLAM. |
| `estimated_c2w` | Camera-to-world pose used by the mapper for this frame. |
| `RGBMapper` | Owns live geometry, color/normal fusion, CLIP feature streaming, and instance runtime. |
| `point_ids_after` | Per-frame `H x W` image mapping pixels to global map-point ids after matching and birth. |
| `DenseCLIPExtractor` | CLIP ViT feature extractor that produces dense per-pixel features, optionally region-aware via SAM labels. |
| `SAMMaskExtractor` | Automatic mask generator that turns an RGB frame into a local SAM label map. |
| `SAMInstanceRuntime` | Maintains global instance gids and per-point gid slots. |
| `SAM2VideoTracker` | Tracks seed-frame gid masks through following non-seed frames. |
| `InstanceBucket` | Per-gid support and point-count bookkeeping. |
| `instance_gid_slots.npy` | Full per-point multi-instance membership. |
| `instance_labels.npy` | Collapsed one-label-per-point view. |
| `clip_feats.npy` | Birth-time float16 CLIP feature row per map point. |
