# Default `build_rgb_map.py` pipeline for your command

## Scope and effective defaults

This document assumes exactly the path you asked for:

- `--dataset_name ScanNet`
- `--config_path` left at its default (`configs/ovo.yaml`)

Important: this explanation is for the current cleaned repo state:

- there is no `--use-inst-gt`
- there is no old OVO-training / `--ovo-style-feature` / `--ovo-online-tracking` path anymore
- the only path described here is the default runtime that `build_rgb_map.py` actually executes

The effective defaults come from three places:

- map-build knobs such as `map_every`, `point_sample_stride`, `match_distance_th`, and `max_frame_points` come from the CLI defaults in `build_rgb_map.py`
- pose-backend defaults come from `configs/ovo.yaml`
- SAM/SAM2 defaults live down in the runtime wrappers:
  - `map_runtime/sam_masks.py`
  - `map_runtime/sam2_tracking.py`

So for your command, the effective defaults are:

- pose backend: `vanilla`
- `map_every = 8`
- `point_sample_stride = 2`
- `match_distance_th = 0.03` meters
- `max_frame_points = 5_000_000`
- seed instance extractor: SAM/SAM2 AMG at level `24`
- textregion extractor: same level `24`
- SAM2 tracker level: `24`

Important semantic point:

- `point_sample_stride` does **not** resize the dataset stream, the SAM input, or the CLIP input globally
- it only stride-samples the seed-frame geometry/fusion tensors right before normal estimation, RGB fusion, new-point insertion, and point-level instance attachment

Under the default `ScanNet` + `vanilla` path:

- each dataset item provides `(frame_id, rgb, depth, c2w, color_data)`
- `rgb` is the resized/cropped RGB image used by the mapper
- `depth` is in meters
- `c2w` is the pose loaded from `data/input/ScanNet/<scene>/pose/*.txt`

One structural fact controls everything below:

- only **seed frames** mutate the stored 3D map
- a seed frame is any frame with `frame_id % 8 == 0`
- non-seed frames do **not** append points, do **not** update RGB, do **not** update normals, and do **not** update CLIP features
- non-seed frames **do** use the tracker to propagate already-existing gids onto visible existing points

Also, the three saved point-aligned outputs use the same point ordering:

- point `i` in `rgb_map.ply`
- row `i` in `clip_feats.npy`
- entry `i` in `instance_labels.npy`

## (a) Geometry: world-space 3D points

### What is stored

The map stores one world-space 3D point per kept sample:

- `self.points[i] = [x, y, z]` in world coordinates

This point is written to `rgb_map.ply`.

### How a frame is processed for geometry

For each frame:

1. The mapper receives `depth`, `rgb`, and `c2w`.
2. It creates a full-resolution validity mask:
   - `mask = (depth > 0)`
3. It allocates a full-resolution image-sized point-id buffer:
   - `point_ids_full[v, u] = -1` initially

If the map already has points, it first tries to explain the current depth image with existing map points:

4. It computes the current camera frustum from the frame depth range and the current `c2w`.
5. It keeps only map points inside that frustum.
6. Each candidate map point is projected into the current image.
7. A projected map point is considered a match if:
   - it lands inside image bounds
   - the current depth pixel is nonzero
   - the point depth and image depth agree within `0.03 m`

Concretely, for an existing world point `p_world`, the code does:

```text
p_cam = w2c * [p_world, 1]
(u, v) = round( K * p_cam[:3] / p_cam.z )
accept if |p_cam.z - depth(v, u)| < 0.03
```

8. For matched pixels:
   - `point_ids_full[v, u]` is filled with the matched global point index
   - `mask[v, u]` is set to `False`

At this point, `mask` marks depth pixels that are still unexplained by the current map.

### When new 3D points are created

New 3D points are created **only on seed frames**.

On a seed frame:

10. The mapper samples the seed-frame fusion tensors by simple stride slicing:
    - `depth = depth[::2, ::2]`
    - `mask = mask[::2, ::2]`
11. It computes surface normals from this sampled depth grid.
12. It further restricts insertion to pixels with valid normals:
    - `mask = mask & normal_valid`

So a new 3D point is inserted only if all of the following are true at the sampled pixel:

- depth is nonzero
- no existing map point matched that pixel
- the local normal estimate is valid

If more than `5_000_000` such candidates exist, they are linearly subsampled.

### How the 3D coordinates are computed

For each kept sampled pixel `(u, v)` with depth `z`, the mapper backprojects using the camera intrinsics:

```text
x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
z_cam = z
```

Then it converts camera coordinates to world coordinates:

```text
p_world = c2w * [x_cam, y_cam, z_cam, 1]^T
```

The stored point is `p_world[:3]`.

### How geometry is updated later

Existing point positions are **never moved** and **never averaged**.

- once a point is appended to `self.points`, its XYZ stays fixed forever
- later observations only update that point’s RGB and normal

So geometry here is:

- append-only on seed frames
- no fusion into a new XYZ
- no pose-graph-style repositioning of old points in this mapper

## (b) RGB

### What is stored

For each map point, the mapper stores:

- `self.colors[i]`: current fused RGB, `uint8`
- `self.color_sum[i]`: accumulated float RGB sum
- `self.obs_count[i]`: number of RGB/normal fusion observations

Only `self.colors` is written to `rgb_map.ply`.

### Initial color at point birth

When a new point is appended on a seed frame:

1. The mapper samples the stride-sampled RGB image at the same pixels used for new-point insertion.
2. That sampled RGB triplet becomes the point’s initial color.
3. Internally:
   - `colors[i] = sampled_rgb`
   - `color_sum[i] = sampled_rgb`
   - `obs_count[i] = 1`

### How existing point colors are updated

Existing point colors are updated **only on seed frames**, and only for stride-sampled matched pixels whose normals are valid.

The relevant set is:

- `visible_existing = (point_ids_sampled >= 0) & normal_valid`

So even if a point matched geometrically at full resolution, it only contributes to RGB fusion if:

- that match survives the `point_sample_stride = 2` sampling
- a valid normal could be computed there

For every such visible sampled point:

1. The current RGB sample is added into `color_sum`.
2. `obs_count` is incremented by 1.
3. The stored RGB is replaced by the rounded running mean:

```text
color_mean = color_sum / obs_count
stored_color = round(color_mean)
```

### What does not happen

- no RGB update on non-seed frames
- no RGB update for unmatched points
- no RGB update for sampled pixels with invalid normals
- no weighted fusion beyond a simple arithmetic mean

So RGB is:

- initialized once when a point is born
- later fused by unweighted averaging on seed frames only

## (c) Normals

### What is stored

For each map point, the mapper stores:

- `self.normals[i]`: current fused unit normal in **world coordinates**
- `self.normal_sum[i]`: accumulated float sum of world normals

Only `self.normals` is written to `rgb_map.ply`.

### How per-frame normals are computed

Normals are computed from the sampled seed-frame depth grid on seed frames.

For each sampled pixel, the code first builds a camera-space vertex map:

```text
V(u, v) = [
  (u - cx) * z(u, v) / fx,
  (v - cy) * z(u, v) / fy,
  z(u, v)
]
```

Then, for interior pixels only, it forms finite differences:

```text
dx = V(u + 1, v) - V(u - 1, v)
dy = V(u, v + 1) - V(u, v - 1)
n_cam_raw = cross(dy, dx)
```

Then it normalizes:

```text
n_cam = n_cam_raw / ||n_cam_raw||
```

A normal is valid only if:

- the center depth is valid
- left/right/up/down neighbor depths are valid
- `||n_cam_raw|| > 1e-8`

### Normal orientation

The normal is flipped to face the camera.

The code uses the camera-space 3D point `V(u, v)` itself as the center ray direction. If

```text
dot(n_cam, V(u, v)) > 0
```

then the normal points away from the camera, so it is negated.

After this step, the valid normal is a camera-facing unit normal in camera coordinates.

### New-point normals

When a new point is born on a seed frame:

1. its camera-space normal is taken from `normals_cam`
2. that normal is rotated into world coordinates using only the rotation block of `c2w`

```text
n_world = R_c2w * n_cam
n_world = n_world / ||n_world||
```

3. the point is appended with:
   - `normals[i] = n_world`
   - `normal_sum[i] = n_world`

### Existing-point normal updates

Existing point normals are updated **only on seed frames**, and only for `visible_existing`.

For each such observation:

1. compute camera-space normal at the sampled pixel
2. rotate it into world coordinates
3. add it into `normal_sum`
4. increment `obs_count`
5. replace the stored normal with the normalized running mean

Concretely:

```text
mean_normal = normal_sum / obs_count
stored_normal = mean_normal / ||mean_normal||
```

### What does not happen

- no normal update on non-seed frames
- no normal update if the local depth neighborhood is invalid
- no geometry-based re-estimation of old point positions from normals

So normals are:

- computed from local depth geometry on seed frames
- stored in world coordinates
- fused by averaging then renormalizing

## (d) CLIP feature

### What is stored

In the current codepath, the mapper always uses `DenseCLIPExtractor`. There is no separate old OVO-style feature-bank path anymore.

For each point, it stores one dense CLIP feature vector:

- row `i` in `clip_feats.npy` corresponds to map point `i`
- dtype is `float16`
- no extra L2 normalization is applied before storage in this default path

Important: in this default path, these stored dense CLIP features are **not updated after point birth**.

### Which labels supervise dense CLIP regions

On seed frames, the dense CLIP extractor receives a per-pixel integer label map called `tr_labels_full`.

Under the exact default path here:

- `sam_model_level_inst == sam_model_level_textregion == 24`
- the same SAM extractor object is reused for both
- therefore `extract_textregion_labels(...)` returns the already computed seed labels

So by default, the dense CLIP regions are derived from the **same SAM segmentation** that seeded the instance pipeline on that seed frame.

### Dense CLIP extraction step by step

This happens only on seed frames.

`point_sample_stride` does not change the CLIP extractor resolution. CLIP still runs on the full mapper RGB frame and only the final point birth pixels are stride-sampled.

Given the current mapper RGB image:

1. Convert RGB to float in `[0, 1]`.
2. Resize so the shorter side is `1024`, preserving aspect ratio.
3. Normalize with CLIP mean/std.
4. Pad to a multiple of the ViT patch size.
5. Run the OpenCLIP visual encoder up to the last transformer block.
6. Manually extract the value branch from the last attention block to obtain per-patch features.
7. Project those into the CLIP feature space to get a baseline patch grid.

Then the textregion labels are used to region-condition those patch features:

8. Convert the pixel label map into per-region patch weights.
9. Remove patches that behave like overly global/background patches.
10. For each region, pool the last-block value features with those patch weights to get a region feature.
11. Push those region features back onto the patch grid.
12. For patches not covered by any remaining region, keep the baseline CLIP patch feature.
13. Bilinearly upsample the patch-grid feature map back to image resolution.

The result is a dense feature image:

- shape: `H x W x D`
- aligned to the mapper RGB image for that frame

### How point features are assigned

After dense extraction:

1. the mapper identifies the new-point pixels `(x_keep, y_keep)` on the seed frame
2. for each such new point, it samples:

```text
feature_i = dense_clip[y_keep[i], x_keep[i]]
```

3. that feature is cast to `float16`
4. it is appended to a temporary binary file in point order

At save time, that temporary stream is copied into `clip_feats.npy`.

### What does not happen

- no dense CLIP extraction on non-seed frames
- no CLIP update for existing points
- no temporal averaging of CLIP features across observations

So the dense CLIP feature for point `i` is simply:

- the dense feature sampled at the pixel where point `i` was first inserted
- from the seed frame where that point was born

## (e) Global instance IDs

### What is stored during mapping

The runtime now keeps two instance representations:

1. per-point multi-instance storage
   - `self.instance_manager.point_multi_labels[i, j]`
   - fixed width `K = 10`
   - each slot is either a global instance id `gid` or `-1`
2. a global bucket per surviving `gid`
   - `support_frames`
   - `num_points`
   - `last_support_frame`

This means a point can belong to multiple gids during mapping.

### What is saved

Two files are written:

- `instance_labels.npy`
  - the public single-label view consumed by the existing viewer / metrics scripts
  - one final gid per point, or `-1`
- `instance_state.npy`
  - the diagnostic dump of the full multi-label table plus bucket state

### Seed-frame logic

Seed frames are the only frames that decide instance identity.

For each local SAM mask on a seed frame:

1. Projected existing map points provide `point_ids_full`, so the mask can see which 3D points it covers.
2. Split those covered points into:
   - labeled points already carrying one or more gids
   - background points whose row is still all `-1`
3. Try to match the mask to an existing gid using the current low-level thresholds:
   - high in-mask coverage among labeled points
   - low outside-mask leakage among other visible labeled points
   - enough in-mask support points
4. If an existing gid is accepted:
   - only the background / unlabeled points under that seed mask are filled with that gid
   - already-labeled points are left as they are
5. If no existing gid is accepted:
   - a brand-new gid is created
   - that new gid is added to all visible 3D points under the seed mask
6. In both cases:
   - the gid gets one frame of support
   - the seed-frame per-pixel gid image is updated for tracker reseeding

### Non-seed logic

Non-seed frames do not birth new gids.

They only:

1. run SAM2 tracking from the latest seed-frame reseed
2. obtain gid-keyed tracked masks
3. for each tracked gid, add that gid onto visible 3D points under the tracked mask if those points do not already contain it
4. record one frame of support for that gid

So non-seed frames can update existing point-instance memberships, but they never create new 3D points or new gids.

### Pruning

Pruning is support-driven, not tracker-state-driven.

A gid is removed if it becomes stale or too weak according to the low-level constants in `map_runtime/instance_pipeline.py`.

When a gid is dropped:

- it is removed from the global bucket
- it is removed from every point’s 10-slot row
- rows are compacted so `-1` values move to the right

### Final public collapse

The public `instance_labels.npy` is produced only at save time.

For each point:

1. look at all valid gids remaining in its 10-slot row
2. keep the gid with:
   - highest `support_frames`
   - then highest `num_points`
   - then most recent `last_support_frame`
3. if the row is empty, save `-1`

## Short summary by saved output

- `rgb_map.ply`
  - stores final world XYZ, fused RGB, fused world normals
- `clip_feats.npy`
  - stores one dense CLIP feature per point, sampled once at point birth
- `instance_labels.npy`
  - stores the final single-label public collapse of the multi-instance state
- `instance_state.npy`
  - stores the full multi-instance diagnostic state
