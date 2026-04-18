# Standalone SAM Instance Debugger Algorithm

## Scope
This document describes the shared SAM instance runtime in [map_runtime/sam_instance_runtime.py](/robodata/smodak/repos/ovo/map_runtime/sam_instance_runtime.py).

- `build_rgb_map.py` uses that runtime directly on the live RGBMapper scene build.
- [map_runtime/sam_instance_debug.py](/robodata/smodak/repos/ovo/map_runtime/sam_instance_debug.py) subclasses that same runtime and adds cache stepping, visualization, and notebook controls on top of cached geometry from `build_rgb_cache.py`.

The runtime never changes geometry. It only consumes `point_ids_after` and updates instance state.

## Core State
- `point_gids`: shape `(n_points, point_gid_slots)`, dtype `int32`, initialized to `-1`.
- `buckets`: `gid -> InstanceBucket(gid, support_frames, point_count, last_support_frame, birth_frame)`.
- `next_gid`: next global instance id to allocate.
- `seeded_gids`: gids currently seeded into the SAM2 tracker.
- `current_view`: per-frame snapshot used by the notebook widget.
- `_gid_scores`: packed bucket scores used only for collapsed-gid visualization and candidate tie-breaking.

### Meaning of bucket fields
- `support_frames`: number of seed-frame detector validations for this gid. Tracker frames do not increment this.
- `point_count`: number of 3D points whose `point_gids` row currently contains this gid.
- `last_support_frame`: most recent seed frame that validated this gid. Tracker frames do not update this.
- `birth_frame`: frame where the gid was created.

## Cached Frame Input
Each cached frame contains:
- `frame_id`
- `is_seed_frame`
- `valid_pose`
- `rgb`
- `depth`
- `c2w`
- `point_ids_before`
- `point_ids_after`
- `new_point_mask`

The debugger uses:
- `rgb`
- `is_seed_frame`
- `valid_pose`
- `point_ids_after`
- `new_point_mask`

`point_ids_before` is kept in `current_view` for inspection, but the instance update logic itself uses `point_ids_after`.

## Per-Frame Flow
For each `step()`:
1. Load the next cached frame.
2. Snapshot `bucket_snapshot_before` and a full copy of `point_gids` for later before/after diffs.
3. Compute:
   - `projected_before` by collapsing visible `point_gids` rows to one primary gid per point
   - `all_gids_before` by projecting the full `K`-slot gid rows per visible point
4. Run one of:
   - invalid-seed special case
   - seed-frame logic
   - non-seed tracking logic
5. Run pruning via `_prune(frame_id)`.
6. Compute:
   - `projected_after`
   - `all_gids_after`
   - `point_gid_changes` by diffing the copied `point_gids` against the current one
7. Store everything into `current_view`, including:
   - decisions
   - `bucket_snapshot_before`
   - post-prune `bucket_snapshot`
   - full post-prune `bucket_snapshot_after`
   - `point_gid_changes`

Important implementation detail:
- pruning happens after the seed/tracker update for that frame
- so a gid can appear in `decisions` for that frame and still appear in `pruned_gids` for the same frame

## Invalid Seed Frame Special Case
If a frame is both:
- `is_seed_frame == True`
- `valid_pose == False`

then the debugger:
1. closes the active tracker if it exists
2. clears `seeded_gids`
3. resets `tracker_frame_idx` to `0`
4. does not run seed extraction

There is no analogous special-case skip for invalid non-seed frames. Non-seed frames still go through the tracker path; they just typically have no valid visible points in `point_ids_after`.

## Seed Frame Logic
Fresh SAM masks are extracted with:
- `SAMMaskExtractor.extract_labels(rgb)`

The result is a local-id label map. `build_label_masks(...)` converts it to:
- `(local_mask_id, binary_mask)` pairs
- sorted by descending mask area

### Seed mask preprocessing
For each local SAM mask:
1. Collect unique visible 3D point ids under the mask from `point_ids_after`.
2. Drop invalid point ids `< 0`.
3. If the mask has fewer than `min_mask_points` visible points:
   - keep it as a record with no candidate gid
   - it will later become `noop_no_points`
4. Split its visible points into:
   - `labeled_points`: points whose `point_gids` row contains at least one gid
   - `background_points`: points whose `point_gids` row is entirely `-1`
5. Propose `candidate_gid`:
   - flatten all gids found on `labeled_points`
   - count occurrences per gid
   - compute `inside_frac = count(gid) / len(point_ids)`
   - unlabeled/background visible points therefore count against reuse
   - keep only gids with `inside_frac >= reuse_inside_frac_th`
   - choose the best kept gid by bucket score, tie-breaking by smaller gid

### Grouped reuse check
All seed masks proposing the same `candidate_gid` are grouped together before the final reuse decision.

For each candidate gid group:
1. Union all binary masks in the group.
2. Union all visible points in the group.
3. Union all background points in the group.
4. Recompute:
   - `inside_frac`: fraction of all union visible points whose `point_gids` row already contains that gid
   - `outside_frac`: fraction of all other visible points in the frame whose `point_gids` row contains that gid
5. Reuse succeeds iff:
   - union visible points are non-empty
   - `inside_frac >= reuse_inside_frac_th`
   - `outside_frac <= reuse_outside_frac_th`
6. On reuse:
   - add the gid only to union background points via `_add_gid_to_points(..., background_only=True)`
   - if `len(union_points) >= min_track_visible_points`, call `_record_support(gid, frame_id)`
   - store the unioned binary mask in `gid_to_mask` for tracker seeding
   - emit one `reuse_existing` decision per original local mask in the group
   - per-decision `added_points` is reported per local mask, not as the union total
7. If the grouped reuse check fails, every record in that group falls back to the birth/no-op path below

### Birth / no-op path
For every seed record not reused:
1. If it has zero visible points:
   - emit `noop_no_points`
2. Else:
   - allocate a new gid with `_create_gid(frame_id)`
   - add that gid to all visible points in the mask via `_add_gid_to_points(..., background_only=False)`
   - if `visible_points >= min_track_visible_points`, call `_record_support(gid, frame_id)`
   - store the raw local binary mask in `gid_to_mask` for tracker seeding
   - emit `birth_new`

### Tracker reseed after a seed frame
After all seed masks are processed:
1. Collect `gid -> binary mask` for all reused and birthed masks.
2. Keep only the largest `sam2_max_num_objects` masks by area.
3. Reset or restart `SAM2VideoTracker` on the current RGB frame with those masks.
4. Set `seeded_gids` to exactly the gids that were actually seeded into SAM2.
5. Build `tracker_labels` from those seeded masks by painting larger masks first.

Also:
- `last_seed_labels` is updated to the raw seed SAM local-id map
- `last_seed_frame_id` is updated to this frame id

## Non-Seed Tracking Logic
No fresh detector extraction runs.

1. If there is no active tracker or `seeded_gids` is empty, do nothing.
2. Append the current RGB frame to `SAM2VideoTracker`.
3. Run tracker inference for this frame.
4. Build `tracker_labels` from the tracked gid masks by painting larger masks first.
5. For each gid in sorted `seeded_gids`:
   - if the gid no longer exists in `buckets`, skip it
   - if the tracker returned no mask for that gid, skip it
   - otherwise collect unique visible 3D point ids under that tracked mask from `point_ids_after`
   - add that gid to those points via `_add_gid_to_points(..., background_only=False)`
   - emit:
     - `track_assign` if any visible points were found
     - `track_no_points` if zero visible points were found

Tracker frames do not call `_record_support(...)`.

## Point Update Rule
`_add_gid_to_points(point_ids, gid, background_only)` does:
1. Drop invalid point ids.
2. Deduplicate point ids.
3. Read their current `point_gids` rows.
4. Skip any point that already contains `gid`.
5. If `background_only=True`, only rows that are entirely `-1` are eligible.
6. If `background_only=False`, every row is eligible, but insertion still requires at least one free slot.
7. Insert `gid` into the first free slot of each eligible row.
8. Count rows with no free slot as `overflow_points`.
9. Increase `buckets[gid].point_count` by the number of successful new insertions.

Important:
- `point_count` only increases on successful new slot insertions
- re-observing a point that already has the gid does not change `point_count`
- if a row is already full and missing this gid, that point contributes to `overflow_points`

## Support Bookkeeping
`_record_support(gid, frame_id)`:
- increments `support_frames` only once per frame
- sets `last_support_frame = frame_id`

This is called only on seed-frame detector validation:
- grouped `reuse_existing` if `len(union_points) >= min_track_visible_points`
- `birth_new` if `visible_points >= min_track_visible_points`

Tracker propagation does not increment `support_frames` and does not update `last_support_frame`.

## Pruning
Pruning runs only if:
- `prune_every_frames > 0`
- `frame_id > 0`
- `frame_id % prune_every_frames == 0`

Definitions:
- `min_support_frames = ceil(prune_min_support_ratio * total_frames)`
- `mature = frame_id - birth_frame >= min_support_frames`
- `stale = frame_id - last_support_frame > prune_stale_gap_frames`
- `low_support = mature and support_frames < min_support_frames`
- `low_points = mature and point_count < prune_min_points`

A gid is dropped iff:
- `stale or low_support or low_points`

Important:
- because `last_support_frame` is detector-only, tracker-only frames do not reset staleness
- because `support_frames` is detector-only, tracker-only propagation does not help the support-ratio criterion

Dropping a gid does:
1. Remove that gid from every row in `point_gids`
2. Left-compact each affected row so all `-1` values move to the end
3. Remove the gid from `seeded_gids`
4. Remove the gid bucket
5. Refresh `_gid_scores`

## Primary GID Collapse For Visualization
The algorithm keeps full multi-gid membership per point. The widget's collapsed projected views need one gid per visible point, so `_project_primary_labels(...)` collapses each visible point row to one primary gid.

### Bucket score
Each gid gets a packed score:
- `(support_frames << 42) | (point_count << 21) | last_support_frame`

So the ordering is:
1. larger `support_frames`
2. then larger `point_count`
3. then larger `last_support_frame`

### Collapse rule
For a visible point:
1. Read its `point_gids` row.
2. Ignore `-1` entries.
3. Pick the gid with the maximum packed bucket score.
4. If every slot is `-1`, project `-1`.
5. If two gids have exactly the same packed score, `np.argmax` keeps the first occurrence in that point's row.

This primary gid is for visualization only. The full `K`-slot row remains the actual instance state.

Important:
- `debugger.show(gid=...)` does not use the collapsed primary gid
- it shows all currently visible points whose `point_gids` row contains that gid anywhere

## Debugger Metrics Collapse
`CachedSAMInstanceDebugger.get_metrics(...)` computes instance metrics from a temporary `N x 1` point-label view derived from `point_gids`.

### Normal mode
If `use_optimal_collapse=False`:
1. Collapse every point row with `_collapse_point_gid_labels()`.
2. Apply `min_component_size` filtering.
3. Relabel kept gids contiguously.

### Optimal mode
If `use_optimal_collapse=True`:
1. Compute the usual collapsed labels only for diagnostics.
2. Match every predicted point to its nearest GT vertex and read that GT instance id.
3. For each gid:
   - collect all predicted points whose `point_gids` row contains that gid anywhere
   - collect those points' GT instance ids
   - set the gid's `target_gt_instance` to the majority GT instance
   - compute:
     - `intersection = number of member points on target_gt_instance`
     - `purity = intersection / number of gid member points`
4. For each GT instance, choose one canonical gid by ranking gids with:
   1. larger `intersection`
   2. larger `purity`
   3. larger `support_frames`
   4. larger `point_count`
   5. smaller gid
5. Build the final `N x 1` labels:
   - start from all `-1`
   - for each point:
     - read its GT instance id
     - find the canonical gid for that GT instance
     - if that canonical gid is present anywhere in the point's `K` slots, assign that gid
     - otherwise leave the point as `-1`
6. Apply `min_component_size` filtering.
7. Relabel kept gids contiguously.

Important:
- in optimal mode there is no normal-collapse fallback in the final labels
- points that do not contain their GT instance's canonical gid stay `-1`

### Optimal-collapse diagnostics
When `use_optimal_collapse=True`, `get_metrics(...)` also reports:
- `use_optimal_collapse`: whether oracle collapse mode was used
- `optimal_collapse_gt_instances`: number of GT instances that received a canonical gid
- `optimal_collapse_points_matched_canonical`: number of predicted points whose row contains their GT instance's canonical gid
- `optimal_collapse_points_overridden`: number of those points whose canonical gid differs from the usual collapse result

### Metrics video artifact
Every `get_metrics(...)` call writes an `.mp4` under:
- `cache_dir/debug_videos/<mode>_collapse_frame_<frame>_mincomp_<min_component_size>.mp4`

Here:
- `<mode>` is `optimal` when `use_optimal_collapse=True`
- `<mode>` is `normal` when `use_optimal_collapse=False`

This video shows, for every frame from `0` to the current debugger frame:
- the exact same projected predicted-point set in both panes
- left pane: the exact point labels that were used for metrics
- right pane: GT instance labels transferred onto those same predicted points by nearest GT point
- rendered as a side-by-side label-map view
- no RGB overlay

The CLI metrics path in `get_metrics_map.py` also writes:
- `scene_output/debug_videos/instance_labels_mincomp_<min_component_size>.mp4`

That CLI video uses the final RGBMapper instance labels after `min_component_size` filtering.
Its GT pane uses GT instance labels transferred onto the same predicted points by nearest GT point.

## Concrete Optimal-Collapse Example
Assume:
- GT instance `0` = object `A`
- GT instance `1` = object `B`
- predicted points `P1..P5` are nearest to GT instance `0`
- predicted points `P6..P10` are nearest to GT instance `1`

Assume the current `N=10 x K=10` `point_gids` rows are:
- `P1: [2, 4, -1, ...]`
- `P2: [2, 5, -1, ...]`
- `P3: [2, -1, -1, ...]`
- `P4: [2, 5, -1, ...]`
- `P5: [5, -1, -1, ...]`
- `P6: [7, 9, -1, ...]`
- `P7: [7, -1, -1, ...]`
- `P8: [7, 9, -1, ...]`
- `P9: [7, 9, -1, ...]`
- `P10: [9, -1, -1, ...]`

Assume the usual collapse would pick:
- `P1 -> 4`
- `P2 -> 5`
- `P3 -> 2`
- `P4 -> 5`
- `P5 -> 5`
- `P6 -> 9`
- `P7 -> 7`
- `P8 -> 9`
- `P9 -> 9`
- `P10 -> 9`

So the usual collapsed labels are:
- `[4, 5, 2, 5, 5, 9, 7, 9, 9, 9]`

For each gid, gather all predicted points that contain it:
- `gid 2` -> `P1 P2 P3 P4`
- `gid 4` -> `P1`
- `gid 5` -> `P2 P4 P5`
- `gid 7` -> `P6 P7 P8 P9`
- `gid 9` -> `P6 P8 P9 P10`

Map those member points to GT instance ids:
- `gid 2` -> GT `0, 0, 0, 0`
- `gid 4` -> GT `0`
- `gid 5` -> GT `0, 0, 0`
- `gid 7` -> GT `1, 1, 1, 1`
- `gid 9` -> GT `1, 1, 1, 1`

Canonical-gid selection:
- GT instance `0` gets canonical gid `2`
- GT instance `1` gets canonical gid `7`

Final optimal collapse:
- `P1` has canonical gid `2` in its row -> assign `2`
- `P2` has canonical gid `2` in its row -> assign `2`
- `P3` has canonical gid `2` in its row -> assign `2`
- `P4` has canonical gid `2` in its row -> assign `2`
- `P5` does not have canonical gid `2` -> assign `-1`
- `P6` has canonical gid `7` in its row -> assign `7`
- `P7` has canonical gid `7` in its row -> assign `7`
- `P8` has canonical gid `7` in its row -> assign `7`
- `P9` has canonical gid `7` in its row -> assign `7`
- `P10` does not have canonical gid `7` -> assign `-1`

So the final optimal labels are:
- `[2, 2, 2, 2, -1, 7, 7, 7, 7, -1]`

Diagnostics in this example:
- `optimal_collapse_gt_instances = 2`
- `optimal_collapse_points_matched_canonical = 8`
- `optimal_collapse_points_overridden = 6`

## Notebook State-Transition Snapshot
For debugging and UI only, every `current_view` also stores:
- `bucket_snapshot_before`
- `bucket_snapshot_after`
- `point_gid_changes`

These are derived by diffing the pre-step copied state against the post-step state. They are not additional algorithm state; they are just debugging snapshots for the widget.
