# Standalone SAM Instance Debugger Algorithm

## State
- `point_gids`: shape `(n_points, point_gid_slots)`, dtype `int32`, initialized to `-1`.
- `buckets`: `gid -> InstanceBucket(gid, support_frames, point_count, last_support_frame, birth_frame)`.
- `next_gid`: next global instance id to allocate.
- `seeded_gids`: gids currently seeded into the SAM2 tracker.
- `current_view`: per-frame snapshot used by the notebook widget.

## Frame Input
Each cached frame provides:
- `rgb`
- `is_seed_frame`
- `valid_pose`
- `point_ids_after`: per-pixel 3D point ids after the geometry stage
- `new_point_mask`

The debugger never changes geometry. It only updates instance state on top of cached 3D point ids.

## Per-Frame Flow
1. Load the next cached frame.
2. Compute `projected_before` by collapsing each visible point's `K` gids to one primary gid for visualization only.
3. Run either seed-frame logic or non-seed tracking logic.
4. Run pruning via `_prune(frame_id)`.
5. Compute `projected_after`.
6. Store the full per-frame snapshot in `current_view`.

## Seed Frame Logic
Fresh SAM masks are extracted with `SAMMaskExtractor.extract_labels(rgb)`.

### Seed mask preprocessing
For each local SAM mask:
1. Collect unique visible 3D point ids under the mask from `point_ids_after`.
2. If the mask has fewer than `min_mask_points` visible points, keep it as a no-op record.
3. Split mask points into:
   - `labeled_points`: points whose `point_gids` row contains at least one gid
   - `background_points`: points whose `point_gids` row is all `-1`
4. Propose `candidate_gid`:
   - Flatten all gids found on `labeled_points`
   - Count occurrences per gid
   - Compute `inside_frac = count(gid) / len(point_ids)`
   - This means unlabeled/background visible points count against reuse
   - Keep gids with `inside_frac >= reuse_inside_frac_th`
   - Choose the best kept gid by bucket score, tie-breaking by smaller gid

### Grouped reuse check
All seed masks proposing the same `candidate_gid` are unioned before the final reuse decision.

For each candidate gid group:
1. Union all masks, visible points, labeled points, and background points in the group.
2. Recompute:
   - `inside_frac`: fraction of all union visible points that contain that gid
   - `outside_frac`: fraction of all other visible points in the frame that contain that gid
3. Reuse the gid iff:
   - union visible points are non-empty
   - `inside_frac >= reuse_inside_frac_th`
   - `outside_frac <= reuse_outside_frac_th`
4. On reuse:
   - Add the gid only to union background points via `_add_gid_to_points(..., background_only=True)`
   - Record support if `len(union_points) >= min_track_visible_points`
   - Store the unioned binary mask for tracker seeding
   - Emit one `reuse_existing` decision per original local mask in the group
   - `added_points` in each decision is reported per local mask, not as the full union count
5. If the grouped reuse check fails, all records in that group fall back to the birth path below.

### Birth / no-op path
For every record not reused:
1. If it has zero visible points:
   - emit `noop_no_points`
2. Else:
   - allocate a new gid via `_create_gid(frame_id)`
   - add that gid to all visible points in the mask via `_add_gid_to_points(..., background_only=False)`
   - record support if `visible_points >= min_track_visible_points`
   - store the raw local mask for tracker seeding
   - emit `birth_new`

### Tracker reseed after a seed frame
After all seed masks are processed:
1. Collect `gid -> binary mask` for reused / birthed masks.
2. Keep only the largest `sam2_max_num_objects` masks by area.
3. Reset / restart `SAM2VideoTracker` on the current RGB frame with those masks.
4. Set `seeded_gids` to exactly the gids that were actually seeded.

## Non-Seed Frame Logic
No fresh SAM extraction is run.

1. If there is no active tracker or no `seeded_gids`, do nothing.
2. Append the RGB frame to `SAM2VideoTracker` and run tracking for this frame.
3. For each gid in `seeded_gids` that still exists in `buckets`:
   - get the tracked binary mask for that gid
   - collect unique visible 3D point ids under that mask
   - add the gid to those points via `_add_gid_to_points(..., background_only=False)`
   - record support if `visible_points >= min_track_visible_points`
   - emit:
     - `track_assign` if any visible points were found
     - `track_no_points` otherwise

## Point Update Rule
`_add_gid_to_points(point_ids, gid, background_only)` does:
1. Drop invalid point ids.
2. Deduplicate point ids.
3. Skip any point that already contains `gid`.
4. If `background_only=True`, only points whose entire row is `-1` are eligible.
5. If `background_only=False`, any point with at least one free slot is eligible.
6. Insert `gid` into the first free slot of each eligible point.
7. Count points with no free slot as `overflow_points`.
8. Increase `buckets[gid].point_count` by the number of successful insertions.

## Support Bookkeeping
`_record_support(gid, frame_id)`:
- increments `support_frames` only once per frame
- sets `last_support_frame = frame_id`

This is called:
- on seed reuse/birth if the mask has at least `min_track_visible_points` visible points
- on non-seed tracker assignment if the tracked mask has at least `min_track_visible_points` visible points

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

A gid is dropped if `stale or low_support or low_points`.

Dropping a gid does:
1. Remove that gid from every row in `point_gids`
2. Left-compact each affected row so all `-1` values move to the end
3. Remove the gid from `seeded_gids`
4. Remove the gid bucket

## Primary GID Collapse For Visualization
The algorithm keeps full multi-gid membership per point. The widget's projected views need a single gid per visible point, so `_project_primary_labels(...)` collapses each visible point row to one primary gid.

### Bucket score
Each gid gets a score:
- `(support_frames << 42) | (point_count << 21) | last_support_frame`

This means the ordering is:
1. larger `support_frames`
2. then larger `point_count`
3. then larger `last_support_frame`

### Collapse rule
For a visible point:
1. Read its `point_gids` row
2. Ignore `-1` entries
3. Pick the gid with the maximum bucket score
4. If every slot is `-1`, project `-1`
5. If two gids have exactly the same score, `argmax` keeps the first occurrence in that point's row

This primary gid is for visualization only. The full `K`-slot membership is still the actual instance state.
