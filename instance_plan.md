# Instance Pipeline Refactor Plan

This is the pipeline I will implement to replace the current SAM/SAM2 instance path in `build_rgb_map.py`.

The design goal is:

- remove the current ad hoc tentative / confirmed / inactive tracker-state machine
- make instance existence support-driven and pruning-driven
- allow a point to belong to multiple global instances during mapping
- still write the final public `instance_labels.npy` in the single-label format that the existing viewer / metrics scripts already consume
- save the richer multi-instance state separately for diagnostics

## 1. Core simplification

The tracker will stop being the source of truth for whether an instance exists.

Instead:

- seed frames decide instance identity
- non-seed frames only propagate already-chosen gids onto visible points
- repeated frame support validates a gid
- periodic pruning deletes weak or stale gids

The current logic based on:

- tentative / confirmed / inactive
- miss counts
- seed hit counters
- tracker reactivation

will be removed.

## 2. Canonical in-memory representation

During mapping, the canonical instance state will be:

### 2.1 Global instance bucket

For each global instance id `gid`, maintain:

- `support_frames`: number of unique frames that supported this gid
- `num_points`: number of 3D points that currently contain this gid
- `last_support_frame`: most recent frame id that supported this gid

This is the only confidence state for an instance.

### 2.2 Point instance labels

For each 3D point, maintain a fixed-width list of `K = 10` gids:

- shape: `[n_points, 10]`
- dtype: `int32`
- unused slots are `-1`
- rows are left-compacted, then `-1` padded
- the same gid is never stored twice in one row

This representation allows a point to belong to multiple instances, including hierarchical cases.

### 2.3 Tracker state

The tracker only stores the current seed-to-seed propagation state.

- tracker objects will be keyed directly by `gid`
- the tracker is not allowed to birth, delete, confirm, or invalidate instances
- it only proposes tracked masks for already-existing gids

## 3. Geometry-to-instance contract

The instance stage runs after geometry on seed frames.

The geometry stage must provide a full-resolution `point_ids_full` image for the current frame:

- existing matched points write their global point ids into `point_ids_full`
- newly birthed geometry points also write their new point ids back into `point_ids_full` at their sampled seed-frame pixel locations

The instance stage will only operate on pixels whose `point_ids_full` value is non-negative.

That means:

- no 3D point => ignored by the instance updater
- geometry decides what points exist
- the instance pipeline only assigns gids to those existing points

## 4. Seed-frame pipeline

For each seed frame:

1. Run geometry first.
2. Run SAM on the full-resolution RGB frame to get flattened local masks.
3. Ignore local mask id `-1`.
4. For each local mask `L`:
   - collect the unique visible point ids under `L`; call this set `X`
   - ignore pixels in `L` with no 3D point
   - if `|X|` is too small, skip the mask entirely
5. Define:
   - `X_labeled`: points in `X` whose K-row is not all `-1`
   - `Y`: all other visible point ids in the frame outside `L`
   - `Y_labeled`: points in `Y` whose K-row is not all `-1`
6. Candidate gids are the gids that appear anywhere in `X_labeled`.
7. For each candidate gid, compute:
   - `in_mask_coverage = (# points in X_labeled containing gid) / max(1, |X_labeled|)`
   - `outside_share = (# points in Y_labeled containing gid) / max(1, #points in X_labeled containing gid + #points in Y_labeled containing gid)`
8. Accept the best existing gid if:
   - `in_mask_coverage >= INSTANCE_MATCH_IN_FRAC_TH`
   - `outside_share <= INSTANCE_MATCH_OUT_FRAC_TH`
   - in-mask support points for that gid exceed `INSTANCE_MATCH_MIN_SUPPORT_POINTS`
9. If an existing gid is accepted:
   - that gid becomes `gid_L`
   - add `gid_L` only to the currently background points in `X`
   - record support for `gid_L` in this frame
10. If no existing gid is accepted:
   - allocate a new gid
   - that gid becomes `gid_L`
   - add `gid_L` to every point in `X`
   - initialize its bucket entry
   - record support for it in this frame

Important correction:

Once a gid is accepted for an existing-mask match, I will only fill the currently background in-mask points.

Reason:

- that follows the original rule you asked for
- it avoids gratuitously adding a strong old gid onto points that already carry other instance structure

This also naturally allows hierarchical memberships:

- a broad seed mask can create a parent gid over points that already carry child gids
- a later narrow seed mask can add a child gid onto points that already carry a parent gid

## 5. Non-seed-frame pipeline

For each non-seed frame:

1. Do not run new SAM births.
2. Use SAM2 only to propagate the previously chosen gid-keyed masks from the latest seed frame.
3. For each tracked gid-mask pair:
   - collect the unique visible point ids under the tracked mask; call this set `X`
   - ignore pixels with no 3D point
   - if `|X|` is too small, ignore this mask for support
   - for each point in `X` that does not already contain that gid, add it
   - record one frame of support for that gid

Important simplification:

- no tracker-driven births
- no tracker-driven kills
- no tracker-driven reactivation logic

If tracking degrades, that only reduces future support accumulation. Pruning handles removal.

## 6. Support bookkeeping

Support is counted at most once per gid per frame.

If a gid appears in multiple masks in the same frame:

- `support_frames` increments only once
- `last_support_frame` is updated once to the current frame

`num_points` changes only when a point truly gains or loses membership in that gid.

## 7. Pruning

Pruning runs every `INSTANCE_PRUNE_EVERY_FRAMES`.

A gid will be dropped if any of the following hold:

- `current_frame - last_support_frame > INSTANCE_PRUNE_MAX_AGE_FRAMES`
- `num_points < INSTANCE_PRUNE_MIN_POINTS`
- it is mathematically impossible for the gid to ever reach the required support count by the end of the sequence

The support threshold uses:

- `min_support_frames = ceil(INSTANCE_PRUNE_MIN_SUPPORT_FRAC * total_num_frames)`

I will not naively prune based on `support_frames < min_support_frames` in the middle of the run, because that would kill instances too early.

Instead, during the run:

- `remaining_frames = total_num_frames - current_frame - 1`
- prune if `support_frames + remaining_frames < min_support_frames`

At the end of the scene:

- prune any remaining gid whose `support_frames < min_support_frames`

When dropping a gid:

- remove it from the global bucket
- remove it from every row of the point-instance table
- compact each affected row left
- fill trailing slots with `-1`
- update `num_points` consistently
- stop reseeding / accepting tracker output for that gid

Gids will not be recycled after pruning.

## 8. K=10 overflow policy

Each point has exactly 10 slots.

If a point already has 10 gids and a new gid wants to be inserted:

- do not evict old labels
- drop the new insertion attempt
- increment a diagnostic counter such as `instance_label_overflow_drops`

This is the safest and simplest policy.

## 9. Hyperparameters

These will live as low-level constants in the instance module, not as top-level CLI arguments.

Planned constants:

- `INSTANCE_LABEL_SLOTS = 10`
- `INSTANCE_MATCH_IN_FRAC_TH = 0.90`
- `INSTANCE_MATCH_OUT_FRAC_TH = 0.10`
- `INSTANCE_MATCH_MIN_SUPPORT_POINTS`
- `INSTANCE_SEED_MIN_VISIBLE_POINTS`
- `INSTANCE_TRACK_MIN_VISIBLE_POINTS`
- `INSTANCE_PRUNE_EVERY_FRAMES`
- `INSTANCE_PRUNE_MAX_AGE_FRAMES = 2000`
- `INSTANCE_PRUNE_MIN_SUPPORT_FRAC = 0.05`
- `INSTANCE_PRUNE_MIN_POINTS = 5000`

These are the actual knobs required by the algorithm.

## 10. Final save contract

The mapping runtime will keep the full multi-instance representation in memory until the end of the scene and perform final pruning before writing outputs.

### 10.1 Public output used by existing scripts

The existing viewer / metrics scripts expect one integer instance label per point.

So before writing the public `instance_labels.npy`, I will collapse each point’s 10-slot row down to one final gid:

- among that point’s valid gids, choose the gid with the highest `support_frames`
- if there is a tie, break it by larger `num_points`
- if there is still a tie, break it by more recent `last_support_frame`
- if the row has no valid gids, save `-1`

This gives a final per-point single-label array:

- shape: `[n_points]`
- dtype: `int32`

That means:

- `visualize_rgb_map.py`
- `topdown_vis.py`
- `get_metrics_map.py`

can continue consuming `instance_labels.npy` without needing a representation change.

### 10.2 Diagnostic output with the full representation

In addition to the public single-label file, I will save a separate diagnostic `.npy` dump containing the full state built by the mapper after final pruning.

Planned contents:

- full per-point multi-label table: shape `[n_points, 10]`
- global bucket state for every surviving gid:
  - gid
  - support_frames
  - num_points
  - last_support_frame

Since you explicitly asked for an `.npy` file, the simplest implementation is to save a Python dict as an object `.npy` via `np.save(...)`.

Planned structure:

```python
{
    "point_instance_labels_multi": ...,
    "bucket_gid": ...,
    "bucket_support_frames": ...,
    "bucket_num_points": ...,
    "bucket_last_support_frame": ...,
}
```

This file is for debugging / later inspection and is not intended to be consumed by the current viewer / metrics code.

## 11. What will be removed from the current implementation

The following concepts will be deleted from the active instance path:

- single-label canonical `point_labels`
- tentative / confirmed / inactive statuses
- tracker miss counts
- tracker-driven instance death
- seed-hit counters
- birth-seed counters
- reactivation logic
- dominant-gid-under-mask as the sole identity rule
- any logic that assumes a point belongs to only one instance during mapping

## 12. Short summary

The final behavior will be:

- seed frames determine instance identity
- non-seed frames only propagate that identity onto visible 3D points
- support accumulation, not tracker state, decides whether a gid remains meaningful
- pruning is the only deletion mechanism
- multi-membership is first-class during mapping
- the final saved `instance_labels.npy` is collapsed back to one label per point so existing downstream scripts stay compatible
- the full multi-instance state is saved separately for diagnostics
