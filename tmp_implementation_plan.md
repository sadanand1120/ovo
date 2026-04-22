## Goal

Remove `K -> 1` collapse from **instance AP evaluation**.

After this change:

- default mode (`use_postpruning_mode=None`) will evaluate **all surviving gids directly** as overlapping predicted instances.
- `optimal` mode will use the exact offline oracle pruning rule from `correct_collapse.md`.
- `learned` mode will predict the same **keep / prune** signal as that oracle-selected gid set, then evaluate the learned-kept gids directly.

`K -> 1` collapse will remain only where a single label per point is still genuinely needed. In those places, the default collapse rule is the existing support-score ordering. Concretely, the map build should export both:

- the full multi-slot instance representation
- the support-score-collapsed `N x 1` representation for downstream consumers that genuinely require one label per point

---

## Core design decisions

### 1. Split the metric path into two separate concepts

Right now the code wrongly reuses one collapsed `metric_point_instance_labels` array for both:

- instance AP
- semantic OVO-style metrics

I will split those.

After the refactor:

- **instance AP path**
  - works on a **set of predicted gids**
  - each gid owns a binary membership mask over points
  - no disjointness assumption
  - no `K -> 1` collapse

- **semantic OVO-style path**
  - uses the support-score-collapsed single-label-per-point view
  - this becomes explicitly separate from instance AP
  - `use_postpruning_mode` will no longer affect semantic OVO-style metrics

### 2. Define the prediction unit for instance AP

For instance AP, one predicted instance = one surviving global gid.

For gid `g`, its predicted support on the predicted map is:

- all point ids `p` such that `point_gids[p]` contains `g` in any slot

Its score is:

- `support_frames(g) / num_seed_frames_processed`

Tie-break order for equal scores:

- smaller gid first

That gives a deterministic prediction list.

### 3. Define “surviving gid”

A gid survives the pre-AP filtering stage iff:

- it exists in `buckets`
- its predicted-map point membership count is at least `min_component_size`
- and, in learned mode only, it is also in the learned-kept gid set
- and, in optimal mode only, it is also in the oracle-kept gid set

Equivalently:

- `use_postpruning_mode=None`
  - keep all gids that pass the base size filter
- `use_postpruning_mode="optimal"`
  - keep only the oracle-selected subset of those gids
- `use_postpruning_mode="learned"`
  - keep only the learned-selected subset of those gids

Important:

- the `min_component_size` filter is still applied on predicted-map membership count
- this happens **before** AP evaluation

### 4. Optimal mode is no longer a pointwise label override

Current “optimal collapse” is pointwise canonical relabeling. That entire idea goes away.

New `optimal` post-pruning mode:

1. Build the same base predicted gid set as the default mode.
2. Compute the IoU matrix between predicted gids and GT instances.
3. Solve the exact oracle pruning problem from `correct_collapse.md` at one explicit IoU threshold.
4. Keep only the oracle-selected gids.
5. Run the normal class-agnostic AP computation on that pruned gid set.

This means:

- optimal mode selects a **subset of gids**
- it does **not** rewrite point labels

### 5. Learned mode copies the oracle keep / prune signal

The learned model returns to the earlier single-head keep classifier.

Training target:

- `1` if gid is in the oracle-selected gid set
- `0` otherwise

At metric time in learned mode:

1. use the already-fit model to predict kept gids
2. drop the rejected gids
3. run the same overlapping-instance AP evaluation as the default mode

No post-pruning support-collapse step.

### 6. Oracle threshold must be explicit

`correct_collapse.md` is threshold-specific.

So I will introduce one explicit threshold parameter for the oracle selection step, with default:

- `oracle_prune_iou_th = 0.25`

Meaning:

- `optimal` mode optimizes AP@25 selection
- the selected gid subset is then evaluated with the existing reporting pipeline for:
  - `ap_25`
  - `ap_50`
  - ...
  - mean `ap`

This also becomes the supervision source for learned mode.

---

## Required code changes

### A. Replace the instance-AP representation

#### File
- `map_runtime/sam_instance_runtime.py`

#### Current problem

These functions are built around `N x 1` labels:

- `_collapse_point_gid_labels(...)`
- `_resolve_optimal_metric_instance_labels(...)`
- `_resolve_metric_instance_gid_labels(...)`

For AP they are the wrong abstraction.

#### Change

Keep `_collapse_point_gid_labels(...)` only for:

- debug panel projected primary-label views
- any semantic-only path that still needs one label per point

Remove it from instance AP logic.

Add new shared helpers in `SAMInstanceRuntime`:

1. `_collect_metric_gid_candidates(...)`
   - inputs:
     - `min_component_size`
     - optional `allowed_gids`
   - outputs per gid:
     - `gid`
     - `point_ids`
     - `score`
     - `point_count`

2. `_transfer_gid_candidates_to_gt(...)`
   - inputs:
     - predicted map points
     - candidate gid point memberships
     - GT points
   - output:
     - GT-space binary membership masks or equivalent sparse representation, one per gid
   - implementation:
     - build one shared 5-NN from GT points to valid predicted points
     - for each gid, apply majority vote on boolean gid-membership over those 5 neighbors
   - this preserves overlapping predicted instances

3. `_build_gid_iou_matrix(...)`
   - inputs:
     - GT instance labels on GT points
     - GT-space predicted gid memberships
   - output:
     - `iou[pred_gid_idx, gt_idx]`
     - ordered `pred_gids`
     - ordered `gt_instance_ids`

4. `_solve_oracle_pruning_for_ap(...)`
   - exact CP-SAT implementation from `correct_collapse.md`
   - deterministic ordering:
     - descending score
     - then smaller gid
   - deterministic GT preference:
     - larger IoU
     - then smaller GT id
   - output:
     - selected gid set
     - matched GT ids
     - solver diagnostics

5. `_resolve_metric_instance_candidates(...)`
   - replaces `_resolve_metric_instance_gid_labels(...)`
   - returns the **selected predicted gid set** plus the GT-space IoU inputs needed by AP
   - modes:
     - `None`
     - `optimal`
     - `learned`

### B. Add a direct class-agnostic AP path from IoU + scores

#### Files
- `map_runtime/metrics_utils.py`
- possibly small wrapper in `get_metrics_map.py`

#### Current problem

`compute_instance_metrics(...)` currently expects:

- `pred_instance_labels` as one label per GT point
- `pred_instance_scores` indexed by relabeled ids

That assumes disjoint predictions.

#### Change

Add a new helper:

- `compute_instance_metrics_from_iou(...)`

Inputs:

- `iou` matrix directly
- `pred_scores`
- ordered GT ids
- ordered predicted gid ids

Behavior:

- uses the existing `compute_instance_ap_dataset(...)`
- feeds one class-agnostic entry exactly as before
- no single-label relabeling step

Then:

- keep the old `compute_instance_metrics(...)` only if still needed elsewhere
- switch debugger + `get_metrics_map.py` to the new IoU-based path

### C. Change debugger metric evaluation

#### File
- `map_runtime/sam_instance_debug.py`

#### Current problem

`debugger.get_metrics(...)` currently:

1. resolves `metric_gid_labels`
2. calls `finalize_instance_labels_and_scores(...)`
3. calls `transfer_instance_labels_ovo_style(...)`
4. computes AP

This is the collapse-based path that needs to go away for instance AP.

#### Change

In `debugger.get_metrics(...)`:

1. keep the current GT / map loading
2. keep the current semantic OVO path, but make it explicitly separate
3. replace the instance path with:
   - collect candidate gids
   - apply default / optimal / learned post-pruning selection
   - transfer selected gid memberships to GT geometry
   - build IoU matrix
   - compute AP directly from IoU + gid scores

Diagnostics to update:

- `pred_instance_count`
  - becomes number of selected surviving gids used in AP
- replace old collapse diagnostics with selection diagnostics

Examples:

- remove:
  - `optimal_collapse_gt_instances`
  - `optimal_collapse_points_matched_canonical`
  - `optimal_collapse_points_overridden`
- add:
  - `selected_pred_gids`
  - `selected_pred_instance_count`
  - `oracle_prune_iou_th`
  - solver stats in optimal mode

### D. Change learned fit supervision

#### File
- `map_runtime/sam_instance_debug.py`

#### Current problem

The fit path currently still assumes the old optimal-selection summary shape.

#### Change

`debugger.fit(...)` will:

1. resolve the oracle-selected gid set using the new optimal selection path
2. build gid-level features exactly as before
3. train the single-head keep classifier against that selected gid set
4. store:
   - `target_selected_gids`
   - `learned_pred_selected_gids`
   - probability table per gid

No ranking head.
No ranking diagnostics.
No collapse-based supervision.

### E. Change `get_metrics_map.py`

#### File
- `get_metrics_map.py`

#### Current problem

It currently loads one scalar label per predicted point and then finalizes / relabels / transfers that.

That path must change for instance AP.

#### Change

Split the offline evaluation into:

1. **semantic OVO-style path**
   - use the saved support-score-collapsed single-label representation directly
   - do not reuse the instance AP post-pruning mode here

2. **instance AP path**
   - load raw gid slots
   - load support scores
   - collect candidate gids
   - run `use_postpruning_mode=None/"optimal"/"learned"` selection logic as appropriate
   - transfer selected gid memberships to GT geometry
   - compute IoU + AP directly

For the main CLI path:

- the normal pipeline will effectively use `use_postpruning_mode=None`
- if later we expose optimal / learned offline evaluation in the CLI, the shared helper path is already ready

### F. Change saved map outputs

#### File
- `build_rgb_map.py`

#### Current problem

It currently writes only one collapsed scalar label export. That is not enough anymore because:

- instance AP needs the full multi-slot representation
- semantic OVO-style still wants a single-label-per-point view

#### Change

Persist both representations explicitly:

- `instance_gid_slots.npy`
  - full raw shape `(N, K)`
  - source of truth for instance AP
- `instance_labels.npy`
  - support-score-collapsed shape `(N,)`
  - used wherever a single-label-per-point representation is still needed

Implementation:

- add `export_point_gids()` in `SAMInstanceRuntime`
- keep `export_collapsed_labels()`, but make it explicitly “support-score collapsed”
- write both files from `build_rgb_map.py`
- make `stats.json` describe both paths clearly

Files/scripts that must be updated accordingly:

- `get_metrics_map.py`
  - raw slots for instance AP
  - collapsed labels for semantic OVO-style if needed
- `visualize_rgb_map.py`
  - can keep using `instance_labels.npy` if it wants one color per point
- docs that describe the saved outputs

### G. Update visualization / video behavior

#### Files
- `map_runtime/instance_label_video.py`
- `map_runtime/sam_instance_debug.py`

#### Current problem

The current instance-label video path assumes one label per point.

With overlapping predicted instances, that is no longer the exact evaluation state.

#### Change

Do **not** keep pretending the single-label video is exact for instance AP.

Recommended implementation:

- remove the current metric-video dependency on collapsed `metric_point_instance_labels`
- either:
  - temporarily skip metric instance video export in the AP path, or
  - replace it with an explicitly labeled overlay renderer later

I will take the first route in the implementation pass unless you want a separate overlay design immediately.

Reason:

- a single-color-per-pixel video is not an exact representation of overlapping predicted instances

### H. Update debugger / notebook plumbing

#### Files
- `sam_instance_debugger.ipynb`
- `instance_algo.md`
- `temp_explain.md` if we still keep it

#### Change

Notebook:

- update comments and examples so `use_postpruning_mode=None/"optimal"/"learned"` are described as **gid selection modes**, not point-collapse modes
- learned mode note must say:
  - “requires `debugger.fit(...)` first”
  - “learned mode prunes gids only; instance AP is still computed on overlapping gid memberships”

Docs:

- rewrite the metric section of `instance_algo.md`
- remove any description of optimal mode as pointwise canonical reassignment
- document the new oracle pruning rule and the explicit `oracle_prune_iou_th`

---

## Exact algorithm after the refactor

### Default mode (`use_postpruning_mode=None`)

1. Start from raw `point_gids` (`N x K`).
2. For each gid:
   - membership = all points whose row contains gid
   - score = seed support ratio
3. Drop gids whose predicted-map membership count is below `min_component_size`.
4. Transfer each remaining gid independently to GT geometry.
5. Build IoU matrix between:
   - predicted gids
   - GT instances
6. Compute AP with the normal greedy score-ordered matcher.

No collapse anywhere in this path.

### Optimal mode (`use_postpruning_mode="optimal"`)

1. Build the same surviving gid set as the default mode.
2. Build the same IoU matrix.
3. Run the exact oracle pruning solver from `correct_collapse.md` at `oracle_prune_iou_th`.
4. Keep only that selected gid subset.
5. Compute AP with the normal greedy matcher on that selected subset.

No pointwise reassignment.

### Learned mode (`use_postpruning_mode="learned"`)

1. `debugger.fit(...)` learns keep / prune on gids using oracle-selected gids as targets.
2. `debugger.get_metrics(..., use_postpruning_mode="learned")`:
   - loads the learned predicted selected gid set
   - drops rejected gids
   - runs the same overlapping-instance AP path as the default mode

No post-pruning collapse.

---

## Implementation order

1. Add the new shared gid-candidate / GT-transfer / IoU / oracle-prune helpers.
2. Switch debugger instance AP to the new path.
3. Switch learned fit supervision to the new oracle-selected gid set.
4. Switch `get_metrics_map.py` instance AP to the new path.
5. Change `build_rgb_map.py` export to raw gid slots + support-collapsed labels.
6. Update any visualization / docs to distinguish:
   - `instance_gid_slots.npy` as the raw source of truth
   - `instance_labels.npy` as the support-collapsed single-label view
7. Remove dead collapse-only metric helpers and stale diagnostics.

---

## Main risks to watch while implementing

1. **Semantic OVO coupling**
   - must not accidentally change semantic OVO-style behavior while changing instance AP

2. **Disk format change**
   - every reader must be updated to use the correct file:
     - raw slots for instance AP
     - support-collapsed labels for single-label consumers

3. **Video exactness**
   - current metric video code becomes logically invalid under overlapping-instance AP

4. **Oracle threshold ambiguity**
   - must keep one explicit `oracle_prune_iou_th`
   - learned supervision must use that exact same threshold

5. **Determinism**
   - solver input ordering must be fixed exactly:
     - score descending
     - gid ascending
   - GT preference must be:
     - IoU descending
     - GT id ascending
