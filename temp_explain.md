# Concrete Toy Example: `support` Collapse vs `optimal` Collapse, then `ap_25`

This matches the current code path in:

- `map_runtime/sam_instance_runtime.py`
  - `_collapse_point_gid_labels`
  - `_resolve_optimal_metric_instance_labels`
  - `_raw_instance_support_scores`
- `map_runtime/metrics_utils.py`
  - `finalize_instance_labels_and_scores`
  - `compute_instance_ap_dataset`
- `get_metrics_map.py`
  - `transfer_instance_labels_ovo_style`
  - `compute_instance_metrics`

## 1. Exact objects involved

At metric time, the instance pipeline uses these objects:

1. `point_gids`
   - Shape: `N_pred_points x K`
   - For each predicted map point, stores up to `K` global instance ids.
   - Empty slots are `-1`.

2. `buckets[gid]`
   - For each global instance id `gid`, stores:
     - `support_frames`
     - `point_count`
     - `last_support_frame`
     - `birth_frame`

3. `raw_instance_scores`
   - Computed from buckets by `_raw_instance_support_scores()`
   - `raw_instance_scores[gid] = support_frames(gid) / num_seed_frames_processed`

4. `metric_gid_labels`
   - One label per predicted point after collapse.
   - Still uses original global gids at this stage.

5. `metric_point_instance_labels`, `metric_instance_scores`
   - Output of `finalize_instance_labels_and_scores(...)`
   - Point labels are relabeled to contiguous ids `0, 1, 2, ...`
   - `metric_instance_scores[new_label]` is the support-ratio score of the original gid that survived.

6. GT-side transferred labels
   - `transfer_instance_labels_ovo_style(...)` projects predicted point-instance labels onto GT points using 5-NN majority vote.

7. AP inputs
   - `gt_instance_labels` on GT points
   - `transferred_instance_labels` on GT points
   - `metric_instance_scores`

## 2. Toy assumptions

Assume:

- `K = 3` for readability.
- There are `8` predicted map points: `P0 ... P7`.
- There are `10` GT points: `G0 ... G9`.
- There are `2` GT instances:
  - GT instance `A`: `G0,G1,G2,G3,G4`
  - GT instance `B`: `G5,G6,G7,G8,G9`
- There are `3` global gids in the map: `7`, `12`, `20`.
- Current number of processed seed frames is `10`.
- `min_component_size = 2`.

Current bucket state:

| gid | support_frames | point_count | last_support_frame | support ratio |
|---|---:|---:|---:|---:|
| 7  | 9 | 6 | 64 | 0.9 |
| 12 | 6 | 5 | 64 | 0.6 |
| 20 | 2 | 4 | 56 | 0.2 |

So:

- `raw_instance_scores[7]  = 0.9`
- `raw_instance_scores[12] = 0.6`
- `raw_instance_scores[20] = 0.2`

Current learned map representation:

| predicted point | `point_gids` row |
|---|---|
| `P0` | `[7, 12, -1]` |
| `P1` | `[7, -1, -1]` |
| `P2` | `[7, 20, -1]` |
| `P3` | `[12, 7, -1]` |
| `P4` | `[12, -1, -1]` |
| `P5` | `[20, 12, -1]` |
| `P6` | `[20, -1, -1]` |
| `P7` | `[20, 12, -1]` |

Also assume each predicted point has a nearest GT point, used by `pred_to_gt_idx` in the `optimal` path:

| predicted point | nearest GT point | nearest GT instance |
|---|---|---|
| `P0` | `G0` | `A` |
| `P1` | `G1` | `A` |
| `P2` | `G2` | `A` |
| `P3` | `G5` | `B` |
| `P4` | `G6` | `B` |
| `P5` | `G7` | `B` |
| `P6` | `G8` | `B` |
| `P7` | `G9` | `B` |

## 3. `support` collapse: exact flow

The code path is:

1. `_resolve_metric_instance_gid_labels(collapse_mode="support")`
2. `_collapse_point_gid_labels()`
3. `finalize_instance_labels_and_scores(...)`
4. `transfer_instance_labels_ovo_style(...)`
5. `compute_instance_metrics(...)`

### 3.1 Per-point collapse in `support` mode

`_collapse_point_gid_labels()` picks the gid with the largest packed score:

`(support_frames << 42) | (point_count << 21) | last_support_frame`

So the priority is:

1. higher `support_frames`
2. if tied, higher `point_count`
3. if tied, higher `last_support_frame`

In this toy example:

`gid 7 > gid 12 > gid 20`

So each point collapses like this:

| point | row | chosen gid in `support` mode |
|---|---|---:|
| `P0` | `[7, 12, -1]` | `7` |
| `P1` | `[7, -1, -1]` | `7` |
| `P2` | `[7, 20, -1]` | `7` |
| `P3` | `[12, 7, -1]` | `7` |
| `P4` | `[12, -1, -1]` | `12` |
| `P5` | `[20, 12, -1]` | `12` |
| `P6` | `[20, -1, -1]` | `20` |
| `P7` | `[20, 12, -1]` | `12` |

So the raw collapsed labels are:

`support_raw = [7, 7, 7, 7, 12, 12, 20, 12]`

### 3.2 Finalization after `support` collapse

`finalize_instance_labels_and_scores(...)` does two things:

1. Drops any component whose size is `< min_component_size`
2. Relabels surviving gids to contiguous ids `0,1,2,...`

Counts in `support_raw`:

- gid `7`: 4 points
- gid `12`: 3 points
- gid `20`: 1 point

Since `min_component_size = 2`, gid `20` is removed.

After removal:

`support_filtered = [7, 7, 7, 7, 12, 12, -1, 12]`

Surviving original gids, in sorted order:

`[7, 12]`

They are relabeled to:

- original gid `7`  -> metric label `0`
- original gid `12` -> metric label `1`

So:

`metric_point_instance_labels_support = [0, 0, 0, 0, 1, 1, -1, 1]`

And the metric scores become:

`metric_instance_scores_support = [0.9, 0.6]`

Those scores are still support ratios. Nothing GT-aware has happened yet.

## 4. `optimal` collapse: exact flow

The code path is:

1. `_resolve_metric_instance_gid_labels(collapse_mode="optimal", pred_to_gt_idx, gt_instance_labels)`
2. `_resolve_optimal_metric_instance_labels(...)`
3. `finalize_instance_labels_and_scores(...)`
4. `transfer_instance_labels_ovo_style(...)`
5. `compute_instance_metrics(...)`

### 4.1 Canonical gid selection per GT instance

`_resolve_optimal_metric_instance_labels(...)` first finds, for each gid, which GT instance it mostly corresponds to.

For each gid, collect all predicted points that contain that gid anywhere in their row.

#### gid `7`

Points containing gid `7`:

- `P0, P1, P2, P3`

Their nearest GT instances:

- `A, A, A, B`

Counts:

- `A: 3`
- `B: 1`

So gid `7` votes for GT instance `A`, with:

- `intersection = 3`
- `purity = 3 / 4 = 0.75`
- `support_frames = 9`
- `point_count = 6`

Its rank tuple is:

`(3, 0.75, 9, 6, -7)`

#### gid `12`

Points containing gid `12`:

- `P0, P3, P4, P5, P7`

Their nearest GT instances:

- `A, B, B, B, B`

Counts:

- `A: 1`
- `B: 4`

So gid `12` votes for GT instance `B`, with:

- `intersection = 4`
- `purity = 4 / 5 = 0.8`
- `support_frames = 6`
- `point_count = 5`

Its rank tuple is:

`(4, 0.8, 6, 5, -12)`

#### gid `20`

Points containing gid `20`:

- `P2, P5, P6, P7`

Their nearest GT instances:

- `A, B, B, B`

Counts:

- `A: 1`
- `B: 3`

So gid `20` votes for GT instance `B`, with:

- `intersection = 3`
- `purity = 3 / 4 = 0.75`
- `support_frames = 2`
- `point_count = 4`

Its rank tuple is:

`(3, 0.75, 2, 4, -20)`

### 4.2 Canonical gid chosen for each GT instance

For GT instance `A`, only gid `7` voted for it, so:

- canonical gid for `A` = `7`

For GT instance `B`, gid `12` and gid `20` voted for it.

Compare their rank tuples:

- gid `12`: `(4, 0.8, 6, 5, -12)`
- gid `20`: `(3, 0.75, 2, 4, -20)`

So:

- canonical gid for `B` = `12`

### 4.3 Per-point assignment in `optimal` mode

Now each predicted point checks:

1. what GT instance its nearest GT point belongs to
2. what canonical gid was chosen for that GT instance
3. whether that canonical gid is present in this point's own `point_gids` row

If yes: assign that canonical gid.

If no: assign `-1`.

Per point:

| point | nearest GT instance | canonical gid | row | assigned in `optimal` mode |
|---|---|---:|---|---:|
| `P0` | `A` | `7`  | `[7, 12, -1]` | `7` |
| `P1` | `A` | `7`  | `[7, -1, -1]` | `7` |
| `P2` | `A` | `7`  | `[7, 20, -1]` | `7` |
| `P3` | `B` | `12` | `[12, 7, -1]` | `12` |
| `P4` | `B` | `12` | `[12, -1, -1]` | `12` |
| `P5` | `B` | `12` | `[20, 12, -1]` | `12` |
| `P6` | `B` | `12` | `[20, -1, -1]` | `-1` |
| `P7` | `B` | `12` | `[20, 12, -1]` | `12` |

So:

`optimal_raw = [7, 7, 7, 12, 12, 12, -1, 12]`

### 4.4 Finalization after `optimal` collapse

Counts in `optimal_raw`:

- gid `7`: 3 points
- gid `12`: 4 points

Both survive `min_component_size = 2`.

Sorted surviving original gids:

`[7, 12]`

Relabel:

- `7 -> 0`
- `12 -> 1`

So:

`metric_point_instance_labels_optimal = [0, 0, 0, 1, 1, 1, -1, 1]`

And the metric scores are:

`metric_instance_scores_optimal = [0.9, 0.6]`

Again, scores are still support ratios. `optimal` changes labels, not scores.

## 5. Transfer predicted instance labels onto GT points

This is the function:

`transfer_instance_labels_ovo_style(pred_points, pred_instance_labels, gt_points)`

It:

1. keeps only predicted points with label `>= 0`
2. builds a KD-tree on those valid predicted points
3. for each GT point, queries `k = min(5, num_valid_pred_points)` nearest predicted points
4. assigns the GT point the majority instance label among those `k` labels

Below, I will not use coordinates. I will directly state the 5-NN label sets returned by the KD-tree, because those are the only values that matter to the algorithm after neighbor search.

## 6. GT transfer for the `support`-collapsed labels

After `support` collapse, valid predicted point labels are:

| predicted point | metric label |
|---|---:|
| `P0` | `0` |
| `P1` | `0` |
| `P2` | `0` |
| `P3` | `0` |
| `P4` | `1` |
| `P5` | `1` |
| `P7` | `1` |

Assume the 5 nearest predicted labels for each GT point are:

| GT point | GT instance | 5-NN predicted labels | majority label |
|---|---|---|---:|
| `G0` | `A` | `[0,0,0,1,1]` | `0` |
| `G1` | `A` | `[0,0,0,1,1]` | `0` |
| `G2` | `A` | `[0,0,0,1,1]` | `0` |
| `G3` | `A` | `[0,0,0,1,1]` | `0` |
| `G4` | `A` | `[1,1,1,0,0]` | `1` |
| `G5` | `B` | `[1,1,1,0,0]` | `1` |
| `G6` | `B` | `[1,1,1,0,0]` | `1` |
| `G7` | `B` | `[1,1,1,0,0]` | `1` |
| `G8` | `B` | `[1,1,1,0,0]` | `1` |
| `G9` | `B` | `[1,1,1,0,0]` | `1` |

So the transferred predicted instance labels on GT points are:

`pred_on_gt_support = [0,0,0,0,1,1,1,1,1,1]`

This means:

- predicted instance `0` covers `G0,G1,G2,G3`
- predicted instance `1` covers `G4,G5,G6,G7,G8,G9`

## 7. `ap_25` for the `support` case

### 7.1 IoU matrix

GT instances:

- `A = {G0,G1,G2,G3,G4}` (5 points)
- `B = {G5,G6,G7,G8,G9}` (5 points)

Pred instances on GT:

- pred `0` = `{G0,G1,G2,G3}` (4 points)
- pred `1` = `{G4,G5,G6,G7,G8,G9}` (6 points)

IoUs:

#### GT `A` vs pred `0`

- intersection = 4
- union = 5 + 4 - 4 = 5
- IoU = `4 / 5 = 0.8`

#### GT `A` vs pred `1`

- intersection = 1 (`G4`)
- union = 5 + 6 - 1 = 10
- IoU = `1 / 10 = 0.1`

#### GT `B` vs pred `0`

- intersection = 0
- IoU = `0`

#### GT `B` vs pred `1`

- intersection = 5
- union = 5 + 6 - 5 = 6
- IoU = `5 / 6 ≈ 0.8333`

So the IoU matrix is:

| GT \\ Pred | pred `0` | pred `1` |
|---|---:|---:|
| `A` | `0.8` | `0.1` |
| `B` | `0.0` | `0.8333` |

### 7.2 Scores used for AP ranking

`compute_instance_metrics(...)` uses:

- `pred_instance_scores = metric_instance_scores_support`

So:

- pred `0` score = `0.9`
- pred `1` score = `0.6`

These are support ratios, not IoU values.

### 7.3 Matching at IoU threshold `0.25`

Sort predictions by score descending:

1. pred `0`, score `0.9`
2. pred `1`, score `0.6`

Now match greedily:

#### pred `0`

- IoU with `A` = `0.8 >= 0.25`
- IoU with `B` = `0.0`

Best available GT match is `A`.

So:

- pred `0` = TP
- GT `A` becomes matched

#### pred `1`

- IoU with `A` = `0.1 < 0.25`
- IoU with `B` = `0.8333 >= 0.25`

Best available GT match is `B`.

So:

- pred `1` = TP
- GT `B` becomes matched

### 7.4 Precision / recall sequence

There are `2` GT instances total.

Ranked TP/FP:

- pred `0`: TP
- pred `1`: TP

So:

- `tp = [1, 1]`
- `fp = [0, 0]`

Cumulative:

- `tp_cum = [1, 2]`
- `fp_cum = [0, 0]`

Recall:

- `[1/2, 2/2] = [0.5, 1.0]`

Precision:

- `[1/1, 2/2] = [1.0, 1.0]`

So:

- `ap_25 = 1.0`

## 8. GT transfer for the `optimal`-collapsed labels

After `optimal` collapse, valid predicted point labels are:

| predicted point | metric label |
|---|---:|
| `P0` | `0` |
| `P1` | `0` |
| `P2` | `0` |
| `P3` | `1` |
| `P4` | `1` |
| `P5` | `1` |
| `P7` | `1` |

Assume the 5 nearest predicted labels for each GT point are:

| GT point | GT instance | 5-NN predicted labels | majority label |
|---|---|---|---:|
| `G0` | `A` | `[0,0,0,1,1]` | `0` |
| `G1` | `A` | `[0,0,0,1,1]` | `0` |
| `G2` | `A` | `[0,0,0,1,1]` | `0` |
| `G3` | `A` | `[0,0,0,1,1]` | `0` |
| `G4` | `A` | `[0,0,0,1,1]` | `0` |
| `G5` | `B` | `[1,1,1,0,0]` | `1` |
| `G6` | `B` | `[1,1,1,0,0]` | `1` |
| `G7` | `B` | `[1,1,1,0,0]` | `1` |
| `G8` | `B` | `[1,1,1,0,0]` | `1` |
| `G9` | `B` | `[1,1,1,0,0]` | `1` |

So:

`pred_on_gt_optimal = [0,0,0,0,0,1,1,1,1,1]`

Pred instances on GT:

- pred `0` = `{G0,G1,G2,G3,G4}`
- pred `1` = `{G5,G6,G7,G8,G9}`

## 9. `ap_25` for the `optimal` case

### 9.1 IoU matrix

#### GT `A` vs pred `0`

- intersection = 5
- union = 5 + 5 - 5 = 5
- IoU = `1.0`

#### GT `B` vs pred `1`

- intersection = 5
- union = 5 + 5 - 5 = 5
- IoU = `1.0`

Cross IoUs are `0`.

So:

| GT \\ Pred | pred `0` | pred `1` |
|---|---:|---:|
| `A` | `1.0` | `0.0` |
| `B` | `0.0` | `1.0` |

### 9.2 Scores used for AP ranking

Scores are still support-ratio scores from the surviving original gids:

- pred `0` score = `0.9`
- pred `1` score = `0.6`

### 9.3 Matching at IoU threshold `0.25`

Sorted order:

1. pred `0`, score `0.9`
2. pred `1`, score `0.6`

Matching:

- pred `0` matches GT `A` -> TP
- pred `1` matches GT `B` -> TP

So:

- `ap_25 = 1.0`

## 10. What changed between `support` and `optimal`

Only the point-level collapse labels changed.

Everything after that is the same:

1. component filtering
2. contiguous relabeling
3. support-ratio score assignment
4. 5-NN transfer to GT points
5. IoU matrix construction
6. greedy AP matching by score

The practical meaning is:

- `support` collapse chooses the strongest gid already present in each point's row.
- `optimal` collapse uses GT only to decide a canonical gid per GT instance, then assigns a predicted point to that canonical gid only if that gid is already present in that point's row.

So `optimal` is GT-guided point-label selection over the already learned `K` slots. It does not invent new gids and it does not change per-gid scores.

## 11. One important detail about `ap_25`

`ap_25` in the code means:

- run AP at a single IoU threshold of `0.25`

The main reported `ap` is different:

- mean of AP at thresholds `0.50, 0.55, 0.60, ..., 0.95`

So:

- `ap_25` is only the `0.25` slice
- `ap` is the stricter multi-threshold mean
