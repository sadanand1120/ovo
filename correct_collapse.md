## Problem statement

We have:

* a set of **predicted instances** (P_1,\dots,P_n), each with a confidence score
* a set of **ground-truth instances** (G_1,\dots,G_m)
* an IoU matrix
  [
  I_{ij} = \mathrm{IoU}(P_i, G_j)
  ]
* an IoU threshold (\tau), where (\tau = X/100) for **AP@X**

Goal:

> Find a **deterministic pruning strategy** that removes some predicted instances before evaluation, so that the final **AP@X** on the pruned predictions is maximized.

This is an **offline oracle** problem, because pruning is allowed to use the GT / IoU matrix.

---

# Evaluation rule we are optimizing against

We assume the usual AP-style matching rule:

1. Sort predictions by:

   * **higher score first**
   * if scores tie, **smaller original prediction index first**

2. Process predictions in that order.

3. For each prediction:

   * among all **currently unmatched** GT instances with IoU (\ge \tau),
   * match to the one with **largest IoU**
   * if there is a tie in IoU, choose the **smaller GT index**
   * if no such GT exists, that prediction is an **FP**

This fixes everything deterministically.

---

# Key insight

The main insight is:

> In the oracle pruning problem, keeping a prediction that becomes an FP can never help.

Why:

* it does not create any TP
* it does not help any later prediction
* it only lowers precision

So in an optimal pruned set:

* **every kept prediction is a TP**
* therefore precision is always 1 over the kept list
* hence **AP reduces to recall**

So the real problem becomes:

> Keep a subset of predictions such that, under the standard greedy score-ordered AP matching rule, every kept prediction is a TP, and the total number of kept TPs is maximized.

If the number of kept TPs is (K), and there are (m) GT instances, then

[
\mathrm{AP}^*_{\tau} = \frac{K}{m}
]

So the optimization target is simply:

[
\max K
]

subject to greedy AP matching consistency.

---

# Why Hungarian matching is not the right solution

Hungarian matching solves:

> global one-to-one assignment maximizing total IoU or total number of matches

But AP evaluation does **not** do that.

AP evaluation is:

* **score-ordered**
* **greedy**
* each prediction claims its **best currently unmatched GT**

So Hungarian can produce assignments that are impossible to realize under actual AP evaluation.

Therefore:

* **do not use Hungarian**
* **do not use `scipy.optimize.linear_sum_assignment`**

It is solving the wrong problem.

---

# Exact solution that scales to your regime

For around:

* (m \approx 40) GTs
* (n \approx 200) predictions

the bitmask DP over GT subsets is too expensive.

The right exact offline formulation is:

> **0-1 integer / constraint optimization**, solved with **OR-Tools CP-SAT**

This is free, deterministic, and practical enough offline.

---

# Exact optimization model

## Variables

Let predictions already be sorted by:

* descending score
* then smaller original prediction index

Define binary variables:

[
y_{ij} \in {0,1}
]

for every pair ((i,j)) such that (I_{ij} \ge \tau).

Interpretation:

[
y_{ij} = 1
]

means:

> prediction (i) is kept, and under the final AP evaluation it matches GT (j).

---

## Constraint 1: each prediction matches at most one GT

[
\sum_j y_{ij} \le 1 \qquad \forall i
]

---

## Constraint 2: each GT is matched at most once

[
\sum_i y_{ij} \le 1 \qquad \forall j
]

---

## Constraint 3: greedy AP consistency

This is the important part.

If prediction (i) is matched to GT (j), then every GT that (i) would prefer over (j) must already have been consumed by an earlier prediction.

Define the **blocker set** for ((i,j)):

[
B_{ij}
======

\left{
k \neq j :
\begin{array}{l}
I_{ik} \ge \tau, \
\text{and } k \text{ is preferred over } j
\end{array}
\right}
]

Using the deterministic tie-break above, “preferred” means:

* (I_{ik} > I_{ij}), or
* (I_{ik} = I_{ij}) and (k < j)

So:

[
k \in B_{ij}
\iff
\left(I_{ik} > I_{ij}\right)
;\text{or};
\left(I_{ik} = I_{ij} \text{ and } k < j\right)
]

with the additional requirement (I_{ik} \ge \tau).

Then for every blocker (k \in B_{ij}), impose:

[
y_{ij}
\le
\sum_{h < i} y_{hk}
]

Interpretation:

* if (y_{ij}=1), then GT (k) must already have been matched by some earlier prediction
* otherwise prediction (i) would have chosen (k), not (j)

This exactly encodes the real greedy AP rule.

---

## Objective

Maximize total number of matched kept predictions:

[
\max \sum_{i,j} y_{ij}
]

That is exactly the oracle objective.

Then:

[
\mathrm{AP}^*_{\tau}
====================

\frac{1}{m}
\sum_{i,j} y_{ij}
]

---

# Preprocessing for speed

Do this before building the solver model.

## 1) Sort predictions deterministically

Sort by:

1. larger score first
2. if equal score, smaller original prediction index first

This must match evaluation exactly.

---

## 2) Drop predictions with no eligible GT

If a prediction has no GT with

[
I_{ij} \ge \tau
]

it can never become TP, so remove it immediately.

---

## 3) Create variables only for eligible pairs

Only create (y_{ij}) for pairs with

[
I_{ij} \ge \tau
]

---

## 4) Safe fixed-point pruning of impossible edges

A pair ((i,j)) is impossible if it needs some blocker GT (k) to have been matched earlier, but there is no earlier feasible prediction that could ever match (k).

So iteratively remove impossible pairs until convergence.

This can substantially shrink the model.

---

# End-to-end pseudocode

## Input

* IoU matrix `iou[n_pred][n_gt]`
* prediction scores `scores[n_pred]`
* threshold `tau`

---

## Deterministic conventions

* prediction order: `(-score, original_pred_index)`
* GT preference for a given prediction:

  * larger IoU first
  * if tied IoU, smaller GT index first

---

## Pseudocode

```text
INPUT:
    iou[n_pred][n_gt]
    scores[n_pred]
    tau

STEP 1: sort predictions
    order predictions by:
        1) descending score
        2) ascending original prediction index
    reorder iou rows accordingly

STEP 2: build eligible pairs
    feasible_pairs = {}
    for each prediction i:
        for each GT j:
            if iou[i][j] >= tau:
                feasible_pairs.add((i, j))

STEP 3: remove predictions with no eligible GT
    delete any prediction row with zero feasible pairs

STEP 4: compute blocker sets
    for each feasible pair (i, j):
        blockers[i, j] = {}
        for each GT k != j:
            if iou[i][k] < tau:
                continue
            if iou[i][k] > iou[i][j]:
                blockers[i, j].add(k)
            else if iou[i][k] == iou[i][j] and k < j:
                blockers[i, j].add(k)

STEP 5: fixed-point impossible-edge pruning
    repeat until no changes:
        for each feasible pair (i, j):
            valid = True
            for each blocker k in blockers[i, j]:
                if there is NO earlier feasible pair (h, k) with h < i:
                    valid = False
                    break
            if not valid:
                remove (i, j) from feasible_pairs

STEP 6: build binary optimization model
    create binary variable y[i, j] for every remaining feasible pair (i, j)

    constraints:
        (a) each pred matches at most one GT
            for each i:
                sum_j y[i, j] <= 1

        (b) each GT matched at most once
            for each j:
                sum_i y[i, j] <= 1

        (c) greedy consistency
            for each feasible pair (i, j):
                for each blocker k in blockers[i, j]:
                    y[i, j] <= sum_{h < i} y[h, k]

    objective:
        maximize sum_{i, j} y[i, j]

STEP 7: solve with CP-SAT

STEP 8: reconstruct pruned prediction set
    kept predictions = all i such that some y[i, j] = 1
    matched GT for prediction i = unique j with y[i, j] = 1

STEP 9: oracle AP
    K = number of kept predictions
    AP_oracle = K / n_gt

OUTPUT:
    kept predictions
    matched GT per kept prediction
    oracle AP
```

---

# Python recommendation

Use exactly this stack:

```python
import numpy as np
from ortools.sat.python import cp_model
```

Install:

```bash
pip install ortools numpy
```

That is the recommendation. No paid solver, no alternate branch to choose unless this fails.

---

# Exact Python implementation

```python
from __future__ import annotations
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model


def optimal_pruning_for_ap(
    iou: np.ndarray,
    scores: np.ndarray,
    tau: float,
    max_time_s: float | None = None,
    num_workers: int = 8,
):
    """
    Exact offline oracle pruning for AP@X, where tau = X / 100.

    Deterministic conventions used here:
      - prediction order: descending score, then smaller original pred index
      - GT choice for a prediction: highest IoU among unmatched GTs;
        if tied IoU, smaller GT index

    Args:
        iou:    shape (n_pred, n_gt)
        scores: shape (n_pred,)
        tau:    IoU threshold, e.g. 0.25 for AP@25
        max_time_s: optional solver time limit
        num_workers: CP-SAT threads

    Returns:
        dict with:
          - status
          - proven_optimal
          - kept_pred_indices_original
          - matched_gt_indices
          - optimal_tp_count
          - optimal_ap
          - sorted_order
    """
    iou = np.asarray(iou, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pred, n_gt = iou.shape
    assert scores.shape == (n_pred,)

    if n_gt == 0:
        return {
            "status": "trivial_no_gt",
            "proven_optimal": True,
            "kept_pred_indices_original": [],
            "matched_gt_indices": [],
            "optimal_tp_count": 0,
            "optimal_ap": 0.0,
            "sorted_order": np.array([], dtype=int),
        }

    # Sort predictions deterministically:
    # higher score first, then smaller original index
    order = np.lexsort((np.arange(n_pred), -scores))
    iou = iou[order]
    scores = scores[order]

    # Eligible GTs per prediction
    eligible = []
    for i in range(n_pred):
        js = [j for j in range(n_gt) if iou[i, j] >= tau]
        eligible.append(js)

    # Remove predictions with no eligible GT
    keep_pred = np.array([len(js) > 0 for js in eligible], dtype=bool)
    iou = iou[keep_pred]
    order_kept = order[keep_pred]
    n_pred = iou.shape[0]

    if n_pred == 0:
        return {
            "status": "no_eligible_predictions",
            "proven_optimal": True,
            "kept_pred_indices_original": [],
            "matched_gt_indices": [],
            "optimal_tp_count": 0,
            "optimal_ap": 0.0,
            "sorted_order": order_kept,
        }

    eligible = []
    for i in range(n_pred):
        js = [j for j in range(n_gt) if iou[i, j] >= tau]
        eligible.append(js)

    # Blocker sets:
    # For pair (i, j), blockers are GTs that prediction i would prefer over j
    # using:
    #   larger IoU better
    #   if equal IoU, smaller GT index better
    blockers = {}
    feasible = set()
    for i in range(n_pred):
        row = iou[i]
        for j in eligible[i]:
            feasible.add((i, j))
            b = []
            for k in eligible[i]:
                if k == j:
                    continue
                if row[k] > row[j]:
                    b.append(k)
                elif row[k] == row[j] and k < j:
                    b.append(k)
            blockers[(i, j)] = tuple(sorted(b))

    # Fixed-point pruning of impossible feasible pairs
    changed = True
    while changed:
        changed = False

        earlier_feasible_for_gt = [set() for _ in range(n_gt)]
        for (i, j) in feasible:
            earlier_feasible_for_gt[j].add(i)

        to_remove = []
        for (i, j) in feasible:
            ok = True
            for k in blockers[(i, j)]:
                found = False
                for h in earlier_feasible_for_gt[k]:
                    if h < i:
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if not ok:
                to_remove.append((i, j))

        if to_remove:
            changed = True
            for x in to_remove:
                feasible.remove(x)

    if not feasible:
        return {
            "status": "no_feasible_pairs_after_pruning",
            "proven_optimal": True,
            "kept_pred_indices_original": [],
            "matched_gt_indices": [],
            "optimal_tp_count": 0,
            "optimal_ap": 0.0,
            "sorted_order": order_kept,
        }

    feasible_by_pred = defaultdict(list)
    feasible_by_gt = defaultdict(list)
    for (i, j) in feasible:
        feasible_by_pred[i].append(j)
        feasible_by_gt[j].append(i)

    for i in feasible_by_pred:
        feasible_by_pred[i].sort()
    for j in feasible_by_gt:
        feasible_by_gt[j].sort()

    model = cp_model.CpModel()

    y = {}
    for (i, j) in feasible:
        y[(i, j)] = model.NewBoolVar(f"y_{i}_{j}")

    # Each pred matches at most one GT
    for i in range(n_pred):
        vars_i = [y[(i, j)] for j in feasible_by_pred.get(i, [])]
        if vars_i:
            model.AddAtMostOne(vars_i)

    # Each GT matched at most once
    for j in range(n_gt):
        vars_j = [y[(i, j)] for i in feasible_by_gt.get(j, [])]
        if vars_j:
            model.AddAtMostOne(vars_j)

    # Prefix availability lists for blocker constraints
    earlier_vars_for_gt = {}
    for j in range(n_gt):
        seen = []
        for i in range(n_pred):
            earlier_vars_for_gt[(i, j)] = list(seen)
            if (i, j) in y:
                seen.append(y[(i, j)])

    # Greedy AP consistency constraints
    for (i, j), var in y.items():
        for k in blockers[(i, j)]:
            rhs = earlier_vars_for_gt[(i, k)]
            if not rhs:
                model.Add(var == 0)
            else:
                model.Add(var <= sum(rhs))

    # Maximize number of kept TPs
    model.Maximize(sum(y.values()))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = num_workers
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 1
    if max_time_s is not None:
        solver.parameters.max_time_in_seconds = max_time_s

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": status_name,
            "proven_optimal": False,
            "kept_pred_indices_original": None,
            "matched_gt_indices": None,
            "optimal_tp_count": None,
            "optimal_ap": None,
            "sorted_order": order_kept,
        }

    chosen = []
    for (i, j), var in y.items():
        if solver.Value(var) == 1:
            chosen.append((i, j))
    chosen.sort()

    kept_pred_indices_original = [int(order_kept[i]) for (i, j) in chosen]
    matched_gt_indices = [int(j) for (i, j) in chosen]
    optimal_tp_count = len(chosen)
    optimal_ap = optimal_tp_count / n_gt

    return {
        "status": status_name,
        "proven_optimal": status == cp_model.OPTIMAL,
        "kept_pred_indices_original": kept_pred_indices_original,
        "matched_gt_indices": matched_gt_indices,
        "optimal_tp_count": optimal_tp_count,
        "optimal_ap": optimal_ap,
        "sorted_order": order_kept,
    }
```

---

# How to use it

If you want oracle pruning for AP@25:

```python
tau = 0.25
res = optimal_pruning_for_ap(iou, scores, tau, max_time_s=600, num_workers=16)
```

If you want oracle pruning for AP@50:

```python
tau = 0.50
res = optimal_pruning_for_ap(iou, scores, tau, max_time_s=600, num_workers=16)
```

Then:

* `res["kept_pred_indices_original"]` tells you which predictions to keep
* `res["matched_gt_indices"]` tells you which GT each kept prediction matches
* `res["optimal_ap"]` is the oracle AP at that threshold

---

# Practical recommendations

Use this exact setup:

## Libraries

* `numpy`
* `ortools`

## Sort rule

* higher score first
* tie: smaller original prediction index

## GT-choice rule

* larger IoU first
* tie: smaller GT index

## Solver

* OR-Tools CP-SAT
* `num_workers = 8` or `16`
* give it a time limit, e.g. `600` seconds for offline runs

## Do not use

* Hungarian
* `linear_sum_assignment`
* bitmask DP for (m \sim 40)

---

# Final summary

## What problem are we solving?

We want the **best deterministic pruning of predicted instances** so that the final **AP@X** after pruning is maximized.

## Key insight

In the oracle setting, optimal pruning never keeps FPs. Therefore the objective becomes:

> maximize the number of TPs realizable under the actual score-ordered greedy AP matching rule.

## Exact solution

Formulate the problem as a **binary optimization problem** with:

* one variable per feasible prediction-GT pair
* one-to-one matching constraints
* extra constraints enforcing the real greedy AP behavior

Then solve with **OR-Tools CP-SAT**.

## Why this is the correct tool

It matches the real AP evaluation semantics. Hungarian does not.

## What to implement

Use the exact Python function above. That is the recommended path.
