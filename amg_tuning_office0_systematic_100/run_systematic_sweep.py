#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import threading
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.ensemble import RandomForestRegressor


ROOT_DIR = Path("/robodata/smodak/repos/ovo")
SWEEP_DIR = ROOT_DIR / "amg_tuning_office0_systematic_100"
RESULT_DIR = SWEEP_DIR / "results"
LOG_DIR = SWEEP_DIR / "logs"
TSV_PATH = SWEEP_DIR / "experiment_dump.tsv"
FRONTIER_TSV_PATH = SWEEP_DIR / "pareto_frontier.tsv"
SUMMARY_JSON_PATH = SWEEP_DIR / "summary.json"
PLOT_PATH = SWEEP_DIR / "pareto_time_vs_metric.png"
PLAN_PATH = SWEEP_DIR / "search_plan.json"

DATASET_NAME = "Replica"
SCENE_NAME = "office0"
FRAME_SAMPLES = 100
FRAME_SAMPLE_SEED = 0
TOTAL_BATCHES = 8
RUNS_PER_BATCH = 16
ALLOWED_GPUS = (1, 3, 4, 5, 6, 7, 8, 9)
RANDOM_SEED = 7

SEARCH_SPACE: dict[str, list[Any]] = {
    "sam_model_level_inst": [13, 22, 23, 24],
    "sam_points_per_side": [8, 12, 16, 20, 24],
    "sam_points_per_batch": [32, 64, 128, 256],
    "sam_crop_n_layers": [0, 1, 2],
    "sam_pred_iou_thresh": [0.84, 0.88, 0.92],
    "sam_stability_score_thresh": [0.90, 0.92, 0.95],
}

DEFAULT_CONFIG = {
    "sam_model_level_inst": 24,
    "sam_points_per_side": 20,
    "sam_points_per_batch": 64,
    "sam_crop_n_layers": 0,
    "sam_pred_iou_thresh": 0.88,
    "sam_stability_score_thresh": 0.92,
}

TSV_COLUMNS = [
    "run_id",
    "batch_idx",
    "role",
    "status",
    "gpu_id",
    "runtime_sec",
    "metric_value",
    "started_at_utc",
    "finished_at_utc",
    *SEARCH_SPACE.keys(),
    "result_json",
    "stdout_log",
    "stderr_log",
    "comment",
]


@dataclass(frozen=True)
class RunSpec:
    run_id: int
    batch_idx: int
    role: str
    config: dict[str, Any]
    comment: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(config[name] for name in SEARCH_SPACE)


def config_to_cli_args(config: dict[str, Any]) -> list[str]:
    args = [
        "--dataset_name",
        DATASET_NAME,
        "--scene_name",
        SCENE_NAME,
        "--frame_samples",
        str(FRAME_SAMPLES),
        "--frame_sample_seed",
        str(FRAME_SAMPLE_SEED),
    ]
    for name in SEARCH_SPACE:
        cli_name = "--" + name.replace("_", "-")
        args.extend([cli_name, str(config[name])])
    return args


def parse_gpu_list(raw: str) -> list[int]:
    text = raw.strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def run_host_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def query_free_gpus() -> list[int]:
    history = parse_gpu_list(
        run_host_command(
            [
                "python3",
                "/robodata/smodak/gpu_free_state.py",
                "history",
                "--minutes",
                "120",
                "--quiet",
            ]
        )
    )
    snapshot = parse_gpu_list(
        run_host_command(
            ["python3", "/robodata/smodak/gpu_free_state.py", "snapshot", "--quiet"]
        )
    )
    history = [gpu for gpu in history if gpu in ALLOWED_GPUS]
    snapshot = [gpu for gpu in snapshot if gpu in ALLOWED_GPUS]
    preferred = [gpu for gpu in history if gpu in snapshot]
    fallback = [gpu for gpu in snapshot if gpu not in preferred]
    return preferred + fallback


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    decoder = json.JSONDecoder()
    best = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if best is None or end > best[0]:
            best = (end, obj)
    if best is None:
        raise ValueError("Could not parse JSON summary from stdout")
    return best[1]


def config_to_vector(config: dict[str, Any]) -> np.ndarray:
    values = []
    for name, choices in SEARCH_SPACE.items():
        idx = choices.index(config[name])
        denom = max(1, len(choices) - 1)
        values.append(float(idx) / float(denom))
    return np.asarray(values, dtype=np.float32)


def greedy_diverse_selection(
    candidates: list[dict[str, Any]],
    n_select: int,
    *,
    score_lookup: dict[tuple[Any, ...], float] | None = None,
    existing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if n_select <= 0 or not candidates:
        return []
    existing_vecs = []
    if existing:
        existing_vecs = [config_to_vector(cfg) for cfg in existing]
    candidate_vecs = {config_key(cfg): config_to_vector(cfg) for cfg in candidates}
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < n_select:
        best_cfg = None
        best_score = None
        for cfg in remaining:
            key = config_key(cfg)
            vec = candidate_vecs[key]
            if selected:
                dist_selected = min(
                    float(np.linalg.norm(vec - candidate_vecs[config_key(sel)]))
                    for sel in selected
                )
            else:
                dist_selected = 1.0
            if existing_vecs:
                dist_existing = min(float(np.linalg.norm(vec - prev)) for prev in existing_vecs)
            else:
                dist_existing = 1.0
            diversity = min(dist_selected, dist_existing)
            score = diversity
            if score_lookup is not None:
                score += 0.15 * float(score_lookup[key])
            if best_score is None or score > best_score:
                best_score = score
                best_cfg = cfg
        selected.append(best_cfg)
        remaining = [cfg for cfg in remaining if config_key(cfg) != config_key(best_cfg)]
    return selected


def pareto_front_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if row["status"] == "ok"]
    frontier = []
    for row in valid_rows:
        dominated = False
        for other in valid_rows:
            if other is row:
                continue
            better_or_equal_time = float(other["runtime_sec"]) <= float(row["runtime_sec"])
            better_or_equal_metric = float(other["metric_value"]) >= float(row["metric_value"])
            strictly_better = (
                float(other["runtime_sec"]) < float(row["runtime_sec"])
                or float(other["metric_value"]) > float(row["metric_value"])
            )
            if better_or_equal_time and better_or_equal_metric and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (float(row["runtime_sec"]), -float(row["metric_value"])))
    return frontier


def enumerate_candidates() -> list[dict[str, Any]]:
    keys = list(SEARCH_SPACE)
    candidates = []
    for combo in itertools.product(*(SEARCH_SPACE[key] for key in keys)):
        candidates.append(dict(zip(keys, combo)))
    return candidates


def initial_batch_candidates(
    candidates: list[dict[str, Any]],
    observed_configs: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    remaining = [cfg for cfg in candidates if config_key(cfg) not in observed_configs]
    seeded: list[dict[str, Any]] = []
    for model_level in SEARCH_SPACE["sam_model_level_inst"]:
        for crop_n_layers in SEARCH_SPACE["sam_crop_n_layers"]:
            pool = [
                cfg
                for cfg in remaining
                if cfg["sam_model_level_inst"] == model_level
                and cfg["sam_crop_n_layers"] == crop_n_layers
            ]
            if not pool:
                continue
            best = min(
                pool,
                key=lambda cfg: np.linalg.norm(config_to_vector(cfg) - config_to_vector(DEFAULT_CONFIG)),
            )
            seeded.append(best)
    seeded_keys = {config_key(cfg) for cfg in seeded}
    residual = [cfg for cfg in remaining if config_key(cfg) not in seeded_keys]
    score_lookup = {config_key(cfg): 0.0 for cfg in residual}
    seeded.extend(
        greedy_diverse_selection(
            residual,
            RUNS_PER_BATCH - len(seeded),
            existing=seeded + [DEFAULT_CONFIG],
            score_lookup=score_lookup,
        )
    )
    return seeded[:RUNS_PER_BATCH]


def fit_surrogates(observed_rows: list[dict[str, Any]]) -> tuple[RandomForestRegressor, RandomForestRegressor]:
    x = np.stack([config_to_vector({name: row[name] for name in SEARCH_SPACE}) for row in observed_rows], axis=0)
    y_metric = np.asarray([float(row["metric_value"]) for row in observed_rows], dtype=np.float32)
    y_time = np.log1p(np.asarray([float(row["runtime_sec"]) for row in observed_rows], dtype=np.float32))
    metric_model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_SEED,
        min_samples_leaf=2,
        n_jobs=1,
    )
    time_model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_SEED + 1,
        min_samples_leaf=2,
        n_jobs=1,
    )
    metric_model.fit(x, y_metric)
    time_model.fit(x, y_time)
    return metric_model, time_model


def forest_mean_std(model: RandomForestRegressor, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    per_tree = np.stack([tree.predict(x) for tree in model.estimators_], axis=0)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def neighbor_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for name, choices in SEARCH_SPACE.items():
        idx = choices.index(config[name])
        for delta in (-1, 1):
            new_idx = idx + delta
            if new_idx < 0 or new_idx >= len(choices):
                continue
            neighbor = dict(config)
            neighbor[name] = choices[new_idx]
            neighbors.append(neighbor)
    return neighbors


def adaptive_batch_candidates(
    all_candidates: list[dict[str, Any]],
    observed_rows: list[dict[str, Any]],
    batch_idx: int,
) -> list[dict[str, Any]]:
    observed_configs = {config_key({name: row[name] for name in SEARCH_SPACE}) for row in observed_rows}
    remaining = [cfg for cfg in all_candidates if config_key(cfg) not in observed_configs]
    if not remaining:
        return []
    if len(observed_rows) <= 1:
        return initial_batch_candidates(all_candidates, observed_configs)

    metric_model, time_model = fit_surrogates(observed_rows)
    x_remaining = np.stack([config_to_vector(cfg) for cfg in remaining], axis=0)
    pred_metric_mean, pred_metric_std = forest_mean_std(metric_model, x_remaining)
    pred_time_mean_log, pred_time_std_log = forest_mean_std(time_model, x_remaining)
    pred_time_mean = np.expm1(pred_time_mean_log)
    pred_time_std = np.maximum(0.0, np.expm1(pred_time_mean_log + pred_time_std_log) - pred_time_mean)

    predicted_rows = []
    for cfg, metric_mean, metric_std, time_mean, time_std in zip(
        remaining,
        pred_metric_mean,
        pred_metric_std,
        pred_time_mean,
        pred_time_std,
    ):
        predicted_rows.append(
            {
                "config": cfg,
                "pred_metric_mean": float(metric_mean),
                "pred_metric_std": float(metric_std),
                "pred_time_mean": float(time_mean),
                "pred_time_std": float(time_std),
            }
        )

    baseline_row = next((row for row in observed_rows if row["role"] == "baseline" and row["status"] == "ok"), None)
    runtime_cap_sec = None if baseline_row is None else 2.0 * float(baseline_row["runtime_sec"])
    predicted_rows = [row for row in predicted_rows if row["config"]["sam_crop_n_layers"] <= 1]
    if runtime_cap_sec is not None:
        capped_rows = [row for row in predicted_rows if row["pred_time_mean"] <= runtime_cap_sec]
        if capped_rows:
            predicted_rows = capped_rows

    predicted_frontier = []
    for row in predicted_rows:
        dominated = False
        for other in predicted_rows:
            if other is row:
                continue
            if (
                other["pred_time_mean"] <= row["pred_time_mean"]
                and other["pred_metric_mean"] >= row["pred_metric_mean"]
                and (
                    other["pred_time_mean"] < row["pred_time_mean"]
                    or other["pred_metric_mean"] > row["pred_metric_mean"]
                )
            ):
                dominated = True
                break
        if not dominated:
            predicted_frontier.append(row)

    observed_frontier = pareto_front_rows(observed_rows)
    frontier_neighbor_pool: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in observed_frontier:
        cfg = {name: row[name] for name in SEARCH_SPACE}
        for neighbor in neighbor_configs(cfg):
            key = config_key(neighbor)
            if key in observed_configs:
                continue
            frontier_neighbor_pool[key] = neighbor

    score_lookup = {}
    for row in predicted_rows:
        key = config_key(row["config"])
        score_lookup[key] = (
            4.0 * row["pred_metric_mean"]
            + 0.8 * row["pred_metric_std"]
            - 0.3 * math.log1p(max(0.0, row["pred_time_mean"]))
        )
        if key in frontier_neighbor_pool:
            score_lookup[key] += 0.12

    frontier_candidates = [row["config"] for row in predicted_frontier]
    exploration_candidates = sorted(
        (row["config"] for row in predicted_rows),
        key=lambda cfg: score_lookup[config_key(cfg)],
        reverse=True,
    )

    selected = greedy_diverse_selection(
        frontier_candidates,
        RUNS_PER_BATCH // 2,
        score_lookup=score_lookup,
        existing=[{name: row[name] for name in SEARCH_SPACE} for row in observed_rows],
    )
    selected_keys = {config_key(cfg) for cfg in selected}

    crop1_pool = [
        row["config"]
        for row in predicted_rows
        if row["config"]["sam_crop_n_layers"] == 1 and config_key(row["config"]) not in selected_keys
    ]
    crop1_target = min(2, RUNS_PER_BATCH - len(selected))
    if crop1_pool and crop1_target > 0:
        crop1_selected = greedy_diverse_selection(
            crop1_pool,
            crop1_target,
            score_lookup=score_lookup,
            existing=selected + [{name: row[name] for name in SEARCH_SPACE} for row in observed_rows],
        )
        selected.extend(crop1_selected)
        selected_keys = {config_key(cfg) for cfg in selected}

    exploration_pool = [cfg for cfg in exploration_candidates if config_key(cfg) not in selected_keys]
    selected.extend(
        greedy_diverse_selection(
            exploration_pool,
            RUNS_PER_BATCH - len(selected),
            score_lookup=score_lookup,
            existing=selected + [{name: row[name] for name in SEARCH_SPACE} for row in observed_rows],
        )
    )
    return selected[:RUNS_PER_BATCH]


class SweepRunner:
    def __init__(self) -> None:
        self.random = random.Random(RANDOM_SEED)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.tsv_lock = threading.Lock()
        self.all_candidates = enumerate_candidates()
        self.observed_rows: list[dict[str, Any]] = []
        self.next_run_id = 1
        self._load_existing_results()
        self._prune_incomplete_tail_batch()
        self._write_plan_file()

    def _write_plan_file(self) -> None:
        payload = {
            "dataset_name": DATASET_NAME,
            "scene_name": SCENE_NAME,
            "frame_samples": FRAME_SAMPLES,
            "frame_sample_seed": FRAME_SAMPLE_SEED,
            "total_batches": TOTAL_BATCHES,
            "runs_per_batch": RUNS_PER_BATCH,
            "allowed_gpus": list(ALLOWED_GPUS),
            "default_config": DEFAULT_CONFIG,
            "search_space": SEARCH_SPACE,
        }
        PLAN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_existing_results(self) -> None:
        self.observed_rows = []
        self.next_run_id = 1
        if not TSV_PATH.exists():
            return
        with open(TSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                parsed = dict(row)
                parsed["run_id"] = int(parsed["run_id"])
                parsed["batch_idx"] = int(parsed["batch_idx"])
                parsed["runtime_sec"] = float(parsed["runtime_sec"]) if parsed["runtime_sec"] else float("nan")
                parsed["metric_value"] = float(parsed["metric_value"]) if parsed["metric_value"] else float("nan")
                parsed["gpu_id"] = int(parsed["gpu_id"]) if parsed["gpu_id"] else -1
                for name in SEARCH_SPACE:
                    value = row[name]
                    example = SEARCH_SPACE[name][0]
                    parsed[name] = type(example)(value)
                self.observed_rows.append(parsed)
                self.next_run_id = max(self.next_run_id, parsed["run_id"] + 1)

    def _rewrite_tsv(self) -> None:
        with open(TSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in sorted(self.observed_rows, key=lambda item: item["run_id"]):
                writer.writerow({key: row.get(key, "") for key in TSV_COLUMNS})

    def _prune_incomplete_tail_batch(self) -> None:
        search_rows = [row for row in self.observed_rows if row["role"] == "search"]
        if not search_rows:
            return
        batch_counts: dict[int, int] = {}
        for row in search_rows:
            batch_counts[row["batch_idx"]] = batch_counts.get(row["batch_idx"], 0) + 1
        tail_batch_idx = max(batch_counts)
        tail_batch_count = batch_counts[tail_batch_idx]
        if tail_batch_count >= RUNS_PER_BATCH:
            return

        print(
            f"Pruning incomplete tail batch {tail_batch_idx} with {tail_batch_count} rows before resume.",
            flush=True,
        )
        kept_rows = []
        removed_rows = []
        for row in self.observed_rows:
            if row["role"] == "search" and row["batch_idx"] == tail_batch_idx:
                removed_rows.append(row)
            else:
                kept_rows.append(row)

        for row in removed_rows:
            for key in ("result_json", "stdout_log", "stderr_log"):
                raw_path = row.get(key)
                if raw_path:
                    Path(raw_path).unlink(missing_ok=True)

        self.observed_rows = kept_rows
        self.next_run_id = 1
        for row in self.observed_rows:
            self.next_run_id = max(self.next_run_id, row["run_id"] + 1)
        self._rewrite_tsv()
        FRONTIER_TSV_PATH.unlink(missing_ok=True)
        SUMMARY_JSON_PATH.unlink(missing_ok=True)
        PLOT_PATH.unlink(missing_ok=True)

    def _ensure_tsv_header(self) -> None:
        if TSV_PATH.exists():
            return
        with open(TSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writeheader()

    def _append_row(self, row: dict[str, Any]) -> None:
        with self.tsv_lock:
            self._ensure_tsv_header()
            with open(TSV_PATH, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
                writer.writerow({key: row.get(key, "") for key in TSV_COLUMNS})

    def _next_run_spec(self, batch_idx: int, role: str, config: dict[str, Any], comment: str) -> RunSpec:
        spec = RunSpec(
            run_id=self.next_run_id,
            batch_idx=batch_idx,
            role=role,
            config=dict(config),
            comment=comment,
        )
        self.next_run_id += 1
        return spec

    def _build_eval_command(self, spec: RunSpec) -> list[str]:
        return [
            "python3",
            str(ROOT_DIR / "eval_sam_amg.py"),
            *config_to_cli_args(spec.config),
        ]

    def _run_single(self, gpu_id: int, spec: RunSpec) -> dict[str, Any]:
        run_prefix = f"run{spec.run_id:03d}"
        stdout_log = LOG_DIR / f"{run_prefix}.stdout.log"
        stderr_log = LOG_DIR / f"{run_prefix}.stderr.log"
        result_json = RESULT_DIR / f"{run_prefix}.json"
        started_at = utc_now()
        command = self._build_eval_command(spec)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(
            f"[batch {spec.batch_idx:02d}] launching run {spec.run_id:03d} on gpu {gpu_id}: "
            f"{spec.comment} | {spec.config}",
            flush=True,
        )
        wall_start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT_DIR, env=env)
        runtime_sec = time.perf_counter() - wall_start
        finished_at = utc_now()
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        row = {
            "run_id": spec.run_id,
            "batch_idx": spec.batch_idx,
            "role": spec.role,
            "gpu_id": gpu_id,
            "runtime_sec": runtime_sec,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "result_json": str(result_json),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "comment": spec.comment,
            **spec.config,
        }
        if completed.returncode != 0:
            row["status"] = "failed"
            row["metric_value"] = ""
            result_json.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "returncode": completed.returncode,
                        "command": command,
                        "stdout_path": str(stdout_log),
                        "stderr_path": str(stderr_log),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._append_row(row)
            print(
                f"[batch {spec.batch_idx:02d}] run {spec.run_id:03d} failed on gpu {gpu_id} "
                f"after {runtime_sec:.1f}s",
                flush=True,
            )
            return row

        summary = extract_json_object(completed.stdout)
        summary["runtime_sec"] = runtime_sec
        summary["status"] = "ok"
        summary["run_id"] = spec.run_id
        summary["batch_idx"] = spec.batch_idx
        summary["role"] = spec.role
        summary["gpu_id"] = gpu_id
        summary["started_at_utc"] = started_at
        summary["finished_at_utc"] = finished_at
        summary["comment"] = spec.comment
        summary["command"] = command
        result_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        row["status"] = "ok"
        row["metric_value"] = float(summary["metric_value"])
        self._append_row(row)
        print(
            f"[batch {spec.batch_idx:02d}] run {spec.run_id:03d} done on gpu {gpu_id}: "
            f"metric={float(summary['metric_value']):.6f} runtime={runtime_sec:.1f}s",
            flush=True,
        )
        return row

    def _schedule_specs(self, specs: list[RunSpec]) -> list[dict[str, Any]]:
        completed_rows: list[dict[str, Any]] = []
        pending = list(specs)
        while pending:
            free_gpus = query_free_gpus()
            if not free_gpus:
                print("No free GPUs from allowed pool; sleeping 60s before retry.", flush=True)
                time.sleep(60.0)
                continue
            wave_slots = min(len(free_gpus), len(pending))
            gpu_wave = free_gpus[:wave_slots]
            spec_wave = [pending.pop(0) for _ in range(wave_slots)]
            print(
                f"Launching wave with GPUs {gpu_wave} for run IDs {[spec.run_id for spec in spec_wave]}",
                flush=True,
            )
            results: list[dict[str, Any]] = []
            threads = []
            result_lock = threading.Lock()

            def _worker(gpu_id: int, spec: RunSpec) -> None:
                row = self._run_single(gpu_id, spec)
                with result_lock:
                    results.append(row)

            for gpu_id, spec in zip(gpu_wave, spec_wave):
                thread = threading.Thread(target=_worker, args=(gpu_id, spec), daemon=False)
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            results.sort(key=lambda row: row["run_id"])
            completed_rows.extend(results)
            for row in results:
                if row["status"] == "ok":
                    self.observed_rows.append(row)
            self._write_summary()
        return completed_rows

    def _write_summary(self) -> None:
        frontier = pareto_front_rows(self.observed_rows)
        payload = {
            "n_completed": sum(1 for row in self.observed_rows if row["status"] == "ok"),
            "frontier_run_ids": [row["run_id"] for row in frontier],
            "best_metric_run_id": max(self.observed_rows, key=lambda row: float(row["metric_value"]))["run_id"]
            if self.observed_rows
            else None,
        }
        SUMMARY_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _planned_specs_for_batch(self, batch_idx: int) -> list[RunSpec]:
        observed_configs = {config_key({name: row[name] for name in SEARCH_SPACE}) for row in self.observed_rows}
        if batch_idx == 0:
            baseline_key = config_key(DEFAULT_CONFIG)
            if baseline_key in observed_configs:
                return []
            return [
                self._next_run_spec(
                    batch_idx=0,
                    role="baseline",
                    config=DEFAULT_CONFIG,
                    comment="baseline current defaults",
                )
            ]
        configs = adaptive_batch_candidates(self.all_candidates, self.observed_rows, batch_idx)
        return [
            self._next_run_spec(
                batch_idx=batch_idx,
                role="search",
                config=cfg,
                comment=f"adaptive batch {batch_idx}",
            )
            for cfg in configs
        ]

    def run(self) -> None:
        self._ensure_tsv_header()
        baseline_specs = self._planned_specs_for_batch(0)
        if baseline_specs:
            self._schedule_specs(baseline_specs)

        completed_search_batches = sorted(
            {
                int(row["batch_idx"])
                for row in self.observed_rows
                if row["status"] == "ok" and row["role"] == "search"
            }
        )
        start_batch_idx = 1 if not completed_search_batches else max(completed_search_batches) + 1
        for batch_idx in range(start_batch_idx, TOTAL_BATCHES + 1):
            print(f"Preparing batch {batch_idx}/{TOTAL_BATCHES}", flush=True)
            specs = self._planned_specs_for_batch(batch_idx)
            if not specs:
                print(f"No remaining configs to schedule for batch {batch_idx}", flush=True)
                break
            self._schedule_specs(specs)
            self._write_frontier_files()
            self._write_plot()

        self._write_frontier_files()
        self._write_plot()

    def _write_frontier_files(self) -> None:
        frontier = pareto_front_rows(self.observed_rows)
        with open(FRONTIER_TSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "runtime_sec", "metric_value", *SEARCH_SPACE.keys()],
                delimiter="\t",
            )
            writer.writeheader()
            for row in frontier:
                writer.writerow(
                    {
                        "run_id": row["run_id"],
                        "runtime_sec": row["runtime_sec"],
                        "metric_value": row["metric_value"],
                        **{name: row[name] for name in SEARCH_SPACE},
                    }
                )

    def _write_plot(self) -> None:
        valid_rows = [row for row in self.observed_rows if row["status"] == "ok"]
        if not valid_rows:
            return
        frontier = pareto_front_rows(valid_rows)
        frontier_ids = {row["run_id"] for row in frontier}
        baseline = next((row for row in valid_rows if row["role"] == "baseline"), None)

        plt.close("all")
        fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
        ax.set_title("SAM AMG Office0 Systematic Sweep\nRuntime vs Mean AP", fontsize=18, weight="bold")
        ax.set_xlabel("Runtime (minutes)", fontsize=13)
        ax.set_ylabel("Mean AP", fontsize=13)
        ax.grid(True, which="major", alpha=0.2, linestyle="--")
        ax.grid(True, which="minor", alpha=0.08, linestyle=":")
        ax.minorticks_on()

        x = np.asarray([float(row["runtime_sec"]) / 60.0 for row in valid_rows], dtype=np.float32)
        y = np.asarray([float(row["metric_value"]) for row in valid_rows], dtype=np.float32)
        colors = []
        sizes = []
        for row in valid_rows:
            if baseline is not None and row["run_id"] == baseline["run_id"]:
                colors.append("#111111")
                sizes.append(220)
            elif row["run_id"] in frontier_ids:
                colors.append("#d95f02")
                sizes.append(120)
            else:
                colors.append("#1f77b4")
                sizes.append(70)

        ax.scatter(x, y, c=colors, s=sizes, alpha=0.9, edgecolors="white", linewidths=0.8, zorder=3)
        frontier_sorted = sorted(frontier, key=lambda row: float(row["runtime_sec"]))
        frontier_x = [float(row["runtime_sec"]) / 60.0 for row in frontier_sorted]
        frontier_y = [float(row["metric_value"]) for row in frontier_sorted]
        ax.plot(frontier_x, frontier_y, color="#d95f02", linewidth=2.0, alpha=0.8, zorder=2)

        for row in valid_rows:
            ax.annotate(
                str(row["run_id"]),
                (float(row["runtime_sec"]) / 60.0, float(row["metric_value"])),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                fontsize=7.5,
                color="#202020",
                alpha=0.9,
                zorder=4,
            )

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="All runs", markerfacecolor="#1f77b4", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Pareto frontier", markerfacecolor="#d95f02", markersize=10),
        ]
        if baseline is not None:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w", label="Baseline", markerfacecolor="#111111", markersize=12)
            )
        ax.legend(handles=legend_handles, loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, bbox_inches="tight")


def main() -> None:
    runner = SweepRunner()
    runner.run()


if __name__ == "__main__":
    main()
