# Repository Guidelines

## Project Structure & Module Organization
- `build_rgb_map.py` is the main entry point for the RGB+normal+CLIP+instance map workflow. `topdown_vis.py`, `visualize_rgb_map.py`, `visualize_gt.py`, and `get_metrics_map.py` inspect those outputs, while `get_ovo_style_eval.py` handles the dataset-level OVO-style report workflow.
- `map_runtime/` holds the first-party runtime used by the main scripts: config loading, datasets, geometry, SAM/SAM2 wrappers, and SLAM backends.
- `configs/` holds the runtime config and dataset metadata: `ovo.yaml`, `replica.yaml`, `scannet.yaml`, `replica_eval.yaml`, and `scannet_eval.yaml`.
- `data/input/` holds datasets and checkpoints. `data/output/` holds generated runs. `thirdParty/` is reserved for external dependencies.

## Build, Test, and Development Commands
- Use `README.md` as the source of truth for environment setup, checkpoint download, ScanNet decoding, backend build steps, and the map build/eval/viewer commands.
- `git submodule update --init --recursive` fetches `ORB_SLAM3` and `segment-anything-2`.
- Run all commands associated to this repo inside container named 'humble' and inside a conda (/opt/miniconda3) env called 'ovo'. Use docker exec to run commands inside the container.

## Coding Style & Naming Conventions
- Use 4-space indentation, `snake_case` for functions and config keys, and `PascalCase` for classes.
- Keep changes local and readable. Match nearby NumPy, Torch, and path-handling patterns instead of adding new abstraction layers.
- Avoid broad reformatting. There is no committed formatter config.

## Testing Guidelines
- There is no first-party unit test suite; validation is command-driven.
- For backend or mapping changes, run at least one small `build_rgb_map.py` scene using the exact command patterns in `README.md`. For viewer changes, rerun the relevant viewer (`topdown_vis.py`, `visualize_rgb_map.py`, or `visualize_gt.py`).
- If you touch config loading, dataset prep, or output writing, confirm `rgb_map.ply`, `clip_feats.npy`, `instance_labels.npy`, `stats.json`, and `timing.json` appear under the output scene directory.
- `vanilla` is deterministic here and is the best regression check. `orbslam` is functional but not bit-stable run to run.
- If a long-running validation command is interrupted or a tool session aborts, verify whether the process is still running and kill it directly by PID or `pkill` before continuing. This applies especially to `docker exec ... python ...` wrappers, which may survive a client-side abort.

## Commit & Pull Request Guidelines
- Recent commits use short, action-oriented subjects such as `Flatten config layout`.
- Keep PRs scoped to one logical change. List the affected dataset/backend, exact verification commands, and any metric deltas.
- Do not commit generated outputs, decoded ScanNet frames, checkpoints, or other large artifacts.
