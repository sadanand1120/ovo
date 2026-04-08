# Repository Guidelines

## Project Structure & Module Organization
- `run_eval.py` is the main entry point for mapping, projection, and metrics. `visualize_scene.py` inspects saved runs. `scannet_decode_sens.py` prepares ScanNet RGB-D folders and `semantic_gt`.
- `ovo/` is intentionally flat. `ovomapping.py` orchestrates runs, `ovo.py` manages 3D instances, `clip_generator.py` / `mask_generator.py` wrap the semantic backbones, `vanilla_mapper.py` and `orbslam.py` provide the geometry backends, and the remaining modules are focused helpers.
- `configs/` holds the runtime config and dataset metadata: `ovo.yaml`, `replica.yaml`, `scannet.yaml`, `replica_eval.yaml`, and `scannet_eval.yaml`.
- `data/input/` holds datasets and checkpoints. `data/output/` holds generated runs. `thirdParty/` is reserved for external dependencies.

## Build, Test, and Development Commands
- Use `README.md` as the source of truth for environment setup, checkpoint download, ScanNet decoding, backend build steps, `run_eval.py` commands, metrics inspection, and visualization commands.
- `git submodule update --init --recursive` fetches `ORB_SLAM3` and `segment-anything-2`.
- Run all commands associated to this repo inside container named 'humble' and inside a conda (/opt/miniconda3) env called 'ovo'. Use docker exec to run commands inside the container.

## Coding Style & Naming Conventions
- Use 4-space indentation, `snake_case` for functions and config keys, and `PascalCase` for classes.
- Keep changes local and readable. Match nearby NumPy, Torch, and path-handling patterns instead of adding new abstraction layers.
- Avoid broad reformatting. There is no committed formatter config.

## Testing Guidelines
- There is no first-party unit test suite; validation is command-driven.
- For backend or mapping changes, run at least one small `run_eval.py` scene using the exact command patterns in `README.md`. For viewer changes, rerun `visualize_scene.py`.
- If you touch config loading, dataset prep, or output writing, confirm `config.yaml` and `ovo_map.ckpt` appear under `data/output/<Dataset>/<experiment>/<scene>/`.
- `vanilla` is deterministic here and is the best regression check. `orbslam` is functional but not bit-stable run to run.
- `--use-inst-gt` is only for isolating supervision-source issues while developing the instance pipeline; use it to separate SAM-side errors from instance-tracking bugs. This is NOT to be used to cheat in any sneaky way, or to sidestep the real intent of what the code is trying to do.
- If a long-running validation command is interrupted or a tool session aborts, verify whether the process is still running and kill it directly by PID or `pkill` before continuing. This applies especially to `docker exec ... python ...` wrappers, which may survive a client-side abort.

## Commit & Pull Request Guidelines
- Recent commits use short, action-oriented subjects such as `Flatten config layout`.
- Keep PRs scoped to one logical change. List the affected dataset/backend, exact verification commands, and any metric deltas.
- Do not commit generated outputs, decoded ScanNet frames, checkpoints, or other large artifacts.
