# Repository Guidelines

## Project Structure & Module Organization
- `run_eval.py` is the main entry point for mapping, projection, and metrics. `visualize_scene.py` inspects saved runs. `scannet_decode_sens.py` prepares ScanNet RGB-D folders and `semantic_gt`.
- `ovo/` is intentionally flat. `ovomapping.py` orchestrates runs, `ovo.py` manages 3D instances, `clip_generator.py` / `mask_generator.py` wrap the semantic backbones, `vanilla_mapper.py` and `orbslam.py` provide the geometry backends, and the remaining modules are focused helpers.
- `configs/` holds the runtime config and dataset metadata: `ovo.yaml`, `replica.yaml`, `scannet.yaml`, `replica_eval.yaml`, and `scannet_eval.yaml`.
- `data/input/` holds datasets and checkpoints. `data/output/` holds generated runs. `thirdParty/` is reserved for external dependencies.

## Build, Test, and Development Commands
- `git submodule update --init --recursive` fetches `ORB_SLAM3` and `segment-anything-2`.
- Follow `README.md` to create the `ovo` conda env, install Python deps, run `./download_ckpts.sh`, and build `thirdParty/ORB_SLAM3`.
- `python scannet_decode_sens.py --scans_root /path/to/ScanNet/scans --output_root data/input/ScanNet --write_semantic_gt --link_pcds` prepares ScanNet data in the format OVO expects.
- `python run_eval.py --dataset_name ScanNet --experiment_name dev_run --run --segment --eval --scenes scene0011_00 --slam_module vanilla` is the smallest full validation path.
- `python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 --working_dir . --visualize_semantic_pred` is the quickest semantic viewer sanity check.

## Coding Style & Naming Conventions
- Use 4-space indentation, `snake_case` for functions and config keys, and `PascalCase` for classes.
- Keep changes local and readable. Match nearby NumPy, Torch, and path-handling patterns instead of adding new abstraction layers.
- Avoid broad reformatting. There is no committed formatter config.

## Testing Guidelines
- There is no first-party unit test suite; validation is command-driven.
- For backend or mapping changes, run at least one small `run_eval.py` scene. For viewer changes, rerun `visualize_scene.py`.
- If you touch config loading, dataset prep, or output writing, confirm `config.yaml` and `ovo_map.ckpt` appear under `data/output/<Dataset>/<experiment>/<scene>/`.
- `vanilla` is deterministic here and is the best regression check. `orbslam` is functional but not bit-stable run to run.

## Commit & Pull Request Guidelines
- Recent commits use short, action-oriented subjects such as `Flatten config layout`.
- Keep PRs scoped to one logical change. List the affected dataset/backend, exact verification commands, and any metric deltas.
- Do not commit generated outputs, decoded ScanNet frames, checkpoints, or other large artifacts.
