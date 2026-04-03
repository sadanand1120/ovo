# Repository Guidelines

## Project Structure & Module Organization
- `run_eval.py` is the main entry point for mapping and metrics. `visualize_scene.py` inspects outputs. `scripts/scannet_preprocess.py` prepares labels.
- `ovo/entities/` contains the semantic pipeline: `ovomapping.py` orchestrates runs, `ovo.py` manages 3D instances and text queries, and `clip_generator.py` / `mask_generator.py` wrap descriptor and mask extraction.
- `ovo/slam/` selects the geometry backend: `vanilla_mapper.py` uses GT poses, and `orbslam.py` wraps ORB-SLAM3.
- `ovo/utils/` contains config, geometry, segmentation, evaluation, and visualization helpers. `data/working/configs/` is layered by dataset and backend.
- Inputs and checkpoints live in `data/input/`; generated runs belong in `data/output/`. `thirdParty/` is for external dependencies.

## Build, Test, and Development Commands
- `git submodule update --init --recursive` fetches required third-party code.
- Follow `README.md` to create the `ovo` Conda environment and install editable dependencies. ORB-SLAM3 is expected under `thirdParty/ORB_SLAM3` when used.
- `python scripts/scannet_preprocess.py --data_path /path/to/ScanNet --link_pcds` prepares ScanNet labels and links meshes.
- `python run_eval.py --dataset_name Replica --experiment_name dev_run --run --segment --eval --scenes office0` runs the full local pipeline on one scene.
- `python visualize_scene.py data/output/Replica/dev_run/office0 --visualize_obj` inspects saved outputs.
- Set `DISABLE_WANDB=true` to disable WandB during local debugging.

## Coding Style & Naming Conventions
- Use 4-space indentation, type hints on public functions, and short docstrings where behavior is not obvious.
- Follow existing Python naming: `snake_case` for functions, variables, and YAML keys; `PascalCase` for classes.
- Keep changes focused and readable. Match nearby NumPy, Torch, YAML-merging, and path-handling patterns instead of introducing new abstractions.
- There is no committed formatter config; do not submit broad reformatting changes.

## Testing Guidelines
- There is no first-party unit test suite or CI config at the repository root; validation is command-driven.
- Before opening a PR, run the smallest scene or script that exercises your change. Use a single scene through `run_eval.py`; rerun `visualize_scene.py` for viewer changes.
- If you touch config loading, dataset preparation, or output writing, confirm the expected files appear under `data/output/<Dataset>/<experiment>/` and that `config.yaml` plus `ovo_map.ckpt` are written.

## Commit & Pull Request Guidelines
- Recent commits use short, capitalized summaries such as `Update configs`. Keep subjects brief and action-oriented.
- PRs should state the affected dataset or backend, list any config changes, include exact verification commands, and report metric deltas or screenshots when outputs change.
- Keep each PR scoped to one logical change. Do not include generated data, weights, or experiment outputs in commits.
