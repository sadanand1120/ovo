# Open-Vocabulary Online Semantic Mapping for SLAM

This fork is reduced to the validated reproduction path:

- datasets: `Replica`, `ScanNet`
- backends: `vanilla`, `orbslam`, `cuvslam`
- semantic stack: `SAM2.1 + SigLIP + learned fusion`

`cuvslam` is integrated as a pose provider only. Dense depth unprojection and semantic fusion stay in OVO.

## Layout

- `ovo/`: flat first-party package
- `configs/`: runtime config and dataset metadata
- `data/input/`: datasets and checkpoints
- `data/output/`: generated runs
- `thirdParty/`: `ORB_SLAM3` and `segment-anything-2`

Main entry points:

- `run_eval.py`
- `visualize_scene.py`
- `scannet_decode_sens.py`
- `download_ckpts.sh`

## Container Setup

These steps assume you already have a compatible container and the repo is mounted somewhere inside it.

Start the container and open a shell:

```bash
docker start <container_name>
docker exec -it <container_name> bash
```

Inside the container:

```bash
conda create -y -n ovo python=3.10
conda activate ovo

pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

pip install pyyaml tqdm psutil wandb==0.19.8 plyfile matplotlib seaborn imageio \
  scipy==1.15.2 scikit-learn==1.6.1 pandas open3d==0.19.0 \
  open_clip_torch==2.32.0 huggingface-hub==0.30.1 einops==0.8.1 \
  timm==1.0.15 ftfy regex transformers==4.51.0 hydra-core==1.3.2 \
  iopath==0.1.10 omegaconf==2.3.0 tokenizers==0.21.1 \
  sentencepiece==0.2.0 blobfile tiktoken==0.9.0

conda install -y -c conda-forge py-opencv=4.11
```

Sanity check:

```bash
python - <<'PY'
import torch, torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Expected:

- `2.11.0+cu128`
- `0.26.0+cu128`
- `12.8`
- `True`

## Repo Setup

```bash
cd <repo_root>
git submodule update --init --recursive

cd thirdParty/segment-anything-2
pip install --no-build-isolation -e .
cd ../..

./download_ckpts.sh
```

This downloads:

- `data/input/sam_ckpts/sam2.1_hiera_large.pt`
- `data/input/weights_predictor/model.pt`

Tracked in git:

- `data/input/weights_predictor/hparams.yaml`

The first SigLIP run downloads weights into the container default Hugging Face cache, typically `/root/.cache/huggingface`.

## ORB-SLAM3 Setup

Install build dependencies in the same env:

```bash
conda activate ovo

conda install -y -c conda-forge \
  cxx-compiler glew eigen=3.4 pangolin-opengl=0.9.2 \
  boost pkg-config \
  libegl libegl-devel libgl libgl-devel \
  libglvnd libglvnd-devel libglx libglx-devel libopengl libopengl-devel
```

This container needed one env-local OpenGL fix before CMake could find the unversioned library name:

```bash
ln -sfn "$CONDA_PREFIX/lib/libOpenGL.so.0.0.0" \
  "$CONDA_PREFIX/lib/libOpenGL.so"
```

Use the conda `py-opencv` package, not the pip `opencv-python` wheel. In this container, the wheel mixed badly with the ORB runtime libraries.

Build ORB-SLAM3:

```bash
conda activate ovo

cd <repo_root>
./patch_orbslam_submodule.sh

cd thirdParty/ORB_SLAM3
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
bash build.sh
cd ../..
```

That patch script applies the two container-specific CMake fixes:

- build the Python bindings against the active Python version instead of hardcoded `3.11`
- resolve Pangolin's OpenGL imported targets cleanly in the container build

Import check:

```bash
python - <<'PY'
import orbslam3
print(orbslam3.__file__)
PY
```

## cuVSLAM Setup

`cuvslam` is optional. This repo uses it as an odometry-only backend first; no pose-graph rewrite or loop-closure integration is forced into OVO.

Install the official wheel for Ubuntu 22.04, Python 3.10, CUDA 12:

```bash
conda activate ovo

pip install \
  https://github.com/nvidia-isaac/cuVSLAM/releases/download/v15.0.0/cuvslam-15.0.0%2Bcu12-cp310-cp310-manylinux_2_35_x86_64.whl
```

Import check:

```bash
python - <<'PY'
import cuvslam
print(cuvslam.__version__)
PY
```

## Data

### Replica

Expected layout:

```text
data/input/Replica/
  semantic_gt/
  office0/
    results/
    traj.txt
  office0_mesh.ply
  ...
```

### ScanNet

OVO does not read `.sens` directly. Decode to:

```text
data/input/ScanNet/
  semantic_gt/
  scene0011_00/
    color/
    depth/
    pose/
    intrinsic/
    scene0011_00_vh_clean_2.labels.ply
```

Decode with:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python scannet_decode_sens.py \
  --scans_root /path/to/scannet_v2/scans \
  --output_root data/input/ScanNet \
  --write_semantic_gt \
  --link_pcds
```

## Smoke Tests

Short ScanNet smokes:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name smoke_vanilla --run \
  --scenes scene0011_00 --slam_module vanilla --frame_limit 5

export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name smoke_orbslam --run \
  --scenes scene0011_00 --slam_module orbslam --frame_limit 5

export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name smoke_cuvslam --run \
  --scenes scene0011_00 --slam_module cuvslam --frame_limit 5
```

Each successful run should produce:

- `data/output/ScanNet/<experiment>/<scene>/config.yaml`
- `data/output/ScanNet/<experiment>/<scene>/ovo_map.ckpt`
- `data/output/ScanNet/<experiment>/<scene>/estimated_c2w.npy`

## Reproduction

Single-scene end-to-end run:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name dev_run \
  --run --segment --eval --scenes scene0011_00 --slam_module vanilla
```

Run only the mapping stage:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name dev_run \
  --run --scenes scene0011_00 --slam_module vanilla
```

Segment and evaluate an existing run:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name dev_run \
  --segment --eval --scenes scene0011_00 --slam_module vanilla
```

Switch backends by changing only `--slam_module`:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name dev_run_orb \
  --run --segment --eval --scenes scene0011_00 --slam_module orbslam

export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name dev_run_cuv \
  --run --segment --eval --scenes scene0011_00 --slam_module cuvslam
```

Full 5-scene HVS sweep:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python run_eval.py --dataset_name ScanNet --experiment_name scannet_hvs \
  --run --segment --eval \
  --scenes scene0011_00 scene0050_00 scene0231_00 scene0378_00 scene0518_00 \
  --slam_module vanilla
```

## Metrics

Each run writes scene outputs under:

```text
data/output/<Dataset>/<experiment>/<scene>/
```

Useful files:

- `config.yaml`: resolved config used for that scene
- `estimated_c2w.npy`: predicted trajectory
- `ovo_map.ckpt`: saved semantic map
- `logger/avg_fps.log`: end-to-end scene FPS

Evaluation outputs are written under:

```text
data/output/<Dataset>/<experiment>/scannetv2/
```

To inspect final metrics for a run:

```bash
cat data/output/ScanNet/dev_run/scannetv2/statistics.txt
```

To inspect just the scene FPS logs:

```bash
for s in scene0011_00 scene0050_00 scene0231_00 scene0378_00 scene0518_00; do
  echo "== $s =="
  cat data/output/ScanNet/scannet_hvs/$s/logger/avg_fps.log
done
```

For the RGB+normal+CLIP+instance map path:

```bash
python get_metrics_map.py data/output/rgb_maps/ScanNet/scene0011_00 --save_json
```

`get_metrics_map.py` writes `metrics.json` with these metrics:

- `geometry`
  - `chamfer_l1_m`: mean bidirectional nearest-neighbor surface distance in meters. Range `[0, inf)`, lower is better.
  - `fscore_3cm`: reconstruction F-score at a `3 cm` distance threshold. Range `[0, 1]`, higher is better.
  - `coverage`: fraction of GT mesh vertices within `match_distance_th` of the predicted map. Range `[0, 1]`, higher is better.
- `rgb`
  - `psnr`: PSNR between predicted RGB and GT RGB after transferring colors onto GT mesh vertices. Range `[0, inf)`, higher is better.
- `normals`
  - `mean_angle_deg`: mean angular error between predicted and GT normals on GT mesh vertices. Range `[0, 180]`, lower is better.
- `feature`
  - `mIoU`: mean IoU of zero-shot semantic labels predicted from CLIP features on GT mesh vertices. Range `[0, 1]`, higher is better.
  - `mAcc`: mean class accuracy of those zero-shot semantic labels. Range `[0, 1]`, higher is better.
- `instance`
  - `mwcov`: mean weighted coverage of predicted instances against GT instances. Range `[0, 1]`, higher is better.
  - `f1_50`: instance F1 score at IoU `0.50`. Range `[0, 1]`, higher is better.

## Visualization

Semantic prediction view:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 \
  --working_dir . --visualize_semantic_pred
```

Instance view:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 \
  --working_dir . --visualize_obj
```

GT-vs-prediction view:

```bash
export CUDA_LAUNCH_BLOCKING=1 && python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 \
  --working_dir . --visualize_gt_vs_pre
```

Validated 5-scene HVS reproduction numbers are recorded in `validation_results.md`.
