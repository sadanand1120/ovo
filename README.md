# RGB-Map Runtime

This repo is built around:

- `build_rgb_map.py`
- `topdown_vis.py`
- `visualize_rgb_map.py`
- `get_metrics_map.py`
- `get_ovo_style_eval.py`

Supported datasets:

- `ScanNet`
- `Replica`

Supported pose backends:

- `vanilla`
- `orbslam`
- `cuvslam`

## Layout

- `map_runtime/`: first-party runtime used by the main scripts
- `configs/`: base runtime config plus dataset metadata
- `data/input/`: datasets and checkpoints
- `data/output/`: generated outputs
- `thirdParty/`: `ORB_SLAM3` and `segment-anything-2`

## Environment

All repo commands should be run inside the `humble` container and the `ovo` conda env.

```bash
docker exec -it humble bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ovo
cd /home/dynamo/AMRL_Research/repos/ovo
```

If you need to create the environment from scratch inside the container:

```bash
conda create -y -n ovo python=3.10
conda activate ovo

pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

pip install pyyaml tqdm psutil plyfile scipy==1.15.2 scikit-learn==1.6.1 \
  open3d==0.19.0 open_clip_torch==2.32.0 huggingface-hub==0.30.1 \
  einops==0.8.1 timm==1.0.15 ftfy regex transformers==4.51.0 \
  hydra-core==1.3.2 iopath==0.1.10 omegaconf==2.3.0 \
  tokenizers==0.21.1 sentencepiece==0.2.0 blobfile tiktoken==0.9.0

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
git submodule update --init --recursive

cd thirdParty/segment-anything-2
pip install --no-build-isolation -e .
cd ../..

./download_ckpts.sh
```

`download_ckpts.sh` fetches the SAM1 and SAM2 checkpoints used by the pipeline.

## Backends

### ORB-SLAM3

Use the existing patch helper and build the submodule:

```bash
conda activate ovo

conda install -y -c conda-forge \
  cxx-compiler glew eigen=3.4 pangolin-opengl=0.9.2 \
  boost pkg-config \
  libegl libegl-devel libgl libgl-devel \
  libglvnd libglvnd-devel libglx libglx-devel libopengl libopengl-devel

ln -sfn "$CONDA_PREFIX/lib/libOpenGL.so.0.0.0" \
  "$CONDA_PREFIX/lib/libOpenGL.so"

./patch_orbslam_submodule.sh

cd thirdParty/ORB_SLAM3
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
bash build.sh
cd ../..
```

Import check:

```bash
python - <<'PY'
import orbslam3
print(orbslam3.__file__)
PY
```

### cuVSLAM

`cuvslam` is optional and is used as a pose provider.

Install the wheel for Ubuntu 22.04, Python 3.10, CUDA 12:

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

## Data Layout

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

Decode `.sens` into:

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

Use:

```bash
python scannet_decode_sens.py \
  --scans_root /path/to/scannet_v2/scans \
  --output_root data/input/ScanNet \
  --write_semantic_gt \
  --link_pcds
```

## Main Commands

### Build a map

ScanNet:

```bash
python build_rgb_map.py \
  --dataset_name ScanNet \
  --scene_name scene0011_00 \
  --slam_module vanilla
```

Replica:

```bash
python build_rgb_map.py \
  --dataset_name Replica \
  --scene_name office0 \
  --slam_module vanilla
```

The output scene directory contains:

- `rgb_map.ply`
- `clip_feats.npy`
- `instance_labels.npy`
- `stats.json`
- `timing.json`

### Render top-down incremental videos

```bash
python topdown_vis.py \
  --dataset_name ScanNet \
  --scene_name scene0011_00 \
  --load-view top_view_scene11.json
```

### Visualize a built map

```bash
python visualize_rgb_map.py data/output/rgb_maps/ScanNet/scene0011_00 --mode rgb
python visualize_rgb_map.py data/output/rgb_maps/ScanNet/scene0011_00 --mode normals
python visualize_rgb_map.py data/output/rgb_maps/ScanNet/scene0011_00 --mode feat
python visualize_rgb_map.py data/output/rgb_maps/ScanNet/scene0011_00 --mode instances
```

### Compute metrics

ScanNet:

```bash
python get_metrics_map.py data/output/rgb_maps/ScanNet/scene0011_00 \
  --scannet_raw_root /path/to/scannet_v2/scans --save_json
```

Replica:

```bash
python get_metrics_map.py data/output/rgb_maps/Replica/office0 \
  --replica_root data/input/Replica --save_json
```

For Replica, metrics that require unavailable GT signals are skipped cleanly.

### ScanNet HVS report

`get_ovo_style_eval.py` is for the ScanNet HVS report:

```bash
python get_ovo_style_eval.py \
  --dataset_name ScanNet \
  --scannet_raw_root /path/to/scannet_v2/scans
```
