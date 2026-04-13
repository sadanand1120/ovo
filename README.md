# RGB-Map Runtime

This repo is built around:

- `build_rgb_map.py`
- `topdown_vis.py`
- `visualize_rgb_map.py`
- `visualize_gt.py`
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
cd /robodata/smodak/repos/ovo
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

NICE-SLAM Replica root:

```text
<replica_niceslam_root>/
  office0/
    results/
      frame000000.jpg
      depth000000.png
      ...
    traj.txt
  office1/
  office2/
  office3/
  office4/
  room0/
  room1/
  room2/
  office0_mesh.ply
  office1_mesh.ply
  office2_mesh.ply
  office3_mesh.ply
  office4_mesh.ply
  room0_mesh.ply
  room1_mesh.ply
  room2_mesh.ply
```

Full Replica root:

```text
<replica_full_root>/
  office_0/
    habitat/
      mesh_semantic.ply
      info_semantic.json
      ...
    mesh.ply
    semantic.bin
    semantic.json
    preseg.bin
    preseg.json
    glass.sur
    textures/
  office_1/
  ...
```

Runtime layout:

```text
data/input/Replica/
  semantic_gt/
  office0/
    results/
    traj.txt
    label-filt/
    instance-filt/
    habitat/
    mesh.ply
    semantic.bin
    semantic.json
    preseg.bin
    preseg.json
    glass.sur
    textures/
  office0_mesh.ply
  ...
```

Setup flow:

1. Download the NICE-SLAM Replica bundle. This provides the RGB-D frames, poses, and root-level `<scene>_mesh.ply` files used by the mapping pipeline.

```bash
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm Replica.zip
```

2. Download the full Replica dataset. This provides the additional official scene assets that NICE-SLAM does not include, including the Habitat semantic/instance files such as `habitat/mesh_semantic.ply` and `habitat/info_semantic.json`.

```bash
git clone https://github.com/facebookresearch/Replica-Dataset /tmp/Replica-Dataset
cd /tmp/Replica-Dataset
./download.sh <replica_full_root>
```

The full Replica download uses scene names like `office_0` and `room_0`; `replica_decode.py` maps those automatically to the NICE-SLAM names `office0` and `room0`.

3. Replica semantic GT for the standard scenes lives in `data/input/Replica/semantic_gt`. In this repo, those are the `ovo-semantics` labels. The default Replica `semantics` and `instances` come from the full Replica Habitat assets.

4. Stage Replica into the runtime layout used by this repo. This merges the NICE-SLAM trajectories with the full Replica per-scene assets and generates per-frame `label-filt/*.png` and `instance-filt/*.png` files from the Habitat semantic mesh.

By default this creates symlinks in `data/input/Replica`:

```bash
python replica_decode.py \
  --source_root <replica_niceslam_root> \
  --full_replica_root <replica_full_root>
```

If you want actual copies instead of symlinks:

```bash
python replica_decode.py \
  --source_root <replica_niceslam_root> \
  --full_replica_root <replica_full_root> \
  --copy
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

### Visualize GT assets

```bash
python visualize_gt.py --dataset_name ScanNet scene0011_00 \
  --scannet_raw_root /path/to/scannet_v2/scans --mode semantics

python visualize_gt.py --dataset_name Replica office0 \
  --replica_root data/input/Replica --mode ovo-semantics
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

### ScanNet HVS report

Use `get_ovo_style_eval.py` for the ScanNet/Replica dataset-level OVO-style report:

```bash
python get_ovo_style_eval.py \
  --dataset_name ScanNet \
  --scannet_raw_root /path/to/scannet_v2/scans
```
