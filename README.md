# Open-Vocabulary Online Semantic Mapping for SLAM

<a href="https://tberriel.github.io/">Tomas Berriel-Martins</a>,
<a href="https://oswaldm.github.io/">Martin R. Oswald</a>,
<a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en">Javier Civera</a>

<div align="left">
    <a href='https://arxiv.org/abs/2411.15043'><img src='https://img.shields.io/badge/arXiv-2404.06836-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://tberriel.github.io/ovo/'><img src='https://img.shields.io/badge/Web-Page-green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>

This repo is now intentionally trimmed to the validated reproduction path:

- `Replica` and `ScanNet` dataset support only
- `vanilla` backend for the paper-style GT-pose row
- `orbslam` backend via ORB-SLAM3
- paper-style semantic defaults in [data/working/configs/ovo.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ovo.yaml)

Gaussian-SLAM and the newer TextRegion / Perception Encoder path have been removed.

## Install

These are the exact environment steps that worked on this machine.

Clone with submodules:

```bash
git clone git@github.com:tberriel/OVO.git --recursive
cd OVO
```

Create the environment:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda create -n ovo python=3.11 -y
conda activate ovo

pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

pip install pyyaml tqdm psutil wandb==0.19.8 plyfile matplotlib seaborn imageio \
  scipy==1.15.2 scikit-learn==1.6.1 pandas open3d==0.19.0 \
  open_clip_torch==2.32.0 huggingface-hub==0.30.1 einops==0.8.1 \
  timm==1.0.15 ftfy regex transformers==4.51.0 hydra-core==1.3.2 \
  iopath==0.1.10 omegaconf==2.3.0 tokenizers==0.21.1 \
  sentencepiece==0.2.0 opencv-python==4.11.0.86 blobfile tiktoken==0.9.0

cd thirdParty/segment-anything-2
pip install --no-build-isolation -e .
cd ../..
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

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`

If `torch.cuda.is_available()` is `False`, check the driver before touching repo code:

```bash
python - <<'PY'
import ctypes
cuda = ctypes.CDLL('libcuda.so')
cuInit = cuda.cuInit
cuInit.argtypes = [ctypes.c_uint]
cuInit.restype = ctypes.c_int
print(cuInit(0))
PY
```

Expected output is `0`. If it returns `999` and `dmesg` shows `Xid 31` followed by `Node Reboot Required`, the driver is in a bad state and the machine must be rebooted before any CUDA-backed smoke run will work.

## Checkpoints

Download the manual checkpoints:

```bash
./download_ckpts.sh
```

This places:

- `data/input/sam_ckpts/sam2.1_hiera_large.pt`
- `data/input/weights_predictor/base/model.pt`

`hparams.yaml` for the weights predictor is already tracked. Hugging Face assets used by `open_clip` download into the default global Hugging Face cache on first use.

## ORB-SLAM3

ORB-SLAM3 is included as a submodule:

```bash
git submodule update --init --recursive
```

Runtime settings are generated automatically from the decoded scene intrinsics, so there are no committed per-scene ORB YAML files left in this repo.

### Build dependencies

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

conda install cxx-compiler -c conda-forge
conda install libegl libegl-devel libgl libgl-devel libgles libgles-devel \
  libglvnd libglvnd-devel libglx libglx-devel libopengl libopengl-devel -c conda-forge
conda install glew eigen=3.4 pangolin-opengl=0.9.2 libopencv=4.11 numpy=2.4 boost -c conda-forge
```

### Boost.Python 3.11

The missing piece on this machine was `boost_python311`. Building Boost locally into the env fixed it:

```bash
cd thirdParty
wget -c https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
tar -xf boost_1_86_0.tar.gz

source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

cd boost_1_86_0
./bootstrap.sh \
  --prefix=/home/dynamo/anaconda3/envs/ovo \
  --with-python=/home/dynamo/anaconda3/envs/ovo/bin/python \
  --with-python-version=3.11 \
  --with-libraries=python,serialization

./b2 -j4 variant=release link=shared runtime-link=shared \
  threading=multi cxxflags=-fPIC install
cd ../..
```

On this machine, `b2` may end with a `boost_numpy311` failure even though the two libraries ORB-SLAM3 actually needs were installed correctly. If both files below exist, continue:

```bash
ls /home/dynamo/anaconda3/envs/ovo/lib/libboost_python311.so.1.86.0
ls /home/dynamo/anaconda3/envs/ovo/lib/libboost_serialization.so.1.86.0
```

### Build ORB-SLAM3

This machine used an existing Pangolin build at:

- `/home/dynamo/AMRL_Research/competitions/RSS2023_SafeAutonomy/repos/Pangolin/build`

Build command:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

export CUDA_HOME=/usr/local/cuda-12.8
export Pangolin_DIR=/home/dynamo/AMRL_Research/competitions/RSS2023_SafeAutonomy/repos/Pangolin/build
export BOOST_ROOT=/home/dynamo/anaconda3/envs/ovo
export BOOST_INCLUDEDIR=/home/dynamo/anaconda3/envs/ovo/include
export BOOST_LIBRARYDIR=/home/dynamo/anaconda3/envs/ovo/lib
export CMAKE_PREFIX_PATH="$Pangolin_DIR:/home/dynamo/anaconda3/envs/ovo:${CMAKE_PREFIX_PATH}"

cd thirdParty/ORB_SLAM3
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
bash build.sh
cd ../..
```

### Pangolin linker hook

`orbslam3` did not import cleanly until Pangolin was added to `LD_LIBRARY_PATH` on env activation.

Create the hooks:

```bash
mkdir -p /home/dynamo/anaconda3/envs/ovo/etc/conda/activate.d
mkdir -p /home/dynamo/anaconda3/envs/ovo/etc/conda/deactivate.d
```

`/home/dynamo/anaconda3/envs/ovo/etc/conda/activate.d/ovo_orbslam.sh`

```bash
export OVO_ORBSLAM_PANGOLIN_DIR="/home/dynamo/AMRL_Research/competitions/RSS2023_SafeAutonomy/repos/Pangolin/build"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${OVO_ORBSLAM_PANGOLIN_DIR}:"*) ;;
  *) export LD_LIBRARY_PATH="${OVO_ORBSLAM_PANGOLIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
```

`/home/dynamo/anaconda3/envs/ovo/etc/conda/deactivate.d/ovo_orbslam.sh`

```bash
if [ -n "${OVO_ORBSLAM_PANGOLIN_DIR:-}" ]; then
  export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | sed "s#${OVO_ORBSLAM_PANGOLIN_DIR}:##; s#:${OVO_ORBSLAM_PANGOLIN_DIR}##; s#${OVO_ORBSLAM_PANGOLIN_DIR}##")"
  unset OVO_ORBSLAM_PANGOLIN_DIR
fi
```

### `libffi` fix

After fixing Pangolin, `orbslam3` still failed on this machine until `libffi.so.7` inside the env was pointed at the system copy:

```bash
cd /home/dynamo/anaconda3/envs/ovo/lib
mv -f libffi.so.7 libffi.so.7.conda-backup
ln -sfn /lib/x86_64-linux-gnu/libffi.so.7.1.0 libffi.so.7
```

Import sanity check:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo
python - <<'PY'
import orbslam3
print(orbslam3.__file__)
PY
```

## Stable runtime environment

These env vars were the stable runtime setup used here:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

export CUDA_HOME=/usr/local/cuda-12.8
export DISABLE_WANDB=true
export CUDA_LAUNCH_BLOCKING=1
```

`CUDA_LAUNCH_BLOCKING=1` mattered on this machine. Without it, SAM2 occasionally hit CUDA illegal-instruction / accelerator errors.

## Data

### Replica

OVO expects the NICE-SLAM processed Replica trajectories plus semantic GT:

```text
/<ovo_path>/data/input/Datasets/Replica/
  semantic_gt/
  office0/
    results/
    traj.txt
  office0_mesh.ply
  ...
```

Download and link:

```bash
cd /<ovo_path>/data/input/Datasets
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm Replica.zip
cd Replica
ln -s /<ovo_abs_path>/data/input/replica_semantic_gt semantic_gt
```

### ScanNet

OVO does not run directly from `.sens`. It expects decoded per-frame folders:

```text
/<ovo_path>/data/input/Datasets/ScanNet/
  semantic_gt/
  scene0011_00/
    color/
    depth/
    pose/
    intrinsic/
    scene0011_00_vh_clean_2.labels.ply
  ...
```

Starting from a ScanNet root like:

```text
/<ScanNet_data_path>/
  scans/
  data/val/
```

Decode, generate `semantic_gt`, and link meshes with one command:

```bash
cd /<ovo_path>
conda activate ovo
python scripts/scannet_decode_sens.py \
  --scans_root /<ScanNet_data_path>/scans \
  --output_root /<ScanNet_data_path>/data/val \
  --write_semantic_gt --link_pcds
```

Then link the decoded validation split:

```bash
ln -s /<ScanNet_data_path>/data/val /<ovo_path>/data/input/Datasets/ScanNet
```

### Repo-local ScanNet decode helper

If you want to decode raw `.sens` scenes into this repo without touching the source dataset:

```bash
python scripts/scannet_decode_sens.py \
  --scans_root /home/dynamo/AMRL_Research/dataset/scannet_v2/scans \
  --output_root /home/dynamo/AMRL_Research/repos/ovo/data/input/scannet_v2_ovo/data/val \
  --scenes scene0000_00 scene0002_00 \
  --write_semantic_gt --link_pcds \
  --min_free_gb 100
```

Then expose the decoded split to OVO:

```bash
ln -sfn /home/dynamo/AMRL_Research/repos/ovo/data/input/scannet_v2_ovo/data/val \
  /home/dynamo/AMRL_Research/repos/ovo/data/input/Datasets/ScanNet
```

## Defaults and reproduction

The repo defaults now match the validated reproduction path. The important semantic settings are already the defaults in [data/working/configs/ovo.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ovo.yaml):

- `slam_module: vanilla`
- `embed_type: learned`
- `model_card: SigLIP-384`
- `fusion: l1_medoid`
- `k_top_views: 10`
- `mask_res: 384`
- `sam_version: 2.1`
- `sam_encoder: hiera_l`

Backend notes:

- `vanilla` is the correct backend for the paper’s GT-pose `OVO-mapping` row.
- `orbslam` here uses ORB-SLAM3 and reaches the same ballpark as the paper’s ORB-SLAM2 row.

The paper HVS subset is:

- `scene0011_00`
- `scene0050_00`
- `scene0231_00`
- `scene0378_00`
- `scene0518_00`

Those scenes are already listed in [data/working/configs/ScanNet/eval_info.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ScanNet/eval_info.yaml).

## Run OVO

Use `run_eval.py` for mapping, segmentation, and evaluation.

Key flags:

- `--dataset_name`: `Replica` or `ScanNet`
- `--experiment_name`: output folder name under `data/output/<dataset_name>/`
- `--run`: run mapping
- `--segment`: project labels onto the GT point cloud
- `--eval`: compute final metrics
- `--dataset_info_file`: defaults to `eval_info.yaml`
- `--scenes`: explicit scene list
- `--scenes_list`: text file containing one scene per line
- `--slam_module`: override backend, e.g. `vanilla` or `orbslam`
- `--config_path`: override the base config, defaults to `data/working/configs/ovo.yaml`

Examples:

```bash
python run_eval.py --dataset_name Replica --experiment_name ovo_mapping \
  --run --segment --eval --scenes office0
```

```bash
python run_eval.py --dataset_name ScanNet --experiment_name ovo_mapping \
  --run --segment --eval --scenes scene0011_00
```

### Reproduce the 5-scene HVS subset

Vanilla:

```bash
python run_eval.py \
  --dataset_name ScanNet \
  --experiment_name scannet_hvs_siglip_vanilla \
  --run --segment --eval \
  --scenes scene0011_00 scene0050_00 scene0231_00 scene0378_00 scene0518_00 \
  --slam_module vanilla
```

ORB:

```bash
python run_eval.py \
  --dataset_name ScanNet \
  --experiment_name scannet_hvs_siglip_orbslam \
  --run --segment --eval \
  --scenes scene0011_00 scene0050_00 scene0231_00 scene0378_00 scene0518_00 \
  --slam_module orbslam
```

## Validated HVS results

These are the validated 5-scene ScanNet HVS numbers from this repo.

### Vanilla

Paper `OVO-mapping`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 38.1 | 50.5 | 57.6 | 70.5 |

Reproduced `vanilla`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 35.00 | 46.90 | 56.00 | 71.10 |

### ORB

Paper `OVO-ORB-SLAM2`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 31.3 | 45.2 | 45.8 | 61.2 |

Reproduced `orbslam3`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 29.80 | 40.90 | 49.50 | 64.30 |

## Known pitfalls

These were the actual setup failures that had to be solved here:

1. ORB-SLAM3 did not build out of the box because Python 3.11 Boost bindings were missing.
2. `orbslam3` imported only after adding Pangolin to `LD_LIBRARY_PATH`.
3. Even then, `libffi` in the conda env broke GTK / Pangolin transitively; the env’s `libffi.so.7` symlink had to be replaced with the system `libffi.so.7.1.0`.
4. SAM2 was unstable on this stack unless flash attention stayed disabled in the repo and `CUDA_LAUNCH_BLOCKING=1` was used.

## Changelog

- 11 January 2026: update environment to Python 3.11 and NumPy 2.4
- 7 October 2025: switch from ORB-SLAM2 to ORB-SLAM3
- 10 June 2025: improve ORB-SLAM2 integration and add loop closure support

## Citation

```bibtex
@article{martins2024ovo,
  title={Open-Vocabulary Online Semantic Mapping for SLAM},
  author={Martins, Tomas Berriel and Oswald, Martin R. and Civera, Javier},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
}
```
