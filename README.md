# Open-Vocabulary Online Semantic Mapping for SLAM

<a href="https://tberriel.github.io/">Tomas Berriel-Martins</a>,
<a href="https://oswaldm.github.io/">Martin R. Oswald</a>,
<a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en">Javier Civera</a>

<div align="left">
  <a href="https://arxiv.org/abs/2411.15043"><img src="https://img.shields.io/badge/arXiv-2404.06836-b31b1b.svg"></a>
  <a href="https://tberriel.github.io/ovo/"><img src="https://img.shields.io/badge/Web-Page-green"></a>
</div>

This fork is trimmed to the validated reproduction path:

- datasets: `Replica`, `ScanNet`
- backends: `vanilla`, `orbslam` (ORB-SLAM3)
- semantic stack: paper-style `SAM2.1 + SigLIP + learned fusion`

Gaussian-SLAM and the newer TextRegion / Perception Encoder path are intentionally removed.

## Layout

- [ovo](/home/dynamo/AMRL_Research/repos/ovo/ovo): flat first-party package
- [configs](/home/dynamo/AMRL_Research/repos/ovo/configs): base config plus dataset metadata
- [data/input](/home/dynamo/AMRL_Research/repos/ovo/data/input): datasets and checkpoints
- [data/output](/home/dynamo/AMRL_Research/repos/ovo/data/output): run outputs
- [thirdParty](/home/dynamo/AMRL_Research/repos/ovo/thirdParty): external dependencies

Main entry points:

- [run_eval.py](/home/dynamo/AMRL_Research/repos/ovo/run_eval.py)
- [visualize_scene.py](/home/dynamo/AMRL_Research/repos/ovo/visualize_scene.py)
- [scannet_decode_sens.py](/home/dynamo/AMRL_Research/repos/ovo/scannet_decode_sens.py)
- [download_ckpts.sh](/home/dynamo/AMRL_Research/repos/ovo/download_ckpts.sh)

## Environment

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

If CUDA is down, check the driver before touching repo code:

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

If that returns `999`, reboot the machine.

## Checkpoints

```bash
./download_ckpts.sh
```

This downloads:

- `data/input/sam_ckpts/sam2.1_hiera_large.pt`
- `data/input/weights_predictor/model.pt`

Tracked in git:

- `data/input/weights_predictor/hparams.yaml`

`open_clip` uses the default global Hugging Face cache on first use.

## ORB-SLAM3

Initialize submodules:

```bash
git submodule update --init --recursive
```

Install build deps:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

conda install cxx-compiler -c conda-forge
conda install libegl libegl-devel libgl libgl-devel libgles libgles-devel \
  libglvnd libglvnd-devel libglx libglx-devel libopengl libopengl-devel -c conda-forge
conda install glew eigen=3.4 pangolin-opengl=0.9.2 libopencv=4.11 numpy=2.4 boost -c conda-forge
```

On this machine, ORB-SLAM3 also needed local Boost Python 3.11:

```bash
cd thirdParty
wget -c https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
tar -xf boost_1_86_0.tar.gz

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

Build ORB-SLAM3:

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

This env also needed two local fixes.

Pangolin activation hook:

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

`libffi` fix:

```bash
cd /home/dynamo/anaconda3/envs/ovo/lib
mv -f libffi.so.7 libffi.so.7.conda-backup
ln -sfn /lib/x86_64-linux-gnu/libffi.so.7.1.0 libffi.so.7
```

Import check:

```bash
python - <<'PY'
import orbslam3
print(orbslam3.__file__)
PY
```

## Runtime env

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

export CUDA_HOME=/usr/local/cuda-12.8
export DISABLE_WANDB=true
export CUDA_LAUNCH_BLOCKING=1
```

`CUDA_LAUNCH_BLOCKING=1` mattered on this machine.

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

Minimal setup:

```bash
cd data/input
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm Replica.zip
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

Decode from raw ScanNet:

```bash
python scannet_decode_sens.py \
  --scans_root /path/to/scannet/scans \
  --output_root data/input/ScanNet \
  --write_semantic_gt --link_pcds
```

For selected scenes only:

```bash
python scannet_decode_sens.py \
  --scans_root /path/to/scannet/scans \
  --output_root data/input/ScanNet \
  --scenes scene0011_00 scene0518_00 \
  --write_semantic_gt --link_pcds \
  --min_free_gb 60
```

## Defaults

The validated defaults are already in [configs/ovo.yaml](/home/dynamo/AMRL_Research/repos/ovo/configs/ovo.yaml):

- `slam_module: vanilla`
- `embed_type: learned`
- `model_card: SigLIP-384`
- `fusion: l1_medoid`
- `k_top_views: 10`
- `mask_res: 384`
- `sam_version: 2.1`
- `sam_encoder: hiera_l`

Dataset metadata:

- [configs/replica.yaml](/home/dynamo/AMRL_Research/repos/ovo/configs/replica.yaml)
- [configs/scannet.yaml](/home/dynamo/AMRL_Research/repos/ovo/configs/scannet.yaml)
- [configs/replica_eval.yaml](/home/dynamo/AMRL_Research/repos/ovo/configs/replica_eval.yaml)
- [configs/scannet_eval.yaml](/home/dynamo/AMRL_Research/repos/ovo/configs/scannet_eval.yaml)

## Run

Single-scene examples:

```bash
python run_eval.py --dataset_name Replica --experiment_name dev_run \
  --run --segment --eval --scenes office0
```

```bash
python run_eval.py --dataset_name ScanNet --experiment_name dev_run \
  --run --segment --eval --scenes scene0011_00 --slam_module vanilla
```

```bash
python run_eval.py --dataset_name ScanNet --experiment_name dev_run \
  --run --segment --eval --scenes scene0011_00 --slam_module orbslam
```

Useful flags:

- `--dataset_info_file`: defaults to `eval.yaml`
- `--scenes_list`: text file with one scene per line
- `--frame_limit`: short smoke runs
- `--config_path`: override base config

## Visualize

```bash
python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 \
  --working_dir . --visualize_obj
```

```bash
python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 \
  --working_dir . --visualize_semantic_pred
```

```bash
python visualize_scene.py data/output/ScanNet/dev_run/scene0011_00 \
  --working_dir . --visualize_interactive_query
```

## Reproduce the ScanNet HVS subset

Scenes:

- `scene0011_00`
- `scene0050_00`
- `scene0231_00`
- `scene0378_00`
- `scene0518_00`

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

## Reference results

Validated reproduction numbers are in [validation_results.md](/home/dynamo/AMRL_Research/repos/ovo/validation_results.md).

Summary:

- `vanilla`: `35.0 / 46.9 / 56.0 / 71.1`
- `orbslam3`: `29.8 / 40.9 / 49.5 / 64.3`

`vanilla` is deterministic here. `orbslam` is functional but not bit-stable run to run.
