# Environment Setup Log

This file records the setup that was required on this machine to get OVO running with:

- `vanilla` backend
- `orbslam` backend via `thirdParty/ORB_SLAM3`

It reflects the working state as of `2026-04-01` in `/home/dynamo/AMRL_Research/repos/ovo`.

## 1. Base Conda Environment

I created and used:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda create -n ovo python=3.11 -y
conda activate ovo
```

## 2. Python / CUDA Stack

I installed the CUDA 12.8 PyTorch build. I installed packages in a few batches while debugging; the command below is the known-good equivalent of the final state:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

pip install \
  pyyaml tqdm psutil wandb==0.19.8 plyfile matplotlib seaborn imageio \
  scipy==1.15.2 scikit-learn==1.6.1 pandas open3d==0.19.0 \
  open_clip_torch==2.32.0 huggingface-hub==0.30.1 einops==0.8.1 \
  timm==1.0.15 ftfy regex transformers==4.51.0 hydra-core==1.3.2 \
  iopath==0.1.10 omegaconf==2.3.0 tokenizers==0.21.1 \
  sentencepiece==0.2.0 opencv-python==4.11.0.86 blobfile \
  tiktoken==0.9.0
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

Working result here:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`

## 3. Editable Third-Party Python Packages

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

pip install -e thirdParty/perception_models --no-deps
pip install --no-build-isolation -e thirdParty/segment-anything-2
```

## 4. Checkpoints / Caches

Manual checkpoints should be downloaded with:

```bash
cd /home/dynamo/AMRL_Research/repos/ovo
./download_ckpts.sh
```

This places:

- SAM checkpoint at `data/input/sam_ckpts/sam2.1_hiera_large.pt`
- CLIP merging weights predictor at `data/input/weights_predictor/base/model.pt`

Hugging Face assets used by `open_clip` / Perception Encoder are auto-downloaded on first run into the default global Hugging Face cache.

## 5. ORB-SLAM3 Native Build

Clone:

```bash
cd /home/dynamo/AMRL_Research/repos/ovo/thirdParty
git clone https://github.com/tberriel/ORB_SLAM3
```

### 5.1 Boost.Python 3.11

The main blocker was missing `boost_python311`. I built Boost locally into the `ovo` env:

```bash
set -e
cd /home/dynamo/AMRL_Research/repos/ovo/thirdParty
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
```

### 5.2 ORB-SLAM3 Build

This repo used an existing Pangolin build at:

- `/home/dynamo/AMRL_Research/competitions/RSS2023_SafeAutonomy/repos/Pangolin/build`

Build command:

```bash
set -e
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

export CUDA_HOME=/usr/local/cuda-12.8
export Pangolin_DIR=/home/dynamo/AMRL_Research/competitions/RSS2023_SafeAutonomy/repos/Pangolin/build
export BOOST_ROOT=/home/dynamo/anaconda3/envs/ovo
export BOOST_INCLUDEDIR=/home/dynamo/anaconda3/envs/ovo/include
export BOOST_LIBRARYDIR=/home/dynamo/anaconda3/envs/ovo/lib
export CMAKE_PREFIX_PATH="$Pangolin_DIR:/home/dynamo/anaconda3/envs/ovo:${CMAKE_PREFIX_PATH}"

cd /home/dynamo/AMRL_Research/repos/ovo/thirdParty/ORB_SLAM3
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
bash build.sh
```

This installed:

- `/home/dynamo/anaconda3/envs/ovo/lib/python3.11/site-packages/orbslam3.so`

## 6. Runtime Linker Fixes

### 6.1 Pangolin in `LD_LIBRARY_PATH`

`orbslam3` initially failed to import because `libpango_vars.so` was not found. I fixed that by adding a conda activation hook:

- `/home/dynamo/anaconda3/envs/ovo/etc/conda/activate.d/ovo_orbslam.sh`
- `/home/dynamo/anaconda3/envs/ovo/etc/conda/deactivate.d/ovo_orbslam.sh`

These prepend/remove:

- `/home/dynamo/AMRL_Research/competitions/RSS2023_SafeAutonomy/repos/Pangolin/build`

### 6.2 `libffi` / `libp11-kit` mismatch

After fixing Pangolin, `import orbslam3` failed with:

- `/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`

Cause:

- the conda env had `libffi.so.7 -> libffi.so.8.1.2`
- `orbslam3.so` RUNPATH searches the env `lib/` first
- GTK / Pangolin transitively loaded the wrong `libffi.so.7`

Fix applied:

```bash
cd /home/dynamo/anaconda3/envs/ovo/lib
mv -f libffi.so.7 libffi.so.7.conda-backup
ln -sfn /lib/x86_64-linux-gnu/libffi.so.7.1.0 libffi.so.7
```

After that:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo
python - <<'PY'
import orbslam3
print(orbslam3.__file__)
PY
```

worked.

## 7. Repo-Side Changes Needed For This Machine

These were required to make the current repo run cleanly:

- `ovo/utils/segment_utils.py`
  - disabled SAM2 flash attention
  - forced math kernel path
- `run_eval.py`
  - added `--slam_module`
  - added `--frame_limit`
- `data/working/configs/slam/orbslam3/`
  - added `ScanNet.yaml`
  - added per-scene configs for `scene0000_00` and `scene0002_00`
  - added `vocabulary/ORBvoc.txt` symlink to `thirdParty/ORB_SLAM3/Vocabulary/ORBvoc.txt`

## 8. Stable Runtime Environment

For stable runs on this machine, I used:

```bash
source /home/dynamo/anaconda3/etc/profile.d/conda.sh
conda activate ovo

export CUDA_HOME=/usr/local/cuda-12.8
export DISABLE_WANDB=true
export CUDA_LAUNCH_BLOCKING=1
```

`CUDA_LAUNCH_BLOCKING=1` was important. Without it, SAM2 occasionally failed with CUDA illegal-instruction / accelerator errors in this stack.

## 9. Known-Good Run Commands

Vanilla:

```bash
python run_eval.py \
  --dataset_name ScanNet \
  --experiment_name vanilla_scene0002_blocking \
  --run --segment --eval \
  --scenes scene0002_00 \
  --slam_module vanilla
```

ORB:

```bash
python run_eval.py \
  --dataset_name ScanNet \
  --experiment_name orbslam_scene0002_blocking \
  --run --segment --eval \
  --scenes scene0002_00 \
  --slam_module orbslam
```

## 10. ScanNet Local Data Prep Used Here

This is not strictly env setup, but it was required to make the repo runnable against local ScanNet data without touching the source dataset.

Decode `.sens` into repo-local folders:

```bash
python scripts/scannet_decode_sens.py \
  --scans_root /home/dynamo/AMRL_Research/dataset/scannet_v2/scans \
  --output_root /home/dynamo/AMRL_Research/repos/ovo/data/input/scannet_v2_ovo/data/val \
  --scenes scene0000_00 scene0002_00 \
  --min_free_gb 100
```

## 11. Generate labels / point-cloud links

```bash
python scripts/scannet_preprocess.py \
  --data_path /home/dynamo/AMRL_Research/repos/ovo/data/input/scannet_v2_ovo \
  --link_pcds
```

Then make OVO see the decoded set:

```bash
ln -sfn \
  /home/dynamo/AMRL_Research/repos/ovo/data/input/scannet_v2_ovo/data/val \
  /home/dynamo/AMRL_Research/repos/ovo/data/input/Datasets/ScanNet
```

## 12. Hiccups Encountered

1. ORB-SLAM3 did not build out of the box because Python 3.11 Boost bindings were missing.
2. `orbslam3` imported only after adding Pangolin to `LD_LIBRARY_PATH`.
3. Even then, `libffi` in the conda env broke GTK / Pangolin transitively; I had to replace the env’s `libffi.so.7` symlink with the system `libffi.so.7.1.0`.
4. SAM2 was unstable on this stack unless flash attention was disabled and `CUDA_LAUNCH_BLOCKING=1` was used.
5. Current `main` is newer than the paper setup. It now defaults to TextRegion + Perception Encoder and ORB-SLAM3, so paper-faithful reproduction is a separate question from “getting current main to run.”

## 13. Final Working State

At the end of setup, all of these imported successfully inside `conda activate ovo`:

- `torch`
- `torchvision`
- `open3d`
- `open_clip`
- `transformers`
- `cv2`
- `orbslam3`

And the repo ran end-to-end on ScanNet with both:

- `vanilla`
- `orbslam`
