# cuVSLAM Integration Analysis

## Current Status

The `humble` container is now the right environment for cuVSLAM:

- Ubuntu `22.04.5`
- CUDA toolkit `12.8`
- NVIDIA driver `580.126.09`
- ROS 2 Humble installed at `/opt/ros/humble`
- conda under `/opt/miniconda3`
- repo mounted at `/home/dynamo/AMRL_Research/repos/ovo`

The container `ovo` env is now aligned with the official wheel path:

- Python `3.10`
- `cuvslam 15.0.0+cu12` installed from the official wheel

References:

- https://github.com/nvidia-isaac/cuVSLAM
- https://nvlabs.github.io/PyCuVSLAM/

## Integration Shape

The current repo integration is intentionally narrow:

- `cuvslam` owns pose estimation
- OVO still owns dense depth unprojection
- OVO still owns semantic fusion
- no ORB-style keyframe emulation
- no cuVSLAM loop-closure / pose-graph rewrite logic in OVO

This is implemented as a first-pass `cuvslam` backend in `ovo/cuvslam.py`.

Short ScanNet smoke validation has already passed on `scene0011_00`.

Practically, the current contract is:

- input: RGB image, depth image, synthetic dataset timestamp, intrinsics
- output: per-frame `c2w` pose
- mapping: existing `VanillaMapper`

## Why This Middle-Ground Is Better

It avoids the two bad extremes:

- forcing cuVSLAM to imitate ORB-SLAM internals
- rewriting OVO around cuVSLAM-specific SLAM abstractions too early

The current design keeps the backend boundary simple and testable:

- pose provider backend
- existing OVO geometry + semantics path unchanged

## What Is Still Open

1. Decide whether to keep cuVSLAM odometry-only or add optional SLAM mode later.
- recommended next step: keep odometry-only until basic validation is stable

2. Decide whether ORB should eventually be simplified to the same pose-provider abstraction.
- recommended: yes, later

3. Validate whether RGB input is the best mode for these offline datasets, or whether grayscale preprocessing gives better stability.
- this is a tuning question, not an architectural blocker

4. Decide whether Replica should be validated immediately after ScanNet, or only after ScanNet behavior is acceptable.
- recommended: ScanNet first

## Bottom Line

The old blocker was platform mismatch. That is gone in the `humble` container.

The remaining work is no longer environment architecture. It is now normal backend validation and tuning:

- confirm short smoke runs
- measure ScanNet behavior
- only then consider optional SLAM-mode integration
