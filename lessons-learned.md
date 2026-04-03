# Lessons Learned

## What mattered for reproduction

- Use the paper-style semantic config, not current `main` defaults. The relevant file is [data/working/configs/ovo_paper_scan.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ovo_paper_scan.yaml).
- The paper-like path on current `main` is:
  - `embed_type: learned`
  - `model_card: SigLIP-384`
  - `fusion: l1_medoid`
  - `k_top_views: 10`
  - `mask_res: 384`
  - `SAM 2.1` with `hiera_l`
- Current default `main` config uses `TextRegion + PE`, which is not the right starting point for reproducing the paper numbers.

## Dataset requirements

- ScanNet must be decoded into per-frame folders:
  - `color/`
  - `depth/`
  - `pose/`
  - `intrinsic/`
- OVO does not run directly from `.sens`.
- `semantic_gt/<scene>.txt` must be generated before evaluation.

## Backend-specific notes

- `vanilla` is the correct backend for the paper’s GT-pose `OVO-mapping` row.
- `orbslam` on current `main` can get into the paper ballpark, but it needed one code fix: [ovo/entities/ovomapping.py](/home/dynamo/AMRL_Research/repos/ovo/ovo/entities/ovomapping.py) was still special-casing `orbslam2` when deciding when to map. That had to be generalized to `orbslam*`.
- ORB-SLAM3 was close enough to the paper’s ORB-SLAM2 row that checking out an older commit was not necessary for a ballpark reproduction.

## Useful reproduction target

- The paper HVS subset already exists in [data/working/configs/ScanNet/eval_info.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ScanNet/eval_info.yaml):
  - `scene0011_00`
  - `scene0050_00`
  - `scene0231_00`
  - `scene0378_00`
  - `scene0518_00`

## Result summary

- `vanilla` reached a near-reproduction of the paper `OVO-mapping` row.
- `orbslam` reached the same ballpark as the paper `OVO-ORB-SLAM2` row, but remained clearly worse than `vanilla` on this subset.
