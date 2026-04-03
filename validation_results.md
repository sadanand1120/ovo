# Validation Results

These numbers compare the paper's 5-scene ScanNet HVS subset against the reproduced runs in this repo.

Scenes:
- `scene0011_00`
- `scene0050_00`
- `scene0231_00`
- `scene0378_00`
- `scene0518_00`

Config used for reproduction:
- [data/working/configs/ovo_paper_scan.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ovo_paper_scan.yaml)

## Vanilla

Paper `OVO-mapping`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 38.1 | 50.5 | 57.6 | 70.5 |

Reproduced `vanilla`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 35.00 | 46.90 | 56.00 | 71.10 |

Delta vs paper:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| -3.10 | -3.60 | -1.60 | +0.60 |

## ORB

Paper `OVO-ORB-SLAM2`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 31.3 | 45.2 | 45.8 | 61.2 |

Reproduced `orbslam3`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 29.80 | 40.90 | 49.50 | 64.30 |

Delta vs paper:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| -1.50 | -4.30 | +3.70 | +3.10 |

## Notes

- The `orbslam` reproduction here uses ORB-SLAM3, while the paper reports ORB-SLAM2.
- The reproduced runs are in:
  - [scannet_hvs_siglip_vanilla](/home/dynamo/AMRL_Research/repos/ovo/data/output/ScanNet/scannet_hvs_siglip_vanilla)
  - [scannet_hvs_siglip_orbslam](/home/dynamo/AMRL_Research/repos/ovo/data/output/ScanNet/scannet_hvs_siglip_orbslam)
