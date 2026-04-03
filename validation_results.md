# Validation Results

These numbers compare the paper's 5-scene ScanNet HVS subset against the reproduced runs in this repo.

Scenes:
- `scene0011_00`
- `scene0050_00`
- `scene0231_00`
- `scene0378_00`
- `scene0518_00`

Config used for reproduction:
- [data/working/configs/ovo.yaml](/home/dynamo/AMRL_Research/repos/ovo/data/working/configs/ovo.yaml)

## Vanilla

Paper `OVO-mapping`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 38.1 | 50.5 | 57.6 | 70.5 |

Reproduced `vanilla`:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| 35.00 | 46.90 | 56.00 | 71.10 |

Per-scene reproduced `vanilla`:

| Scene | mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- | --- |
| `scene0011_00` | 42.74 | 52.88 | 63.32 | 75.78 |
| `scene0050_00` | 23.71 | 30.31 | 49.80 | 60.36 |
| `scene0231_00` | 30.27 | 39.80 | 62.24 | 74.75 |
| `scene0378_00` | 44.61 | 56.08 | 62.67 | 73.32 |
| `scene0518_00` | 31.14 | 44.21 | 48.61 | 65.23 |

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

Per-scene reproduced `orbslam3`:

| Scene | mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- | --- |
| `scene0011_00` | 36.52 | 47.73 | 55.02 | 69.69 |
| `scene0050_00` | 18.34 | 23.66 | 40.79 | 50.25 |
| `scene0231_00` | 28.10 | 36.27 | 57.67 | 71.17 |
| `scene0378_00` | 30.40 | 40.44 | 46.21 | 59.30 |
| `scene0518_00` | 28.68 | 42.07 | 46.27 | 62.75 |

Delta vs paper:

| mIoU | mAcc | f-mIoU | f-mAcc |
| --- | --- | --- | --- |
| -1.50 | -4.30 | +3.70 | +3.10 |

## Notes

- The `orbslam` reproduction here uses ORB-SLAM3, while the paper reports ORB-SLAM2.
- The reproduced runs are in:
  - [scannet_hvs_siglip_vanilla.preserved](/home/dynamo/AMRL_Research/repos/ovo/data/output/ScanNet/scannet_hvs_siglip_vanilla.preserved)
  - [scannet_hvs_siglip_orbslam.preserved](/home/dynamo/AMRL_Research/repos/ovo/data/output/ScanNet/scannet_hvs_siglip_orbslam.preserved)
