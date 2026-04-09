#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$ROOT_DIR/data/input"
SAM_DIR="$INPUT_DIR/sam_ckpts"
PRED_DIR="$INPUT_DIR/weights_predictor"

mkdir -p "$SAM_DIR" "$PRED_DIR"

echo "Downloading SAM 2.1 large checkpoint..."
wget -c \
  -O "$SAM_DIR/sam2.1_hiera_large.pt" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

echo "Downloading SAM 2.1 base+ checkpoint..."
wget -c \
  -O "$SAM_DIR/sam2.1_hiera_base_plus.pt" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

echo "Downloading SAM 2.1 small checkpoint..."
wget -c \
  -O "$SAM_DIR/sam2.1_hiera_small.pt" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

echo "Downloading SAM 2.1 tiny checkpoint..."
wget -c \
  -O "$SAM_DIR/sam2.1_hiera_tiny.pt" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

echo "Downloading SAM ViT-H checkpoint..."
wget -c \
  -O "$SAM_DIR/sam_vit_h_4b8939.pth" \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

echo "Downloading SAM ViT-L checkpoint..."
wget -c \
  -O "$SAM_DIR/sam_vit_l_0b3195.pth" \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"

echo "Downloading SAM ViT-B checkpoint..."
wget -c \
  -O "$SAM_DIR/sam_vit_b_01ec64.pth" \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown..."
  python -m pip install gdown
fi

echo "Downloading CLIP merging weights predictor..."
gdown --fuzzy \
  "https://drive.google.com/file/d/186wZ2mLES_QjUjW8l2DmVmlWDpNl2fSY/view" \
  -O "$PRED_DIR/model.pt"

echo "Done."
