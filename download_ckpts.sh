#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$ROOT_DIR/data/input"
SAM_DIR="$INPUT_DIR/sam_ckpts"
PRED_DIR="$INPUT_DIR/weights_predictor/base"

mkdir -p "$SAM_DIR" "$PRED_DIR"

echo "Downloading SAM 2.1 large checkpoint..."
wget -c \
  -O "$SAM_DIR/sam2.1_hiera_large.pt" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown..."
  python -m pip install gdown
fi

echo "Downloading CLIP merging weights predictor..."
gdown --fuzzy \
  "https://drive.google.com/file/d/186wZ2mLES_QjUjW8l2DmVmlWDpNl2fSY/view" \
  -O "$PRED_DIR/model.pt"

echo "Done."
