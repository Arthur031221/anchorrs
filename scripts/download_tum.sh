#!/usr/bin/env bash
# Downloads the four fr3 walking sequences of TUM RGB-D from the official server,
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
# Usage (from the repository root): bash scripts/download_tum.sh [data/tum]
set -euo pipefail

DEST=${1:-data/tum}
mkdir -p "$DEST"
for name in walking_static walking_xyz walking_halfsphere walking_rpy; do
  seq=rgbd_dataset_freiburg3_$name
  if [ -d "$DEST/$seq/rgb" ]; then
    continue
  fi
  wget -c "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/$seq.tgz" -P "$DEST"
  tar -xzf "$DEST/$seq.tgz" -C "$DEST"
  rm "$DEST/$seq.tgz"
done
