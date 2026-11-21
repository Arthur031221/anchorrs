#!/usr/bin/env bash
# Downloads the eight Bonn RGB-D Dynamic sequences used in the paper from the official server,
# https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset
# Usage (from the repository root): bash scripts/download_bonn.sh [data/bonn]
set -euo pipefail

DEST=${1:-data/bonn}
mkdir -p "$DEST"
for name in balloon balloon2 crowd crowd2 person_tracking person_tracking2 \
    moving_nonobstructing_box moving_nonobstructing_box2; do
  seq=rgbd_bonn_$name
  if [ -d "$DEST/$seq/rgb" ]; then
    continue
  fi
  wget -c "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/$seq.zip" -P "$DEST"
  unzip -q "$DEST/$seq.zip" -d "$DEST"
  rm "$DEST/$seq.zip"
done
