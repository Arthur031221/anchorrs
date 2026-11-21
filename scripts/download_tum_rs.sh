#!/usr/bin/env bash
# Downloads the ten sequences of the TUM rolling-shutter dataset (EuRoC export) from the
# official server, https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset
# Usage (from the repository root): bash scripts/download_tum_rs.sh [data/tum_rs]
set -euo pipefail

DEST=${1:-data/tum_rs}
mkdir -p "$DEST"
for n in 1 2 3 4 5 6 7 8 9 10; do
  if [ -d "$DEST/dataset-seq$n/mav0" ]; then
    continue
  fi
  wget -c "https://cdn3.vision.in.tum.de/rolling/exported/euroc/dataset-seq$n.tar" -P "$DEST"
  tar -xf "$DEST/dataset-seq$n.tar" -C "$DEST"
  rm "$DEST/dataset-seq$n.tar"
done
