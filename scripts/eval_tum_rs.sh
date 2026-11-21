#!/usr/bin/env bash
# TUM rolling-shutter rows of the paper: the rolling-shutter camera cam1 of the ten sequences,
# seeds 0, 1 and 2, with the rolling-shutter model and with tau = 0. Prints the mean ATE (cm)
# of each. The undistorted pinhole images are written to <sequence>/pinhole_cam1.
# Usage (from the repository root): bash scripts/eval_tum_rs.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
CKPT=${CKPT:-yyfz233/Pi3}
DATA=${DATA:-data/tum_rs}
CONFIG=configs/tum_rs.yaml
WIN=windows/tum_rs
OUT=outputs/tum_rs
CAM=1
# Thread counts of the paper runs. They fix the rounding of the reductions.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$WIN" "$OUT"
SEQS=$($PYTHON -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['sequences'])" \
  "$CONFIG")

# Windows step: pinhole images, then pi3 on the anchored windows of each video, on the GPU.
for s in $SEQS; do
  if [ ! -f "$WIN/${s}_cam$CAM.npz" ]; then
    $PYTHON - "$DATA/$s" "$WIN/${s}_cam$CAM.npz" "$CKPT" "$CAM" <<'EOF'
import sys
import run
frames, _ = run.load_frames("tum_rs", sys.argv[1], int(sys.argv[4]), None)
run.compute_windows(frames, sys.argv[2], sys.argv[3])
EOF
  fi
done

# Solver on the CPU: three seeds, with the rolling-shutter model and with tau = 0.
for seed in 0 1 2; do
  for s in $SEQS; do
    for arm in rs gs; do
      if [ "$arm" = rs ]; then
        out=$OUT/${s}_s$seed.npz; flag=()
      else
        out=$OUT/${s}_gs_s$seed.npz; flag=(--global-shutter)
      fi
      if [ ! -f "$out" ]; then
        CUDA_VISIBLE_DEVICES= $PYTHON run.py --config "$CONFIG" --seq "$DATA/$s" --cam "$CAM" \
          --windows "$WIN/${s}_cam$CAM.npz" --seed "$seed" "${flag[@]}" --out "$out"
      fi
    done
  done
done

echo "AnchorRS"
$PYTHON evaluate.py --config "$CONFIG" --data-root "$DATA" --cam "$CAM" \
  --traj "$OUT/{seq}_s{seed}.npz" --seeds 0 1 2
echo "AnchorRS with tau = 0"
$PYTHON evaluate.py --config "$CONFIG" --data-root "$DATA" --cam "$CAM" \
  --traj "$OUT/{seq}_gs_s{seed}.npz" --seeds 0 1 2
