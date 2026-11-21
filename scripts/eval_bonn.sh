#!/usr/bin/env bash
# Bonn RGB-D Dynamic rows of the paper table: eight sequences, seeds 0, 1 and 2, with the
# rolling-shutter model and with tau = 0. Prints the mean ATE (cm) of each.
# Usage (from the repository root): bash scripts/eval_bonn.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
CKPT=${CKPT:-yyfz233/Pi3}
DATA=${DATA:-data/bonn}
CONFIG=configs/bonn.yaml
WIN=windows/bonn
OUT=outputs/bonn
# Thread counts of the paper runs. They fix the rounding of the reductions.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$WIN" "$OUT"
SEQS=$($PYTHON -c "import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))['sequences'])" \
  "$CONFIG")

# Windows step: pi3 on the anchored windows of each video, on the GPU.
for s in $SEQS; do
  if [ ! -f "$WIN/$s.npz" ]; then
    $PYTHON - "$DATA/$s" "$WIN/$s.npz" "$CKPT" <<'EOF'
import sys
import run
frames, _ = run.load_frames("bonn", sys.argv[1], 1, None)
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
        CUDA_VISIBLE_DEVICES= $PYTHON run.py --config "$CONFIG" --seq "$DATA/$s" \
          --windows "$WIN/$s.npz" --seed "$seed" "${flag[@]}" --out "$out"
      fi
    done
  done
done

echo "AnchorRS"
$PYTHON evaluate.py --config "$CONFIG" --data-root "$DATA" --traj "$OUT/{seq}_s{seed}.npz" \
  --seeds 0 1 2
echo "AnchorRS with tau = 0"
$PYTHON evaluate.py --config "$CONFIG" --data-root "$DATA" --traj "$OUT/{seq}_gs_s{seed}.npz" \
  --seeds 0 1 2
