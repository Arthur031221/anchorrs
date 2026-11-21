"""Reconstruct the camera trajectory of one TUM RGB-D, Bonn or TUM rolling-shutter sequence.

Example:
    python run.py --config configs/tum.yaml --seq data/tum/rgbd_dataset_freiburg3_walking_static \
        --out outputs/tum/walking_static.npz

The pi3 windows are computed on the GPU and cached next to the output (or read from
--windows). The solver then runs on the CPU.
"""
import os

# Thread counts fix the float rounding of the reductions, so results repeat exactly.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from anchorrs import data, tum_rs  # noqa: E402
from anchorrs.pipeline import Camera, SolverConfig, reconstruct  # noqa: E402
from anchorrs.rolling_shutter import RowModel  # noqa: E402


def _plain(value):
    """JSON form of numpy values in the solver statistics."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def load_frames(dataset: str, seq: str, cam: int, prepared: str | None) -> tuple[list, Camera]:
    """Image paths in video order and the camera model of a sequence."""
    if dataset == "tum_rs":
        out_dir = prepared or os.path.join(seq, f"pinhole_cam{cam}")
        prep = tum_rs.prepare(seq, cam, out_dir)
        cal = prep["calib"]
        rows = RowModel(cal["principal_row"], cal["rows_total"],
                        cal["row_map"].astype(np.float64))
        return prep["frames"], Camera(cal["K"], None, rows)
    sequence = data.load_sequence(seq)
    key = sequence["camera"]
    return sequence["frames"], Camera(data.INTRINSICS[key], data.DISTORTION[key])


def compute_windows(frames: list[str], path: str, checkpoint: str) -> None:
    """Run pi3 on the anchored windows of the video and save them to path."""
    from anchorrs.windows import Pi3Frontend, infer_windows

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    infer_windows(frames, path, Pi3Frontend(checkpoint))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="configs/{tum,bonn,tum_rs}.yaml")
    ap.add_argument("--seq", required=True, help="sequence folder of the official dataset")
    ap.add_argument("--out", required=True, help="output trajectory (.npz)")
    ap.add_argument("--windows", default=None, help="pi3 windows (.npz); computed if missing")
    ap.add_argument("--checkpoint", default="yyfz233/Pi3", help="pi3 weights (path or hub id)")
    ap.add_argument("--cam", type=int, default=1, help="TUM rolling-shutter camera (0 or 1)")
    ap.add_argument("--prepared", default=None, help="folder for the undistorted TUM-RS images")
    ap.add_argument("--seed", type=int, default=None, help="match sampling seed (paper: 0, 1, 2)")
    ap.add_argument("--global-shutter", action="store_true",
                    help="ablation without the rolling-shutter model (tau = 0)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_all = yaml.safe_load(f)
    cfg = SolverConfig.from_dict(cfg_all.get("solver", {}))
    if args.seed is not None:
        cfg.seed = args.seed
    if args.global_shutter:
        cfg.readout_grid_ms = None
    frames, camera = load_frames(cfg_all["dataset"], args.seq, args.cam, args.prepared)
    windows = args.windows or os.path.splitext(args.out)[0] + "_windows.npz"
    if not os.path.exists(windows):
        compute_windows(frames, windows, args.checkpoint)

    rec = reconstruct(windows, frames, camera, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, frames=np.array(rec.frames), c2w=rec.c2w,
                        readout_ms=rec.readout_ms, seed=cfg.seed,
                        global_shutter=cfg.readout_grid_ms is None,
                        stats=json.dumps(rec.stats, default=_plain))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
