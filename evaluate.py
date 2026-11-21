"""Absolute trajectory error (cm) of saved trajectories after Sim(3) alignment.

One trajectory:
    python evaluate.py --config configs/tum.yaml --seq data/tum/<sequence> --traj out.npz
A paper table (mean over the sequences of the config, then over the seeds):
    python evaluate.py --config configs/tum.yaml --data-root data/tum \
        --traj "outputs/tum/{seq}_s{seed}.npz" --seeds 0 1 2
"""
import argparse
import os

import numpy as np
import yaml

from anchorrs import data, tum_rs
from anchorrs.metrics import ate


def ground_truth(dataset: str, seq: str, frames: list[str], cam: int
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """Ground-truth camera-to-world poses (m) of the trajectory frames and the usable mask.

    TUM-RS frames are compared at the time of the principal row. A frame counts only if the
    ground truth exists both at its stamp and at that time.
    """
    if dataset == "tum_rs":
        stamps, _ = tum_rs.frame_stamps(seq, cam)
        names = [tum_rs.stamp_name(t) + ".png" for t in stamps]
        if names != frames:
            raise ValueError(f"{seq}: trajectory frames differ from the dataset frames")
        _, valid_stamp = tum_rs.ground_truth(seq, cam, stamps, 0.0)
        gt, valid = tum_rs.ground_truth(seq, cam, stamps, tum_rs.evaluation_offset_ms(seq, cam))
        return gt, valid & valid_stamp
    sequence = data.load_sequence(seq)
    if [os.path.basename(f) for f in sequence["frames"]] != frames:
        raise ValueError(f"{seq}: trajectory frames differ from the dataset frames")
    return sequence["gt_c2w"], None


def trajectory_ate_cm(dataset: str, seq: str, traj: str, cam: int) -> float:
    """ATE (cm) of one saved trajectory."""
    z = np.load(traj, allow_pickle=False)
    frames = [str(x) for x in z["frames"]]
    gt, mask = ground_truth(dataset, seq, frames, cam)
    return 100.0 * ate(z["c2w"], gt, mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--traj", required=True,
                    help="trajectory .npz, or a pattern with {seq} and {seed} for a table")
    ap.add_argument("--seq", default=None, help="sequence folder (one trajectory)")
    ap.add_argument("--data-root", default=None, help="folder holding the config's sequences")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cam", type=int, default=1, help="TUM rolling-shutter camera")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["dataset"]

    if args.seq is not None:
        print(f"ATE {trajectory_ate_cm(dataset, args.seq, args.traj, args.cam):.6f} cm")
        return
    if args.data_root is None:
        ap.error("give --seq for one trajectory or --data-root for a table")
    seed_means = []
    for seed in args.seeds:
        values = []
        for name in cfg["sequences"]:
            path = args.traj.format(seq=name, seed=seed)
            value = trajectory_ate_cm(dataset, os.path.join(args.data_root, name), path, args.cam)
            values.append(value)
            print(f"seed {seed}  {name:45s} {value:9.4f} cm")
        seed_means.append(float(np.mean(values)))
        print(f"seed {seed}  mean {seed_means[-1]:.6f} cm")
    print(f"mean over {len(args.seeds)} seeds: {np.mean(seed_means):.6f} cm")


if __name__ == "__main__":
    main()
