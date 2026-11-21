"""Frame lists, ground truth and camera calibration for TUM RGB-D and Bonn RGB-D Dynamic."""
from __future__ import annotations

import bisect
import os

import numpy as np
from scipy.spatial.transform import Rotation

# Calibrated pinhole intrinsics (pixels) published with each dataset.
INTRINSICS = {
    "tum_fr3": np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]]),
    "bonn": np.array([[542.822841, 0.0, 315.593520],
                      [0.0, 542.576870, 237.756098],
                      [0.0, 0.0, 1.0]]),
}
# OpenCV distortion order (k1, k2, p1, p2, k3). TUM fr3 images are already undistorted.
DISTORTION = {
    "tum_fr3": None,
    "bonn": np.array([0.039903, -0.099343, -0.000730, -0.000144, 0.0]),
}


def read_tum_list(path: str) -> tuple[list[float], list[list[str]]]:
    """Read a TUM-format text file (rgb.txt, groundtruth.txt) into time stamps and rows."""
    stamps, rows = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            stamps.append(float(parts[0]))
            rows.append(parts[1:])
    return stamps, rows


def associate(ts_a: list[float], ts_b: list[float], max_diff: float = 0.02
              ) -> list[tuple[int, int]]:
    """Greedy nearest time stamp association of the TUM benchmark tools (seconds)."""
    order_b = sorted(range(len(ts_b)), key=lambda i: ts_b[i])
    tb = [ts_b[i] for i in order_b]
    cand = []
    for ia, a in enumerate(ts_a):
        lo = bisect.bisect_left(tb, a - max_diff)
        hi = bisect.bisect_right(tb, a + max_diff)
        for k in range(lo, hi):
            d = abs(a - tb[k])
            if d < max_diff:
                cand.append((d, a, tb[k], ia, order_b[k]))
    cand.sort(key=lambda t: (t[0], t[1], t[2]))
    used_a, used_b, out = set(), set(), []
    for _, a, _, ia, ib in cand:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        out.append((a, ia, ib))
    out.sort()
    return [(ia, ib) for _, ia, ib in out]


def camera_of(seq_dir: str) -> str:
    """Calibration key of a sequence folder."""
    return "bonn" if "rgbd_bonn" in os.path.basename(os.path.normpath(seq_dir)) else "tum_fr3"


def load_sequence(seq_dir: str, max_diff: float = 0.02) -> dict:
    """Frames of a TUM or Bonn sequence that have a ground-truth pose within max_diff seconds.

    Returns a dict with the image paths, their time stamps (s), the matching ground-truth
    camera-to-world matrices (metres) and the calibration key.
    """
    ts_rgb, rows_rgb = read_tum_list(os.path.join(seq_dir, "rgb.txt"))
    ts_gt, rows_gt = read_tum_list(os.path.join(seq_dir, "groundtruth.txt"))
    pairs = associate(ts_rgb, ts_gt, max_diff)
    frames = [os.path.join(seq_dir, rows_rgb[i][0]) for i, _ in pairs]
    gt = np.array([[float(x) for x in rows_gt[j][:7]] for _, j in pairs])
    return dict(
        frames=frames,
        stamps=np.array([ts_rgb[i] for i, _ in pairs]),
        gt_c2w=tum_to_c2w(gt),
        camera=camera_of(seq_dir),
    )


def tum_to_c2w(rows: np.ndarray) -> np.ndarray:
    """Rows (tx ty tz qx qy qz qw) to (N, 4, 4) camera-to-world matrices."""
    out = np.tile(np.eye(4), (len(rows), 1, 1))
    out[:, :3, 3] = rows[:, :3]
    out[:, :3, :3] = Rotation.from_quat(rows[:, 3:7]).as_matrix()
    return out


def frame_time(path: str) -> float:
    """Time stamp (s) encoded in a TUM or Bonn image file name."""
    return float(os.path.splitext(os.path.basename(path))[0])
