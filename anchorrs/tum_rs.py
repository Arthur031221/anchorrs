"""TUM rolling-shutter dataset: pinhole undistortion, row map and ground truth (Sec. 4).

Each sequence folder (dataset-seqN, EuRoC export) holds mav0/cam<C>/data.csv with nanosecond
stamps, 16-bit fisheye images in mav0/cam<C>/data/, and dso/camchain.yaml plus dso/gt_imu.csv.
cam0 has a global shutter and cam1 a rolling shutter.
"""
from __future__ import annotations

import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

WIDTH, HEIGHT, FOCAL = 640, 512, 320.0
# Readout time of the cam1 sensor (29.47 us per row times 1024 rows).
# It is used only to time the ground truth.
READOUT_MS = 30.17728
MAX_MOCAP_GAP_NS = 25_000_000


def pinhole_K(focal: float = FOCAL) -> np.ndarray:
    """Intrinsics of the undistorted 640x512 images (90 degree horizontal field of view)."""
    return np.array([[focal, 0.0, (WIDTH - 1) / 2.0], [0.0, focal, (HEIGHT - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def read_camchain(seq_dir: str, cam: int) -> dict:
    """Kalibr calibration of one camera: K, equidistant distortion, resolution, T_cam_imu."""
    with open(os.path.join(seq_dir, "dso", "camchain.yaml")) as fh:
        cc = yaml.safe_load(fh)[f"cam{cam}"]
    fu, fv, pu, pv = cc["intrinsics"]
    return dict(K=np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]]),
                D=np.array(cc["distortion_coeffs"], np.float64).reshape(4, 1),
                size=tuple(cc["resolution"]), T_cam_imu=np.array(cc["T_cam_imu"], np.float64))


def frame_stamps(seq_dir: str, cam: int) -> tuple[np.ndarray, list[str]]:
    """Frame time stamps (int64 ns) and raw image names from mav0/cam<C>/data.csv."""
    rows = []
    with open(os.path.join(seq_dir, "mav0", f"cam{cam}", "data.csv")) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                rows.append(line.strip().split(","))
    return np.array([int(r[0]) for r in rows], np.int64), [r[1] for r in rows]


def stamp_name(ns: int) -> str:
    """File stem in seconds with nine decimals, so float(stem) is the time stamp in seconds."""
    s = str(int(ns))
    return s[:-9] + "." + s[-9:]


def prepare(seq_dir: str, cam: int, out_dir: str) -> dict:
    """Undistort every frame to the 640x512 pinhole camera and save the row map.

    The row map holds, for every pinhole pixel, the row of the original 1024-row sensor it
    comes from. The rolling-shutter model uses it to time each keypoint. Existing outputs
    are reused. Returns frames, stamps (ns) and the calibration dict of the pipeline.
    """
    import cv2

    cal = read_camchain(seq_dir, cam)
    stamps, names = frame_stamps(seq_dir, cam)
    frames = [os.path.join(out_dir, stamp_name(t) + ".png") for t in stamps]
    row_path = os.path.join(out_dir, "row_map.npy")
    if not (os.path.exists(row_path) and all(os.path.exists(f) for f in frames)):
        os.makedirs(out_dir, exist_ok=True)
        mx, my = cv2.fisheye.initUndistortRectifyMap(cal["K"], cal["D"], np.eye(3), pinhole_K(),
                                                     (WIDTH, HEIGHT), cv2.CV_32FC1)
        mx, my = mx.astype(np.float32), my.astype(np.float32)
        divisor = None
        for name, path in zip(names, frames):
            raw = cv2.imread(os.path.join(seq_dir, "mav0", f"cam{cam}", "data", name),
                             cv2.IMREAD_UNCHANGED)
            if divisor is None:
                divisor = 256.0 if int((raw & 255).max()) == 0 else 257.0
            img = cv2.remap(raw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0).astype(np.float64)
            img = np.clip(np.round(img / divisor), 0, 255).astype(np.uint8)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        np.save(row_path, my)
    calib = dict(K=pinhole_K(), dist=None, row_map=np.load(row_path),
                 rows_total=float(cal["size"][1]), principal_row=float(cal["K"][1, 2]))
    return dict(frames=frames, stamps=stamps, calib=calib)


def _mocap(seq_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Motion-capture poses of the IMU frame: stamps (ns), positions (m), quaternions (xyzw)."""
    rows = []
    with open(os.path.join(seq_dir, "dso", "gt_imu.csv")) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                rows.append(line.strip().split(","))
    ns = np.array([int(r[0]) for r in rows], np.int64)
    v = np.array([[float(x) for x in r[1:]] for r in rows])
    return ns, v[:, :3], v[:, [4, 5, 6, 3]]


def ground_truth(seq_dir: str, cam: int, stamps: np.ndarray, offset_ms: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world ground truth (m) at stamps + offset_ms and the mask of usable frames.

    T_w_cam(t) = T_w_imu(t) inv(T_cam_imu), with linear interpolation of the position and
    slerp of the rotation. A frame is usable if t lies inside the motion-capture span and the
    two samples around t are at most 25 ms apart.
    """
    ns, pos, quat = _mocap(seq_dir)
    t0 = int(stamps[0])
    tg = (ns - t0).astype(np.float64)
    tq = (stamps - t0).astype(np.float64) + float(offset_ms) * 1e6
    inside = (tq >= tg[0]) & (tq <= tg[-1])
    lo = np.clip(np.searchsorted(tg, tq, side="right") - 1, 0, len(tg) - 1)
    hi = np.clip(np.searchsorted(tg, tq, side="left"), 0, len(tg) - 1)
    gap = np.where(inside, (ns[hi] - ns[lo]).astype(np.float64), np.inf)
    valid = inside & (gap <= MAX_MOCAP_GAP_NS)
    T = np.full((len(tq), 4, 4), np.nan)
    tg_s, tq_s = tg * 1e-9, tq * 1e-9
    T[inside, :3, :3] = Slerp(tg_s, Rotation.from_quat(quat))(tq_s[inside]).as_matrix()
    T[inside, :3, 3] = np.stack([np.interp(tq_s[inside], tg_s, pos[:, k]) for k in range(3)], 1)
    T[inside, 3] = [0, 0, 0, 1]
    return T @ np.linalg.inv(read_camchain(seq_dir, cam)["T_cam_imu"]), valid


def evaluation_offset_ms(seq_dir: str, cam: int) -> float:
    """Time of the principal row after the frame stamp (ms): zero for the global-shutter cam0."""
    if cam == 0:
        return 0.0
    cal = read_camchain(seq_dir, cam)
    return READOUT_MS * float(cal["K"][1, 2]) / float(cal["size"][1])
