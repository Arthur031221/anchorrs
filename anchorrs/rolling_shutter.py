"""Rolling-shutter model of Eq. (5) and readout time from RANSAC consensus, Eq. (6).

Row y of a frame is exposed at the offset dt(y) = tau (y - c_y) / H from the frame time stamp,
with tau the readout time in ms. Angular and linear velocities are central differences of the
synchronized trajectory.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence

import cv2
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from .features import Features, PutativeMatch
from .geometry import rotation_mean

cv2.setNumThreads(4)

READOUT_GRID_TUM = (0.0, 33.34, 4.7619)
READOUT_GRID_TUM_RS = (0.0, 47.62, 4.7619)


class RowModel(NamedTuple):
    """Sensor row of a keypoint for Eq. (5).

    centre_row is c_y (px). height is H (px), or None for the image height. row_map (H x W, float64)
    gives the original sensor row of each pixel of a rectified image (TUM-RS fisheye undistortion),
    read by bilinear interpolation at the raw keypoint.
    """

    centre_row: float
    height: float | None = None
    row_map: np.ndarray | None = None

    def rows(self, raw_xy: np.ndarray) -> np.ndarray:
        """Sensor row (px) of each raw keypoint (n, 2)."""
        if self.row_map is None:
            return raw_xy[:, 1]
        xy = np.asarray(raw_xy, np.float64).reshape(-1, 2)
        return map_coordinates(self.row_map, [xy[:, 1], xy[:, 0]], order=1, mode="nearest",
                               output=np.float64)

    def image_height(self, shape: Sequence[int]) -> float:
        """H (px) for an image of the given shape."""
        return shape[0] if self.height is None else self.height


def frame_velocities(poses: dict, windows: Iterable[int], times: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Angular (rad/s, camera axes) and linear (scene units/s) velocity of every frame.

    poses[k][f] = (R, c) is the synchronized camera-to-world rotation and centre of frame f in
    window k. Per frame, the rotations of all windows are averaged (chordal mean) and the centres
    are averaged, then both are differenced centrally over the time stamps (s), clamped at the
    sequence ends.
    """
    rotations: dict[int, list] = {}
    centres: dict[int, list] = {}
    for k in windows:
        for f, (R, c) in poses[k].items():
            rotations.setdefault(f, []).append(R)
            centres.setdefault(f, []).append(c)
    n = len(times)
    mean_rot = np.stack([rotation_mean(torch.as_tensor(np.stack(rotations[f])),
                                       torch.ones(len(rotations[f]), dtype=torch.float64)).numpy()
                         for f in range(n)])
    mean_ctr = np.stack([np.mean(np.stack(centres[f]), 0) for f in range(n)])
    omega = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    for f in range(n):
        f0, f1 = max(f - 1, 0), min(f + 1, n - 1)
        dt = max(times[f1] - times[f0], 1e-6)
        omega[f] = cv2.Rodrigues(mean_rot[f0].T @ mean_rot[f1])[0].ravel() / dt
    for f in range(n):
        f0, f1 = max(f - 1, 0), min(f + 1, n - 1)
        velocity[f] = (mean_ctr[f1] - mean_ctr[f0]) / max(times[f1] - times[f0], 1e-6)
    return omega, velocity


def rectify(xy: np.ndarray, rows: np.ndarray, omega: np.ndarray, readout_ms: float,
            centre_row: float, height: float, K: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """Rotation part of Eq. (5): rotate the ray of each pixel xy (n, 2) by exp([omega dt(y)]x)."""
    theta = omega[None, :] * (readout_ms / 1000.0 / height * (rows - centre_row))[:, None]
    v = np.c_[xy, np.ones(len(xy))] @ K_inv.T
    angle = np.linalg.norm(theta, axis=1, keepdims=True)
    axis = theta / np.maximum(angle, 1e-12)
    ca, sa = np.cos(angle), np.sin(angle)
    v = v * ca + np.cross(axis, v) * sa + axis * (axis * v).sum(1, keepdims=True) * (1 - ca)
    return (v[:, :2] / v[:, 2:3]) @ K[:2, :2].T + K[:2, 2]


def point_shift(velocity: np.ndarray, rows: np.ndarray, readout_ms: float, centre_row: float,
                height: float | np.ndarray) -> np.ndarray:
    """Translation part of Eq. (5): the shift v_f dt(y) (n, 3) that is subtracted from the point."""
    return velocity * (readout_ms / 1000.0 * (rows - centre_row) / height)[:, None]


def rectify_frame(feat: Features, omega: np.ndarray, readout_ms: float, K: np.ndarray,
                  row_model: RowModel) -> Features:
    """Features of one frame with the rotation part of Eq. (5) applied to undistorted keypoints."""
    if not len(feat.xy):
        return feat
    xy = rectify(feat.xy, row_model.rows(feat.raw_xy), omega, readout_ms, row_model.centre_row,
                 row_model.image_height(feat.shape), K, np.linalg.inv(K))
    return Features(xy, feat.descriptors, feat.raw_xy, feat.shape)


def rectify_features(feats: dict[int, Features], omega: np.ndarray, readout_ms: float,
                     K: np.ndarray, row_model: RowModel) -> dict[int, Features]:
    """rectify_frame for every frame f of feats, with the angular velocity omega[f]."""
    return {f: rectify_frame(feat, omega[f], readout_ms, K, row_model) for f, feat in feats.items()}


def estimate_readout(feats: dict[int, Features], matches: list[PutativeMatch], omega: np.ndarray,
                     K: np.ndarray, row_model: RowModel,
                     grid: tuple[float, float, float] = READOUT_GRID_TUM) -> tuple[float, dict]:
    """Readout time tau (ms) of Eq. (6).

    For each tau on the grid (lo, hi, step), all features are rectified with Eq. (5) and the
    essential-matrix RANSAC inliers are summed over the putative matches of the keyframe pairs
    (pairs with fewer than 15 inliers count zero). The maximum is refined by a parabola when it
    lies inside the grid.
    Returns tau and a dict with the grid, the inlier counts and whether the maximum lies on the
    boundary.
    """
    lo, hi, step = grid
    taus = np.arange(lo, hi + 1e-9, step)
    frames = sorted({f for m in matches for f in (m.frame_a, m.frame_b)})
    K_inv = np.linalg.inv(K)
    counts = []
    for tau in taus:
        xy = {}
        for f in frames:
            feat = feats[f]
            if tau and len(feat.xy):
                xy[f] = rectify(feat.xy, row_model.rows(feat.raw_xy), omega[f], tau,
                                row_model.centre_row, row_model.image_height(feat.shape), K, K_inv)
            else:
                xy[f] = feat.xy
        total = 0
        for m in matches:
            E, mask = cv2.findEssentialMat(xy[m.frame_a][m.idx_a], xy[m.frame_b][m.idx_b], K,
                                           cv2.RANSAC, 0.999, 1.0)
            if E is None or mask is None or mask.sum() < 15:
                continue
            total += int(mask.sum())
        counts.append(total)
    n = np.array(counts, float)
    i = int(np.argmax(n))
    tau_hat = float(taus[i])
    if 0 < i < len(n) - 1:
        den = n[i - 1] - 2 * n[i] + n[i + 1]
        if den < 0:
            tau_hat = float(taus[i] + 0.5 * step * (n[i - 1] - n[i + 1]) / den)
    info = dict(grid=taus.tolist(), inliers=counts, readout_ms=tau_hat,
                boundary=bool(i in (0, len(n) - 1)), pairs_matched=len(matches))
    return tau_hat, info
