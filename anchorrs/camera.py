"""Pinhole projection of a corrected camera, its Jacobians and DLT triangulation.

A pose correction is the 7-vector d = (w, log s, t): rotation vector w, log scale and
translation t. It maps a camera (R, c) about a reference centre cb to
R' = exp([w]x) R and c' = cb + s exp([w]x) (c - cb) + t.
"""
import cv2
import numpy as np
import torch

DTYPE = torch.float64

cv2.setNumThreads(4)
torch.set_num_threads(4)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrix exp([w]x) of a rotation vector w (radians), Rodrigues formula."""
    th2 = (w * w).sum()
    small = th2 < 1e-8
    t2 = torch.where(small, torch.ones_like(th2), th2)
    th = torch.sqrt(t2)
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / t2)
    z = torch.zeros_like(w[0])
    wx = torch.stack([torch.stack([z, -w[2], w[1]]),
                      torch.stack([w[2], z, -w[0]]),
                      torch.stack([-w[1], w[0], z])])
    return torch.eye(3, dtype=w.dtype) + a * wx + b * (wx @ wx)


def rotation_matrix(w: np.ndarray) -> np.ndarray:
    """exp([w]x) as a float64 numpy array."""
    return so3_exp(torch.as_tensor(w, dtype=DTYPE)).numpy()


def rotation_matrices(ws: np.ndarray) -> np.ndarray:
    """Stack of exp([w]x) for the rows of ws (n, >=3), using the first three entries."""
    return np.stack([rotation_matrix(x[:3]) for x in ws])


def project(X: torch.Tensor, d: torch.Tensor, R: torch.Tensor, c: torch.Tensor,
            cb: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(u, v, depth) of point X in the camera (R, c) corrected by d about cb, Eq. (4)."""
    Rk = so3_exp(d[:3])
    Rp = Rk @ R
    cp = cb + torch.exp(d[3]) * (Rk @ (c - cb)) + d[4:]
    y = Rp.T @ (X - cp)
    return torch.stack([K[0, 0] * y[0] / y[2] + K[0, 2], K[1, 1] * y[1] / y[2] + K[1, 2], y[2]])


project_batch = torch.func.vmap(project)
"""Batched project over the leading axis of all arguments."""

project_jacobians = torch.func.vmap(torch.func.jacrev(project, argnums=(0, 1)))
"""Batched Jacobians of project: d(u, v, depth)/dX (n, 3, 3) and d(u, v, depth)/dd (n, 3, 7)."""


def projection_matrix(K: np.ndarray, R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """3x4 matrix K [R^T | -R^T c] of the camera-to-world pose (R, c)."""
    return K @ np.concatenate([R.T, -R.T @ c[:, None]], 1)


def triangulate(Ps: list[np.ndarray], us: np.ndarray) -> np.ndarray:
    """Multi-view DLT point from 3x4 projection matrices Ps and pixels us (m, 2)."""
    A = np.concatenate([np.stack([u[0] * P[2] - P[0], u[1] * P[2] - P[1]])
                        for P, u in zip(Ps, us)])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]
