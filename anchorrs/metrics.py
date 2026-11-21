"""Absolute trajectory error after Sim(3) alignment (Umeyama), as in the evo ape translation
metric with alignment and scale correction."""
import numpy as np


def umeyama(x: np.ndarray, y: np.ndarray, with_scale: bool = True
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation r, translation t and scale c with y ~ c r x + t for point sets x, y (3, n)."""
    m, n = x.shape
    mx, my = x.mean(1), y.mean(1)
    xc, yc = x - mx[:, None], y - my[:, None]
    sx = (xc ** 2).sum() / n
    cov = yc @ xc.T / n
    u, d, vt = np.linalg.svd(cov)
    if np.linalg.matrix_rank(cov) < m - 1:
        raise ValueError("degenerate covariance rank, Umeyama alignment is not possible")
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[m - 1, m - 1] = -1
    r = u @ s @ vt
    c = (d * np.diag(s)).sum() / sx if with_scale else 1.0
    t = my - c * r @ mx
    return r, t, c


def align_sim3(est: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Camera-to-world poses est (N, 4, 4) aligned onto ref by the Sim(3) fit of their centres."""
    r, t, c = umeyama(est[:, :3, 3].T, ref[:, :3, 3].T, True)
    out = est.copy()
    out[:, :3, 3] = est[:, :3, 3] * c
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T @ out, c


def ate(c2w_est: np.ndarray, c2w_gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMSE (in the ground-truth unit, metres) of the camera centres after Sim(3) alignment.

    Both inputs are (N, 4, 4) camera-to-world in the same frame order. With mask, only the
    selected frames enter the alignment and the error.
    """
    P = np.asarray(c2w_est, np.float64)
    Q = np.asarray(c2w_gt, np.float64)
    if mask is not None:
        P, Q = P[mask], Q[mask]
    assert P.shape == Q.shape and P.ndim == 3 and P.shape[1:] == (4, 4), (P.shape, Q.shape)
    Pa, _ = align_sim3(P, Q)
    return float(np.sqrt(((Pa[:, :3, 3] - Q[:, :3, 3]) ** 2).sum(1).mean()))
