"""Priors on the pose corrections of Eq. (4): the smoothness prior across block boundaries and the
two-level prior that ties keyframe corrections to their block correction.

Corrections are 7-vectors (w, log s, t), see camera.py. Node q is the keyframe (window, frame)
nodes[q]; poses[q] = (R camera-to-world, centre).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

CHI2_3_MEDIAN = 2.3660


def block_motion_jacobian(y: np.ndarray) -> np.ndarray:
    """First-order motion (n, 3, 7) of points y = x - c under a correction (w, log s, t) about c,
    that is w x y + (log s) y + t."""
    A = np.zeros((len(y), 3, 7))
    A[:, 0, 1], A[:, 0, 2], A[:, 1, 0] = y[:, 2], -y[:, 1], -y[:, 2]
    A[:, 1, 2], A[:, 2, 0], A[:, 2, 1] = y[:, 0], y[:, 1], -y[:, 0]
    A[:, :, 3] = y
    A[:, :, 4:] = np.eye(3)
    return A


def _left_jacobian_inv(p: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian J^-1 of SO(3) at the rotation vector p.

    log(exp(e) exp(p)) = p + J^-1 e + O(e^2).
    """
    p = np.asarray(p, float)
    th = float(np.linalg.norm(p))
    K = np.array([[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]])
    c = 1.0 / 12.0 if th < 1e-6 else 1.0 / th ** 2 - (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))
    return np.eye(3) - 0.5 * K + c * (K @ K)


@dataclass
class SmoothnessPrior:
    """Quadratic prior 1/2 d^T H d + g^T d + const, linearised at the synchronized poses.

    H1, g1 act on the block corrections (7 per block), H2, g2 on the keyframe corrections
    (7 per node). const is the same for both.
    """

    H1: sp.csr_matrix
    g1: np.ndarray
    H2: sp.csr_matrix
    g2: np.ndarray
    const: float
    num_boundaries: int


def smoothness_prior(nodes: list[tuple[int, int]], poses: dict, stamps: np.ndarray,
                     block_of: dict, centres: dict, num_blocks: int,
                     weight: float = 1.0) -> SmoothnessPrior:
    """White-noise-on-acceleration prior on the keyframe triples that straddle each boundary
    between time-consecutive blocks.

    The finite-difference accelerations a_c (centres) and a_r (rotations) of a triple have
    variance sigma^2 nu with nu = (dt12 + dt23) / (2 dbar); sigma^2 is the median of |a|^2 / nu over
    the triples inside the blocks divided by the chi^2_3 median. stamps[q] are node times (s).
    """
    tq = stamps
    nq = len(nodes)
    by_window: dict = {}
    for q, (kk, f) in enumerate(nodes):
        by_window.setdefault(kk, []).append(q)
    for kk in by_window:
        by_window[kk].sort(key=lambda q: (tq[q], q))
    order = sorted(by_window, key=lambda kk: (tq[by_window[kk][0]], block_of[kk]))
    R0 = np.stack([np.asarray(poses[q][0], float) for q in range(nq)])
    C0 = np.stack([np.asarray(poses[q][1], float) for q in range(nq)])
    inner = [tuple(Lq[i:i + 3]) for Lq in by_window.values() for i in range(len(Lq) - 2)
             if tq[Lq[i]] < tq[Lq[i + 1]] < tq[Lq[i + 2]]]
    gaps = np.array([tq[Lq[i + 1]] - tq[Lq[i]] for Lq in by_window.values()
                     for i in range(len(Lq) - 1) if tq[Lq[i + 1]] > tq[Lq[i]]])
    dbar = float(np.median(gaps))
    straddle = []
    for ka, kb in zip(order[:-1], order[1:]):
        La = [q for q in by_window[ka] if tq[q] < tq[by_window[kb][0]]]
        Lb = by_window[kb]
        if len(La) >= 2:
            straddle.append((La[-2], La[-1], Lb[0]))
        if len(La) >= 1 and len(Lb) >= 2 and tq[Lb[1]] > tq[Lb[0]]:
            straddle.append((La[-1], Lb[0], Lb[1]))

    def nu(T):
        return (tq[T[2]] - tq[T[0]]) / (2.0 * dbar)

    def acc(T, R, C):
        q1, q2, q3 = T
        d12, d23 = tq[q2] - tq[q1], tq[q3] - tq[q2]
        ac = (C[q3] - C[q2]) / d23 - (C[q2] - C[q1]) / d12
        ar = (Rotation.from_matrix(R[q3] @ R[q2].T).as_rotvec() / d23
              - Rotation.from_matrix(R[q2] @ R[q1].T).as_rotvec() / d12)
        return ac, ar

    inner_acc = [acc(T, R0, C0) for T in inner]
    s2c = float(np.median([np.sum(x[0] ** 2) / nu(T) for x, T in zip(inner_acc, inner)])
                / CHI2_3_MEDIAN)
    s2r = float(np.median([np.sum(x[1] ** 2) / nu(T) for x, T in zip(inner_acc, inner)])
                / CHI2_3_MEDIAN)
    m = len(straddle)
    a0 = np.zeros(6 * m)
    w = np.zeros(6 * m)
    r1, c1, v1, r2, c2, v2 = [], [], [], [], [], []
    for t, T in enumerate(straddle):
        ac, ar = acc(T, R0, C0)
        a0[6 * t:6 * t + 3] = ac
        a0[6 * t + 3:6 * t + 6] = ar
        w[6 * t:6 * t + 3] = weight / (s2c * nu(T))
        w[6 * t + 3:6 * t + 6] = weight / (s2r * nu(T))
        d12, d23 = tq[T[1]] - tq[T[0]], tq[T[2]] - tq[T[1]]
        coef = (1.0 / d12, -1.0 / d12 - 1.0 / d23, 1.0 / d23)
        P12 = R0[T[1]] @ R0[T[0]].T
        P23 = R0[T[2]] @ R0[T[1]].T
        L12 = _left_jacobian_inv(Rotation.from_matrix(P12).as_rotvec())
        L23 = _left_jacobian_inv(Rotation.from_matrix(P23).as_rotvec())
        # exact first-order derivative of a_r with respect to left perturbations of the 3 rotations
        drot = (L12 @ P12 / d12, -L23 @ P23 / d23 - L12 / d12, L23 / d23)
        for q, cf, B in zip(T, coef, drot):
            kb_ = block_of[nodes[q][0]]
            A = block_motion_jacobian((C0[q] - centres[nodes[q][0]])[None])[0]
            for i in range(3):
                for j in range(7):
                    if A[i, j] != 0:
                        r1.append(6 * t + i)
                        c1.append(7 * kb_ + j)
                        v1.append(cf * A[i, j])
                r2.append(6 * t + i)
                c2.append(7 * q + 4 + i)
                v2.append(cf)
                for j in range(3):
                    r1.append(6 * t + 3 + i)
                    c1.append(7 * kb_ + j)
                    v1.append(B[i, j])
                    r2.append(6 * t + 3 + i)
                    c2.append(7 * q + j)
                    v2.append(B[i, j])
    J1 = sp.csr_matrix((v1, (r1, c1)), shape=(6 * m, 7 * num_blocks))
    J2 = sp.csr_matrix((v2, (r2, c2)), shape=(6 * m, 7 * nq))
    Wd = sp.diags(w)
    return SmoothnessPrior(
        H1=(J1.T @ Wd @ J1).tocsr(), g1=np.asarray(J1.T @ (w * a0)).ravel(),
        H2=(J2.T @ Wd @ J2).tocsr(), g2=np.asarray(J2.T @ (w * a0)).ravel(),
        const=float(0.5 * a0 @ (w * a0)), num_boundaries=m)


def two_level_prior(nodes: list[tuple[int, int]], centres: dict, poses: dict,
                    sigma_block: np.ndarray, sigma_key: np.ndarray,
                    block_factor: float = 1.0) -> sp.csr_matrix:
    """Information matrix (7 per node) of the keyframe corrections d_q = A_q D_k + r_q with
    D_k ~ N(0, (block_factor sigma_block)^2) and r_q ~ N(0, sigma_key^2), D_k marginalised:
    H_k = S_r^-1 - S_r^-1 A (S_D^-1 + A^T S_r^-1 A)^-1 A^T S_r^-1, one dense block per window."""
    iSr = 1.0 / np.asarray(sigma_key, float) ** 2
    iSD = 1.0 / (block_factor * np.asarray(sigma_block, float)) ** 2
    by_window: dict = {}
    rows, cols, vals = [], [], []
    for q, (kk, f) in enumerate(nodes):
        by_window.setdefault(kk, []).append(q)
    for kk, qs in by_window.items():
        y = np.stack([np.asarray(poses[q][1], float) - centres[kk] for q in qs])
        m = len(qs)
        A = np.zeros((m, 7, 7))
        A[:, 0:3, 0:3] = np.eye(3)
        A[:, 4:7, :] = block_motion_jacobian(y)
        A = A.reshape(7 * m, 7)
        iS = np.tile(iSr, m)
        SA = iS[:, None] * A
        Hk = np.diag(iS) - SA @ np.linalg.solve(np.diag(iSD) + A.T @ SA, SA.T)
        idx = (7 * np.asarray(qs)[:, None] + np.arange(7)).ravel()
        rows.append(np.repeat(idx, 7 * m))
        cols.append(np.tile(idx, 7 * m))
        vals.append(Hk.ravel())
    n = 7 * len(nodes)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))
