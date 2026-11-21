"""Bundle adjustment of Eq. (4): Levenberg-Marquardt with the Schur complement on the points,
solved by continuation over block corrections and then over all keyframe corrections."""
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.spatial.transform import Rotation

from .camera import (DTYPE, project_batch, project_jacobians, projection_matrix,
                     rotation_matrices, triangulate)
from .priors import SmoothnessPrior, two_level_prior
from .rolling_shutter import RowModel, point_shift


@dataclass
class KeyframeGraph:
    """Keyframes of the synchronized windows.

    poses[k][f] = (R camera-to-world, centre) of core frame f in window k after synchronization,
    centres[k] the mean core centre of window k, zbar the median scene depth (scene units).
    Node q is (window, frame) = nodes[q]; block_of[k] is the block index of window k.
    """

    windows: list
    keyframes: dict
    poses: dict
    centres: dict
    zbar: float
    K: np.ndarray
    nodes: list = field(init=False)
    node_index: dict = field(init=False)
    block_of: dict = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = [(kk, f) for kk in self.windows for f in self.keyframes[kk]]
        self.node_index = {nd: i for i, nd in enumerate(self.nodes)}
        self.block_of = {kk: i for i, kk in enumerate(self.windows)}

    @property
    def num_blocks(self) -> int:
        return len(self.windows)

    def node_poses(self) -> dict:
        """Synchronized pose of every node."""
        return {q: self.poses[kk][f] for q, (kk, f) in enumerate(self.nodes)}

    def block_corrected_poses(self, dk: np.ndarray) -> dict:
        """Node poses after the block corrections dk (num_blocks, 7)."""
        Rk = rotation_matrices(dk)
        s = np.exp(dk[:, 3])
        tau = dk[:, 4:]
        out = {}
        for q, (kk, f) in enumerate(self.nodes):
            b, (R, c), cb = self.block_of[kk], self.poses[kk][f], self.centres[kk]
            out[q] = (Rk[b] @ R, cb + s[b] * Rk[b] @ (c - cb) + tau[b])
        return out


@dataclass
class ObservationSet:
    """Triangulated track points X0 and their observations: point id, node id, pixel U (n, 2)
    after undistortion and rotation rectification, global feature id E."""

    X0: np.ndarray
    point: np.ndarray
    node: np.ndarray
    U: np.ndarray
    feature: np.ndarray


@dataclass
class Observations:
    """Per-observation inputs of bundle_adjust: correction index k, reference camera (R, c), centre
    cb of the correction, intrinsics K, pixel u and the rolling-shutter point shift (or None)."""

    point: np.ndarray
    k: np.ndarray
    R: np.ndarray
    c: np.ndarray
    cb: np.ndarray
    K: np.ndarray
    u: np.ndarray
    node: np.ndarray
    shift: np.ndarray | None = None


def observation_set(tracks: list, node_of: np.ndarray, kp_of: np.ndarray, features: dict,
                    graph: KeyframeGraph, poses: dict, init_px: float = 40.0) -> ObservationSet:
    """DLT points of cross-window tracks under the node poses. An observation is kept if its depth
    exceeds 0.05 zbar and its reprojection error is below init_px; a point needs 3 kept
    observations in at least 2 windows."""
    K, nodes, zbar = graph.K, graph.nodes, graph.zbar
    Pm = {}
    X0, obs = [], []
    for grp in tracks:
        q = node_of[grp]
        us = np.stack([features[nodes[x][1]][0][kp_of[e]] for x, e in zip(q, grp)])
        for x in q:
            if x not in Pm:
                Pm[x] = projection_matrix(K, *poses[x])
        X = triangulate([Pm[x] for x in q], us)
        keep = []
        for j, (x, uu) in enumerate(zip(q, us)):
            R, c = poses[x]
            y = R.T @ (X - c)
            err = np.linalg.norm(K[:2, :2] @ (y[:2] / y[2]) + K[:2, 2] - uu)
            if y[2] > 0.05 * zbar and err < init_px:
                keep.append(j)
        if len(keep) < 3 or len({nodes[q[j]][0] for j in keep}) < 2:
            continue
        pid = len(X0)
        for j in keep:
            obs.append((pid, int(q[j]), us[j], int(grp[j])))
        X0.append(X)
    return ObservationSet(
        X0=np.array(X0).reshape(-1, 3), point=np.array([o[0] for o in obs], dtype=int),
        node=np.array([o[1] for o in obs], dtype=int),
        U=np.array([o[2] for o in obs]).reshape(-1, 2),
        feature=np.array([o[3] for o in obs], dtype=int))


def block_observations(S: ObservationSet, graph: KeyframeGraph, poses: dict,
                       shift: np.ndarray | None) -> Observations:
    """Observations of the first continuation stage: one correction per block about its centre."""
    q = S.node
    nodes = graph.nodes
    R = np.array([poses[x][0] for x in q]).reshape(-1, 3, 3)
    c = np.array([poses[x][1] for x in q]).reshape(-1, 3)
    k = np.array([graph.block_of[nodes[x][0]] for x in q], dtype=int)
    cb = np.array([graph.centres[nodes[x][0]] for x in q]).reshape(-1, 3)
    return Observations(point=S.point, k=k, R=R, c=c, cb=cb, K=np.repeat(graph.K[None], len(q), 0),
                        u=S.U, node=q, shift=shift if len(q) else None)


def observation_shift(S: ObservationSet, graph: KeyframeGraph, kp_of: np.ndarray, features: dict,
                      velocity: np.ndarray, readout_ms: float, row_model: RowModel) -> np.ndarray:
    """Translation part of Eq. (5) for every observation: v_f dt(y) at the sensor row of its raw
    keypoint (n, 3). The point of an observation is X - shift."""
    fq = np.array([graph.nodes[x][1] for x in S.node])
    kq = kp_of[S.feature]
    rows = row_model.rows(np.array([features[f].raw_xy[k] for f, k in zip(fq, kq)]).reshape(-1, 2))
    H = np.array([row_model.image_height(features[f].shape) for f in fq], float)
    return point_shift(velocity[fq], rows, readout_ms, row_model.centre_row, H)


def bundle_adjust(X0: np.ndarray, d0: np.ndarray, ob: Observations, prior_info: sp.spmatrix,
                  prior_grad: np.ndarray, prior_const: float, zmin: float, cauchy_px: float = 2.0,
                  iters: int = 40, cap_px: float = 50.0, dense_max: int = 4200
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimise Eq. (4) over points X (P, 3) and corrections d (nb, 7).

    Pixel residuals have a Cauchy loss of scale cauchy_px (px), an observation at depth <= zmin
    costs as a cap_px residual, and the corrections have the quadratic prior
    1/2 d^T prior_info d + prior_grad^T d + prior_const. The reduced camera system is solved
    densely up to dense_max unknowns. Returns X, d and the cost before and after.
    """
    p, k = ob.point, ob.k
    P, nb = len(X0), len(d0)
    n = len(p)
    fs, Hp, gl = cauchy_px, prior_info, prior_grad

    def T(a):
        return torch.as_tensor(a, dtype=DTYPE)

    Rj, cj, cb, Kj, u = T(ob.R), T(ob.c), T(ob.cb), T(ob.K), ob.u
    shift = ob.shift
    invalid_cost = 0.5 * fs ** 2 * np.log1p(cap_px ** 2 / fs ** 2)

    def shifted(X):
        return X[p] - shift if shift is not None else X[p]

    def cost_of(X, d):
        y = project_batch(T(shifted(X)), T(d[k]), Rj, cj, cb, Kj).numpy()
        ok = y[:, 2] > zmin
        e2 = ((y[:, :2] - u) ** 2).sum(1)
        c = (0.5 * fs ** 2 * np.log1p(e2[ok] / fs ** 2).sum() + invalid_cost * (~ok).sum()
             + 0.5 * d.ravel() @ Hp @ d.ravel())
        c += float(gl @ d.ravel()) + prior_const
        return c, e2, ok

    X, d = X0.copy(), d0.copy()
    c0, e2, ok = cost_of(X, d)
    lam = 1e-4
    hist = [c0]
    rows = np.repeat(np.arange(3 * P).reshape(P, 3), 3, 1).reshape(-1)
    cols = np.tile(np.arange(3 * P).reshape(P, 1, 3), (1, 3, 1)).reshape(-1)
    bi = (7 * np.arange(nb)[:, None, None] + np.arange(7)[None, :, None]).repeat(7, 2).reshape(-1)
    bj = (7 * np.arange(nb)[:, None, None] + np.arange(7)[None, None, :]).repeat(7, 1).reshape(-1)
    for it in range(iters):
        Xp, dp = T(shifted(X)), T(d[k])
        y = project_batch(Xp, dp, Rj, cj, cb, Kj).numpy()
        JX, Jd = project_jacobians(Xp, dp, Rj, cj, cb, Kj)
        JX = JX.numpy().copy()
        Jd = Jd.numpy().copy()
        r = np.zeros((n, 3))
        r[:, :2] = y[:, :2] - u
        e2 = (r[:, :2] ** 2).sum(1)
        okm = y[:, 2] > zmin
        Wt = np.zeros((n, 3))
        Wt[:, 0] = Wt[:, 1] = np.where(okm, 1.0 / (1.0 + e2 / fs ** 2), 0.0)
        JXw = JX * Wt[:, :, None]
        Jdw = Jd * Wt[:, :, None]
        Hpp = np.zeros((P, 3, 3))
        np.add.at(Hpp, p, np.einsum("nai,naj->nij", JXw, JX))
        gp = np.zeros((P, 3))
        np.add.at(gp, p, np.einsum("nai,na->ni", JXw, r))
        Hdd = np.zeros((nb, 7, 7))
        np.add.at(Hdd, k, np.einsum("nai,naj->nij", Jdw, Jd))
        gd = np.zeros((nb, 7))
        np.add.at(gd, k, np.einsum("nai,na->ni", Jdw, r))
        gd += (Hp @ d.ravel()).reshape(nb, 7)
        gd += gl.reshape(nb, 7)
        Hpd_o = np.einsum("nai,naj->nij", JXw, Jd)
        pk = p * nb + k
        upk, inv = np.unique(pk, return_inverse=True)
        Hpd = np.zeros((len(upk), 3, 7))
        np.add.at(Hpd, inv, Hpd_o)
        up, uk = upk // nb, upk % nb
        HD = (sp.csr_matrix((Hdd.reshape(-1), (bi, bj)), shape=(7 * nb, 7 * nb))
              + sp.csr_matrix(Hp)).tocsr()
        rr = (3 * up[:, None, None] + np.arange(3)[None, :, None]).repeat(7, 2)
        cc = (7 * uk[:, None, None] + np.arange(7)[None, None, :]).repeat(3, 1)
        B = sp.csr_matrix((Hpd.reshape(-1), (rr.reshape(-1), cc.reshape(-1))),
                          shape=(3 * P, 7 * nb))
        while True:
            Hl = Hpp + lam * np.einsum("pii->pi", Hpp)[:, :, None] * np.eye(3) + 1e-12 * np.eye(3)
            Wi = np.linalg.inv(Hl)
            Wm = sp.csr_matrix((Wi.reshape(-1), (rows, cols)), shape=(3 * P, 3 * P))
            WB = Wm @ B
            S = HD + lam * sp.diags(HD.diagonal()) - B.T @ WB
            rhs = gd.reshape(-1) - B.T @ (Wm @ gp.reshape(-1))
            dd = -(np.linalg.solve(S.toarray(), rhs) if 7 * nb <= dense_max
                   else spla.spsolve(S.tocsc(), rhs))
            dX = -(Wm @ (gp.reshape(-1) + B @ dd)).reshape(P, 3)
            c1, _, okn = cost_of(X + dX, d + dd.reshape(nb, 7))
            if c1 < hist[-1]:
                X, d = X + dX, d + dd.reshape(nb, 7)
                ok = okn
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5
            if lam > 1e6:
                break
        rel = (hist[-1] - c1) / max(hist[-1], 1e-12)
        hist.append(min(c1, hist[-1]))
        if lam > 1e6 or rel < 1e-7:
            break
    c1, e2, ok = cost_of(X, d)
    info = dict(cost0=float(c0), cost=float(c1), iters=it + 1,
                med_err=float(np.sqrt(np.median(e2[ok]))),
                p90_err=float(np.sqrt(np.percentile(e2[ok], 90))))
    return X, d, info


def prior_sigmas(zbar: float, rot_deg: float, log_scale: float, trans: float) -> np.ndarray:
    """Prior standard deviations of a 7-vector correction: rotation (deg), log scale, translation
    as a fraction of zbar."""
    return np.array([math.radians(rot_deg)] * 3 + [log_scale] + [trans * zbar] * 3)


def solve_blocks(S: ObservationSet, graph: KeyframeGraph, shift: np.ndarray | None,
                 sigma: np.ndarray, smooth: SmoothnessPrior, **lm) -> tuple:
    """First continuation stage: one Sim(3) correction per block (Eq. (4) restricted to d_k).
    Returns points, block corrections (nb, 7), the observations and the solver info."""
    nb = graph.num_blocks
    ob = block_observations(S, graph, graph.node_poses(), shift)
    Hp = (sp.kron(sp.identity(nb), sp.diags(1.0 / sigma ** 2)) + smooth.H1).tocsr()
    X, dk, info = bundle_adjust(S.X0, np.zeros((nb, 7)), ob, Hp, smooth.g1, smooth.const,
                                0.05 * graph.zbar, **lm)
    return X, dk, ob, info


def solve_joint(X1: np.ndarray, ob1: Observations, graph: KeyframeGraph, poses1: dict,
                sigma_block: np.ndarray, sigma_key: np.ndarray, smooth: SmoothnessPrior,
                block_factor: float = 1.0, **lm) -> tuple:
    """Second continuation stage: MAP of Eq. (4) over all keyframe corrections, referenced at the
    synchronized poses and started from the first-stage solution poses1.

    Returns points, keyframe corrections dk2 (nq, 7) relative to poses1, the observations and info.
    """
    nq = len(graph.nodes)
    poses0 = graph.node_poses()
    R0a = np.stack([poses0[q][0] for q in range(nq)])
    c0a = np.stack([poses0[q][1] for q in range(nq)])
    R1a = np.stack([poses1[q][0] for q in range(nq)])
    c1a = np.stack([poses1[q][1] for q in range(nq)])
    d0 = np.zeros((nq, 7))
    d0[:, :3] = Rotation.from_matrix(np.einsum("nij,nkj->nik", R1a, R0a)).as_rotvec()
    d0[:, 4:] = c1a - c0a
    qq = ob1.node
    ob2 = Observations(point=ob1.point, k=qq, R=R0a[qq], c=c0a[qq], cb=c0a[qq], K=ob1.K,
                       u=ob1.u, node=qq, shift=ob1.shift)
    HJ = (two_level_prior(graph.nodes, graph.centres, poses0, sigma_block, sigma_key, block_factor)
          + smooth.H2).tocsr()
    X2, dj, info = bundle_adjust(X1, d0, ob2, HJ, smooth.g2, smooth.const, 0.05 * graph.zbar, **lm)
    Rf = np.einsum("nij,njk->nik", rotation_matrices(dj[:, :3]), R0a)
    cf = c0a + dj[:, 4:]
    dk2 = np.zeros((nq, 7))
    dk2[:, :3] = Rotation.from_matrix(np.einsum("nij,nkj->nik", Rf, R1a)).as_rotvec()
    dk2[:, 4:] = cf - c1a
    return X2, dk2, ob2, info
