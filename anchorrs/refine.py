"""Per-frame trajectory from the solved keyframes: block and interpolated keyframe corrections on
every core frame of every window, then pose-only refinement of each non-keyframe against the
solved points, and the mean over the windows that contain a frame."""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

from .ba import KeyframeGraph, ObservationSet, Observations
from .camera import DTYPE, project_batch, project_jacobians, rotation_matrices, rotation_matrix
from .features import MIN_INLIERS, MIN_MATCHES, Features, ratio_matches, seeded_order
from .geometry import rotation_mean
from .rolling_shutter import RowModel, point_shift


@dataclass
class FrameEntries:
    """One entry per (window, core frame): block index, frame index, synchronized pose (R0, C0)
    and the block centres CB (nb, 3)."""

    block: np.ndarray
    frame: np.ndarray
    R0: np.ndarray
    C0: np.ndarray
    CB: np.ndarray

    @classmethod
    def from_graph(cls, graph: KeyframeGraph) -> "FrameEntries":
        sel, PL = graph.windows, graph.poses
        return cls(
            block=np.array([graph.block_of[kk] for kk in sel for f in sorted(PL[kk])]),
            frame=np.array([f for kk in sel for f in sorted(PL[kk])]),
            R0=np.stack([PL[kk][f][0] for kk in sel for f in sorted(PL[kk])]),
            C0=np.stack([PL[kk][f][1] for kk in sel for f in sorted(PL[kk])]),
            CB=np.stack([graph.centres[kk] for kk in sel]))


def interpolated_corrections(graph: KeyframeGraph, dk2: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Keyframe rotation and translation corrections linearly interpolated over the core frames
    of each window, in entry order."""
    PL, KF = graph.poses, graph.keyframes
    n = sum(len(PL[kk]) for kk in graph.windows)
    W2, T2 = np.zeros((n, 3)), np.zeros((n, 3))
    i = 0
    for kk in graph.windows:
        Fk = sorted(PL[kk])
        kq = [graph.node_index[(kk, f)] for f in KF[kk]]
        if kq:
            for j in range(3):
                W2[i:i + len(Fk), j] = np.interp(Fk, KF[kk], dk2[kq, j])
                T2[i:i + len(Fk), j] = np.interp(Fk, KF[kk], dk2[kq, 4 + j])
        i += len(Fk)
    return W2, T2


def entry_poses(graph: KeyframeGraph, ent: FrameEntries, dk: np.ndarray,
                dk2: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotations and centres of all entries after the block corrections dk and, if given, the
    interpolated keyframe corrections dk2."""
    Rk = rotation_matrices(dk)
    s = np.exp(dk[:, 3])
    tau = dk[:, 4:]
    b = ent.block
    Rn = np.einsum("nij,njk->nik", Rk[b], ent.R0)
    cn = ent.CB[b] + s[b, None] * np.einsum("nij,nj->ni", Rk[b], ent.C0 - ent.CB[b]) + tau[b]
    if dk2 is not None:
        W2, T2 = interpolated_corrections(graph, dk2)
        Rn = np.einsum("nij,njk->nik", rotation_matrices(W2), Rn)
        cn = cn + T2
    return Rn, cn


def frame_trajectory(frame: np.ndarray, Rn: np.ndarray, cn: np.ndarray) -> np.ndarray:
    """Camera-to-world (T, 4, 4) per frame: rotation mean and mean centre over its entries."""
    fids = sorted(set(frame.tolist()))
    c2w = np.zeros((len(fids), 4, 4))
    c2w[:, 3, 3] = 1
    for i, f in enumerate(fids):
        m = frame == f
        c2w[i, :3, :3] = rotation_mean(torch.as_tensor(Rn[m]),
                                       torch.ones(int(m.sum()), dtype=DTYPE)).numpy()
        c2w[i, :3, 3] = cn[m].mean(0)
    assert fids == list(range(len(fids))), "trajectory must cover every frame"
    return c2w


def pose_only_lm(ent: np.ndarray, Xo: np.ndarray, uo: np.ndarray, Rr: np.ndarray, cr: np.ndarray,
                 sigma: np.ndarray, K: np.ndarray, zmin: float, cauchy_px: float = 2.0,
                 cap_px: float = 50.0, iters: int = 40) -> tuple[np.ndarray, dict]:
    """Independent Levenberg-Marquardt per pose with the points fixed: Cauchy pixel residuals of
    observations ent -> pose, and a prior of standard deviation sigma on the 7-vector correction
    about the reference pose (Rr, cr)."""
    Ne = len(Rr)
    d = np.zeros((Ne, 7))
    lam = np.full(Ne, 1e-4)
    isw2 = 1.0 / sigma ** 2
    fs = cauchy_px

    def T(x):
        return torch.as_tensor(x, dtype=DTYPE)

    Kj, Xp, Rj, cj = T(np.repeat(K[None], len(ent), 0)), T(Xo), T(Rr[ent]), T(cr[ent])
    cinv = 0.5 * fs ** 2 * np.log1p(cap_px ** 2 / fs ** 2)

    def cost(dd):
        y = project_batch(Xp, T(dd[ent]), Rj, cj, cj, Kj).numpy()
        ok = y[:, 2] > zmin
        e2 = ((y[:, :2] - uo) ** 2).sum(1)
        c = np.zeros(Ne)
        np.add.at(c, ent, np.where(ok, 0.5 * fs ** 2 * np.log1p(e2 / fs ** 2), cinv))
        return c + 0.5 * (dd ** 2 * isw2).sum(1), e2, ok

    c, _, _ = cost(d)
    c0 = c.copy()
    it = 0
    for it in range(iters):
        dp = T(d[ent])
        y = project_batch(Xp, dp, Rj, cj, cj, Kj).numpy()
        _, Jd = project_jacobians(Xp, dp, Rj, cj, cj, Kj)
        Jd = Jd.numpy()[:, :2, :]
        r = y[:, :2] - uo
        w = np.where(y[:, 2] > zmin, 1.0 / (1.0 + (r ** 2).sum(1) / fs ** 2), 0.0)
        Jw = Jd * w[:, None, None]
        H = np.zeros((Ne, 7, 7))
        np.add.at(H, ent, np.einsum("nai,naj->nij", Jw, Jd))
        H += np.diag(isw2)[None]
        g = np.zeros((Ne, 7))
        np.add.at(g, ent, np.einsum("nai,na->ni", Jw, r))
        g += d * isw2
        Hl = H + lam[:, None, None] * np.einsum("nii->ni", H)[:, :, None] * np.eye(7)[None]
        st = -np.linalg.solve(Hl, g[:, :, None])[:, :, 0]
        cn_, _, _ = cost(d + st)
        acc = cn_ < c
        d[acc] += st[acc]
        c[acc] = cn_[acc]
        lam = np.where(acc, np.maximum(lam / 3, 1e-9), lam * 5)
        if (lam > 1e6).all() or np.abs(st).max() < 1e-12:
            break
    _, e2, ok = cost(d)
    return d, dict(cost0=float(c0.sum()), cost=float(c.sum()), iters=it + 1,
                   med_err=float(np.sqrt(np.median(e2[ok]))) if ok.any() else float("nan"))


class KeyframeMatcher:
    """Essential-matrix verified matches between a frame and a keyframe (ratio test, at least
    MIN_MATCHES putative and MIN_INLIERS inlier matches), cached per pair."""

    def __init__(self, K: np.ndarray, seed: int = 0) -> None:
        self.K, self.seed = K, seed
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.cache: dict = {}

    def __call__(self, fq: int, fk: int, a: Features, b: Features):
        if (fq, fk) not in self.cache:
            res = None
            if len(a.descriptors) >= 8 and len(b.descriptors) >= 8:
                ia, ib, _ = ratio_matches(self.matcher, a.descriptors, b.descriptors)
                if len(ia) >= MIN_MATCHES:
                    perm = seeded_order(len(ia), self.seed, fq, fk)
                    if perm is not None:
                        ia, ib = ia[perm], ib[perm]
                    E, mask = cv2.findEssentialMat(a.xy[ia], b.xy[ib], self.K, cv2.RANSAC,
                                                   0.999, 1.0)
                    if E is not None and mask is not None and mask.sum() >= MIN_INLIERS:
                        v = mask[:, 0] > 0
                        res = (ia[v], ib[v])
            self.cache[(fq, fk)] = res
        return self.cache[(fq, fk)]


def refine_non_keyframes(graph: KeyframeGraph, ent: FrameEntries, dk: np.ndarray, dk2: np.ndarray,
                         X2: np.ndarray, S: ObservationSet, ob2: Observations, poses1: dict,
                         sigma: np.ndarray, kp_of: np.ndarray, features: dict,
                         frame_features: Callable[[int], Features], matcher: KeyframeMatcher,
                         velocity: np.ndarray, readout_ms: float, row_model: RowModel,
                         neighbours: int = 2, min_corr: int = 15, cauchy_px: float = 2.0,
                         cap_px: float = 50.0, iters: int = 40, init_px: float = 40.0
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pose of every non-keyframe entry by pose-only MAP against the solved points.

    A point is mapped to a keyframe feature when its observation reprojects within 3 cauchy_px
    under the solved keyframe pose. A non-keyframe is matched to the `neighbours` keyframes before
    and after it; a feature that reaches exactly one point gives a correspondence; poses with at
    least min_corr correspondences (reprojection < init_px at the interpolated pose) are refined
    with the keyframe prior sigma. Returns the entry rotations, centres and statistics.
    """
    K, nodes, zbar, KF = graph.K, graph.nodes, graph.zbar, graph.keyframes
    nq = len(nodes)
    R1a = np.stack([poses1[q][0] for q in range(nq)])
    c1a = np.stack([poses1[q][1] for q in range(nq)])
    RF = np.einsum("nij,njk->nik", rotation_matrices(dk2), R1a)
    CF = c1a + dk2[:, 4:]
    pid, qid, U = S.point, S.node, np.asarray(ob2.u)
    Xe = X2[pid] - (ob2.shift if ob2.shift is not None else 0.0)
    y = np.einsum("nji,nj->ni", RF[qid], Xe - CF[qid])
    okz = y[:, 2] > 0.05 * zbar
    pr = (y[:, :2] / np.where(okz, y[:, 2], 1.0)[:, None]) @ K[:2, :2].T + K[:2, 2]
    inl = okz & (np.linalg.norm(pr - U, axis=1) < 3 * cauchy_px)
    point_of: dict = {}
    for o in np.nonzero(inl)[0]:
        key = (int(nodes[qid[o]][1]), int(kp_of[S.feature[o]]))
        p = int(pid[o])
        point_of[key] = p if point_of.get(key, p) == p else -1
    kf_frames = np.array(sorted({int(f) for _, f in nodes}))
    kf_sets = {kk: set(int(x) for x in KF[kk]) for kk in graph.windows}
    Rn, cn = entry_poses(graph, ent, dk, dk2)
    windows = [graph.windows[b] for b in ent.block]
    is_kf = np.array([int(f) in kf_sets[kk] for kk, f in zip(windows, ent.frame)], bool)
    cache: dict = {}

    def feat(f):
        if f in features:
            return features[f]
        if f not in cache:
            cache[f] = frame_features(f)
        return cache[f]

    def correspondences(f):
        j = int(np.searchsorted(kf_frames, f))
        before = [int(g) for g in kf_frames[:j] if g != f][-neighbours:]
        after = [int(g) for g in kf_frames[j:] if g != f][:neighbours]
        M: dict = {}
        for g in before + after:
            mt = matcher(f, g, feat(f), features[g])
            if mt is None:
                continue
            for i, k in zip(mt[0].tolist(), mt[1].tolist()):
                p = point_of.get((g, k), -1)
                if p >= 0:
                    M.setdefault(i, set()).add(p)
        ii = np.array(sorted(i for i, ps in M.items() if len(ps) == 1), int)
        return ii, np.array([next(iter(M[i])) for i in ii], int)

    todo = [(i, int(ent.frame[i]), Rn[i], cn[i]) for i in range(len(ent.frame)) if not is_kf[i]]
    obs_ent, Xo, uo, Rr, cr, solved, ncor = [], [], [], [], [], [], []
    for e, (i, f, Rref, cref) in enumerate(todo):
        ii, pp = correspondences(f)
        fe = feat(f)
        if len(ii):
            Xw = X2[pp]
            if readout_ms:
                Xw = Xw - point_shift(velocity[f], row_model.rows(fe.raw_xy[ii]), readout_ms,
                                      row_model.centre_row, row_model.image_height(fe.shape))
            yy = (Xw - cref) @ Rref
            okk = yy[:, 2] > 0.05 * zbar
            pj = (yy[:, :2] / np.where(okk, yy[:, 2], 1.0)[:, None]) @ K[:2, :2].T + K[:2, 2]
            okk &= np.linalg.norm(pj - fe.xy[ii], axis=1) < init_px
            ii, Xw = ii[okk], Xw[okk]
        ncor.append(len(ii))
        if len(ii) >= min_corr:
            obs_ent.append(np.full(len(ii), len(Rr)))
            Rr.append(Rref)
            cr.append(cref)
            Xo.append(Xw)
            uo.append(fe.xy[ii])
            solved.append(e)
    Rnk, cnk = Rn.copy(), cn.copy()
    info = dict(n=len(todo), solved=len(solved))
    if Rr:
        dd, lm_info = pose_only_lm(np.concatenate(obs_ent), np.concatenate(Xo), np.concatenate(uo),
                                   np.stack(Rr), np.stack(cr), sigma, K, 0.05 * zbar, cauchy_px,
                                   cap_px, iters)
        info.update(lm_info)
        for e, x in zip(solved, dd):
            i = todo[e][0]
            Rnk[i] = rotation_matrix(x[:3]) @ Rn[i]
            cnk[i] = cn[i] + x[4:]
    info.update(nmap=int(sum(v >= 0 for v in point_of.values())),
                ncor_med=float(np.median(ncor)) if ncor else 0.0)
    return Rnk, cnk, info
