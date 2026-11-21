"""Baselines and viewing angles of the synchronized trajectory, and the keyframe graph."""
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import rotation_mean
from .sync import SyncState, WindowSet, synchronize

F64 = torch.float64
NUM_POINTS = 256
# The covisibility pass of the paper runs torch with two threads, the solver with four.
COVISIBILITY_THREADS = 2


def frame_trajectory(windows: list, sync: SyncState) -> tuple[list, np.ndarray]:
    """Per-frame camera-to-world [T, 4, 4], averaged over all windows that hold the frame."""
    acc = {}
    for k, win in enumerate(windows):
        s, Rg, tg = sync.s[k], sync.R[k], sync.t[k]
        R, t = win.R.to(F64), win.t.to(F64)
        Rn = Rg @ R
        tn = s * (t @ Rg.T) + tg
        for a, f in enumerate(win.ids):
            acc.setdefault(f, []).append((Rn[a], tn[a]))
    fids = sorted(acc)
    c2w = np.zeros((len(fids), 4, 4))
    c2w[:, 3, 3] = 1
    for n, f in enumerate(fids):
        Rs = torch.stack([x[0] for x in acc[f]])
        c2w[n, :3, :3] = rotation_mean(Rs, torch.ones(len(Rs), dtype=F64)).numpy()
        c2w[n, :3, 3] = torch.stack([x[1] for x in acc[f]]).mean(0).numpy()
    return fids, c2w


def core_frame_poses(sync: SyncState, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame chordal-mean rotation [T, 3, 3] and mean centre [T, 3] of the core poses."""
    rots, cens = {}, {}
    for k in sorted(sync.poses):
        for f, (R, c) in sync.poses[k].items():
            rots.setdefault(f, []).append(R)
            cens.setdefault(f, []).append(c)
    R = np.stack([rotation_mean(torch.as_tensor(np.stack(rots[f])),
                                torch.ones(len(rots[f]), dtype=F64)).numpy()
                  for f in range(num_frames)])
    C = np.stack([np.mean(np.stack(cens[f]), 0) for f in range(num_frames)])
    return R, C


@dataclass
class Covisibility:
    baseline: np.ndarray     # [T, T] float32 distance between camera centres (world units)
    angle: np.ndarray        # [T, T] float16 angle between the optical axes (degrees)
    scene_depth: np.ndarray  # [T] float64 median depth of the frame's points (world units)

    def relative_baseline(self) -> np.ndarray:
        """Baseline over the scene depth of the first frame, float64 [T, T]."""
        return self.baseline.astype(np.float64) / self.scene_depth[:, None]

    def angle_deg(self) -> np.ndarray:
        return self.angle.astype(np.float64)


def covisibility(ws: WindowSet, sync: SyncState, num_points: int = NUM_POINTS) -> Covisibility:
    """Baselines, axis angles and per-frame scene depth from the synchronized windows.

    Each frame takes its points from one window: the window of its block, or for an anchor frame
    the window whose block median frame is nearest (first window on ties).
    """
    windows = ws.windows
    fids, c2w = frame_trajectory(windows, sync)
    T = len(fids)
    if fids != list(range(T)):
        raise ValueError("the windows must cover every frame")
    rep = {}
    for k, win in enumerate(windows):
        mid = float(np.median([f for f in win.ids if f not in ws.anchors]))
        for pos, f in enumerate(win.ids):
            key = (0, 0.0) if f not in ws.anchors else (1, abs(f - mid))
            if f not in rep or key < rep[f][0]:
                rep[f] = (key, k, pos)
    P = np.zeros((T, num_points, 3))
    for f in range(T):
        _, k, pos = rep[f]
        win = windows[k]
        s, Rg, tg = float(sync.s[k]), sync.R[k].to(F64), sync.t[k].to(F64)
        X = win.X[pos].to(F64)
        ok = (torch.isfinite(X).all(-1) & (X[:, 2] > 0) & (win.sig[pos] < 1e5)).numpy()
        iv = np.flatnonzero(ok)
        if len(iv):
            sel = iv[np.linspace(0, len(iv) - 1, num_points).round().astype(int)]
        else:
            sel = np.zeros(num_points, int)
        Y = X[sel] @ win.R[pos].to(F64).T + win.t[pos].to(F64)
        P[f] = (s * (Y @ Rg.T) + tg).numpy()
    Rw, C = c2w[:, :3, :3], c2w[:, :3, 3]
    fwd = Rw[:, :, 2]
    zmed = np.array([np.median(((P[f] - C[f]) @ Rw[f])[:, 2]) for f in range(T)])
    base = np.linalg.norm(C[:, None] - C[None], axis=-1)
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
    return Covisibility(base.astype(np.float32), ang.astype(np.float16), zmed)


def sequence_covisibility(ws: WindowSet, threads: int = COVISIBILITY_THREADS) -> Covisibility:
    """Synchronize and compute the covisibility with `threads` torch threads, then restore them.

    The thread count changes float rounding in the reductions, so it is fixed for reproducibility.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        return covisibility(ws, synchronize(ws.windows))
    finally:
        torch.set_num_threads(previous)


@dataclass
class KeyframeGraph:
    keyframes: dict   # k -> sorted keyframe ids of window k
    nodes: list       # [(k, frame)] in window order
    pairs: list       # sorted [((ka, fa), (kb, fb))] keyframe pairs to match
    num_low_parallax: int  # cross pairs added by the low-parallax rule


def keyframe_graph(sync: SyncState, cov: Covisibility, step: int = 3, within: tuple = (1, 2, 4),
                   min_baseline: float = 0.08, max_baseline: float = 0.6, max_angle: float = 25.0,
                   num_wide: int = 6, num_low_parallax: int = 6) -> KeyframeGraph:
    """Keyframes (every step-th core frame) and the pairs to match.

    Inside a window: keyframes `within` positions apart. Across windows a and b: the num_wide
    pairs with the largest relative baseline in [min_baseline, max_baseline], and the
    num_low_parallax pairs with the smallest sqrt(baseline^2 + angle^2), angle in radians. Both
    need distinct frames, an axis angle below max_angle degrees and a relative baseline at most
    max_baseline.
    """
    BZ = cov.relative_baseline()
    AN = cov.angle_deg()
    QQ = np.sqrt(BZ ** 2 + np.radians(AN) ** 2)
    sel = sorted(sync.core)
    KF = {k: sorted(sync.core[k])[step // 2::step] for k in sel}
    nodes = [(k, f) for k in sel for f in KF[k]]
    pairs = set()
    nlo = 0
    for k in sel:
        L = KF[k]
        for d in within:
            pairs |= {((k, L[i]), (k, L[i + d])) for i in range(len(L) - d)}
    for i, ka in enumerate(sel):
        fa = np.array(KF[ka])
        for kb in sel[i + 1:]:
            fb = np.array(KF[kb])
            bz = BZ[np.ix_(fa, fb)]
            an = AN[np.ix_(fa, fb)]
            base_ok = (fa[:, None] != fb[None, :]) & (an < max_angle) & (bz <= max_baseline)
            ii, jj = np.nonzero(base_ok & (bz >= min_baseline))
            o = np.argsort(-bz[ii, jj], kind="stable")[:num_wide]
            pairs |= {((ka, int(fa[x])), (kb, int(fb[y]))) for x, y in zip(ii[o], jj[o])}
            if num_low_parallax:
                ii, jj = np.nonzero(base_ok)
                o = np.argsort(QQ[np.ix_(fa, fb)][ii, jj], kind="stable")[:num_low_parallax]
                new = {((ka, int(fa[x])), (kb, int(fb[y]))) for x, y in zip(ii[o], jj[o])} - pairs
                nlo += len(new)
                pairs |= new
    return KeyframeGraph(KF, nodes, sorted(pairs), nlo)
