"""Sim(3) synchronization of the feed-forward windows through their shared anchor frames.

Every window of the anchored design contains the same anchor frames. The same pixels of a shared
frame give 3D points in the coordinates of both windows, so every window pair yields one relative
Sim(3) measurement (weighted Umeyama). All window Sim(3)s are then solved jointly: exact log-scale
least squares, chordal rotation relaxation refined by Gauss-Newton on SO(3), and linear least
squares for the translations. Window 0 fixes the gauge. Poses are camera-to-world, x = R X + t.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import pixel_grid

F64 = torch.float64

# Frozen window model: sigma = kappa * depth / normalised confidence, on a stride-4 grid.
DEPTH_KAPPA = 2.0
GRID_STRIDE = 4
GRID_OFFSET = 2
MIN_CORE_RUN = 8
GN_ITERATIONS = 6


@dataclass
class Window:
    """One feed-forward window on its work grid (all tensors float32, window gauge)."""
    ids: list          # global frame ids [L]
    X: torch.Tensor    # [L, N, 3] camera-frame points (depth times z=1 ray)
    sig: torch.Tensor  # [L, N] point sigma in model units, 1e6 where the depth is invalid
    K: torch.Tensor    # [L, 3, 3] full-resolution intrinsics
    median_depth: float
    R: torch.Tensor    # [L, 3, 3] model camera-to-world rotations
    t: torch.Tensor    # [L, 3] model camera-to-world translations


@dataclass
class WindowSet:
    windows: list      # list[Window]
    anchors: set       # anchor frame ids
    frames: list       # frame file names in sequence order
    image_hw: tuple    # model image size (H, W)


def prepare_window(ids: list, depth: np.ndarray, conf: np.ndarray, K: np.ndarray,
                   c2w: np.ndarray) -> Window:
    """Back-project depth [L, H, W] on the work grid and attach the confidence-based sigma."""
    f = lambda a: torch.as_tensor(a, dtype=torch.float32)
    depth, conf, K, c2w = f(depth), f(conf), f(K), f(c2w)
    L, H, W = depth.shape
    d = depth[:, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
    c = conf[:, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
    h, w = d.shape[1:]
    uu, vv = pixel_grid(H, W, GRID_STRIDE, GRID_OFFSET)
    ray = torch.stack([(uu[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None],
                       (vv[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None],
                       torch.ones(L, h, w)], -1).reshape(L, h * w, 3)
    d = d.reshape(L, -1)
    c = c.reshape(L, -1)
    ok = torch.isfinite(d) & (d > 0)
    median_depth = float(d[ok].median()) if ok.any() else 1.0
    c = torch.nan_to_num(c, nan=1e-3)
    cm = float(c[ok & (c > 0)].median()) if (ok & (c > 0)).any() else 1.0
    c = c / max(cm, 1e-9)
    sig = DEPTH_KAPPA * d.clamp_min(1e-6) / c.clamp_min(1e-3)
    sig = torch.where(ok, sig, torch.full_like(sig, 1e6))
    d = torch.where(ok, d, torch.zeros_like(d))
    X = d[..., None] * ray
    return Window(list(ids), X, sig, K, median_depth, c2w[:, :3, :3].clone(), c2w[:, :3, 3].clone())


def load_window_items(npz_path: str) -> WindowSet:
    """Read the anchored windows written by windows.infer_windows (or the cached ones)."""
    z = np.load(npz_path, allow_pickle=False)
    idx, c2w, K, depth, conf = z["idx"], z["c2w"], z["K"], z["depth"], z["conf"]
    windows = [prepare_window([int(i) for i in idx[k]], depth[k].astype(np.float32),
                              conf[k].astype(np.float32), K[k].astype(np.float32),
                              c2w[k].astype(np.float32)) for k in range(len(idx))]
    frames = [str(x).replace("\\", "/").rsplit("/", 1)[-1] for x in z["frames"]]
    return WindowSet(windows, set(int(x) for x in z["anchors"]), frames, tuple(depth.shape[2:]))


def core_frames(ids: list, min_run: int = MIN_CORE_RUN) -> set:
    """Longest run of consecutive frame ids of a window (its block with the anchors inside it)."""
    s = sorted(ids)
    best, cur = [], [s[0]]
    for a, b in zip(s[:-1], s[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [b]
    if len(cur) > len(best):
        best = cur
    return set(best) if len(best) >= min_run else set()


def shared_points(wa: Window, wb: Window, frames: list) -> tuple:
    """Points of the same pixels of the shared frames in the coordinates of windows a and b.

    Camera centres of the shared frames are appended with the mean point weight.
    Returns (Pa, Pb, w) in float64.
    """
    def world(win: Window, pos: list) -> tuple:
        X = win.X[pos]
        n, N, _ = X.shape
        X = X.reshape(-1, 3)
        sig = win.sig[pos].reshape(-1)
        ok = torch.isfinite(X[:, 2]) & (X[:, 2] > 0) & (sig < 1e5) & torch.isfinite(sig)
        Rp = win.R[pos][:, None].expand(n, N, 3, 3).reshape(-1, 3, 3)
        tp = win.t[pos][:, None].expand(n, N, 3).reshape(-1, 3)
        return torch.einsum("mij,mj->mi", Rp, X) + tp, sig, ok, win.t[pos]

    Ya, sa, oka, ca = world(wa, [wa.ids.index(f) for f in frames])
    Yb, sb, okb, cb = world(wb, [wb.ids.index(f) for f in frames])
    ok = oka & okb
    sb = sb * (wa.median_depth / wb.median_depth)
    om = ok.to(F64) / (sa.to(F64) ** 2 + sb.to(F64) ** 2).clamp_min(1e-30)
    ca, cb = ca.to(F64), cb.to(F64)
    wc = torch.full((len(ca),), float(om.sum()) / len(ca), dtype=F64)
    Pa = torch.cat([Ya.to(F64), ca])
    Pb = torch.cat([Yb.to(F64), cb])
    w = torch.cat([om, wc])
    fin = torch.isfinite(Pa).all(-1) & torch.isfinite(Pb).all(-1)
    return torch.nan_to_num(Pa), torch.nan_to_num(Pb), w * fin.to(F64)


@dataclass
class Edge:
    """Relative Sim(3) of window b into window a (Pa ~ s R Pb + t) with scalar information."""
    a: int
    b: int
    s: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    w_s: torch.Tensor  # log-scale information
    w_R: torch.Tensor  # rotation information [rad^-2]
    w_t: torch.Tensor  # translation information [1 / length^2]


def edge_from_points(a: int, b: int, Pa: torch.Tensor, Pb: torch.Tensor, w: torch.Tensor,
                     factor: float = 1.0) -> Edge:
    """Weighted Umeyama fit Pa ~ s R Pb + t and its information, scaled by the overlap factor."""
    W = w.sum().clamp_min(1e-30)
    mx = (w[:, None] * Pb).sum(0) / W
    my = (w[:, None] * Pa).sum(0) / W
    xc, yc = Pb - mx, Pa - my
    H = torch.einsum("m,mi,mj->ij", w, yc, xc) / W
    U, S, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(U @ Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.stack([torch.ones_like(d), torch.ones_like(d), d])
    R = U @ torch.diag(D) @ Vh
    vx = (w * (xc * xc).sum(-1)).sum() / W
    s = (S * D).sum() / vx.clamp_min(1e-30)
    t = my - s * (R @ mx)
    sxx = (w * (xc * xc).sum(-1)).sum()
    w_s, w_R, w_t = sxx, (2.0 / 3.0) * s * s * sxx, W
    if factor != 1.0:
        w_s, w_R, w_t = w_s * factor, w_R * factor, w_t * factor
    return Edge(a, b, s, R, t, w_s, w_R, w_t)


def window_edges(windows: list) -> list:
    """One edge per window pair with shared frames.

    Its information is scaled by the mean of min(1, 2 / multiplicity) over the shared frames.
    """
    count = {}
    for win in windows:
        for f in win.ids:
            count[f] = count.get(f, 0) + 1
    sets = [set(win.ids) for win in windows]
    edges = []
    for a in range(len(windows)):
        for b in range(a + 1, len(windows)):
            sh = sorted(sets[a] & sets[b])
            if not sh:
                continue
            factor = float(np.mean([min(1.0, 2.0 / count[f]) for f in sh]))
            edges.append(edge_from_points(a, b, *shared_points(windows[a], windows[b], sh), factor))
    return edges


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _exp_so3(phi: torch.Tensor) -> torch.Tensor:
    th2 = (phi * phi).sum(-1, keepdim=True)[..., None]
    th = torch.sqrt(th2 + 1e-30)
    small = th2 < 1e-10
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / (th * th))
    K = _skew(phi)
    return torch.eye(3, dtype=phi.dtype) + A * K + B * (K @ K)


def _project_so3(M: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(M)
    d = torch.sign(torch.linalg.det(U @ Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh


def _incidence(i: torch.Tensor, j: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    E = len(i)
    ar = torch.arange(E)
    B = torch.zeros(E, n, dtype=dtype)
    return B.index_put((ar, j), torch.ones(E, dtype=dtype)).index_put(
        (ar, i), -torch.ones(E, dtype=dtype), accumulate=True)


def _grounded_inverse(L: torch.Tensor) -> torch.Tensor:
    P = torch.zeros_like(L)
    P[1:, 1:] = torch.linalg.inv(L[1:, 1:])
    return P


def _rotation_form(i, j, s_ij, R_ij, t_ij, w_R, w_t, n, s_nodes) -> torch.Tensor:
    """Quadratic form Q [3n, 3n] of the rotation cost tr(X Q X^T), translations eliminated."""
    dt = R_ij.dtype
    I3 = torch.eye(3, dtype=dt)
    Ne = len(i)
    Q4 = torch.zeros(n, n, 3, 3, dtype=dt)
    Q4 = Q4.index_put((i, i), w_R[:, None, None] * I3, accumulate=True)
    Q4 = Q4.index_put((j, j), w_R[:, None, None] * I3, accumulate=True)
    Q4 = Q4.index_put((i, j), -w_R[:, None, None] * R_ij, accumulate=True)
    Q4 = Q4.index_put((j, i), -w_R[:, None, None] * R_ij.transpose(-1, -2), accumulate=True)
    tp = s_nodes[i][:, None] * t_ij
    Gam = torch.zeros(Ne, n, 3, dtype=dt).index_put((torch.arange(Ne), i), tp, accumulate=True)
    Gam = Gam.permute(1, 2, 0).reshape(3 * n, Ne)
    B = _incidence(i, j, n, dt)
    P = _grounded_inverse(B.T @ (w_t[:, None] * B))
    GD = Gam * w_t[None, :]
    QT = GD @ Gam.T - GD @ B @ P @ (B.T @ GD.T)
    Q = Q4.permute(0, 2, 1, 3).reshape(3 * n, 3 * n) + QT
    return 0.5 * (Q + Q.T)


def _rotation_relaxation(Q: torch.Tensor, n: int) -> torch.Tensor:
    dt = Q.dtype
    I3 = torch.eye(3, dtype=dt)
    Y = torch.linalg.solve(Q[3:, 3:] + 1e-14 * torch.eye(3 * n - 3, dtype=dt), -Q[3:, :3] @ I3)
    R = _project_so3(Y.reshape(n - 1, 3, 3).transpose(-1, -2))
    return torch.cat([I3[None], R], 0)


def _rotation_gauss_newton(Q: torch.Tensor, R: torch.Tensor, damp: float = 1e-9) -> torch.Tensor:
    """One Gauss-Newton step on SO(3)^n with window 0 fixed, R_i <- R_i exp(phi_i)."""
    n = R.shape[0]
    dt = R.dtype
    Q4 = Q.reshape(n, 3, n, 3).permute(0, 2, 1, 3)
    Ek = _skew(torch.eye(3, dtype=dt))
    Dm = -torch.einsum("kpq,iqr->ikpr", Ek, R.transpose(-1, -2))
    Y = R.transpose(-1, -2)
    H = torch.einsum("iapq,ijpr,jbrq->iajb", Dm, Q4, Dm).reshape(3 * n, 3 * n)
    g = torch.einsum("iapq,ijpr,jrq->ia", Dm, Q4, Y).reshape(3 * n)
    Hf = H[3:, 3:]
    Hf = Hf + damp * (torch.diagonal(Hf).mean().abs() + 1e-30) * torch.eye(3 * n - 3, dtype=dt)
    phi = torch.linalg.solve(Hf, -g[3:]).reshape(n - 1, 3)
    return torch.cat([R[:1], R[1:] @ _exp_so3(phi)], 0)


def solve_sim3(edges: list, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global Sim(3) of every window (window coordinates to world).

    Returns float64 s [n], R [n, 3, 3], t [n, 3].
    """
    i = torch.tensor([e.a for e in edges], dtype=torch.long)
    j = torch.tensor([e.b for e in edges], dtype=torch.long)
    stack = lambda k: torch.stack([getattr(e, k) for e in edges])
    s_ij, R_ij, t_ij = stack("s"), stack("R"), stack("t")
    w_s, w_R, w_t = stack("w_s"), stack("w_R"), stack("w_t")
    m = torch.log(s_ij)
    L = _incidence(i, j, n, w_s.dtype)
    L = L.T @ (w_s[:, None] * L)
    b = torch.zeros(n, dtype=w_s.dtype).index_add(0, j, w_s * m).index_add(0, i, -w_s * m)
    log_s = torch.cat([torch.zeros(1, dtype=w_s.dtype), torch.linalg.solve(L[1:, 1:], b[1:])])
    s = torch.exp(log_s)
    Q = _rotation_form(i, j, s_ij, R_ij, t_ij, w_R, w_t, n, s)
    R = _rotation_relaxation(Q, n)
    for _ in range(GN_ITERATIONS):
        R = _rotation_gauss_newton(Q, R)
    c = s[i][:, None] * torch.einsum("eab,eb->ea", R[i], t_ij)
    B = _incidence(i, j, n, w_t.dtype)
    Lt = B.T @ (w_t[:, None] * B)
    b = torch.zeros(n, 3, dtype=w_t.dtype).index_add(0, j, w_t[:, None] * c).index_add(
        0, i, -w_t[:, None] * c)
    t = torch.cat([torch.zeros(1, 3, dtype=w_t.dtype), torch.linalg.solve(Lt[1:, 1:], b[1:])], 0)
    return s, R, t


@dataclass
class SyncState:
    """Synchronized windows. Window k maps to world by x = s[k] R[k] X + t[k]."""
    s: torch.Tensor     # [n] float64
    R: torch.Tensor     # [n, 3, 3] float64
    t: torch.Tensor     # [n, 3] float64
    core: dict          # k -> set of core frame ids (block frames and the anchors inside the block)
    poses: dict         # k -> {frame: (R [3, 3], c [3])} synchronized core poses, float64
    median_depths: list  # per window: s[k] times the median core-frame depth
    scene_depth: float  # median of median_depths, the length unit of the priors
    centres: dict       # k -> mean synchronized camera centre of the core frames [3]


def synchronize(windows: list) -> SyncState:
    """Synchronize all windows through their shared anchor frames (Sec. 3.2)."""
    n = len(windows)
    s, Rg, tg = solve_sim3(window_edges(windows), n)
    core = {}
    for k, win in enumerate(windows):
        core[k] = core_frames(win.ids) or set(win.ids)
    poses, median_depths = {}, []
    for k, win in enumerate(windows):
        R, t = win.R.to(F64), win.t.to(F64)
        Rn = (Rg[k] @ R).numpy()
        tn = (s[k] * (t @ Rg[k].T) + tg[k]).numpy()
        ids = [int(f) for f in win.ids]
        ci = [p for p, f in enumerate(ids) if f in core[k]]
        poses[k] = {ids[p]: (Rn[p], tn[p]) for p in ci}
        Z = win.X[ci][..., 2].reshape(-1)
        Z = Z[torch.isfinite(Z) & (Z > 0)]
        median_depths.append(float(s[k]) * float(Z.median()))
    centres = {k: np.mean([v[1] for v in poses[k].values()], 0) for k in range(n)}
    return SyncState(s, Rg, tg, core, poses, median_depths, float(np.median(median_depths)),
                     centres)
