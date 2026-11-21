"""SIFT features, putative matches, geometric verification and multi-view tracks of keyframes."""
from __future__ import annotations

from typing import NamedTuple, Sequence

import cv2
import numpy as np

cv2.setNumThreads(4)

RATIO = 0.8
MIN_MATCHES = 20
MIN_INLIERS = 15
SAMPSON_PX = 8.0


class Features(NamedTuple):
    """SIFT features of one frame.

    xy (n, 2) float64 are the keypoints used for geometry (undistorted, then rectified by Eq. (5)),
    descriptors (n, 128) float32, raw_xy (n, 2) the detected keypoints (px), shape the image shape.
    """

    xy: np.ndarray
    descriptors: np.ndarray
    raw_xy: np.ndarray
    shape: tuple


class PutativeMatch(NamedTuple):
    """Ratio-test matches of one keyframe pair. Nodes are (window, frame), idx index features."""

    node_a: tuple
    node_b: tuple
    idx_a: np.ndarray
    idx_b: np.ndarray
    lowe: np.ndarray

    @property
    def frame_a(self) -> int:
        return self.node_a[1]

    @property
    def frame_b(self) -> int:
        return self.node_b[1]


def detect_sift(path: str, num_features: int) -> tuple[np.ndarray, np.ndarray, tuple]:
    """SIFT keypoints (n, 2) float64, descriptors (n, 128) and image shape of a gray image."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kp, des = cv2.SIFT_create(nfeatures=num_features).detectAndCompute(image, None)
    xy = np.array([k.pt for k in kp], np.float64).reshape(-1, 2)
    return xy, (des if des is not None else np.zeros((0, 128), np.float32)), image.shape


def load_features(path: str, K: np.ndarray, dist: np.ndarray | None,
                  num_features: int = 1500) -> Features:
    """SIFT features of one image, keypoints undistorted with (K, dist) when dist is given."""
    xy, des, shape = detect_sift(path, num_features)
    raw = xy.copy()
    if dist is not None and len(xy):
        xy = cv2.undistortPoints(xy.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    return Features(xy, des, raw, shape)


def extract_features(image_paths: Sequence[str], nodes: list[tuple[int, int]], K: np.ndarray,
                     dist: np.ndarray | None, num_features: int = 1500) -> dict[int, Features]:
    """load_features for every frame that appears in the keyframe nodes (window, frame)."""
    feats: dict[int, Features] = {}
    for _, f in nodes:
        if f not in feats:
            feats[f] = load_features(image_paths[f], K, dist, num_features)
    return feats


def seeded_order(n: int, seed: int, frame_a: int, frame_b: int) -> np.ndarray | None:
    """Permutation of n putative matches for seed > 0 (RANSAC sample order), None for seed 0."""
    if not seed:
        return None
    return np.random.default_rng([seed, frame_a, frame_b]).permutation(n)


def ratio_matches(matcher: cv2.BFMatcher, des_a: np.ndarray, des_b: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matches that pass Lowe's ratio test, with their distance ratio."""
    knn = [m for m in matcher.knnMatch(des_a, des_b, k=2)
           if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    idx_a = np.array([m[0].queryIdx for m in knn])
    idx_b = np.array([m[0].trainIdx for m in knn])
    lowe = np.array([m[0].distance / max(m[1].distance, 1e-12) for m in knn], dtype=float)
    return idx_a, idx_b, lowe


def putative_matches(feats: dict[int, Features], pairs: set, seed: int = 0
                     ) -> list[PutativeMatch]:
    """Ratio-test matches of every keyframe pair (in sorted order) with at least 20 matches."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    out = []
    for na, nb in sorted(pairs):
        da, db = feats[na[1]].descriptors, feats[nb[1]].descriptors
        if len(da) < 8 or len(db) < 8:
            continue
        idx_a, idx_b, lowe = ratio_matches(matcher, da, db)
        if len(idx_a) < MIN_MATCHES:
            continue
        order = seeded_order(len(idx_a), seed, na[1], nb[1])
        if order is not None:
            idx_a, idx_b, lowe = idx_a[order], idx_b[order], lowe[order]
        out.append(PutativeMatch(na, nb, idx_a, idx_b, lowe))
    return out


def sampson_distance(xa: np.ndarray, xb: np.ndarray, K: np.ndarray, R: np.ndarray,
                     t: np.ndarray) -> np.ndarray:
    """First-order geometric distance (px) of matches xa <-> xb to the epipolar geometry of (R, t).
    """
    K_inv = np.linalg.inv(K)
    tn = t / max(np.linalg.norm(t), 1e-12)
    tx = np.array([[0, -tn[2], tn[1]], [tn[2], 0, -tn[0]], [-tn[1], tn[0], 0]])
    F = K_inv.T @ (tx @ R) @ K_inv
    pa = np.c_[xa, np.ones(len(xa))]
    pb = np.c_[xb, np.ones(len(xb))]
    Fa = pa @ F.T
    Fb = pb @ F
    return np.abs(np.einsum("ij,ij->i", pb, Fa)) / np.sqrt(
        Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Fb[:, 0] ** 2 + Fb[:, 1] ** 2 + 1e-300)


def feature_offsets(nodes: list[tuple[int, int]], feats: dict[int, Features]
                    ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Global index of each node's first feature, and node and keypoint of each global index."""
    offset = {}
    total = 0
    for nd in nodes:
        offset[nd] = total
        total += len(feats[nd[1]].xy)
    node_of = np.zeros(total, int)
    keypoint_of = np.zeros(total, int)
    for q, nd in enumerate(nodes):
        n = len(feats[nd[1]].xy)
        node_of[offset[nd]:offset[nd] + n] = q
        keypoint_of[offset[nd]:offset[nd] + n] = np.arange(n)
    return offset, node_of, keypoint_of


def verify_matches(feats: dict[int, Features], matches: list[PutativeMatch], offset: dict,
                   poses: dict, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Geometrically verified matches between global feature indices.

    A match is kept when it is an essential-matrix RANSAC inlier and lies within 8 px (Sampson
    distance) of the epipolar geometry of the synchronized relative pose. A pair with fewer than
    15 kept matches gives none. Returns (idx_i, idx_j, lowe ratio, number of pairs kept).
    """
    ei, ej, er = [], [], []
    pairs_kept = 0
    for m in matches:
        xa, xb = feats[m.frame_a].xy, feats[m.frame_b].xy
        E, mask = cv2.findEssentialMat(xa[m.idx_a], xb[m.idx_b], K, cv2.RANSAC, 0.999, 1.0)
        if E is None or mask is None or mask.sum() < MIN_INLIERS:
            continue
        (Ra, ca), (Rb, cb) = poses[m.node_a[0]][m.node_a[1]], poses[m.node_b[0]][m.node_b[1]]
        keep = (mask[:, 0] > 0) & (sampson_distance(xa[m.idx_a], xb[m.idx_b], K, Rb.T @ Ra,
                                                    Rb.T @ (ca - cb)) < SAMPSON_PX)
        if keep.sum() < MIN_INLIERS:
            continue
        ei.append(offset[m.node_a] + m.idx_a[keep])
        ej.append(offset[m.node_b] + m.idx_b[keep])
        er.append(m.lowe[keep])
        pairs_kept += 1
    if not ei:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), 0
    return np.concatenate(ei), np.concatenate(ej), np.concatenate(er), pairs_kept


def union_find_tracks(ei: np.ndarray, ej: np.ndarray, ratio: np.ndarray, frame_of: np.ndarray
                      ) -> tuple[list[np.ndarray], int]:
    """Conflict-free tracks from pairwise matches.

    Matches are merged in ascending Lowe ratio (stable). Two sets merge only if they contain no
    common frame. Returns the groups (sorted feature indices, ordered by their smallest index) and
    the number of rejected merges.
    """
    parent: dict[int, int] = {}
    frames: dict[int, set] = {}
    members: dict[int, list] = {}
    rejected = 0

    def find(x: int) -> int:
        r = x
        while parent.get(r, r) != r:
            r = parent[r]
        while parent.get(x, x) != r:
            nx = parent[x]
            parent[x] = r
            x = nx
        return r

    for t in np.argsort(ratio, kind="stable"):
        u, v = find(int(ei[t])), find(int(ej[t]))
        if u == v:
            continue
        su = frames.get(u) or {int(frame_of[u])}
        sv = frames.get(v) or {int(frame_of[v])}
        if not su.isdisjoint(sv):
            rejected += 1
            continue
        if len(su) < len(sv):
            u, v, su, sv = v, u, sv, su
        mu = members.pop(u, None) or [u]
        mv = members.pop(v, None) or [v]
        parent[v] = u
        su |= sv
        frames[u] = su
        frames.pop(v, None)
        members[u] = mu + mv
    groups = sorted((np.array(sorted(m), dtype=int) for m in members.values()),
                    key=lambda g: int(g[0]))
    return groups, rejected


def build_tracks(ei: np.ndarray, ej: np.ndarray, ratio: np.ndarray, nodes: list[tuple[int, int]],
                 node_of: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Tracks with at least 3 observations in distinct frames that span at least 2 windows.

    Each track is (global feature indices, True); the flag marks a cross-window track.
    """
    frame_of = np.array([nodes[x][1] for x in node_of])
    groups, _ = union_find_tracks(ei, ej, ratio, frame_of)
    tracks = []
    for grp in groups:
        q = node_of[grp]
        fr = [nodes[x][1] for x in q]
        if len(grp) < 3 or len(set(fr)) < len(fr):
            continue
        if len({nodes[x][0] for x in q}) >= 2:
            tracks.append((grp, True))
    return tracks
