"""Long-video camera trajectory from overlapping feed-forward windows.

The solver runs on CPU in this order: Sim(3) synchronization of the windows through their
shared anchor frames, covisibility and the keyframe match graph, SIFT matching with the
readout-time estimate of Eq. (6) and the rolling-shutter rectification of Eq. (5), the
two-stage bundle adjustment of Eq. (4), and the pose of every non-keyframe against the
solved points.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .ba import KeyframeGraph as SolverGraph
from .ba import observation_set, observation_shift, prior_sigmas, solve_blocks, solve_joint
from .covisibility import keyframe_graph, sequence_covisibility
from .data import frame_time
from .features import (Features, build_tracks, extract_features, feature_offsets,
                       load_features, putative_matches, verify_matches)
from .priors import smoothness_prior
from .refine import (FrameEntries, KeyframeMatcher, entry_poses, frame_trajectory,
                     refine_non_keyframes)
from .rolling_shutter import (RowModel, estimate_readout, frame_velocities, rectify_features,
                              rectify_frame)
from .sync import load_window_items, synchronize


@dataclass
class SolverConfig:
    """Settings of the paper. readout_grid_ms = None runs the global-shutter model (tau = 0)."""

    readout_grid_ms: tuple[float, float, float] | None = (0.0, 33.34, 4.7619)
    seed: int = 0
    num_features: int = 1500
    keyframe_step: int = 3
    within_lags: tuple[int, ...] = (1, 2, 4)
    min_baseline: float = 0.08
    max_baseline: float = 0.6
    max_angle_deg: float = 25.0
    num_wide_pairs: int = 6
    num_low_parallax_pairs: int = 6
    block_rot_sigma_deg: float = 1.0
    block_log_scale_sigma: float = 0.05
    block_trans_sigma: float = 0.02
    key_rot_sigma_deg: float = 0.5
    key_log_scale_sigma: float = 0.05
    key_trans_sigma: float = 0.01
    smoothness_weight: float = 1.0
    block_factor: float = 1.0
    cauchy_px: float = 2.0
    iterations: int = 40
    init_px: float = 40.0
    cap_px: float = 50.0
    dense_max: int = 4200
    neighbour_keyframes: int = 2
    min_correspondences: int = 15

    @classmethod
    def from_dict(cls, values: dict) -> "SolverConfig":
        """Config from a parsed yaml mapping; lists become tuples."""
        known = cls.__dataclass_fields__
        unknown = set(values) - set(known)
        if unknown:
            raise KeyError(f"unknown solver settings: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


@dataclass
class Camera:
    """Pinhole intrinsics K (px), OpenCV distortion or None, and the row timing model."""

    K: np.ndarray
    dist: np.ndarray | None = None
    row_model: RowModel | None = None

    def rows(self) -> RowModel:
        return self.row_model if self.row_model is not None else RowModel(float(self.K[1, 2]))


@dataclass
class Reconstruction:
    """Camera-to-world poses (T, 4, 4) in the frame of the synchronized windows."""

    frames: list[str]
    c2w: np.ndarray
    readout_ms: float
    stats: dict = field(default_factory=dict)


def reconstruct(windows_path: str, frames: list[str], camera: Camera, cfg: SolverConfig,
                log=print) -> Reconstruction:
    """Trajectory of all frames from the pi3 windows of one video.

    frames are the image paths in video order. Their file names encode the time stamps (s).
    """
    t_start = time.time()
    ws = load_window_items(windows_path)
    names = [os.path.basename(f) for f in frames]
    if list(ws.frames) != names:
        raise ValueError("window file and frame list do not match")
    K, dist, rows = camera.K, camera.dist, camera.rows()
    times = np.array([frame_time(f) for f in frames])

    sync = synchronize(ws.windows)
    cov = sequence_covisibility(ws)
    kf = keyframe_graph(sync, cov, cfg.keyframe_step, cfg.within_lags, cfg.min_baseline,
                        cfg.max_baseline, cfg.max_angle_deg, cfg.num_wide_pairs,
                        cfg.num_low_parallax_pairs)
    windows = sorted(sync.poses)
    log(f"sync: {len(windows)} windows, {len(kf.nodes)} keyframes, {len(kf.pairs)} pairs")

    omega, velocity = frame_velocities(sync.poses, windows, times)
    feats = extract_features(frames, kf.nodes, K, dist, cfg.num_features)
    matches = putative_matches(feats, kf.pairs, cfg.seed)
    readout_ms, readout_info = 0.0, {}
    if cfg.readout_grid_ms is not None:
        readout_ms, readout_info = estimate_readout(feats, matches, omega, K, rows,
                                                    tuple(cfg.readout_grid_ms))
        feats = rectify_features(feats, omega, readout_ms, K, rows)
    log(f"readout time {readout_ms:.4f} ms")

    offset, node_of, kp_of = feature_offsets(kf.nodes, feats)
    ei, ej, er, pairs_kept = verify_matches(feats, matches, offset, sync.poses, K)
    tracks = build_tracks(ei, ej, er, kf.nodes, node_of)
    log(f"matching: {pairs_kept} verified pairs, {len(tracks)} tracks")

    graph = SolverGraph(windows=windows, keyframes=kf.keyframes, poses=sync.poses,
                        centres=sync.centres, zbar=sync.scene_depth, K=K)
    cross = [g for g, is_cross in tracks if is_cross]
    obs = observation_set(cross, node_of, kp_of, feats, graph, graph.node_poses(), cfg.init_px)
    shift = (observation_shift(obs, graph, kp_of, feats, velocity, readout_ms, rows)
             if readout_ms else None)
    stamps = np.array([times[f] for _, f in graph.nodes])
    smooth = smoothness_prior(graph.nodes, graph.node_poses(), stamps, graph.block_of,
                              graph.centres, graph.num_blocks, cfg.smoothness_weight)
    lm = dict(cauchy_px=cfg.cauchy_px, iters=cfg.iterations, cap_px=cfg.cap_px,
              dense_max=cfg.dense_max)
    sw = prior_sigmas(graph.zbar, cfg.block_rot_sigma_deg, cfg.block_log_scale_sigma,
                      cfg.block_trans_sigma)
    X1, dk, ob1, info1 = solve_blocks(obs, graph, shift, sw, smooth, **lm)
    log(f"block stage: {info1}")
    sw2 = prior_sigmas(graph.zbar, cfg.key_rot_sigma_deg, cfg.key_log_scale_sigma,
                       cfg.key_trans_sigma)
    poses1 = graph.block_corrected_poses(dk)
    X2, dk2, ob2, info2 = solve_joint(X1, ob1, graph, poses1, sw, sw2, smooth,
                                      cfg.block_factor, **lm)
    log(f"joint stage: {info2}")

    ent = FrameEntries.from_graph(graph)
    cache: dict[int, Features] = {}

    def frame_features(f: int) -> Features:
        if f in feats:
            return feats[f]
        if f not in cache:
            ft = load_features(frames[f], K, dist, cfg.num_features)
            cache[f] = rectify_frame(ft, omega[f], readout_ms, K, rows) if readout_ms else ft
        return cache[f]

    matcher = KeyframeMatcher(K, cfg.seed)
    R_nk, c_nk, info_nk = refine_non_keyframes(
        graph, ent, dk, dk2, X2, obs, ob2, poses1, sw2, kp_of, feats, frame_features, matcher,
        velocity, readout_ms, rows, cfg.neighbour_keyframes, cfg.min_correspondences,
        cfg.cauchy_px, cfg.cap_px, cfg.iterations, cfg.init_px)
    c2w = frame_trajectory(ent.frame, R_nk, c_nk)
    stats = dict(keyframes=len(kf.nodes), pairs=len(kf.pairs), verified_pairs=int(pairs_kept),
                 tracks=len(tracks), block_stage=info1, joint_stage=info2,
                 non_keyframes=info_nk, readout=readout_info,
                 seconds=round(time.time() - t_start, 1))
    log(f"done in {stats['seconds']} s")
    return Reconstruction(frames=names, c2w=c2w, readout_ms=float(readout_ms), stats=stats)
