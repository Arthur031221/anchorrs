"""CPU tests of the core geometry: rotations, triangulation, the shutter model and the ATE."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from anchorrs.camera import projection_matrix, rotation_matrix, triangulate
from anchorrs.geometry import rotation_mean
from anchorrs.metrics import ate
from anchorrs.rolling_shutter import point_shift, rectify

K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])


def test_rotation_matrix_matches_scipy():
    rng = np.random.default_rng(0)
    for w in rng.normal(scale=0.7, size=(20, 3)):
        assert np.allclose(rotation_matrix(w), Rotation.from_rotvec(w).as_matrix(), atol=1e-12)


def test_rotation_mean_of_equal_rotations():
    R = torch.tensor(Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix())
    mean = rotation_mean(R.expand(5, 3, 3).clone(), torch.ones(5, dtype=R.dtype))
    assert torch.allclose(mean, R, atol=1e-10)


def test_triangulation_recovers_point():
    X = np.array([0.2, -0.1, 2.5])
    Ps, us = [], []
    for c in ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, -0.1]):
        R = Rotation.from_rotvec([0.0, 0.05 * len(Ps), 0.0]).as_matrix()
        P = projection_matrix(K, R, np.array(c))
        u = P @ np.append(X, 1.0)
        Ps.append(P)
        us.append(u[:2] / u[2])
    assert np.allclose(triangulate(Ps, np.array(us)), X, atol=1e-8)


def test_shutter_model_is_identity_without_motion():
    xy = np.array([[10.0, 20.0], [300.0, 400.0]])
    rows = xy[:, 1].copy()
    out = rectify(xy, rows, np.zeros(3), 30.0, K[1, 2], 480.0, K, np.linalg.inv(K))
    assert np.allclose(out, xy, atol=1e-9)
    v = np.tile([0.1, 0.0, -0.2], (3, 1))
    shift = point_shift(v, np.array([K[1, 2], K[1, 2] + 240.0, K[1, 2] - 240.0]), 30.0,
                        K[1, 2], 480.0)
    assert np.allclose(shift[0], 0.0)
    assert np.allclose(shift[1], v[0] * 0.015) and np.allclose(shift[2], -v[0] * 0.015)


def test_ate_is_invariant_to_similarity():
    rng = np.random.default_rng(1)
    gt = np.tile(np.eye(4), (50, 1, 1))
    gt[:, :3, 3] = np.cumsum(rng.normal(scale=0.05, size=(50, 3)), axis=0)
    S = np.eye(4)
    S[:3, :3] = 2.5 * Rotation.from_rotvec([0.4, 0.1, -0.3]).as_matrix()
    S[:3, 3] = [1.0, -2.0, 0.5]
    est = S @ gt
    assert ate(est, gt) < 1e-9
    est[:, :3, 3] += 2.5 * rng.normal(scale=0.01, size=(50, 3))
    assert 0.005 < ate(est, gt) < 0.03
