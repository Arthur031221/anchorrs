"""Rotation and similarity helpers shared by the solver stages.

Poses are camera-to-world, a world point is x = R X + t.
Bulk math is float32, 3x3 SVDs run in float64.
"""
import torch


def rotation_mean(Rs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Chordal weighted mean of rotations Rs [n, 3, 3] with weights w [n]."""
    M = (w.double()[:, None, None] * Rs.double()).sum(0)
    U, _, Vh = torch.linalg.svd(M)
    S = torch.eye(3, dtype=torch.float64, device=Rs.device)
    if torch.det(U) * torch.det(Vh) < 0:
        S[2, 2] = -1
    return (U @ S @ Vh).to(Rs.dtype)


def pixel_grid(H: int, W: int, stride: int, offset: int,
               dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-resolution pixel coordinates (u, v) of the strided work grid, each [h, w]."""
    v = torch.arange(offset, H, stride, dtype=dtype)
    u = torch.arange(offset, W, stride, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uu, vv
