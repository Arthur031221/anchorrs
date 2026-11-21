"""Anchored window design and feed-forward inference with pi3 (Sec. 3.1)."""
from __future__ import annotations

import math
import os
import time

import numpy as np

WINDOW = 64
NUM_ANCHORS = 16
PIXEL_LIMIT = 255000


def anchored_windows(num_frames: int, num_anchors: int = NUM_ANCHORS,
                     window: int = WINDOW) -> tuple[list[int], list[list[int]]]:
    """Anchor frames spread evenly over the video and the windows that share them.

    Every window holds all anchors plus one block of window - num_anchors consecutive
    non-anchor frames. Blocks have stride equal to their size and the last block ends at
    the last frame, so every frame is in at least one window.
    """
    grid = np.round(np.linspace(0, num_frames - 1, num_anchors)).astype(int)
    anchors = sorted(set(grid.tolist()))
    anchor_set = set(anchors)
    others = [f for f in range(num_frames) if f not in anchor_set]
    size = window - len(anchors)
    starts = list(range(0, max(len(others) - size, 0) + 1, size))
    if starts[-1] + size < len(others):
        starts.append(len(others) - size)
    windows = [sorted(others[s:s + size] + anchors) for s in starts]
    return anchors, windows


def _resize_target(width: int, height: int) -> tuple[int, int]:
    """pi3 input size: multiples of 14 pixels with at most PIXEL_LIMIT pixels."""
    scale = math.sqrt(PIXEL_LIMIT / (width * height))
    wt, ht = width * scale, height * scale
    k, m = round(wt / 14), round(ht / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


class Pi3Frontend:
    """Runs pi3 on one window and returns poses, depth, confidence and fitted intrinsics."""

    def __init__(self, checkpoint: str = "yyfz233/Pi3", device: str = "cuda",
                 dtype: str = "float16"):
        import torch
        from pi3.models.pi3 import Pi3

        if os.path.isfile(checkpoint):
            from safetensors.torch import load_file
            self.model = Pi3()
            self.model.load_state_dict(load_file(checkpoint))
        else:
            self.model = Pi3.from_pretrained(checkpoint)
        self.model = self.model.to(device).eval()
        self.device = device
        self.dtype = getattr(torch, dtype)

    def load_images(self, paths: list[str]):
        import torch
        import torchvision.transforms as T
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        size = _resize_target(*images[0].size)
        resized = [im.resize(size, Image.Resampling.LANCZOS) for im in images]
        return torch.stack([T.ToTensor()(im) for im in resized])

    def __call__(self, paths: list[str]) -> dict:
        """Camera-to-world poses, z-depth, confidence and pinhole K at the pi3 resolution.

        K has its principal point at the image centre and the focal lengths that best fit the
        local point maps, u - cx = fx X / Z, over pixels with confidence at least
        min(0.5, median confidence).
        """
        import torch

        with torch.no_grad():
            x = self.load_images(paths).to(self.device)
            with torch.autocast("cuda", dtype=self.dtype):
                out = self.model(x[None])
            points = out["local_points"][0].float()
            n, h, w, _ = points.shape
            z = points[..., 2]
            conf = torch.sigmoid(out["conf"][0, ..., 0].float())
            v, u = torch.meshgrid(torch.arange(h, device=points.device) + 0.5 - h / 2,
                                  torch.arange(w, device=points.device) + 0.5 - w / 2,
                                  indexing="ij")
            ks = []
            for i in range(n):
                pos = z[i] > 1e-3
                has = int(pos.sum()) > 0
                thr = min(0.5, float(torch.quantile(conf[i][pos], 0.5))) if has else 0.5
                mask = (conf[i] >= thr) & pos
                if int(mask.sum()) < 100:
                    mask = pos
                xz = (points[i, ..., 0] / z[i])[mask]
                yz = (points[i, ..., 1] / z[i])[mask]
                fx = float((u[mask] * xz).sum() / (xz * xz).sum().clamp_min(1e-9))
                fy = float((v[mask] * yz).sum() / (yz * yz).sum().clamp_min(1e-9))
                if not (np.isfinite(fx) and np.isfinite(fy) and fx > 1 and fy > 1):
                    raise RuntimeError(f"focal fit failed on frame {i}")
                ks.append([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
            return dict(c2w=out["camera_poses"][0].double().cpu().numpy(), depth=z.cpu().numpy(),
                        K=np.array(ks, np.float32), conf=conf.cpu().numpy())


def infer_windows(frames: list[str], out_path: str, frontend: Pi3Frontend) -> None:
    """Run the frontend on every anchored window of a video and save one .npz file.

    Keys: c2w (W, 64, 4, 4), depth and conf (W, 64, h, w) float16, K (W, 64, 3, 3),
    idx (W, 64) frame indices, anchors, frames (file names), sec (seconds per window).
    """
    import torch

    anchors, windows = anchored_windows(len(frames))
    keys = ("c2w", "depth", "conf", "K")
    acc = {k: [] for k in keys}
    sec = []
    for win in windows:
        t0 = time.time()
        r = frontend([frames[i] for i in win])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec.append(time.time() - t0)
        acc["c2w"].append(r["c2w"])
        acc["depth"].append(r["depth"].astype(np.float16))
        acc["conf"].append(r["conf"].astype(np.float16))
        acc["K"].append(r["K"])
    data = {k: np.stack(v) for k, v in acc.items()}
    data.update(idx=np.array(windows, dtype=np.int64), anchors=np.array(anchors, dtype=np.int64),
                frames=np.array([os.path.basename(f) for f in frames]), sec=np.array(sec))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".part.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, out_path)
