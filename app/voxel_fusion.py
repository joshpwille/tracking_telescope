# app/voxel_fusion.py
"""
Sparse voxel fusion with log-odds occupancy, DDA ray traversal, and time-decay.

What this module does (high level)
- Maintains a sparse hash-map grid: key=(ix,iy,iz) -> Cell(logodds, last_update_s)
- Single-camera update ("ray carving"): march a back-projected ray through the grid,
  push "free" (miss) updates along the ray, and a small positive "weak hit" along the ray
  to represent directional uncertainty. 【turn12file1†L1-L13】【turn12file1†L26-L34】
- Stereo/triangulation update: mark free along each camera ray up to X̂ and a strong,
  localized "hit" near X̂, which sharpens depth quickly. 【turn12file1†L15-L24】
- Models uncertainty by widening rays and spreading Gaussian kernels instead of binary
  flips; uses efficient DDA traversal so only voxels on involved rays are touched. 【turn12file1†L38-L46】
- Applies time-decay so stale evidence fades and the map adapts to motion. 【turn12file1†L26-L34】
- Designed for incremental per-frame updates; pair with preallocated buffers/zero-copy
  paths if you later move this to GPU. 【turn12file5†L1-L8】

Inputs you’ll give it each frame
- For single-cam carving: (K, R, t) and a pixel (u,v) [+ optional conf, width]
- For stereo: same per camera, plus triangulated 3D point X̂ (world)
- You can also pass direct rays (C, d) if you’ve already built them.

Conventions
- K (3x3) intrinsics; R (3x3) is world→camera; t (3,) so P = K [R|t]
- Camera center: C = -R^T t ; ray dir (world): d = R^T * (K^{-1}[u v 1]^T) normalized
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, List
import numpy as np
import math
import time
from collections import defaultdict


# ==============================
# Config & cell
# ==============================

@dataclass
class VoxelFusionConfig:
    voxel_size_m: float = 1.0
    # Log-odds parameters
    logodds_hit: float = +1.2       # strong occupancy bump near triangulated X̂
    logodds_weak: float = +0.1      # faint occupancy along single-cam ray
    logodds_miss: float = -0.6      # free-space carve along rays
    clamp_min: float = -4.0
    clamp_max: float = +4.0
    # Uncertainty modeling
    sigma_ray_vox: float = 0.75     # Gaussian σ (voxels) around the ray centerline
    sigma_hit_vox: float = 1.0      # Gaussian σ (voxels) for a triangulated hit blob
    # Ray extents & step control
    max_ray_len_m: float = 5000.0   # safety cap for carving
    max_vox_per_ray: int = 40000    # guardrail
    # Time decay: pull log-odds toward 0 so stale evidence fades
    decay_per_second: float = 0.08  # ΔL per second toward 0
    # Pruning: reclaim cells that settle near 0 and haven’t been touched
    prune_abs_logodds_lt: float = 0.15
    prune_older_than_s: float = 12.0
    # Weighting
    conf_to_scale: float = 1.0      # multiply deltas by detection confidence
    min_ray_angle_deg_for_strong_hit: float = 0.6  # if using multi-cam, low angle → weaken hit
    # Neighborhood caps for Gaussian spill
    max_kernel_radius_vox: int = 3  # limit neighbor loops


@dataclass
class Cell:
    L: float            # log-odds
    t_last: float       # last update time (monotonic seconds)


# ==============================
# Core sparse grid
# ==============================

class SparseVoxelGrid:
    def __init__(self, cfg: VoxelFusionConfig):
        self.cfg = cfg
        self._cells: Dict[Tuple[int,int,int], Cell] = {}
        self._size = float(cfg.voxel_size_m)

    # ---- coordinates ----

    def world_to_idx(self, p: np.ndarray) -> Tuple[int,int,int]:
        s = self._size
        return (int(math.floor(p[0] / s)),
                int(math.floor(p[1] / s)),
                int(math.floor(p[2] / s)))

    def idx_to_center(self, idx: Tuple[int,int,int]) -> np.ndarray:
        s = self._size
        return np.array([(idx[0] + 0.5) * s,
                         (idx[1] + 0.5) * s,
                         (idx[2] + 0.5) * s], dtype=float)

    # ---- decay & pruning ----

    def _decay_cell(self, c: Cell, now: float):
        if self.cfg.decay_per_second <= 0: return
        dt = max(0.0, now - c.t_last)
        if dt <= 0.0: return
        L = c.L
        sign = 1.0 if L > 0 else -1.0
        mag = max(0.0, abs(L) - self.cfg.decay_per_second * dt)
        c.L = max(0.0, mag) * sign

    def prune(self, now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        kill: List[Tuple[int,int,int]] = []
        for k, c in self._cells.items():
            self._decay_cell(c, now)
            if abs(c.L) < self.cfg.prune_abs_logodds_lt and (now - c.t_last) > self.cfg.prune_older_than_s:
                kill.append(k)
        for k in kill:
            self._cells.pop(k, None)

    # ---- update ----

    def add_L(self, idx: Tuple[int,int,int], dL: float, now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        c = self._cells.get(idx)
        if c is None:
            c = Cell(L=0.0, t_last=now)
            self._cells[idx] = c
        else:
            self._decay_cell(c, now)
        c.L = float(max(self.cfg.clamp_min, min(self.cfg.clamp_max, c.L + dL)))
        c.t_last = now

    def items(self) -> Iterable[Tuple[Tuple[int,int,int], Cell]]:
        return self._cells.items()

    # Query helpers
    def top_k(self, k: int = 1) -> List[Tuple[Tuple[int,int,int], Cell]]:
        # naive scan; fine for sparse maps (thousands to low millions)
        return sorted(self._cells.items(), key=lambda kv: kv[1].L, reverse=True)[:k]

    def to_point_cloud(self, min_logodds: float = 0.0) -> np.ndarray:
        pts = [self.idx_to_center(idx) for idx, c in self._cells.items() if c.L >= min_logodds]
        return np.array(pts, dtype=float) if pts else np.zeros((0,3), dtype=float)


# ==============================
# Math helpers (camera, rays, kernels)
# ==============================

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def camera_center_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3,3)
    t = np.asarray(t, dtype=float).reshape(3)
    return -R.T @ t

def bearing_from_pixel(K: np.ndarray, uv: Tuple[float,float]) -> np.ndarray:
    K = np.asarray(K, dtype=float).reshape(3,3)
    u, v = float(uv[0]), float(uv[1])
    x = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    return _normalize(x)

def world_ray_from_obs(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv: Tuple[float,float]) -> Tuple[np.ndarray, np.ndarray]:
    C = camera_center_from_R_t(R, t)
    d_cam = bearing_from_pixel(K, uv)
    d_world = _normalize(R.T @ d_cam)
    return C, d_world

def gaussian_weight(dist_vox: float, sigma_vox: float) -> float:
    if sigma_vox <= 0.0: return 0.0
    return math.exp(-0.5 * (dist_vox / sigma_vox) ** 2)

def ray_aabb_intersect(C: np.ndarray, d: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> Tuple[float,float]:
    """
    Return (tmin, tmax) for intersection with axis-aligned box; if no hit, (inf, -inf).
    Useful when you restrict updates to a ROI box.
    """
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0 = (aabb_min - C) * inv
    t1 = (aabb_max - C) * inv
    tmin = float(np.maximum(np.minimum(t0, t1), 0.0).max())
    tmax = float(np.maximum(t0, t1).min())
    if tmax < tmin: return float("inf"), float("-inf")
    return tmin, tmax


# ==============================
# DDA traversal (grid marching)
# ==============================

def traverse_ray_dda(C: np.ndarray, d: np.ndarray, grid: SparseVoxelGrid,
                     max_len_m: float, max_vox: int) -> Iterable[Tuple[Tuple[int,int,int], np.ndarray]]:
    """
    3D DDA voxel traversal: yields (voxel_idx, voxel_center) along the ray.
    """
    s = grid._size
    # Start at first voxel
    p = C.copy()
    # Advance to the first voxel boundary if we start inside
    idx = grid.world_to_idx(p)
    # Precompute stepping
    step = np.sign(d).astype(int)
    step[step == 0] = 1
    next_boundary = (grid.idx_to_center(idx) + (step * 0.5) * s)
    t_max = np.where(d != 0.0, (next_boundary - p) / d, float("inf"))
    t_delta = np.where(d != 0.0, (step * s) / d, float("inf"))

    # march
    t = 0.0
    for _ in range(max_vox):
        yield (tuple(idx.tolist()), grid.idx_to_center(tuple(idx.tolist())))
        # choose axis of smallest t_max
        axis = int(np.argmin(t_max))
        t = t_max[axis]
        if t > max_len_m:
            break
        idx[axis] += step[axis]
        t_max[axis] += t_delta[axis]


# ==============================
# Fusion orchestrator
# ==============================

class VoxelFusion:
    def __init__(self, cfg: VoxelFusionConfig):
        self.cfg = cfg
        self.grid = SparseVoxelGrid(cfg)

    # ---------- single-camera ray carving (unknown depth) ----------
    def carve_from_pixel(self,
                         K: np.ndarray, R: np.ndarray, t: np.ndarray,
                         uv: Tuple[float,float],
                         *,
                         conf: Optional[float] = None,
                         roi_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        March the back-projected ray through the grid, mark free space (miss) along the ray,
        and add a small positive "weak occupancy" along the centerline to encode directional
        evidence. This is classic ray carving with uncertainty-aware updates. 【turn12file1†L1-L13】
        """
        C, d = world_ray_from_obs(K, R, t, uv)
        self._carve_from_ray(C, d, conf=conf, roi_aabb=roi_aabb)

    def _carve_from_ray(self, C: np.ndarray, d: np.ndarray, *,
                        conf: Optional[float],
                        roi_aabb: Optional[Tuple[np.ndarray, np.ndarray]]):
        cfg = self.cfg
        now = time.monotonic()
        conf_w = float(conf) * cfg.conf_to_scale if conf is not None else 1.0

        # Optional ROI clipping
        tmin, tmax = 0.0, cfg.max_ray_len_m
        if roi_aabb is not None:
            tmn, tmx = ray_aabb_intersect(C, d, roi_aabb[0], roi_aabb[1])
            if not math.isfinite(tmn) or not math.isfinite(tmx) or (tmx < tmn):
                return  # miss ROI entirely
            tmin, tmax = tmn, min(tmx, tmax)

        # March and update
        for idx, center in traverse_ray_dda(C + d * tmin, d, self.grid, tmax - tmin, cfg.max_vox_per_ray):
            # miss (free) along the ray
            self.grid.add_L(idx, cfg.logodds_miss * conf_w, now)
            # weak positive along the centerline to encode bearing evidence
            self._spill_neighbors(idx, cfg.logodds_weak * conf_w, sigma_vox=cfg.sigma_ray_vox, now=now)

    # ---------- stereo / triangulation (known depth) ----------
    def fuse_triangulated(self,
                          Xw: np.ndarray,
                          cam_rays: List[Tuple[np.ndarray, np.ndarray]],
                          *,
                          min_pair_angle_deg: Optional[float] = None,
                          conf: Optional[float] = None):
        """
        Given triangulated point X̂ (world) and per-camera rays (C_i, d_i) that see it,
        mark free along each ray up to X̂, and add a strong localized occupancy bump near X̂. 【turn12file1†L15-L24】
        """
        cfg = self.cfg
        now = time.monotonic()
        conf_w = float(conf) * cfg.conf_to_scale if conf is not None else 1.0

        # Weight hit strength by ray intersection angle (shallower = less confident)
        ang_w = 1.0
        if min_pair_angle_deg is not None and min_pair_angle_deg < cfg.min_ray_angle_deg_for_strong_hit:
            ang_w = max(0.2, min_pair_angle_deg / cfg.min_ray_angle_deg_for_strong_hit)

        # Free space along each ray up to X̂
        for C, d in cam_rays:
            L = max(0.0, min(cfg.max_ray_len_m, float(np.linalg.norm(Xw - C))))
            vox = 0
            for idx, center in traverse_ray_dda(C, d, self.grid, L, cfg.max_vox_per_ray):
                self.grid.add_L(idx, cfg.logodds_miss * conf_w, now)
                vox += 1
                if vox >= cfg.max_vox_per_ray: break

        # Strong local hit around X̂
        idx_center = self.grid.world_to_idx(Xw)
        self._spill_neighbors(idx_center, cfg.logodds_hit * conf_w * ang_w,
                              sigma_vox=cfg.sigma_hit_vox, now=now)

    # ---------- utilities ----------
    def _spill_neighbors(self, idx_center: Tuple[int,int,int], dL_peak: float, *,
                         sigma_vox: float, now: float):
        """
        Deposit a small Gaussian kernel around a center voxel (for ray/point).
        """
        rmax = int(max(0, min(self.cfg.max_kernel_radius_vox, math.ceil(3.0 * sigma_vox))))
        if rmax == 0:
            self.grid.add_L(idx_center, dL_peak, now)
            return
        cx, cy, cz = idx_center
        for dx in range(-rmax, rmax + 1):
            for dy in range(-rmax, rmax + 1):
                for dz in range(-rmax, rmax + 1):
                    ijk = (cx + dx, cy + dy, cz + dz)
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    w = gaussian_weight(dist, sigma_vox)
                    if w < 1e-4: continue
                    self.grid.add_L(ijk, float(dL_peak * w), now)

    def update_and_prune(self):
        """Apply decay & prune unused cells; call occasionally (e.g., once per second)."""
        self.grid.prune()

    # Convenience queries
    def most_likely_point(self, min_logodds: float = 0.0) -> Optional[np.ndarray]:
        top = self.grid.top_k(k=1)
        if not top: return None
        (idx, cell) = top[0]
        if cell.L < min_logodds: return None
        return self.grid.idx_to_center(idx)

    def point_cloud(self, min_logodds: float = 0.0) -> np.ndarray:
        return self.grid.to_point_cloud(min_logodds=min_logodds)


# ==============================
# Example usage / smoke tests
# ==============================

if __name__ == "__main__":
    # Build a fusion map
    cfg = VoxelFusionConfig(voxel_size_m=5.0)
    fusion = VoxelFusion(cfg)

    # Fake cameras: identity K, one camera at origin looking +Z, one translated on X
    K = np.array([[800.0, 0, 640.0],
                  [0, 800.0, 360.0],
                  [0,   0,    1.0]])
    R1 = np.eye(3); t1 = np.zeros(3)
    R2 = np.eye(3); t2 = np.array([-1.0, 0.0, 0.0])  # C2 = -R^T t = +x
    # Single-cam carving
    fusion.carve_from_pixel(K, R1, t1, (640, 360), conf=0.6)

    # Stereo hit at ~ 3km out along +Z
    Xw = np.array([0.0, 0.0, 3000.0])
    C1 = camera_center_from_R_t(R1, t1)
    C2 = camera_center_from_R_t(R2, t2)
    d1 = _normalize(R1.T @ np.array([0,0,1.0]))
    d2 = _normalize(R2.T @ np.array([0,0,1.0]))
    fusion.fuse_triangulated(Xw, [(C1,d1), (C2,d2)], min_pair_angle_deg=2.0, conf=0.9)

    # Query
    p = fusion.most_likely_point()
    print("Top voxel center:", p)
    print("#points with L>=0:", fusion.point_cloud(0.0).shape[0])

