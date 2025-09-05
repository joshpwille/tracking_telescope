# vision/triangulation.py
"""
Triangulation utilities: matched 2D detections across cameras → 3D point.

Pipeline match:
- Back-project pixels with K^{-1} → unit rays in camera frame; map to world via R^T. 【turn11file1†L41-L48】
- Use each camera’s center C (from P = K[R|t]) to form lines ℓ_i(λ)=C_i + λ d_i. 【turn11file3†L13-L17】
- Solve closest-point across 2 rays (midpoint) or N>=2 rays (least-squares).
- Optional Gauss–Newton refinement to minimize reprojection error.
- Return 3D X with quality metrics and per-camera reprojection errors.

Assumptions/conventions
- R is world→camera, t is camera translation in the same convention as P=K[R|t].
- Camera center in world: C = -R^T t.
- Pixels are undistorted (apply your undistort/rectify first if needed).
- Keep K updated when zoom changes; keep R (and lever-arm t) updated from encoders
  before triangulating, as the doc stresses for stable depth. 【turn11file0†L10-L22】

Public API
- triangulate_observations(observations, refine=True, reproj_sigma_px=1.0, ...)
  observations: list of dicts, each with:
    {"uv": (x,y), "K": (3x3), "R": (3x3), "t": (3,) or "C": (3,),
     "weight": optional float (e.g., detection conf)}
- project_point(K,R,t,X) -> (u,v), depth
- rays_from_observations(observations) -> list of (C_i, d_i)

Returns
  {
    "X": np.ndarray shape (3,),                # 3D point (world)
    "method": "rays_ls" | "pair_midpoint" | "gn_refine",
    "per_cam_err_px": [float, ...],            # reprojection errors
    "rms_err_px": float,
    "min_ray_angle_deg": float,                # conditioning (pair or worst pair)
    "in_front_count": int,                     # cheirality
    "success": bool
  }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# -----------------------------
# Basic ops
# -----------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n

def camera_center_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """C_world = -R^T t for P = K [R|t]."""
    R = np.asarray(R, dtype=float).reshape(3,3)
    t = np.asarray(t, dtype=float).reshape(3)
    return -R.T @ t

def bearing_from_pixel(K: np.ndarray, uv: Tuple[float,float]) -> np.ndarray:
    """
    Back-project pixel to a unit ray in camera frame: d_cam ∝ K^{-1} [u v 1]^T. 【turn11file1†L41-L48】
    """
    K = np.asarray(K, dtype=float).reshape(3,3)
    u, v = float(uv[0]), float(uv[1])
    x = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    return _normalize(x)

def world_ray_from_obs(obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (C_world, d_world) for one observation dict.
    Accepts:
      - "C" (preferred) OR "t" with "R".
    """
    K = np.asarray(obs["K"], dtype=float).reshape(3,3)
    R = np.asarray(obs["R"], dtype=float).reshape(3,3)
    if "C" in obs:
        C = np.asarray(obs["C"], dtype=float).reshape(3)
    elif "t" in obs:
        C = camera_center_from_R_t(R, np.asarray(obs["t"], dtype=float))
    else:
        raise ValueError("obs needs 'C' or 't' along with 'R'")
    d_cam = bearing_from_pixel(K, obs["uv"])
    d_world = _normalize(R.T @ d_cam)  # map camera ray to world
    return C, d_world

def project_point(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> Tuple[Tuple[float,float], float]:
    """
    Project X (world) with P = K [R|t] to pixel and return ((u,v), depth_camZ).
    """
    R = np.asarray(R, dtype=float).reshape(3,3)
    t = np.asarray(t, dtype=float).reshape(3,1)
    Xw = np.asarray(X, dtype=float).reshape(3,1)
    Xc = R @ Xw + t
    z = float(Xc[2,0])
    if abs(z) < 1e-9:
        return (float("nan"), float("nan")), z
    x = Xc[:2,0] / z
    uv = (K @ np.array([x[0], x[1], 1.0])).reshape(3)
    return (float(uv[0]), float(uv[1])), z


# -----------------------------
# Ray–ray: midpoint / N-ray least squares
# -----------------------------

def _closest_point_pair(C1, d1, C2, d2) -> Tuple[np.ndarray, float]:
    """
    Closest point between two skew lines; return midpoint and min angle (deg).
    """
    d1 = _normalize(d1); d2 = _normalize(d2)
    r = C2 - C1
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    denom = (a*c - b*b)
    if abs(denom) < 1e-12:
        # nearly parallel → pick mid along average direction
        X = 0.5 * (C1 + C2)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, np.dot(d1, d2)))))
        return X, angle
    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom
    P1 = C1 + s * d1
    P2 = C2 + t * d2
    X = 0.5 * (P1 + P2)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, np.dot(d1, d2)))))
    return X, angle

def _least_squares_rays(Cs: List[np.ndarray], ds: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Solve argmin_X Σ_i || (I - d_i d_i^T) (X - C_i) ||_2^2  (closest to all lines).
    Closed-form normal equations: (Σ A_i) X = Σ A_i C_i, with A_i = W_i (I - d_i d_i^T) W_i.
    """
    n = len(Cs)
    if n < 2:
        raise ValueError("need at least 2 rays")
    A = np.zeros((3,3), dtype=float)
    b = np.zeros(3, dtype=float)
    for i in range(n):
        d = _normalize(ds[i]).reshape(3,1)
        C = Cs[i].reshape(3)
        w = float(weights[i]) if (weights is not None) else 1.0
        Pi = np.eye(3) - d @ d.T
        Ai = w * Pi
        A += Ai
        b += Ai @ C
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, b, rcond=None)[0]
    return X


# -----------------------------
# Main entry: triangulate observations
# -----------------------------

@dataclass
class TriangulationResult:
    X: np.ndarray
    method: str
    per_cam_err_px: List[float]
    rms_err_px: float
    min_ray_angle_deg: float
    in_front_count: int
    success: bool

def triangulate_observations(
    observations: List[Dict],
    *,
    refine: bool = True,
    reproj_sigma_px: float = 1.0,
    min_ok_ray_angle_deg: float = 0.2,
) -> TriangulationResult:
    """
    observations: list of dicts with keys "uv","K","R", and ("t" or "C"), optional "weight"
    Returns TriangulationResult with cheirality & residuals.
    """
    if len(observations) < 2:
        raise ValueError("Need at least two observations for triangulation")

    # Build rays in world frame
    rays = [world_ray_from_obs(o) for o in observations]
    Cs  = [r[0] for r in rays]
    ds  = [r[1] for r in rays]
    wts = [float(o.get("weight", 1.0)) for o in observations]

    # Initial estimate
    if len(observations) == 2:
        X0, angle = _closest_point_pair(Cs[0], ds[0], Cs[1], ds[1])
        min_ang = angle
        method = "pair_midpoint"
    else:
        X0 = _least_squares_rays(Cs, ds, wts)
        # compute minimum pairwise angle for conditioning
        min_ang = 180.0
        for i in range(len(ds)):
            for j in range(i+1, len(ds)):
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_normalize(ds[i]), _normalize(ds[j])))))))
                min_ang = min(min_ang, ang)
        method = "rays_ls"

    # If rays are nearly parallel, note poor conditioning
    if min_ang < min_ok_ray_angle_deg:
        refine = False  # refinement won’t help much; still report

    # Optional Gauss–Newton reprojection refinement
    X = X0.copy()
    if refine:
        X = _gauss_newton_refine(X0, observations, wts, reproj_sigma_px)
        method = "gn_refine"

    # Residuals & cheirality
    per_err = []
    in_front = 0
    for o in observations:
        uv_hat, z = project_point(o["K"], o["R"], o["t"] if "t" in o else -o["R"] @ o["C"], X)
        if z > 0:
            in_front += 1
        per_err.append(float(math.hypot(uv_hat[0] - o["uv"][0], uv_hat[1] - o["uv"][1])))

    rms = float(math.sqrt(max(1e-12, sum(e*e for e in per_err) / len(per_err))))

    return TriangulationResult(
        X=X, method=method, per_cam_err_px=per_err, rms_err_px=rms,
        min_ray_angle_deg=min_ang, in_front_count=in_front,
        success=bool(in_front >= 2)
    )


# -----------------------------
# Gauss–Newton on reprojection error
# -----------------------------

def _gauss_newton_refine(X0: np.ndarray, observations: List[Dict], wts: List[float], sigma_px: float) -> np.ndarray:
    """
    Minimize Σ_i w_i || π_i(X) - u_i ||^2 with 3–5 iterations; π_i uses P_i = K[R|t].
    """
    X = X0.reshape(3,1).astype(float)
    for _ in range(5):
        J_list = []
        r_list = []
        for i, o in enumerate(observations):
            K = np.asarray(o["K"], dtype=float).reshape(3,3)
            R = np.asarray(o["R"], dtype=float).reshape(3,3)
            t = (np.asarray(o["t"], dtype=float).reshape(3,1) if "t" in o else -R @ np.asarray(o["C"], dtype=float).reshape(3,1))
            Xc = R @ X + t
            x, y, z = Xc[0,0], Xc[1,0], Xc[2,0]
            if z <= 1e-8:
                continue  # ignore behind-camera residuals
            # d(u,v)/d(Xc)
            du_dXc = np.array([[1/z, 0, -x/(z*z)],
                               [0, 1/z, -y/(z*z)]], dtype=float)
            # dXc/dX = R
            J_cam = du_dXc @ R
            # Pixel Jacobian: K on homogeneous normalized coords
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            J_pix = np.array([[fx, 0, 0],
                              [0, fy, 0]], dtype=float) @ J_cam
            # residual r = u_meas - u_hat
            uhat = np.array([fx * x/z + cx, fy * y/z + cy]).reshape(2,1)
            umea = np.array([o["uv"][0], o["uv"][1]]).reshape(2,1)
            r = (umea - uhat)
            w = max(1e-6, float(wts[i])) / (sigma_px**2)
            J_list.append(math.sqrt(w) * J_pix)
            r_list.append(math.sqrt(w) * r)
        if not J_list:
            break
        J = np.vstack(J_list)      # (2M x 3)
        r = np.vstack(r_list)      # (2M x 1)
        try:
            # Solve (J^T J) δ = J^T r
            H = J.T @ J
            g = J.T @ r
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, r, rcond=None)[0]
        X += delta
        if float(np.linalg.norm(delta)) < 1e-6:
            break
    return X.reshape(3)


# -----------------------------
# Convenience: build inputs from matched tracks
# -----------------------------

def rays_from_observations(observations: List[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return [(C_i, d_i)] for debugging/visualization."""
    return [world_ray_from_obs(o) for o in observations]

