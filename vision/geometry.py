"""
geometry.py
Robust camera geometry utilities:
- Intrinsics (K) handling + zoom-LUT interpolation
- Encoder-driven extrinsics: pan/tilt -> R, with optional lever-arm offset
- Pixel <-> normalized ray <-> small-angle az/el
- P = K [R | t] projection and back-projection
- Two-ray triangulation (closest point between skew lines)

Conventions:
- Pixels are (x, y) with origin at top-left.
- Normalized camera rays live in camera coords (z > 0).
- Pan = azimuth (rotate about +Z_world), Tilt = elevation (rotate about +X’ or +Y’ depending on mount).
  Here we assume: world Z up, world X to the right, world Y forward; camera looks along +Z_cam.
  Adjust axes to your rig if needed.

Dependencies: numpy only (OpenCV optional for dist/undist in other modules).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional
import numpy as np


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0,      self.cx],
                         [0.0,      self.fy,  self.cy],
                         [0.0,      0.0,      1.0]], dtype=np.float64)

    @staticmethod
    def from_matrix(K: np.ndarray) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=np.float64)
        return CameraIntrinsics(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])


@dataclass
class CameraExtrinsics:
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3x1

    def P(self, K: np.ndarray) -> np.ndarray:
        """Projection matrix P = K [R | t]."""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return K @ Rt


# -----------------------------
# Intrinsics helpers (zoom LUT)
# -----------------------------

def interpolate_intrinsics(zoom_steps: float,
                           lut: Dict[float, Tuple[float, float, float, float]]) -> CameraIntrinsics:
    """
    Linearly interpolate (fx, fy, cx, cy) from a small zoom-step LUT.
    The LUT keys are step positions; values are (fx, fy, cx, cy).

    Guidance: build a tiny LUT over 3–5 zoom points and interpolate K at runtime. :contentReference[oaicite:0]{index=0}
    """
    if not lut:
        raise ValueError("Zoom intrinsics LUT is empty")

    # Sort by step
    steps = np.array(sorted(lut.keys()), dtype=np.float64)
    vals = np.array([lut[s] for s in steps], dtype=np.float64)  # (N,4)

    # Clamp if out of range
    if zoom_steps <= steps[0]:
        fx, fy, cx, cy = vals[0]
        return CameraIntrinsics(fx, fy, cx, cy)
    if zoom_steps >= steps[-1]:
        fx, fy, cx, cy = vals[-1]
        return CameraIntrinsics(fx, fy, cx, cy)

    # Find neighbors
    i_hi = int(np.searchsorted(steps, zoom_steps, side="right"))
    i_lo = i_hi - 1
    s0, s1 = steps[i_lo], steps[i_hi]
    t = float((zoom_steps - s0) / (s1 - s0))
    fx, fy, cx, cy = (1 - t) * vals[i_lo] + t * vals[i_hi]
    return CameraIntrinsics(float(fx), float(fy), float(cx), float(cy))


# -----------------------------
# Rotations (pan/tilt) and lever-arm
# -----------------------------

def Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s ],
                     [0.0,  s,  c ]], dtype=np.float64)

def pan_tilt_to_R(pan_rad: float, tilt_rad: float) -> np.ndarray:
    """
    Simple mount: pan about world +Z, then tilt about camera’s local +X (post-pan).
    If your gimbal’s axes differ, swap Rx/Ry or order accordingly.
    """
    return Rx(tilt_rad) @ Rz(pan_rad)


def extrinsics_from_mount(pivot_world: Iterable[float],
                          offset_cam_from_pivot: Iterable[float],
                          pan_rad: float,
                          tilt_rad: float) -> CameraExtrinsics:
    """
    Compute camera extrinsics given a mount pivot and a fixed lever-arm from pivot to camera center.

    World camera center: C = P + R * o
    Then t = -R * C_world  (if using world->camera: X_cam = R*(X_world) + t)
    The doc’s guidance models this to keep [R|t] consistent while slewing. :contentReference[oaicite:1]{index=1}
    """
    P = np.asarray(pivot_world, dtype=np.float64).reshape(3)
    o = np.asarray(offset_cam_from_pivot, dtype=np.float64).reshape(3)

    R = pan_tilt_to_R(pan_rad, tilt_rad)
    C_world = P + R @ o  # camera center in world
    # Using X_cam = R_wc^T (X_world - C_world). Here take R_cw = R^T; pack as [R_cw | -R_cw*C].
    R_cw = R.T
    t = -R_cw @ C_world
    return CameraExtrinsics(R=R_cw, t=t)


# -----------------------------
# Pixel <-> rays / angles
# -----------------------------

def normalize_pixel(K: np.ndarray, px: Tuple[float, float]) -> np.ndarray:
    """
    Convert pixel to normalized coords: [x_n, y_n, 1] = K^{-1} [x, y, 1].
    This is used before forming a camera-ray or small-angle az/el. :contentReference[oaicite:2]{index=2}
    """
    x, y = px
    Kinv = np.linalg.inv(K)
    v = Kinv @ np.array([x, y, 1.0], dtype=np.float64)
    return v  # (x_n, y_n, 1)

def ray_from_pixel(K: np.ndarray, px: Tuple[float, float]) -> np.ndarray:
    """
    Back-project a pixel to a unit ray in camera coordinates (z>0).
    """
    xn, yn, _ = normalize_pixel(K, px)
    v = np.array([xn, yn, 1.0], dtype=np.float64)
    return v / np.linalg.norm(v)

def angles_from_pixel(K: np.ndarray, px: Tuple[float, float]) -> Tuple[float, float]:
    """
    Small-angle az/el (radians) using atan on normalized coords:
    theta_x = atan(x_n), theta_y = atan(y_n). :contentReference[oaicite:3]{index=3}
    """
    xn, yn, _ = normalize_pixel(K, px)
    return float(np.arctan(xn)), float(np.arctan(yn))

def pixel_from_angles(K: np.ndarray, theta_x: float, theta_y: float) -> Tuple[float, float]:
    """
    Inverse of angles_from_pixel under small-angle model: x_n = tan(theta_x), y_n = tan(theta_y).
    """
    xn = np.tan(theta_x)
    yn = np.tan(theta_y)
    x = K[0, 0] * xn + K[0, 2]
    y = K[1, 1] * yn + K[1, 2]
    return float(x), float(y)


# -----------------------------
# Projection utilities
# -----------------------------

def project_points(P: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """
    Project 3D world points with P = K [R | t].
    X_world: (N,3)
    Returns pixel coords (N,2).
    """
    X = np.asarray(X_world, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])  # (N,4)
    x_h = (P @ X_h.T).T  # (N,3)
    x = x_h[:, :2] / x_h[:, 2:3]
    return x

def camera_ray_in_world(extr: CameraExtrinsics, px_ray_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a unit ray in camera coords, return (origin, direction) in world coords.
    origin = camera center C_world; direction = R_wc^T @ ray_cam (but our extr stores R_cw, so use R_wc = R_cw^T).
    """
    R_cw = extr.R
    R_wc = R_cw.T
    # Recover camera center from t: t = -R_cw * C_world  => C_world = -R_wc * t
    C_world = -R_wc @ extr.t
    d_world = R_wc @ px_ray_cam
    d_world = d_world / np.linalg.norm(d_world)
    return C_world, d_world


# -----------------------------
# Two-ray triangulation (closest point between skew lines)
# -----------------------------

def triangulate_two_rays(o1: np.ndarray, d1: np.ndarray,
                         o2: np.ndarray, d2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the point minimizing distance to both rays L1: o1 + a*d1, L2: o2 + b*d2.
    Returns (X_hat, pairwise_distance).

    If lines are nearly parallel, returns midpoint between closest points with larger uncertainty.

    Notes from doc: multi-view depth via intersecting back-projected rays; sharpens depth vs single-ray carving. :contentReference[oaicite:4]{index=4}
    """
    o1 = np.asarray(o1, dtype=np.float64).reshape(3)
    d1 = np.asarray(d1, dtype=np.float64).reshape(3)
    o2 = np.asarray(o2, dtype=np.float64).reshape(3)
    d2 = np.asarray(d2, dtype=np.float64).reshape(3)

    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    r = o1 - o2

    a11 = np.dot(d1, d1)      # =1
    a22 = np.dot(d2, d2)      # =1
    a12 = np.dot(d1, d2)
    b1 = -np.dot(d1, r)
    b2 =  np.dot(d2, r)

    denom = (a11*a22 - a12*a12)
    if abs(denom) < 1e-9:
        # Nearly parallel: project midpoint between orthogonal projections onto a bisector
        # Fallback: pick midpoint between closest points along vector perpendicular to d1
        # Use component orthogonal to d1
        n = r - np.dot(r, d1) * d1
        if np.linalg.norm(n) < 1e-12:
            # Rays originate almost same line; pick halfway point
            X = (o1 + o2) * 0.5
            return X, float(np.linalg.norm(o1 - o2))
        # Closest points approx:
        o2_to_line1 = o2 + n
        X = 0.5 * (o1 + o2_to_line1)
        return X, float(np.linalg.norm(n))

    a = ( b1*a22 - b2*a12) / denom
    b = (-b1*a12 + b2*a11) / denom
    p1 = o1 + a * d1
    p2 = o2 + b * d2
    X = 0.5 * (p1 + p2)
    dist = float(np.linalg.norm(p1 - p2))
    return X, dist


# -----------------------------
# Convenience wrappers
# -----------------------------

def build_P_from_pan_tilt_K(pivot_world: Iterable[float],
                            offset_cam_from_pivot: Iterable[float],
                            pan_rad: float,
                            tilt_rad: float,
                            K: np.ndarray) -> Tuple[np.ndarray, CameraExtrinsics]:
    """
    Build P and extrinsics from mount state + K.
    """
    extr = extrinsics_from_mount(pivot_world, offset_cam_from_pivot, pan_rad, tilt_rad)
    return extr.P(K), extr


def pixel_to_world_ray(px: Tuple[float, float], K: np.ndarray, extr: CameraExtrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pixel -> unit ray in world coords (origin, direction).
    """
    v_cam = ray_from_pixel(K, px)
    return camera_ray_in_world(extr, v_cam)


# -----------------------------
# Minimal self-test (optional)
# -----------------------------

if __name__ == "__main__":
    # Quick sanity: round-trip angles <-> pixels
    intr = CameraIntrinsics(1000.0, 1000.0, 640.0, 360.0)
    K = intr.K
    thx, thy = 5.0*np.pi/180.0, -2.0*np.pi/180.0
    px = pixel_from_angles(K, thx, thy)
    thx2, thy2 = angles_from_pixel(K, px)
    assert abs(thx - thx2) < 1e-9 and abs(thy - thy2) < 1e-9

