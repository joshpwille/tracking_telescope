# app/pose_update.py
"""
Pose update utilities:
- Read encoders (pan/tilt) and zoom steps each frame
- Smooth & clamp rates for robustness
- Compute extrinsics from a pivot + lever-arm offset (if any)
- Interpolate intrinsics K from a zoom LUT
- Return K, extrinsics [R|t], and projection P = K [R | t]

Matches the doc’s guidance: update R from encoders every frame, model t via
a small lever arm when needed, and update K from a zoom LUT so geometry
stays consistent while slewing/zooming. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math
import time
import numpy as np

# --- geometry helpers (local module) ---
from vision.geometry import (
    CameraIntrinsics,
    CameraExtrinsics,
    interpolate_intrinsics,
    extrinsics_from_mount,
)

# --- optional IO loaders; kept optional to not hard-crash in tests ---
try:
    from io.calib_loader import load_intrinsics  # -> dict or 3x3 matrix
except Exception:
    load_intrinsics = None

try:
    from io.zoom_lut_loader import load_zoom_lut  # -> {steps: (fx,fy,cx,cy)}
except Exception:
    load_zoom_lut = None


# =========================
# Config / State / Result
# =========================

@dataclass
class PoseConfig:
    # Static calibration
    calib_path: str                       # e.g., "io/calib_left.yml"
    zoom_lut_path: Optional[str] = None   # e.g., "io/zoom_lut_left.json"

    # Mount geometry for t modeling (lever arm)
    pivot_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_cam_from_pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Smoothing / limits
    pan_tilt_alpha: float = 0.5           # EMA for angles
    max_rate_deg_s: float = 90.0          # clamp per-frame change
    zoom_alpha: float = 0.5               # EMA for zoom steps

    # Safety / sanity
    zoom_min_steps: Optional[float] = None
    zoom_max_steps: Optional[float] = None


@dataclass
class PoseState:
    # raw sensors (last seen)
    pan_raw: float = 0.0
    tilt_raw: float = 0.0
    steps_raw: float = 0.0

    # filtered
    pan: float = 0.0
    tilt: float = 0.0
    steps: float = 0.0

    # time bookkeeping
    t_last: float = 0.0

    # products
    K: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    P: Optional[np.ndarray] = None

    # LUT cache
    zoom_lut: Optional[Dict[float, Tuple[float, float, float, float]]] = None


@dataclass
class PoseResult:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray
    pan_rad: float
    tilt_rad: float
    steps: float


# =========================
# Public API
# =========================

def init_pose(cfg: PoseConfig) -> PoseState:
    """
    Loads base intrinsics and (optionally) the zoom LUT. Initializes filtered state.
    """
    st = PoseState()

    if load_intrinsics is None:
        raise RuntimeError("io.calib_loader.load_intrinsics not available")

    intr = load_intrinsics(cfg.calib_path)
    if isinstance(intr, dict):
        st.intrinsics = CameraIntrinsics(intr["fx"], intr["fy"], intr["cx"], intr["cy"])
        st.K = st.intrinsics.K
    else:
        K = np.asarray(intr, dtype=np.float64)
        st.intrinsics = CameraIntrinsics.from_matrix(K)
        st.K = K

    if cfg.zoom_lut_path and load_zoom_lut is not None:
        st.zoom_lut = load_zoom_lut(cfg.zoom_lut_path)

    st.t_last = time.time()
    return st


def update_pose(cfg: PoseConfig,
                st: PoseState,
                *,
                read_pan_tilt_rad,     # callable -> (pan_rad, tilt_rad)
                read_zoom_steps=None   # callable -> steps (float) or None
                ) -> PoseResult:
    """
    One per frame: read sensors → smooth & clamp → compute K and [R|t] → build P.

    The lever-arm model computes camera center C = P + R o, then t is derived so
    P = K [R | t] maps world → pixels consistently during slews. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
    """
    # --- time delta ---
    t_now = time.time()
    dt = max(1e-3, t_now - (st.t_last or t_now))
    st.t_last = t_now

    # --- read sensors ---
    pan_raw, tilt_raw = read_pan_tilt_rad()
    st.pan_raw, st.tilt_raw = float(pan_raw), float(tilt_raw)

    if read_zoom_steps is not None:
        z_raw = float(read_zoom_steps())
        # sanity clamp if configured
        if cfg.zoom_min_steps is not None and z_raw < cfg.zoom_min_steps:
            z_raw = cfg.zoom_min_steps
        if cfg.zoom_max_steps is not None and z_raw > cfg.zoom_max_steps:
            z_raw = cfg.zoom_max_steps
        st.steps_raw = z_raw

    # --- smooth & rate-limit angles ---
    if st.pan == 0.0 and st.tilt == 0.0 and st.t_last == t_now:
        # first call: seed filtered values
        st.pan, st.tilt = st.pan_raw, st.tilt_raw
    else:
        a = float(cfg.pan_tilt_alpha)
        st.pan  = a * st.pan  + (1 - a) * st.pan_raw
        st.tilt = a * st.tilt + (1 - a) * st.tilt_raw

        # clamp per-frame delta to avoid encoder spikes
        max_step = math.radians(cfg.max_rate_deg_s) * dt
        st.pan  = _clamp_rate(st.pan,  st.pan_raw,  max_step)
        st.tilt = _clamp_rate(st.tilt, st.tilt_raw, max_step)

    # --- smooth zoom steps ---
    if read_zoom_steps is not None:
        if st.steps == 0.0:
            st.steps = st.steps_raw
        else:
            st.steps = cfg.zoom_alpha * st.steps + (1 - cfg.zoom_alpha) * st.steps_raw

    # --- intrinsics from LUT (if present) ---
    if st.zoom_lut is not None and read_zoom_steps is not None:
        intr = interpolate_intrinsics(st.steps, st.zoom_lut)  # K(steps)
        st.intrinsics = intr
        st.K = intr.K
        # Keeping K updated under zoom is crucial to avoid depth drift. :contentReference[oaicite:4]{index=4}
    else:
        # base K already loaded at init
        intr = st.intrinsics

    # --- extrinsics from mount (lever arm if any) ---
    extr = extrinsics_from_mount(
        cfg.pivot_world,
        cfg.offset_cam_from_pivot,
        st.pan,
        st.tilt
    )
    st.extrinsics = extr  # [R_cw | t_c]

    # --- projection matrix ---
    P = extr.P(st.K)
    st.P = P

    return PoseResult(
        K=st.K,
        R=extr.R,
        t=extr.t,
        P=P,
        pan_rad=st.pan,
        tilt_rad=st.tilt,
        steps=st.steps
    )


# =========================
# Helpers
# =========================

def _clamp_rate(filtered_val: float, raw_val: float, max_step: float) -> float:
    """
    Bring filtered value toward raw by at most max_step per update.
    """
    dv = raw_val - filtered_val
    if abs(dv) <= max_step:
        return raw_val
    return filtered_val + math.copysign(max_step, dv)


# =========================
# Example wiring (optional)
# =========================

if __name__ == "__main__":
    # Dummy sensor functions for a quick smoke test
    class DummySensors:
        def __init__(self): self.t = 0.0
        def read_pan_tilt_rad(self):
            self.t += 0.02
            return 0.3 * math.sin(self.t), 0.15 * math.cos(self.t)
        def read_zoom_steps(self):
            return 1000.0 + 400.0 * math.sin(self.t * 0.2)

    sensors = DummySensors()
    cfg = PoseConfig(
        calib_path="io/calib_left.yml",
        zoom_lut_path=None,
        pivot_world=(0.0, 0.0, 0.0),
        offset_cam_from_pivot=(0.0, 0.0, 0.0),
    )
    st = init_pose(cfg)

    # shim lambdas to match update_pose signature
    for _ in range(5):
        res = update_pose(
            cfg, st,
            read_pan_tilt_rad=sensors.read_pan_tilt_rad,
            read_zoom_steps=sensors.read_zoom_steps
        )
        print(f"K(0,0)={res.K[0,0]:.1f} pan={math.degrees(res.pan_rad):.2f}° tilt={math.degrees(res.tilt_rad):.2f}°")

