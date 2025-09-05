# app/policy.py
"""
Policy for converting image-space error into smooth motor commands.

Implements (from the doc’s control recipe):
- Gain that scales with error magnitude (large error -> bigger steps; small -> gentle) 
- Optional per-axis splitting and scale modulation
- Deadband, hysteresis, and rate limiting to avoid overshoot
- Optional small-angle conversion (Δθ ≈ Δx/f) using intrinsics K

Use:
    cfg = PolicyConfig()
    st  = PolicyState()
    # each frame:
    decision = decide_policy(
        st, cfg,
        prev_bbox=st.last_bbox, curr_bbox=bbox,
        error_px=(dx, dy), conf=conf
    )
    if K is not None:
        dpan, dtilt = command_angles(decision, K, dt=delta_time)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import time


# =========================
# Config & State
# =========================

@dataclass
class PolicyConfig:
    # Basic shaping
    deadband_px: float = 4.0
    error_breakpoints_px: Tuple[float, ...] = (0.0, 20.0, 60.0, 160.0, 400.0)
    gain_multipliers:   Tuple[float, ...] = (0.0, 0.6, 1.0, 1.2, 1.4)
    base_gain: float = 0.40
    max_gain:  float = 1.50
    ema_gain_alpha: float = 0.70          # smooth gain frame-to-frame

    # Axis behavior
    split_axes: bool = True               # weight x/y by their relative error
    scale_mod_ref: float = 120.0          # px (approx bbox height at "neutral"); smaller if zoomed in

    # Hysteresis / model selection via scale change
    delta_scale_small: float = 0.12       # |log(s2/s1)| thresholds
    delta_scale_large: float = 0.35
    model_hold_frames: int = 10           # frames to hold chosen model before switching again

    # Rate limit on angular command
    rate_limit_deg_s: float = 12.0

    # Confidence gating
    min_conf: float = 0.05


@dataclass
class PolicyState:
    model: str = "affine"                 # 'affine' | 'paraperspective' | 'projective'
    frames_since_switch: int = 0
    last_gain: float = 0.0
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_time_s: float = 0.0


# =========================
# Helpers
# =========================

def _bbox_scale(b: Optional[Tuple[int,int,int,int]]) -> float:
    """Proxy for apparent scale; bbox height works well with telescopes."""
    if not b:
        return 0.0
    return float(max(1, int(b[3])))

def _delta_scale(prev: Optional[Tuple[int,int,int,int]],
                 curr: Optional[Tuple[int,int,int,int]]) -> float:
    """
    Symmetric “size change” using log-ratio of heights.
    """
    s1, s2 = _bbox_scale(prev), _bbox_scale(curr)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    return abs(math.log(s2 / s1))

def _piecewise_gain(r: float,
                    xs: Tuple[float, ...],
                    ys: Tuple[float, ...]) -> float:
    """Linear piecewise interpolation; xs ascending."""
    assert len(xs) == len(ys) and len(xs) >= 2
    if r <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if r <= xs[i]:
            x0, x1 = xs[i-1], xs[i]
            y0, y1 = ys[i-1], ys[i]
            t = 0.0 if x1 == x0 else (r - x0) / (x1 - x0)
            return y0 * (1.0 - t) + y1 * t
    return ys[-1]

def _axis_weights(dx: float, dy: float) -> Tuple[float, float]:
    ax = abs(dx); ay = abs(dy)
    s = ax + ay
    if s <= 1e-9:
        return 0.5, 0.5
    return ax / s, ay / s

def _rate_limit_rad(dtheta_x: float, dtheta_y: float, dt: float, limit_deg_s: float) -> Tuple[float, float]:
    limit = math.radians(max(1e-6, limit_deg_s)) * max(1e-3, dt)
    return float(max(-limit, min(+limit, dtheta_x))), float(max(-limit, min(+limit, dtheta_y)))


# =========================
# Core policy
# =========================

def decide_policy(state: PolicyState,
                  cfg: PolicyConfig,
                  prev_bbox: Optional[Tuple[int,int,int,int]],
                  curr_bbox: Optional[Tuple[int,int,int,int]],
                  error_px: Tuple[float, float],
                  conf: Optional[float] = None,
                  now_s: Optional[float] = None) -> Dict:
    """
    Decide control model + gain for this frame given the error and bbox change.

    Returns:
      dict with keys:
        model: str
        gain:  float
        weights_xy: (wx, wy)
        error_px: (dx, dy)
        deadband: float
        scale: float
        delta_scale: float
    """
    if now_s is None:
        now_s = time.time()

    dx, dy = error_px
    r = math.hypot(dx, dy)
    s = _bbox_scale(curr_bbox)
    ds = _delta_scale(prev_bbox, curr_bbox)

    # Deadband → zero gain
    if r < cfg.deadband_px or (conf is not None and conf < cfg.min_conf):
        g = 0.0
    else:
        m = _piecewise_gain(r, cfg.error_breakpoints_px, cfg.gain_multipliers)
        # Scale modulation: when zoomed-in (large s), reduce effective multiplier
        scale_mod = min(1.0, cfg.scale_mod_ref / max(1.0, s))
        g = min(cfg.max_gain, cfg.base_gain * m * scale_mod)
        # Smooth gain
        g = cfg.ema_gain_alpha * state.last_gain + (1.0 - cfg.ema_gain_alpha) * g if state.last_gain > 0 else g

    # Axis weighting
    wx, wy = _axis_weights(dx, dy) if cfg.split_axes else (0.5, 0.5)

    # Model selection with hysteresis via ds
    model = state.model
    hold = state.frames_since_switch

    want = ("affine" if ds <= cfg.delta_scale_small
            else "projective" if ds >= cfg.delta_scale_large
            else "paraperspective")
    if hold >= cfg.model_hold_frames or want == model:
        if want != model:
            model = want
            hold = 0
        else:
            hold += 1
    else:
        hold += 1  # still holding the last model

    # Update state
    state.last_gain = g
    state.last_bbox = curr_bbox
    state.model = model
    state.frames_since_switch = hold
    state.last_time_s = now_s

    return dict(
        model=model,
        gain=g,
        weights_xy=(wx, wy),
        error_px=(dx, dy),
        deadband=cfg.deadband_px,
        scale=s,
        delta_scale=ds,
    )


# =========================
# Angle command (optional)
# =========================

def command_angles(decision: Dict,
                   K,
                   dt: float,
                   cfg: PolicyConfig) -> Tuple[float, float]:
    """
    Convert pixel error to small-angle pan/tilt deltas and apply rate limits.

    Args:
      decision: output from decide_policy()
      K: 3x3 intrinsics (fx, fy)
      dt: seconds since last command
    Returns:
      (dpan_rad, dtilt_rad)
    """
    dx, dy = decision["error_px"]
    fx, fy = float(K[0, 0]), float(K[1, 1])

    # Small-angle: Δθ ≈ Δx / f (radians)
    dtx = dx / max(1e-6, fx)
    dty = dy / max(1e-6, fy)

    # Apply gain and axis weights
    g = decision["gain"]
    wx, wy = decision["weights_xy"]
    dtx *= g * wx
    dty *= g * wy

    # Rate limit
    return _rate_limit_rad(dtx, dty, dt, cfg.rate_limit_deg_s)

