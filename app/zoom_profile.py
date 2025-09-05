# app/zoom_profile.py
"""
Zoom profile manager.

What it does
- Supports zoom modes:
    * "fixed": hold a specific motor step position
    * "band": keep the active target's bbox height within [min_px, max_px]
    * "fov": aim for a desired horizontal FOV (deg)
- Adds deadband + hysteresis to avoid zoom "hunting"
- Rate-limits step changes (steps/s) and enforces soft limits
- Gating: only zoom if (a) detection confidence is high enough and
  (b) the center error is small enough (so you don't zoom while off-target)
- Optional intrinsics update: interpolates K from a zoom→intrinsics LUT
  whenever steps change, so downstream geometry stays correct while zooming

Interfaces nicely with:
- hardware.zoom_motor.ZoomMotor (expects read_steps(), goto_steps())
- io.zoom_lut_loader (load_zoom_lut, interpolate_zoom_lut)
- policy.py / pose_update.py (consume K updates if you wire the callback)

Why this matches the PDF:
- Uses a zoom→K LUT and interpolates K(steps) whenever zoom changes, because
  zooming changes f (fx, fy) and you must keep K current to avoid depth drift. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
- The runtime loop is: track targets → choose an ID → update R from encoders →
  select the correct zoom K profile and keep the object centered/within a
  scale band—exactly the outlined control flow. :contentReference[oaicite:5]{index=5}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import time

try:
    from io.zoom_lut_loader import load_zoom_lut, interpolate_zoom_lut
except Exception:
    # Make LUT optional. You can still use "band" and "fixed" without K updates.
    load_zoom_lut = None   # type: ignore
    interpolate_zoom_lut = None  # type: ignore


# =========================
# Data classes
# =========================

@dataclass
class BandConfig:
    min_px: int = 80           # keep bbox height >= min_px
    max_px: int = 220          # and <= max_px
    dead_px: int = 12          # deadband inside the band before commanding zoom
    px_ema_alpha: float = 0.5  # smooth bbox height frame-to-frame

@dataclass
class FovConfig:
    target_fov_deg: float = 3.0     # desired horizontal FOV
    fov_dead_deg: float = 0.15      # deadband

@dataclass
class GateConfig:
    min_conf: float = 0.20          # don't zoom if detection too uncertain
    max_center_err_px: int = 120    # don't zoom while way off-center (let pan/tilt fix first)

@dataclass
class LimitsConfig:
    min_steps: Optional[int] = 0
    max_steps: Optional[int] = None
    max_rate_steps_s: float = 2500.0   # cap command rate
    min_step_delta: int = 4            # ignore tiny step changes (mechanical slack)

@dataclass
class ZoomProfileConfig:
    mode: str = "band"                 # "band" | "fixed" | "fov"
    fixed_steps: Optional[int] = None
    band: BandConfig = BandConfig()
    fov: FovConfig = FovConfig()
    gates: GateConfig = GateConfig()
    limits: LimitsConfig = LimitsConfig()
    change_cooldown_s: float = 0.15    # minimum time between new targets or big changes

@dataclass
class ZoomProfileState:
    last_cmd_steps: Optional[float] = None
    last_k: Optional[Tuple[float, float, float, float]] = None  # fx, fy, cx, cy
    bbox_h_ema: Optional[float] = None
    last_change_t: float = 0.0


# =========================
# Main manager
# =========================

class ZoomProfile:
    """
    Orchestrates zoom setpoints based on the current mode and track readouts.
    """
    def __init__(self,
                 motor,                              # object with read_steps(), goto_steps()
                 cfg: ZoomProfileConfig = ZoomProfileConfig(),
                 *,
                 lut_path: Optional[str] = None,     # path to zoom LUT (JSON/YAML/CSV)
                 image_size: Optional[Tuple[int,int]] = None,  # (W,H) for FOV math
                 on_intrinsics: Optional[Callable[[Dict[str, float], float], None]] = None  # cb(K_dict, steps)
                 ):
        self.motor = motor
        self.cfg = cfg
        self.image_size = image_size
        self.on_intrinsics = on_intrinsics
        self.state = ZoomProfileState()
        self._lut = None
        if lut_path and load_zoom_lut is not None:
            self._lut = load_zoom_lut(lut_path)  # {steps: (fx,fy,cx,cy)}

    # ---------- public API ----------

    def update(self,
               *,
               bbox_h_px: Optional[float],
               conf: Optional[float],
               center_err_px: Optional[Tuple[float,float]] = None,
               now: Optional[float] = None) -> Dict:
        """
        Call once per frame. Decides a new zoom target if needed and commands the motor.

        Args:
          bbox_h_px: current target bbox height (pixels) or None if no target
          conf: detection confidence (0..1) or None
          center_err_px: (dx, dy) from image center to target center; used for gating
        Returns:
          dict with summary of the decision and (optionally) K update
        """
        t = time.time() if now is None else now
        curr = float(self.motor.read_steps())  # current steps
        changed = False
        new_steps: Optional[float] = None

        # --- gates ---
        if not self._passes_gates(bbox_h_px, conf, center_err_px):
            # Too uncertain or not centered; don't zoom this frame
            # (You still might want to keep K current to this steps.)
            k = self._maybe_intrinsics(curr, force=False)
            return dict(mode=self.cfg.mode, steps=curr, action="gate_hold", K=k)

        # --- decide by mode ---
        if self.cfg.mode == "fixed":
            target = self._clamp(self.cfg.fixed_steps if self.cfg.fixed_steps is not None else curr)
            new_steps = target

        elif self.cfg.mode == "band":
            new_steps = self._decide_band(curr, bbox_h_px, t)

        elif self.cfg.mode == "fov":
            new_steps = self._decide_fov(curr)

        else:
            # Unknown mode: hold
            k = self._maybe_intrinsics(curr, force=False)
            return dict(mode=self.cfg.mode, steps=curr, action="unknown_mode_hold", K=k)

        # --- rate limit & min delta ---
        new_steps = self._rate_limit(curr, new_steps, t)
        if self.state.last_cmd_steps is not None:
            if abs(new_steps - self.state.last_cmd_steps) < self.cfg.limits.min_step_delta:
                k = self._maybe_intrinsics(curr, force=False)
                return dict(mode=self.cfg.mode, steps=curr, action="deadband_hold", K=k)

        # --- command motor ---
        self.motor.goto_steps(new_steps)
        self.state.last_cmd_steps = new_steps
        self.state.last_change_t = t
        changed = True

        # --- intrinsics update (optional) ---
        k = self._maybe_intrinsics(new_steps, force=True)

        return dict(mode=self.cfg.mode, steps=new_steps,
                    action="cmd" if changed else "hold", K=k)

    # ---------- mode impls ----------

    def _decide_band(self, curr: float, bbox_h_px: Optional[float], t: float) -> float:
        # smooth bbox height
        if bbox_h_px is None or bbox_h_px <= 0:
            return curr
        if self.state.bbox_h_ema is None:
            self.state.bbox_h_ema = float(bbox_h_px)
        else:
            a = self.cfg.band.px_ema_alpha
            self.state.bbox_h_ema = a * self.state.bbox_h_ema + (1.0 - a) * float(bbox_h_px)

        h = self.state.bbox_h_ema
        b = self.cfg.band
        # inside band with deadband? -> hold
        if (h >= (b.min_px + b.dead_px)) and (h <= (b.max_px - b.dead_px)):
            return curr

        # proportional step change (pixel → steps heuristic)
        # Positive delta = zoom in (more steps), Negative = zoom out (fewer steps).
        # Heuristic gain: tune k_step_per_px for your mechanism (start with ~1..3).
        k_step_per_px = 2.0
        if h < b.min_px:
            # too small → zoom in
            deficit = b.min_px - h
            target = curr + k_step_per_px * deficit
        else:
            # too large → zoom out
            excess = h - b.max_px
            target = curr - k_step_per_px * excess

        # clamp to limits
        return self._clamp(target)

    def _decide_fov(self, curr: float) -> float:
        """
        Choose steps to achieve target FOV using LUT (requires image_size & LUT).
        """
        if self._lut is None or interpolate_zoom_lut is None or not self.image_size:
            # Can't compute: hold current
            return curr

        W = float(self.image_size[0])
        tgt = max(0.2, float(self.cfg.fov.target_fov_deg))
        fx_needed = W / (2.0 * math.tan(math.radians(tgt) * 0.5))

        # Find steps that yield ~fx_needed by scanning LUT domain
        xs = sorted(self._lut.keys())
        best_s = xs[0]
        best_err = float("inf")
        for s in xs:
            Kd = interpolate_zoom_lut(s, self._lut)
            err = abs(Kd["fx"] - fx_needed)
            if err < best_err:
                best_err = err
                best_s = s

        # deadband in FOV space to avoid chattering
        if self.state.last_k is not None:
            fx_last = self.state.last_k[0]
            fov_last = 2.0 * math.degrees(math.atan(W / (2.0 * fx_last)))
            if abs(fov_last - tgt) < max(0.05, self.cfg.fov.fov_dead_deg):
                return curr

        return self._clamp(float(best_s))

    # ---------- helpers ----------

    def _passes_gates(self, bbox_h_px, conf, center_err_px) -> bool:
        g = self.cfg.gates
        if conf is not None and conf < g.min_conf:
            return False
        if center_err_px is not None:
            dx, dy = center_err_px
            if math.hypot(dx or 0.0, dy or 0.0) > g.max_center_err_px:
                return False
        # Cooldown: avoid rapid flipping decisions
        if (time.time() - self.state.last_change_t) < self.cfg.change_cooldown_s:
            return False
        return True

    def _clamp(self, steps: float) -> float:
        s = float(steps)
        lim = self.cfg.limits
        if lim.min_steps is not None:
            s = max(s, float(lim.min_steps))
        if lim.max_steps is not None:
            s = min(s, float(lim.max_steps))
        return s

    def _rate_limit(self, curr: float, target: float, t: float) -> float:
        # Cap motion per frame based on elapsed time and configured rate
        dt = max(1e-3, t - (self.state.last_change_t or (t - 0.02)))
        max_delta = self.cfg.limits.max_rate_steps_s * dt
        delta = max(-max_delta, min(+max_delta, target - curr))
        return curr + delta

    def _maybe_intrinsics(self, steps: float, *, force: bool) -> Optional[Dict[str, float]]:
        """
        If a LUT is present, return K for current steps and call on_intrinsics.
        """
        if self._lut is None or interpolate_zoom_lut is None:
            return None
        Kd = interpolate_zoom_lut(float(steps), self._lut)
        last = self.state.last_k
        if (not force) and last is not None:
            # small change? skip callback
            if max(abs(Kd["fx"] - last[0]), abs(Kd["fy"] - last[1])) < 1e-3:
                return None
        self.state.last_k = (Kd["fx"], Kd["fy"], Kd["cx"], Kd["cy"])
        if self.on_intrinsics is not None:
            try:
                self.on_intrinsics(Kd, float(steps))
            except Exception:
                pass
        return Kd
# app/zoom_profile.py
"""
Zoom profile manager.

What it does
- Supports zoom modes:
    * "fixed": hold a specific motor step position
    * "band": keep the active target's bbox height within [min_px, max_px]
    * "fov": aim for a desired horizontal FOV (deg)
- Adds deadband + hysteresis to avoid zoom "hunting"
- Rate-limits step changes (steps/s) and enforces soft limits
- Gating: only zoom if (a) detection confidence is high enough and
  (b) the center error is small enough (so you don't zoom while off-target)
- Optional intrinsics update: interpolates K from a zoom→intrinsics LUT
  whenever steps change, so downstream geometry stays correct while zooming

Interfaces nicely with:
- hardware.zoom_motor.ZoomMotor (expects read_steps(), goto_steps())
- io.zoom_lut_loader (load_zoom_lut, interpolate_zoom_lut)
- policy.py / pose_update.py (consume K updates if you wire the callback)

Why this matches the PDF:
- Uses a zoom→K LUT and interpolates K(steps) whenever zoom changes, because
  zooming changes f (fx, fy) and you must keep K current to avoid depth drift. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
- The runtime loop is: track targets → choose an ID → update R from encoders →
  select the correct zoom K profile and keep the object centered/within a
  scale band—exactly the outlined control flow. :contentReference[oaicite:5]{index=5}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import time

try:
    from io.zoom_lut_loader import load_zoom_lut, interpolate_zoom_lut
except Exception:
    # Make LUT optional. You can still use "band" and "fixed" without K updates.
    load_zoom_lut = None   # type: ignore
    interpolate_zoom_lut = None  # type: ignore


# =========================
# Data classes
# =========================

@dataclass
class BandConfig:
    min_px: int = 80           # keep bbox height >= min_px
    max_px: int = 220          # and <= max_px
    dead_px: int = 12          # deadband inside the band before commanding zoom
    px_ema_alpha: float = 0.5  # smooth bbox height frame-to-frame

@dataclass
class FovConfig:
    target_fov_deg: float = 3.0     # desired horizontal FOV
    fov_dead_deg: float = 0.15      # deadband

@dataclass
class GateConfig:
    min_conf: float = 0.20          # don't zoom if detection too uncertain
    max_center_err_px: int = 120    # don't zoom while way off-center (let pan/tilt fix first)

@dataclass
class LimitsConfig:
    min_steps: Optional[int] = 0
    max_steps: Optional[int] = None
    max_rate_steps_s: float = 2500.0   # cap command rate
    min_step_delta: int = 4            # ignore tiny step changes (mechanical slack)

@dataclass
class ZoomProfileConfig:
    mode: str = "band"                 # "band" | "fixed" | "fov"
    fixed_steps: Optional[int] = None
    band: BandConfig = BandConfig()
    fov: FovConfig = FovConfig()
    gates: GateConfig = GateConfig()
    limits: LimitsConfig = LimitsConfig()
    change_cooldown_s: float = 0.15    # minimum time between new targets or big changes

@dataclass
class ZoomProfileState:
    last_cmd_steps: Optional[float] = None
    last_k: Optional[Tuple[float, float, float, float]] = None  # fx, fy, cx, cy
    bbox_h_ema: Optional[float] = None
    last_change_t: float = 0.0


# =========================
# Main manager
# =========================

class ZoomProfile:
    """
    Orchestrates zoom setpoints based on the current mode and track readouts.
    """
    def __init__(self,
                 motor,                              # object with read_steps(), goto_steps()
                 cfg: ZoomProfileConfig = ZoomProfileConfig(),
                 *,
                 lut_path: Optional[str] = None,     # path to zoom LUT (JSON/YAML/CSV)
                 image_size: Optional[Tuple[int,int]] = None,  # (W,H) for FOV math
                 on_intrinsics: Optional[Callable[[Dict[str, float], float], None]] = None  # cb(K_dict, steps)
                 ):
        self.motor = motor
        self.cfg = cfg
        self.image_size = image_size
        self.on_intrinsics = on_intrinsics
        self.state = ZoomProfileState()
        self._lut = None
        if lut_path and load_zoom_lut is not None:
            self._lut = load_zoom_lut(lut_path)  # {steps: (fx,fy,cx,cy)}

    # ---------- public API ----------

    def update(self,
               *,
               bbox_h_px: Optional[float],
               conf: Optional[float],
               center_err_px: Optional[Tuple[float,float]] = None,
               now: Optional[float] = None) -> Dict:
        """
        Call once per frame. Decides a new zoom target if needed and commands the motor.

        Args:
          bbox_h_px: current target bbox height (pixels) or None if no target
          conf: detection confidence (0..1) or None
          center_err_px: (dx, dy) from image center to target center; used for gating
        Returns:
          dict with summary of the decision and (optionally) K update
        """
        t = time.time() if now is None else now
        curr = float(self.motor.read_steps())  # current steps
        changed = False
        new_steps: Optional[float] = None

        # --- gates ---
        if not self._passes_gates(bbox_h_px, conf, center_err_px):
            # Too uncertain or not centered; don't zoom this frame
            # (You still might want to keep K current to this steps.)
            k = self._maybe_intrinsics(curr, force=False)
            return dict(mode=self.cfg.mode, steps=curr, action="gate_hold", K=k)

        # --- decide by mode ---
        if self.cfg.mode == "fixed":
            target = self._clamp(self.cfg.fixed_steps if self.cfg.fixed_steps is not None else curr)
            new_steps = target

        elif self.cfg.mode == "band":
            new_steps = self._decide_band(curr, bbox_h_px, t)

        elif self.cfg.mode == "fov":
            new_steps = self._decide_fov(curr)

        else:
            # Unknown mode: hold
            k = self._maybe_intrinsics(curr, force=False)
            return dict(mode=self.cfg.mode, steps=curr, action="unknown_mode_hold", K=k)

        # --- rate limit & min delta ---
        new_steps = self._rate_limit(curr, new_steps, t)
        if self.state.last_cmd_steps is not None:
            if abs(new_steps - self.state.last_cmd_steps) < self.cfg.limits.min_step_delta:
                k = self._maybe_intrinsics(curr, force=False)
                return dict(mode=self.cfg.mode, steps=curr, action="deadband_hold", K=k)

        # --- command motor ---
        self.motor.goto_steps(new_steps)
        self.state.last_cmd_steps = new_steps
        self.state.last_change_t = t
        changed = True

        # --- intrinsics update (optional) ---
        k = self._maybe_intrinsics(new_steps, force=True)

        return dict(mode=self.cfg.mode, steps=new_steps,
                    action="cmd" if changed else "hold", K=k)

    # ---------- mode impls ----------

    def _decide_band(self, curr: float, bbox_h_px: Optional[float], t: float) -> float:
        # smooth bbox height
        if bbox_h_px is None or bbox_h_px <= 0:
            return curr
        if self.state.bbox_h_ema is None:
            self.state.bbox_h_ema = float(bbox_h_px)
        else:
            a = self.cfg.band.px_ema_alpha
            self.state.bbox_h_ema = a * self.state.bbox_h_ema + (1.0 - a) * float(bbox_h_px)

        h = self.state.bbox_h_ema
        b = self.cfg.band
        # inside band with deadband? -> hold
        if (h >= (b.min_px + b.dead_px)) and (h <= (b.max_px - b.dead_px)):
            return curr

        # proportional step change (pixel → steps heuristic)
        # Positive delta = zoom in (more steps), Negative = zoom out (fewer steps).
        # Heuristic gain: tune k_step_per_px for your mechanism (start with ~1..3).
        k_step_per_px = 2.0
        if h < b.min_px:
            # too small → zoom in
            deficit = b.min_px - h
            target = curr + k_step_per_px * deficit
        else:
            # too large → zoom out
            excess = h - b.max_px
            target = curr - k_step_per_px * excess

        # clamp to limits
        return self._clamp(target)

    def _decide_fov(self, curr: float) -> float:
        """
        Choose steps to achieve target FOV using LUT (requires image_size & LUT).
        """
        if self._lut is None or interpolate_zoom_lut is None or not self.image_size:
            # Can't compute: hold current
            return curr

        W = float(self.image_size[0])
        tgt = max(0.2, float(self.cfg.fov.target_fov_deg))
        fx_needed = W / (2.0 * math.tan(math.radians(tgt) * 0.5))

        # Find steps that yield ~fx_needed by scanning LUT domain
        xs = sorted(self._lut.keys())
        best_s = xs[0]
        best_err = float("inf")
        for s in xs:
            Kd = interpolate_zoom_lut(s, self._lut)
            err = abs(Kd["fx"] - fx_needed)
            if err < best_err:
                best_err = err
                best_s = s

        # deadband in FOV space to avoid chattering
        if self.state.last_k is not None:
            fx_last = self.state.last_k[0]
            fov_last = 2.0 * math.degrees(math.atan(W / (2.0 * fx_last)))
            if abs(fov_last - tgt) < max(0.05, self.cfg.fov.fov_dead_deg):
                return curr

        return self._clamp(float(best_s))

    # ---------- helpers ----------

    def _passes_gates(self, bbox_h_px, conf, center_err_px) -> bool:
        g = self.cfg.gates
        if conf is not None and conf < g.min_conf:
            return False
        if center_err_px is not None:
            dx, dy = center_err_px
            if math.hypot(dx or 0.0, dy or 0.0) > g.max_center_err_px:
                return False
        # Cooldown: avoid rapid flipping decisions
        if (time.time() - self.state.last_change_t) < self.cfg.change_cooldown_s:
            return False
        return True

    def _clamp(self, steps: float) -> float:
        s = float(steps)
        lim = self.cfg.limits
        if lim.min_steps is not None:
            s = max(s, float(lim.min_steps))
        if lim.max_steps is not None:
            s = min(s, float(lim.max_steps))
        return s

    def _rate_limit(self, curr: float, target: float, t: float) -> float:
        # Cap motion per frame based on elapsed time and configured rate
        dt = max(1e-3, t - (self.state.last_change_t or (t - 0.02)))
        max_delta = self.cfg.limits.max_rate_steps_s * dt
        delta = max(-max_delta, min(+max_delta, target - curr))
        return curr + delta

    def _maybe_intrinsics(self, steps: float, *, force: bool) -> Optional[Dict[str, float]]:
        """
        If a LUT is present, return K for current steps and call on_intrinsics.
        """
        if self._lut is None or interpolate_zoom_lut is None:
            return None
        Kd = interpolate_zoom_lut(float(steps), self._lut)
        last = self.state.last_k
        if (not force) and last is not None:
            # small change? skip callback
            if max(abs(Kd["fx"] - last[0]), abs(Kd["fy"] - last[1])) < 1e-3:
                return None
        self.state.last_k = (Kd["fx"], Kd["fy"], Kd["cx"], Kd["cy"])
        if self.on_intrinsics is not None:
            try:
                self.on_intrinsics(Kd, float(steps))
            except Exception:
                pass
        return Kd

