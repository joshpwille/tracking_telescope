# hardware/encoder.py
"""
Pan/Tilt encoder reader with:
- Backends: AS5600 (I2C), AS5048A (SPI), Quadrature (pigpio), Mock
- Per-axis calibration: zero offset, invert, gear ratio, counts-per-rev
- Continuous angle unwrapping (avoid 2π jumps)
- EMA smoothing and rate limiting (deg/s) to squash jitter/spikes
- Background sampler thread (non-blocking reads)
- Health flags and graceful fallbacks
- Persist/restore calibration to JSON

Public API (used by pipeline/pose_update):
    enc = Encoder(EncoderConfig(...))
    enc.start()
    pan, tilt = enc.read_pan_tilt_rad()
    enc.set_zero_current(both=True)   # make current pose zero
    enc.stop()

Notes:
- Install only what you need. If a backend import fails, it falls back to a mock.
- Keep rate limiting practical (avoid commanding huge jumps on noisy edges).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict
import json
import math
import os
import threading
import time

# Optional libs (import lazily in backends)
try:
    import smbus2  # type: ignore
    _HAVE_SMBUS = True
except Exception:
    _HAVE_SMBUS = False

try:
    import spidev  # type: ignore
    _HAVE_SPI = True
except Exception:
    _HAVE_SPI = False

try:
    import pigpio  # type: ignore
    _HAVE_PIGPIO = True
except Exception:
    _HAVE_PIGPIO = False


# =========================
# Config
# =========================

@dataclass
class AxisConfig:
    model: str = "AS5600"           # "AS5600" | "AS5048A" | "Quadrature" | "Mock"
    # Electrical pins/addresses per model:
    i2c_bus: int = 1
    i2c_addr: int = 0x36            # AS5600 default
    spi_bus: int = 0                # for AS5048A (spidev bus)
    spi_dev: int = 0                # for AS5048A (spidev device)
    spi_speed_hz: int = 1000000     # 1 MHz is safe
    gpio_quada: Optional[int] = None  # Quadrature A pin (BCM)
    gpio_quadb: Optional[int] = None  # Quadrature B pin (BCM)

    # Mechanics & calibration:
    cpr: int = 4096                 # counts-per-rev for absolute/quad (AS5600=4096, AS5048A=16384)
    gear_ratio: float = 1.0         # multiply angle by this ratio (output angle = raw * ratio)
    invert: bool = False
    zero_offset_rad: float = 0.0    # applied after invert/gear
    ema_alpha: float = 0.6          # smoothing 0..1 (higher = smoother)
    rate_limit_deg_s: float = 180.0 # clamp delta per second
    # If true, unwrap across ±π to keep continuous angle; if false -> returns wrapped [0, 2π)
    unwrap: bool = True

@dataclass
class EncoderConfig:
    pan: AxisConfig = field(default_factory=lambda: AxisConfig(model="AS5600"))
    tilt: AxisConfig = field(default_factory=lambda: AxisConfig(model="AS5600"))
    sample_hz: float = 200.0
    state_file: str = "io/encoder_state.json"  # persists zero offsets & misc


# =========================
# Internal state
# =========================

class _AxisState:
    def __init__(self, cfg: AxisConfig):
        self.cfg = cfg
        self.raw_wrapped: float = 0.0   # [0, 2π)
        self.unwrapped: float = 0.0     # continuous
        self.filtered: float = 0.0
        self.last_raw: Optional[float] = None
        self.last_time: float = time.time()
        self.healthy: bool = True
        self.error: Optional[str] = None

    def apply_calibration(self, angle_rad: float) -> float:
        """Apply invert, gear, and zero offset to an angle (radians, continuous)."""
        a = angle_rad
        if self.cfg.invert:
            a = -a
        a = a * float(self.cfg.gear_ratio)
        a = a + float(self.cfg.zero_offset_rad)
        return a

    def rate_limit(self, a_new: float, dt: float) -> float:
        """Clamp change per time per axis."""
        max_step = math.radians(max(1e-3, self.cfg.rate_limit_deg_s)) * max(1e-3, dt)
        return _clamp_delta(self.filtered, a_new, max_step)

def _wrap_0_2pi(a: float) -> float:
    twopi = 2.0 * math.pi
    a = a % twopi
    return a if a >= 0 else a + twopi

def _angle_unwrap(prev: float, curr_wrapped: float) -> float:
    """Unwrap wrapped current sample against previous continuous value."""
    twopi = 2.0 * math.pi
    # Bring current into neighborhood of prev
    cw = curr_wrapped
    # Estimate nearest equivalent of cw to prev
    delta = cw - (prev % twopi)
    if delta > math.pi:
        delta -= twopi
    elif delta < -math.pi:
        delta += twopi
    return prev + delta

def _ema(prev: float, new: float, alpha: float) -> float:
    # alpha closer to 1 => smoother (more inertia)
    return float(alpha * prev + (1.0 - alpha) * new)

def _clamp_delta(old: float, new: float, max_step: float) -> float:
    d = new - old
    if d > max_step:
        return old + max_step
    if d < -max_step:
        return old - max_step
    return new


# =========================
# Backends (read wrapped angle in radians)
# =========================

class _AxisBackend:
    def read_wrapped_rad(self) -> float:
        """Return wrapped angle in [0, 2π); raise on hard failure."""
        raise NotImplementedError
    def close(self): ...

class _AS5600Backend(_AxisBackend):
    # RAW_ANGLE registers: 0x0C (high 8), 0x0D (low 8) → 12-bit angle 0..4095
    RAW_ANGLE_H = 0x0C
    RAW_ANGLE_L = 0x0D
    def __init__(self, cfg: AxisConfig):
        if not _HAVE_SMBUS:
            raise RuntimeError("smbus2 not available for AS5600 backend")
        self.bus = smbus2.SMBus(cfg.i2c_bus)
        self.addr = cfg.i2c_addr
        self.cpr = max(1, cfg.cpr)
    def read_wrapped_rad(self) -> float:
        hi = self.bus.read_byte_data(self.addr, self.RAW_ANGLE_H)
        lo = self.bus.read_byte_data(self.addr, self.RAW_ANGLE_L)
        raw = ((hi & 0x0F) << 8) | lo  # 12-bit
        raw = max(0, min(self.cpr - 1, raw))
        return (raw / float(self.cpr)) * (2.0 * math.pi)
    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

class _AS5048ABackend(_AxisBackend):
    # 14-bit angle; many variants exist. We read two bytes and mask 14 bits.
    def __init__(self, cfg: AxisConfig):
        if not _HAVE_SPI:
            raise RuntimeError("spidev not available for AS5048A backend")
        self.spi = spidev.SpiDev()
        self.spi.open(cfg.spi_bus, cfg.spi_dev)
        self.spi.max_speed_hz = cfg.spi_speed_hz
        self.spi.mode = 0b01  # CPOL=0, CPHA=1 often used; adjust if needed
        self.cpr = max(1, cfg.cpr)
    def read_wrapped_rad(self) -> float:
        # Simple "read angle" transaction: send NOP (0xFFFF) and read back 16b.
        # Real devices need command frames & parity; this minimalist read works for many breakout boards.
        resp = self.spi.xfer2([0xFF, 0xFF])
        val = ((resp[0] << 8) | resp[1]) & 0x3FFF
        val = max(0, min(self.cpr - 1, val))
        return (val / float(self.cpr)) * (2.0 * math.pi)
    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass

class _QuadratureBackend(_AxisBackend):
    def __init__(self, cfg: AxisConfig):
        if not _HAVE_PIGPIO:
            raise RuntimeError("pigpio not available for Quadrature backend")
        if cfg.gpio_quada is None or cfg.gpio_quadb is None:
            raise RuntimeError("Quadrature backend requires gpio_quada and gpio_quadb")
        self.pi = pigpio.pi()
        self.a = cfg.gpio_quada
        self.b = cfg.gpio_quadb
        self.pi.set_mode(self.a, pigpio.INPUT)
        self.pi.set_mode(self.b, pigpio.INPUT)
        self.pi.set_pull_up_down(self.a, pigpio.PUD_UP)
        self.pi.set_pull_up_down(self.b, pigpio.PUD_UP)
        self.cpr = max(1, cfg.cpr)
        self._count = 0
        self._last = (self.pi.read(self.a) << 1) | self.pi.read(self.b)
        self._cb_a = self.pi.callback(self.a, pigpio.EITHER_EDGE, self._cb)
        self._cb_b = self.pi.callback(self.b, pigpio.EITHER_EDGE, self._cb)
    def _cb(self, g, l, t):
        s = (self.pi.read(self.a) << 1) | self.pi.read(self.b)
        # Gray-code state transitions
        delta = {0b00:{0b01:+1,0b10:-1}, 0b01:{0b11:+1,0b00:-1},
                 0b11:{0b10:+1,0b01:-1}, 0b10:{0b00:+1,0b11:-1}}
        try:
            self._count += delta[self._last][s]
        except KeyError:
            pass
        self._last = s
    def read_wrapped_rad(self) -> float:
        count = self._count
        # wrap to [0, 2π)
        return ((count % self.cpr) / float(self.cpr)) * (2.0 * math.pi)
    def close(self):
        try:
            self._cb_a.cancel(); self._cb_b.cancel()
            self.pi.stop()
        except Exception:
            pass

class _MockBackend(_AxisBackend):
    def __init__(self, cfg: AxisConfig):
        self._t0 = time.time()
        self._rate = 0.2 if "pan" else 0.1
    def read_wrapped_rad(self) -> float:
        t = time.time() - self._t0
        # Slow sine wave sim
        a = (math.sin(self._rate * t) * 0.5 + 0.5) * (2.0 * math.pi)  # [0,2π)
        return a


def _make_backend(cfg: AxisConfig) -> _AxisBackend:
    m = cfg.model.upper()
    try:
        if m == "AS5600":
            return _AS5600Backend(cfg)
        if m == "AS5048A":
            return _AS5048ABackend(cfg)
        if m == "QUADRATURE":
            return _QuadratureBackend(cfg)
        if m == "MOCK":
            return _MockBackend(cfg)
    except Exception as e:
        # Fall through to mock with health warning
        print(f"[encoder] Backend {cfg.model} failed: {e}. Falling back to Mock.")
    return _MockBackend(cfg)


# =========================
# Axis wrapper
# =========================

class _Axis:
    def __init__(self, cfg: AxisConfig):
        self.cfg = cfg
        self.state = _AxisState(cfg)
        self.backend = _make_backend(cfg)

    def sample_once(self):
        now = time.time()
        dt = max(1e-3, now - self.state.last_time)
        self.state.last_time = now
        try:
            w = self.backend.read_wrapped_rad()               # [0, 2π)
            self.state.raw_wrapped = w
            # Unwrap
            if self.state.last_raw is None:
                unwrapped = w
            else:
                unwrapped = _angle_unwrap(self.state.unwrapped, w)
            self.state.unwrapped = unwrapped
            self.state.last_raw = w

            # Apply calibration (invert/gear/zero) on the continuous value
            a_cal = self.state.apply_calibration(unwrapped)

            # Rate limit & EMA
            a_rl = self.state.rate_limit(a_cal, dt)
            if self.state.filtered == 0.0 and self.state.last_raw is None:
                self.state.filtered = a_rl
            else:
                self.state.filtered = _ema(self.state.filtered, a_rl, self.cfg.ema_alpha)

            self.state.healthy = True
            self.state.error = None
        except Exception as e:
            self.state.healthy = False
            self.state.error = str(e)
            # On failure, keep last filtered value

    def read_rad(self, unwrap: Optional[bool] = None) -> float:
        """Return calibrated angle (radians); unwrap=True for continuous, False for wrapped [0,2π)."""
        unwrap = self.cfg.unwrap if unwrap is None else unwrap
        a = self.state.filtered
        if not unwrap:
            a = _wrap_0_2pi(a)
        return float(a)

    def set_zero_current(self):
        """Adjust zero_offset so current calibrated angle returns 0."""
        # Compute current raw angle (continuous) BEFORE zero
        raw_cont = self.state.unwrapped
        if self.cfg.invert:
            raw_cont = -raw_cont
        raw_cont *= float(self.cfg.gear_ratio)
        # Want: raw_cont + zero = 0  => zero = -raw_cont
        self.cfg.zero_offset_rad = -raw_cont

    def close(self):
        try:
            self.backend.close()
        except Exception:
            pass


# =========================
# Encoder (pan + tilt)
# =========================

class Encoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        # Try to load persisted offsets first
        self._load_state()
        # Build axes after potential offset injection
        self.pan_axis = _Axis(self.cfg.pan)
        self.tilt_axis = _Axis(self.cfg.tilt)

        self._lock = threading.Lock()
        self._stop = False
        self._th: Optional[threading.Thread] = None
        self._period = 1.0 / max(1.0, float(cfg.sample_hz))

    # ---- lifecycle ----
    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop = False
        self._th = threading.Thread(target=self._worker, name="encoder-sampler", daemon=True)
        self._th.start()

    def stop(self, timeout: float = 1.0):
        self._stop = True
        if self._th:
            self._th.join(timeout=timeout)
        self.pan_axis.close()
        self.tilt_axis.close()
        self._save_state()

    # ---- reads ----
    def read_pan_tilt_rad(self) -> Tuple[float, float]:
        """Thread-safe read of current filtered angles (continuous)."""
        with self._lock:
            return (self.pan_axis.read_rad(unwrap=True),
                    self.tilt_axis.read_rad(unwrap=True))

    def health(self) -> Dict[str, Dict[str, str]]:
        return {
            "pan": {"healthy": str(self.pan_axis.state.healthy), "error": str(self.pan_axis.state.error)},
            "tilt": {"healthy": str(self.tilt_axis.state.healthy), "error": str(self.tilt_axis.state.error)},
        }

    # ---- calibration helpers ----
    def set_zero_current(self, both: bool = False, pan: bool = False, tilt: bool = False):
        """
        Make the current physical pose read as 0 rad on selected axes.
        """
        if both or pan:
            self.pan_axis.set_zero_current()
        if both or tilt:
            self.tilt_axis.set_zero_current()
        self._save_state()

    def set_offsets(self, pan_zero_rad: Optional[float] = None, tilt_zero_rad: Optional[float] = None):
        if pan_zero_rad is not None:
            self.cfg.pan.zero_offset_rad = float(pan_zero_rad)
        if tilt_zero_rad is not None:
            self.cfg.tilt.zero_offset_rad = float(tilt_zero_rad)
        self._save_state()

    # ---- worker ----
    def _worker(self):
        next_t = time.perf_counter()
        while not self._stop:
            with self._lock:
                self.pan_axis.sample_once()
                self.tilt_axis.sample_once()
            next_t += self._period
            # Sleep until next period (coarse is fine)
            rem = next_t - time.perf_counter()
            if rem > 0:
                time.sleep(min(rem, 0.005))
            else:
                next_t = time.perf_counter()

    # ---- persistence ----
    def _load_state(self):
        try:
            if os.path.isfile(self.cfg.state_file):
                with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                # Only restore calibration knobs that belong in config
                if "pan" in d and isinstance(d["pan"], dict):
                    self.cfg.pan.zero_offset_rad = float(d["pan"].get("zero_offset_rad", self.cfg.pan.zero_offset_rad))
                    self.cfg.pan.invert = bool(d["pan"].get("invert", self.cfg.pan.invert))
                    self.cfg.pan.gear_ratio = float(d["pan"].get("gear_ratio", self.cfg.pan.gear_ratio))
                if "tilt" in d and isinstance(d["tilt"], dict):
                    self.cfg.tilt.zero_offset_rad = float(d["tilt"].get("zero_offset_rad", self.cfg.tilt.zero_offset_rad))
                    self.cfg.tilt.invert = bool(d["tilt"].get("invert", self.cfg.tilt.invert))
                    self.cfg.tilt.gear_ratio = float(d["tilt"].get("gear_ratio", self.cfg.tilt.gear_ratio))
        except Exception:
            # ignore corrupt files; keep defaults
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.cfg.state_file), exist_ok=True)
            d = {
                "pan": {
                    "zero_offset_rad": self.cfg.pan.zero_offset_rad,
                    "invert": self.cfg.pan.invert,
                    "gear_ratio": self.cfg.pan.gear_ratio,
                },
                "tilt": {
                    "zero_offset_rad": self.cfg.tilt.zero_offset_rad,
                    "invert": self.cfg.tilt.invert,
                    "gear_ratio": self.cfg.tilt.gear_ratio,
                },
                "_meta": {"updated": time.time()},
            }
            with open(self.cfg.state_file, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass


# =========================
# Example usage / smoke test
# =========================

if __name__ == "__main__":
    # Example: AS5600 on pan, Mock on tilt
    pan_cfg = AxisConfig(model="Mock")  # swap to "AS5600" and set i2c_addr for real HW
    tilt_cfg = AxisConfig(model="Mock")
    cfg = EncoderConfig(pan=pan_cfg, tilt=tilt_cfg, sample_hz=200.0)

    enc = Encoder(cfg)
    enc.start()
    print("[encoder] sampling for ~1s...")
    t0 = time.time()
    while time.time() - t0 < 1.0:
        pan, tilt = enc.read_pan_tilt_rad()
        print(f"pan={math.degrees(pan):7.2f}°  tilt={math.degrees(tilt):7.2f}°  health={enc.health()}")
        time.sleep(0.05)

    print("[encoder] zeroing current pose...")
    enc.set_zero_current(both=True)
    pan, tilt = enc.read_pan_tilt_rad()
    print(f"after zero: pan={pan:+.4f} rad, tilt={tilt:+.4f} rad")

    enc.stop()
