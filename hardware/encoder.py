# hardware/encoder.py
from __future__ import annotations
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

from hardware.gpio_backend import GPIOIf, IN, PUD_UP


@dataclass
class EncoderConfig:
    pin_a: int
    pin_b: int
    counts_per_rev: float = 1024.0   # encoder resolution (A/B x4 if needed)
    invert: int = 1                  # +1 normal, -1 invert direction
    poll_hz: float = 2000.0          # polling frequency (fallback to polling across backends)
    use_pullups: bool = True         # enable internal pull-ups if available
    velocity_window: float = 0.100   # seconds of history for velocity smoothing


class QuadratureEncoder:
    """
    Simple quadrature (A/B) encoder reader with background polling.
    - Backend-agnostic (uses gpio_backend GPIOIf).
    - Decodes Gray-code transitions; ignores illegal glitches.
    - Provides counts, degrees, and velocity (deg/s).
    """

    # State machine for quadrature decoding:
    # Index by (prev_state << 2) | curr_state → delta {-1,0,+1}
    _DELTA_LUT = [
        0,  +1,  -1,   0,
        -1,  0,   0,  +1,
        +1,  0,   0,  -1,
        0,  -1, +1,    0,
    ]

    def __init__(self, gpio: GPIOIf, cfg: EncoderConfig):
        self.gpio = gpio
        self.cfg = cfg
        self._counts: int = 0
        self._zero_counts: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # velocity history (ts, counts)
        self._hist: Deque[Tuple[float, int]] = deque()

        # Pin setup
        pull = PUD_UP if cfg.use_pullups else "OFF"  # type: ignore
        self.gpio.setup(cfg.pin_a, IN, pull)  # type: ignore
        self.gpio.setup(cfg.pin_b, IN, pull)  # type: ignore

        # Initialize state
        a = 1 if self.gpio.input(cfg.pin_a) else 0
        b = 1 if self.gpio.input(cfg.pin_b) else 0
        self._state: int = (a << 1) | b
        self._last_sample_t = time.monotonic()

    # -------- lifecycle --------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="QuadratureEncoder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    # -------- core loop --------
    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.poll_hz, 1.0)
        while not self._stop.is_set():
            t = time.monotonic()
            a = 1 if self.gpio.input(self.cfg.pin_a) else 0
            b = 1 if self.gpio.input(self.cfg.pin_b) else 0
            new_state = (a << 1) | b
            idx = ((self._state << 2) | new_state) & 0xF
            delta = self._DELTA_LUT[idx]

            if delta != 0:
                with self._lock:
                    self._counts += self.cfg.invert * delta
                    # push history for velocity estimate
                    self._hist.append((t, self._counts))
                    self._trim_history(t)

                self._state = new_state
            else:
                self._state = new_state

            # sleep to maintain approximate poll rate
            dt = time.monotonic() - t
            sleep_t = period - dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _trim_history(self, now_t: float) -> None:
        """Keep only recent samples within velocity_window."""
        horizon = self.cfg.velocity_window
        while self._hist and (now_t - self._hist[0][0]) > horizon:
            self._hist.popleft()

    # -------- public API --------
    def reset(self) -> None:
        """Zero the encoder (sets current position as 0 deg)."""
        with self._lock:
            self._zero_counts = self._counts
            self._hist.clear()

    def get_counts(self) -> int:
        with self._lock:
            return self._counts - self._zero_counts

    def get_revolutions(self) -> float:
        return self.get_counts() / float(self.cfg.counts_per_rev)

    def get_degrees(self) -> float:
        return self.get_revolutions() * 360.0

    def get_velocity_dps(self) -> float:
        """
        Smoothed angular velocity (deg/s) using linear fit over recent history.
        Falls back to instantaneous estimate if history is too short.
        """
        with self._lock:
            if len(self._hist) < 2:
                return 0.0
            t0, c0 = self._hist[0]
            tn, cn = self._hist[-1]
        dt = tn - t0
        if dt <= 1e-6:
            return 0.0
        dc = cn - c0
        rev_s = (dc / self.cfg.counts_per_rev) / dt
        return rev_s * 360.0

    # convenience: degrees per count (useful for motor mapping)
    @property
    def deg_per_count(self) -> float:
        return 360.0 / float(self.cfg.counts_per_rev)


# --------------- quick demo ---------------
if __name__ == "__main__":
    from hardware.gpio_backend import make_gpio
    # Null backend demo (simulated reads will be 0; replace with rpi/jetson on hardware)
    gpio = make_gpio("null")
    enc = QuadratureEncoder(gpio, EncoderConfig(pin_a=23, pin_b=24, counts_per_rev=2048, poll_hz=2000))
    enc.start()
    try:
        for _ in range(5):
            time.sleep(0.5)
            print(f"counts={enc.get_counts():6d}  deg={enc.get_degrees():8.2f}  vel={enc.get_velocity_dps():7.2f} dps")
    finally:
        enc.stop()

