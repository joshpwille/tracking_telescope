# hardware/zoom_motor.py
"""
Zoom motor driver (step/dir) with:
- Threaded motion (non-blocking), trapezoidal profile (accel → cruise → decel)
- Optional limit switches + homing, soft limits, and JSON-persisted calibration
- Works with RPi.GPIO or pigpio if available; otherwise falls back to a safe mock backend

Typical use:
    cfg = ZoomConfig(step_pin=18, dir_pin=23, en_pin=24,
                     min_switch_pin=5, max_switch_pin=6,
                     steps_per_rev=200, gear_ratio=1.0,
                     max_speed_sps=2500, accel_sps2=8000,
                     state_file="io/zoom_state.json")
    zm = ZoomMotor(cfg)
    zm.start()
    zm.home()                 # find min/max & set zero/limits (if switches wired)
    zm.goto_steps(1350)       # non-blocking
    steps = zm.read_steps()   # float
    zm.stop()

Interface expected by your pipeline:
  - read_steps() -> float
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import json
import os
import time
import threading

# ---------- Optional GPIO backends ----------
try:
    import pigpio  # type: ignore
    _HAVE_PIGPIO = True
except Exception:
    _HAVE_PIGPIO = False

try:
    import RPi.GPIO as GPIO  # type: ignore
    _HAVE_RPI_GPIO = True
except Exception:
    _HAVE_RPI_GPIO = False


# ========================
# Config and State
# ========================

@dataclass
class ZoomConfig:
    # Required motor pins
    step_pin: int
    dir_pin: int
    en_pin: Optional[int] = None  # active-low enable (LOW=en)

    # Optional limit switches (active-low normally-closed recommended)
    min_switch_pin: Optional[int] = None
    max_switch_pin: Optional[int] = None
    switch_pulldown: bool = True     # True -> PULLUP on inputs (expect NC to GND)

    # Mechanics
    steps_per_rev: int = 200
    gear_ratio: float = 1.0          # *multiply* steps per output rev
    microstep: int = 16              # 1, 2, 4, 8, 16...
    backlash_steps: int = 0          # if you want to overshoot and return

    # Motion profile
    max_speed_sps: float = 2500.0    # steps per second
    accel_sps2: float = 8000.0       # steps per second^2
    min_speed_sps: float = 100.0     # floor to avoid super-long pulses

    # Homing
    homing_speed_sps: float = 800.0
    homing_backoff_steps: int = 200
    home_zero_at_min: bool = True    # set 0 at min switch
    seek_max_after_min: bool = True  # discover max limit after homing min

    # Soft limits (set/updated after homing; used even without switches if known)
    soft_min_steps: Optional[int] = 0
    soft_max_steps: Optional[int] = None  # set after finding max

    # Persistence
    state_file: str = "io/zoom_state.json"

    # Backend selection: "pigpio" | "rpi" | "mock" | None (auto)
    backend: Optional[str] = None


class _State:
    def __init__(self):
        self.curr_steps: float = 0.0
        self.target_steps: float = 0.0
        self.busy: bool = False
        self.homed: bool = False
        self.soft_min: Optional[int] = None
        self.soft_max: Optional[int] = None
        self.stop_flag: bool = False


# ========================
# Backends
# ========================

class _Backend:
    def setup(self): ...
    def set_enabled(self, en: bool): ...
    def read_min_switch(self) -> Optional[bool]: return None
    def read_max_switch(self) -> Optional[bool]: return None
    def step_once(self, direction_positive: bool): ...
    def cleanup(self): ...

class _MockBackend(_Backend):
    def __init__(self, cfg: ZoomConfig):
        self.cfg = cfg
        self.enabled = False
        self._min_state = False
        self._max_state = False
    def setup(self): self.enabled = True
    def set_enabled(self, en: bool): self.enabled = en
    def read_min_switch(self): return None if self.cfg.min_switch_pin is None else self._min_state
    def read_max_switch(self): return None if self.cfg.max_switch_pin is None else self._max_state
    def step_once(self, direction_positive: bool):
        # No GPIO; timing handled by controller loop sleeps
        return
    def cleanup(self): self.enabled = False

class _PigpioBackend(_Backend):
    def __init__(self, cfg: ZoomConfig):
        self.cfg = cfg
        self.pi = pigpio.pi()
    def setup(self):
        pi = self.pi
        pi.set_mode(self.cfg.step_pin, pigpio.OUTPUT)
        pi.set_mode(self.cfg.dir_pin, pigpio.OUTPUT)
        if self.cfg.en_pin is not None:
            pi.set_mode(self.cfg.en_pin, pigpio.OUTPUT)
            pi.write(self.cfg.en_pin, 1)  # inactive (HIGH)
        for p in (self.cfg.min_switch_pin, self.cfg.max_switch_pin):
            if p is not None:
                pi.set_mode(p, pigpio.INPUT)
                # pull-up if expecting NC to GND
                pi.set_pull_up_down(p, pigpio.PUD_UP if self.cfg.switch_pulldown else pigpio.PUD_DOWN)
    def set_enabled(self, en: bool):
        if self.cfg.en_pin is None: return
        self.pi.write(self.cfg.en_pin, 0 if en else 1)
    def read_min_switch(self):
        if self.cfg.min_switch_pin is None: return None
        v = self.pi.read(self.cfg.min_switch_pin)
        return (v == 0) if self.cfg.switch_pulldown else (v == 1)
    def read_max_switch(self):
        if self.cfg.max_switch_pin is None: return None
        v = self.pi.read(self.cfg.max_switch_pin)
        return (v == 0) if self.cfg.switch_pulldown else (v == 1)
    def step_once(self, direction_positive: bool):
        self.pi.write(self.cfg.dir_pin, 1 if direction_positive else 0)
        # one pulse (~400 ns high recommended; we’ll use ~100 µs for safety in Python)
        self.pi.write(self.cfg.step_pin, 1)
        # pigpio handles fast toggles; but we still sleep in controller for period timing
        self.pi.write(self.cfg.step_pin, 0)
    def cleanup(self):
        self.set_enabled(False)
        self.pi.stop()

class _RpiBackend(_Backend):
    def __init__(self, cfg: ZoomConfig):
        self.cfg = cfg
    def setup(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.cfg.step_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.cfg.dir_pin, GPIO.OUT, initial=GPIO.LOW)
        if self.cfg.en_pin is not None:
            GPIO.setup(self.cfg.en_pin, GPIO.OUT, initial=GPIO.HIGH)  # inactive
        for p in (self.cfg.min_switch_pin, self.cfg.max_switch_pin):
            if p is not None:
                GPIO.setup(p, GPIO.IN, pull_up_down=GPIO.PUD_UP if self.cfg.switch_pulldown else GPIO.PUD_DOWN)
    def set_enabled(self, en: bool):
        if self.cfg.en_pin is None: return
        GPIO.output(self.cfg.en_pin, GPIO.LOW if en else GPIO.HIGH)
    def read_min_switch(self):
        if self.cfg.min_switch_pin is None: return None
        v = GPIO.input(self.cfg.min_switch_pin)
        return (v == GPIO.LOW) if self.cfg.switch_pulldown else (v == GPIO.HIGH)
    def read_max_switch(self):
        if self.cfg.max_switch_pin is None: return None
        v = GPIO.input(self.cfg.max_switch_pin)
        return (v == GPIO.LOW) if self.cfg.switch_pulldown else (v == GPIO.HIGH)
    def step_once(self, direction_positive: bool):
        GPIO.output(self.cfg.dir_pin, GPIO.HIGH if direction_positive else GPIO.LOW)
        GPIO.output(self.cfg.step_pin, GPIO.HIGH)
        GPIO.output(self.cfg.step_pin, GPIO.LOW)
    def cleanup(self):
        try:
            self.set_enabled(False)
            GPIO.cleanup()
        except Exception:
            pass


def _select_backend(cfg: ZoomConfig) -> _Backend:
    if cfg.backend == "pigpio" or (cfg.backend is None and _HAVE_PIGPIO):
        return _PigpioBackend(cfg)
    if cfg.backend == "rpi" or (cfg.backend is None and _HAVE_RPI_GPIO):
        return _RpiBackend(cfg)
    return _MockBackend(cfg)


# ========================
# ZoomMotor
# ========================

class ZoomMotor:
    def __init__(self, cfg: ZoomConfig):
        self.cfg = cfg
        self.state = _State()
        self._bk = _select_backend(cfg)

        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._bk.setup()
        self._bk.set_enabled(True)

        self._load_state()

    # ---------- Lifecycle ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.state.stop_flag = False
        self._thread = threading.Thread(target=self._worker, name="zoom-motor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0):
        with self._lock:
            self.state.stop_flag = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._bk.cleanup()

    # ---------- Public API ----------
    def read_steps(self) -> float:
        with self._lock:
            return float(self.state.curr_steps)

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self.state.busy)

    def goto_steps(self, steps: float):
        with self._lock:
            s = self._clamp_soft_limits(steps)
            self.state.target_steps = float(s)
            self.state.busy = True
            self._cv.notify_all()

    def move_by(self, delta_steps: float):
        self.goto_steps(self.read_steps() + float(delta_steps))

    def home(self) -> bool:
        """
        Homing routine (blocking):
          1) Seek MIN until switch triggers; set zero (optional).
          2) Back off; approach slowly to latch MIN precisely.
          3) Optionally seek MAX and set soft_max.
        """
        # Require at least a min switch for homing
        if self.cfg.min_switch_pin is None:
            return False

        # Ensure worker is running
        self.start()

        # Fast seek toward min
        self._seek_limit(min_side=True, speed=self.cfg.homing_speed_sps)

        # Back off
        self.move_by(+self.cfg.homing_backoff_steps)
        self._wait_idle()

        # Slow approach for precise latch
        self._seek_limit(min_side=True, speed=max(self.cfg.homing_speed_sps * 0.25, 120.0))

        # Zero at min
        with self._lock:
            self.state.curr_steps = 0.0 if self.cfg.home_zero_at_min else self.state.curr_steps
            self.state.homed = True
            self.state.soft_min = 0 if self.cfg.home_zero_at_min else int(self.state.curr_steps)

        # Optionally discover max
        if self.cfg.max_switch_pin is not None and self.cfg.seek_max_after_min:
            # back off from min a bit before heading to max
            self.move_by(+self.cfg.homing_backoff_steps * 2)
            self._wait_idle()
            # seek max
            start = self.read_steps()
            self._seek_limit(min_side=False, speed=self.cfg.homing_speed_sps)
            max_pos = self.read_steps()
            with self._lock:
                self.state.soft_max = int(max_pos)
        else:
            # If no max switch, keep previous or config soft_max
            with self._lock:
                if self.state.soft_max is None and self.cfg.soft_max_steps is not None:
                    self.state.soft_max = int(self.cfg.soft_max_steps)

        self._save_state()
        return True

    # ---------- Internals ----------
    def _worker(self):
        """
        Motion controller: trapezoidal profile, limit checks, timing.
        """
        # Use high-res timer
        last_tick = time.perf_counter()
        v = 0.0  # current speed (steps/s)
        while True:
            with self._lock:
                # stop?
                if self.state.stop_flag:
                    break

                # target?
                curr = self.state.curr_steps
                tgt = self.state.target_steps
                busy = abs(tgt - curr) > 0.5  # within half-step -> done
                self.state.busy = busy

                # choose direction
                dir_pos = (tgt > curr)
                dist = abs(tgt - curr)

            if not busy:
                # Idle: slow down to 0 and wait
                v = 0.0
                with self._lock:
                    self._cv.wait(timeout=0.05)
                continue

            # Step timing based on trapezoid velocity
            now = time.perf_counter()
            dt = max(1e-4, now - last_tick)
            last_tick = now

            # accelerate/decelerate to meet target
            cfg = self.cfg
            a = cfg.accel_sps2
            vmax = max(cfg.min_speed_sps, cfg.max_speed_sps)

            # distance needed to stop: d = v^2 / (2a)
            d_stop = (v * v) / (2.0 * a) if a > 0 else 0.0
            # choose accel sign
            if dist <= d_stop:
                v = max(0.0, v - a * dt)     # decel
            else:
                v = min(vmax, v + a * dt)    # accel

            # enforce minimum motion
            v = max(cfg.min_speed_sps, min(v, vmax))

            # step period
            period = 1.0 / v

            # Check limits before stepping
            min_hit = self._debounced_switch(min_side=True)
            max_hit = self._debounced_switch(min_side=False)
            if min_hit and not dir_pos:
                # at min: clamp and retarget to current
                with self._lock:
                    self.state.curr_steps = float(self.state.soft_min or 0)
                    self.state.target_steps = self.state.curr_steps
                v = 0.0
                time.sleep(0.002)
                continue
            if max_hit and dir_pos:
                with self._lock:
                    # if we know max, clamp to it; else clamp to current
                    clamp = self.state.soft_max if self.state.soft_max is not None else int(self.state.curr_steps)
                    self.state.curr_steps = float(clamp)
                    self.state.target_steps = self.state.curr_steps
                v = 0.0
                time.sleep(0.002)
                continue

            # Perform one step
            self._bk.step_once(direction_positive=dir_pos)
            # Update position
            with self._lock:
                self.state.curr_steps += +1.0 if dir_pos else -1.0

            # Sleep for the remainder of the step
            # (avoid busy loop; coarse timing is fine for zoom)
            tgt_t = last_tick + period
            while True:
                now2 = time.perf_counter()
                rem = tgt_t - now2
                if rem <= 0:
                    break
                time.sleep(min(rem, 0.001))

        # on exit
        self._bk.set_enabled(False)

    def _seek_limit(self, *, min_side: bool, speed: float):
        """
        Seek toward a limit switch at constant speed, stopping on trigger or soft limit.
        Blocking; uses the same stepping primitives as the worker thread.
        """
        # Temporarily override target and step synchronously
        dir_pos = not min_side  # min -> negative, max -> positive
        sps = max(self.cfg.min_speed_sps, min(speed, self.cfg.max_speed_sps))
        period = 1.0 / sps

        # Safety window
        start = self.read_steps()
        while True:
            # If a soft limit exists and we crossed it, stop
            with self._lock:
                soft_min = self.state.soft_min
                soft_max = self.state.soft_max
                pos = self.state.curr_steps

            if min_side and soft_min is not None and pos <= soft_min:
                break
            if (not min_side) and soft_max is not None and pos >= soft_max:
                break

            # If switch is wired, check it
            trig = self._debounced_switch(min_side=min_side)
            if trig:
                break

            # Step
            self._bk.step_once(direction_positive=dir_pos)
            with self._lock:
                self.state.curr_steps += +1.0 if dir_pos else -1.0

            time.sleep(period)

            # Simple watchdog: avoid infinite run if no switch
            if abs(self.read_steps() - start) > 200000:  # absurd travel
                break

    def _debounced_switch(self, *, min_side: bool, samples: int = 3, gap_s: float = 0.0005) -> bool:
        """
        Return True if the requested switch is asserted (with simple debounce).
        If the switch is not present, returns False.
        """
        read_fn = self._bk.read_min_switch if min_side else self._bk.read_max_switch
        if read_fn() is None:
            return False
        ok = 0
        for _ in range(samples):
            val = read_fn()
            if val:
                ok += 1
            time.sleep(gap_s)
        return ok >= (samples // 2 + 1)

    def _wait_idle(self, timeout: float = 10.0):
        t0 = time.time()
        while self.is_busy():
            if time.time() - t0 > timeout:
                break
            time.sleep(0.01)

    def _clamp_soft_limits(self, steps: float) -> float:
        s = float(steps)
        with self._lock:
            if self.state.soft_min is not None:
                s = max(s, float(self.state.soft_min))
            if self.state.soft_max is not None:
                s = min(s, float(self.state.soft_max))
        return s

    # ---------- Persistence ----------
    def _load_state(self):
        try:
            if os.path.isfile(self.cfg.state_file):
                with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                with self._lock:
                    self.state.curr_steps = float(d.get("curr_steps", 0.0))
                    self.state.homed = bool(d.get("homed", False))
                    self.state.soft_min = d.get("soft_min", self.cfg.soft_min_steps)
                    self.state.soft_max = d.get("soft_max", self.cfg.soft_max_steps)
        except Exception:
            # start fresh on parse errors
            pass

    def _save_state(self):
        try:
            d = dict(
                curr_steps=self.read_steps(),
                homed=self.state.homed,
                soft_min=self.state.soft_min if self.state.soft_min is not None else self.cfg.soft_min_steps,
                soft_max=self.state.soft_max if self.state.soft_max is not None else self.cfg.soft_max_steps,
            )
            os.makedirs(os.path.dirname(self.cfg.state_file), exist_ok=True)
            with open(self.cfg.state_file, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass


# ================
# Quick smoke test
# ================

if __name__ == "__main__":
    # Mock test if no GPIO libs
    cfg = ZoomConfig(step_pin=18, dir_pin=23, en_pin=None,
                     min_switch_pin=None, max_switch_pin=None,
                     state_file="io/zoom_state.json",
                     backend="mock")
    zm = ZoomMotor(cfg)
    zm.start()
    print("Homed?", zm.state.homed)
    zm.goto_steps(1000)
    while zm.is_busy():
        print("pos=", zm.read_steps())
        time.sleep(0.05)
    print("Done at", zm.read_steps())
    zm.stop()

