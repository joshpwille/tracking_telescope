# hardware/e_stop.py
from __future__ import annotations
import time
import threading
from typing import Callable, Optional

from hardware.gpio_backend import GPIOIf, IN, PUD_UP


class EStopMonitor:
    """
    Latching emergency stop:
      - If a GPIO pin is provided, monitors it in a background thread.
      - Debounced input; active-low by default (pull-up recommended).
      - Software latch: once engaged(), stays engaged until clear().
      - Optional callback(state: bool) on changes (True=engaged).

    Typical wiring (active-low):
      Pin -> switch -> GND, with internal pull-up enabled.
    """

    def __init__(
        self,
        gpio: Optional[GPIOIf] = None,
        pin: Optional[int] = None,
        *,
        active_low: bool = True,
        debounce_ms: int = 25,
        poll_hz: float = 200.0,
        on_change: Optional[Callable[[bool], None]] = None,
    ):
        self.gpio = gpio
        self.pin = pin
        self.active_low = active_low
        self.debounce_s = max(debounce_ms, 0) / 1000.0
        self.period = 1.0 / max(poll_hz, 1.0)
        self.on_change = on_change

        self._latched = False
        self._last_hw = None  # type: Optional[bool]
        self._last_change_t = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if self.gpio and self.pin is not None:
            # Enable pull-up so open switch reads HIGH
            self.gpio.setup(self.pin, IN, PUD_UP)  # type: ignore

    # ---------- public API ----------
    def start(self) -> None:
        if self.pin is None or self.gpio is None:
            return  # software-only latch; no thread needed
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="EStopMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_engaged(self) -> bool:
        with self._lock:
            return self._latched

    def engage(self) -> None:
        self._set_state(True)

    def clear(self) -> None:
        self._set_state(False)

    def toggle(self) -> None:
        with self._lock:
            self._set_state(not self._latched)

    # ---------- internals ----------
    def _set_state(self, new_state: bool) -> None:
        changed = False
        with self._lock:
            if self._latched != new_state:
                self._latched = new_state
                changed = True
        if changed and self.on_change:
            try:
                self.on_change(self._latched)
            except Exception:
                pass

    def _read_hw_active(self) -> bool:
        val = 1 if self.gpio.input(self.pin) else 0  # type: ignore[arg-type]
        active = (val == 0) if self.active_low else (val == 1)
        return bool(active)

    def _loop(self) -> None:
        # Initialize debouncer
        self._last_hw = self._read_hw_active()
        self._last_change_t = time.monotonic()

        while not self._stop.is_set():
            t = time.monotonic()
            cur = self._read_hw_active()
            if cur != self._last_hw:
                # potential edge: wait debounce, confirm stable
                if (t - self._last_change_t) >= self.debounce_s:
                    # commit edge
                    self._last_hw = cur
                    self._last_change_t = t
                    if cur:
                        # physical E-STOP pressed => latch ON
                        self._set_state(True)
                # else still bouncing; don't update last_change_t to avoid drift
            time.sleep(self.period)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

