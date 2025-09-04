# hardware/gpio_backend.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Literal, Protocol

# Public constants (normalized)
Mode = Literal["IN", "OUT"]
Pull = Literal["OFF", "UP", "DOWN"]
Level = Literal[0, 1]

IN: Mode = "IN"
OUT: Mode = "OUT"
PUD_OFF: Pull = "OFF"
PUD_UP: Pull = "UP"
PUD_DOWN: Pull = "DOWN"
LOW: Level = 0
HIGH: Level = 1


class GPIOIf(Protocol):
    """Minimal GPIO interface used by the rest of the project."""
    def setup(self, pin: int, mode: Mode, pull: Pull = PUD_OFF) -> None: ...
    def output(self, pin: int, value: Level) -> None: ...
    def output_many(self, pins: Iterable[int], values: Iterable[Level]) -> None: ...
    def input(self, pin: int) -> int: ...
    def cleanup(self, pin: Optional[int] = None) -> None: ...
    def pwm(self, pin: int, freq_hz: float) -> "PWMIf": ...


class PWMIf(Protocol):
    def start(self, duty_cycle: float) -> None: ...
    def ChangeFrequency(self, freq_hz: float) -> None: ...
    def ChangeDutyCycle(self, duty_cycle: float) -> None: ...
    def stop(self) -> None: ...


# -------------------- Null / No-op backend --------------------

class _NullPWM:
    def __init__(self, pin: int, freq: float): self.pin, self.freq, self.dc = pin, freq, 0.0
    def start(self, duty_cycle: float) -> None: self.dc = duty_cycle
    def ChangeFrequency(self, freq_hz: float) -> None: self.freq = freq_hz
    def ChangeDutyCycle(self, duty_cycle: float) -> None: self.dc = duty_cycle
    def stop(self) -> None: pass


class NullGPIO(GPIOIf):
    """No-op backend for development on laptops/servers."""
    def __init__(self) -> None:
        self._state: dict[int, int] = {}

    def setup(self, pin: int, mode: Mode, pull: Pull = PUD_OFF) -> None:
        self._state.setdefault(pin, LOW)

    def output(self, pin: int, value: Level) -> None:
        self._state[pin] = int(1 if value else 0)

    def output_many(self, pins: Iterable[int], values: Iterable[Level]) -> None:
        for p, v in zip(pins, values):
            self.output(p, v)

    def input(self, pin: int) -> int:
        return int(self._state.get(pin, LOW))

    def cleanup(self, pin: Optional[int] = None) -> None:
        if pin is None:
            self._state.clear()
        else:
            self._state.pop(pin, None)

    def pwm(self, pin: int, freq_hz: float) -> PWMIf:
        return _NullPWM(pin, freq_hz)


# -------------------- Raspberry Pi backend (RPi.GPIO) --------------------

class RPiGPIO(GPIOIf):
    def __init__(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except Exception as e:
            raise RuntimeError("RPi.GPIO not available. Are you on a Raspberry Pi?") from e
        self.GPIO = GPIO
        self.GPIO.setmode(self.GPIO.BCM)
        self._pwm_objs: dict[int, "RPiGPIO._PWM"] = {}

    def setup(self, pin: int, mode: Mode, pull: Pull = PUD_OFF) -> None:
        m = self.GPIO.OUT if mode == OUT else self.GPIO.IN
        pud_map = {
            PUD_OFF: self.GPIO.PUD_OFF,
            PUD_UP: self.GPIO.PUD_UP,
            PUD_DOWN: self.GPIO.PUD_DOWN,
        }
        self.GPIO.setup(pin, m, pull_up_down=pud_map[pull])

    def output(self, pin: int, value: Level) -> None:
        self.GPIO.output(pin, self.GPIO.HIGH if value else self.GPIO.LOW)

    def output_many(self, pins: Iterable[int], values: Iterable[Level]) -> None:
        # RPi.GPIO doesn't have a vectorized write; loop.
        for p, v in zip(pins, values):
            self.output(p, v)

    def input(self, pin: int) -> int:
        return int(self.GPIO.input(pin))

    def cleanup(self, pin: Optional[int] = None) -> None:
        if pin is None:
            self.GPIO.cleanup()
        else:
            self.GPIO.cleanup(pin)

    class _PWM:
        def __init__(self, pwm):
            self._pwm = pwm
        def start(self, duty_cycle: float) -> None: self._pwm.start(duty_cycle)
        def ChangeFrequency(self, freq_hz: float) -> None: self._pwm.ChangeFrequency(freq_hz)
        def ChangeDutyCycle(self, duty_cycle: float) -> None: self._pwm.ChangeDutyCycle(duty_cycle)
        def stop(self) -> None: self._pwm.stop()

    def pwm(self, pin: int, freq_hz: float) -> PWMIf:
        pwm = self.GPIO.PWM(pin, freq_hz)
        return RPiGPIO._PWM(pwm)


# -------------------- Jetson backend (Jetson.GPIO) --------------------

class JetsonGPIO(GPIOIf):
    def __init__(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore
        except Exception as e:
            raise RuntimeError("Jetson.GPIO not available. Are you on a Jetson device?") from e
        self.GPIO = GPIO
        # Use BCM numbering to match Pi-style pin maps in code
        try:
            self.GPIO.setmode(self.GPIO.BCM)
        except Exception:
            # Some Jetson images require BOARD mode; fallback if needed
            self.GPIO.setmode(self.GPIO.BOARD)
        self._pwm_objs: dict[int, "JetsonGPIO._PWM"] = {}

    def setup(self, pin: int, mode: Mode, pull: Pull = PUD_OFF) -> None:
        m = self.GPIO.OUT if mode == OUT else self.GPIO.IN
        pud_map = {
            PUD_OFF: self.GPIO.PUD_OFF,
            PUD_UP: self.GPIO.PUD_UP,
            PUD_DOWN: self.GPIO.PUD_DOWN,
        }
        # Not all pins support pull resistors on Jetson; ignore if unsupported
        try:
            self.GPIO.setup(pin, m, pull_up_down=pud_map[pull])
        except Exception:
            self.GPIO.setup(pin, m)

    def output(self, pin: int, value: Level) -> None:
        self.GPIO.output(pin, self.GPIO.HIGH if value else self.GPIO.LOW)

    def output_many(self, pins: Iterable[int], values: Iterable[Level]) -> None:
        for p, v in zip(pins, values):
            self.output(p, v)

    def input(self, pin: int) -> int:
        return int(self.GPIO.input(pin))

    def cleanup(self, pin: Optional[int] = None) -> None:
        try:
            if pin is None:
                self.GPIO.cleanup()
            else:
                self.GPIO.cleanup(pin)
        except Exception:
            pass

    class _PWM:
        def __init__(self, pwm):
            self._pwm = pwm
        def start(self, duty_cycle: float) -> None: self._pwm.start(duty_cycle)
        def ChangeFrequency(self, freq_hz: float) -> None: self._pwm.ChangeFrequency(freq_hz)
        def ChangeDutyCycle(self, duty_cycle: float) -> None: self._pwm.ChangeDutyCycle(duty_cycle)
        def stop(self) -> None: self._pwm.stop()

    def pwm(self, pin: int, freq_hz: float) -> PWMIf:
        # Jetson.GPIO supports software PWM similar to RPi.GPIO
        pwm = self.GPIO.PWM(pin, freq_hz)
        return JetsonGPIO._PWM(pwm)


# -------------------- Factory --------------------

@dataclass
class GPIOFactory:
    """Create a GPIO backend by name: 'null', 'rpi', or 'jetson'."""
    backend: str = "null"  # default safe no-op

    def create(self) -> GPIOIf:
        b = self.backend.strip().lower()
        if b == "null":
            return NullGPIO()
        if b in ("rpi", "raspi", "raspberrypi"):
            return RPiGPIO()
        if b in ("jetson", "nvidia"):
            return JetsonGPIO()
        # Auto-detect if requested
        if b in ("auto", "autodetect"):
            try:
                return RPiGPIO()
            except Exception:
                try:
                    return JetsonGPIO()
                except Exception:
                    return NullGPIO()
        # Fallback to null with warning
        print(f"[gpio_backend] Unknown backend '{self.backend}', using NullGPIO.")
        return NullGPIO()


# -------------------- Convenience --------------------

def make_gpio(backend: str) -> GPIOIf:
    """Shorthand factory function."""
    return GPIOFactory(backend=backend).create()


# Example usage:
# from io.config import get_config
# from hardware.gpio_backend import make_gpio, OUT, IN, HIGH, LOW, PUD_UP
#
# cfg = get_config()
# gpio = make_gpio(cfg.pins.backend)   # "null" / "rpi" / "jetson"
# for pin in cfg.pins.stepper_pan:
#     gpio.setup(pin, OUT)
# gpio.output(cfg.pins.stepper_pan[0], HIGH)
# gpio.cleanup()

