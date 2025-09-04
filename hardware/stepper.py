# hardware/stepper.py
from __future__ import annotations
import time
from typing import List, Sequence, Optional

from hardware.gpio_backend import GPIOIf, OUT, HIGH, LOW


class StepperMotor:
    """
    Simple 4-phase stepper motor driver using GPIO backend.
    Works with ULN2003/L293D driver boards or direct GPIO drivers.

    - Supports full-step or half-step sequences.
    - Speed controlled via step_delay (seconds between steps).
    - Non-blocking stop() available to de-energize coils.
    """

    # Full-step sequence (single coil at a time)
    FULL_SEQ: List[List[int]] = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    # Half-step sequence (finer resolution, smoother motion)
    HALF_SEQ: List[List[int]] = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    def __init__(
        self,
        gpio: GPIOIf,
        pins: Sequence[int],
        mode: str = "half",
        step_delay: float = 0.002,
    ):
        """
        Args:
            gpio: GPIO backend (e.g., from gpio_backend.make_gpio()).
            pins: 4 GPIO pins connected to IN1..IN4 of driver.
            mode: "full" or "half".
            step_delay: seconds to wait between steps (controls speed).
        """
        if len(pins) != 4:
            raise ValueError("Stepper requires exactly 4 pins")

        self.gpio = gpio
        self.pins = list(pins)
        self.mode = mode.lower()
        self.seq = StepperMotor.HALF_SEQ if self.mode == "half" else StepperMotor.FULL_SEQ
        self.step_delay = step_delay
        self._pos = 0  # index in sequence

        # Init pins
        for p in self.pins:
            self.gpio.setup(p, OUT)
            self.gpio.output(p, LOW)

    def step(self, steps: int = 1, direction: int = 1, delay: Optional[float] = None) -> None:
        """
        Step the motor a given number of steps.
        Args:
            steps: number of steps to move
            direction: +1 (forward) or -1 (reverse)
            delay: override step delay in seconds
        """
        d = delay if delay is not None else self.step_delay
        seq_len = len(self.seq)

        for _ in range(steps):
            self._pos = (self._pos + direction) % seq_len
            pattern = self.seq[self._pos]
            self._apply(pattern)
            time.sleep(d)

    def rotate_degrees(self, degrees: float, steps_per_rev: int, direction: int = 1) -> None:
        """
        Rotate a given number of degrees (blocking).
        Args:
            degrees: rotation in degrees
            steps_per_rev: motor's steps per full revolution (e.g., 2048 for 28BYJ-48 half-step)
            direction: +1 or -1
        """
        total_steps = int(steps_per_rev * (degrees / 360.0))
        self.step(total_steps, direction)

    def stop(self) -> None:
        """De-energize coils (release motor)."""
        for p in self.pins:
            self.gpio.output(p, LOW)

    def _apply(self, pattern: List[int]) -> None:
        for p, v in zip(self.pins, pattern):
            self.gpio.output(p, HIGH if v else LOW)

    def cleanup(self) -> None:
        self.stop()
        self.gpio.cleanup()


# --------------- quick demo ----------------
if __name__ == "__main__":
    from hardware.gpio_backend import make_gpio
    gpio = make_gpio("null")  # replace with "rpi" or "jetson" on hardware
    stepper = StepperMotor(gpio, pins=[17, 18, 27, 22], mode="half", step_delay=0.002)

    print("Stepping forward 100 half-steps...")
    stepper.step(100, direction=1)
    print("Stepping backward 100 half-steps...")
    stepper.step(100, direction=-1)

    stepper.stop()
    stepper.cleanup()

