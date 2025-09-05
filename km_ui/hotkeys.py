# ui/hotkeys.py
"""
Hotkey handler that merges OpenCV keypresses and stdin messages into
canonical events for the pipeline.

Usage in your loop:
    hk = HotkeyHandler()
    ...
    # 1) process cv2 key (if you show a window)
    key = cv2.waitKey(1) & 0xFF
    hk.process_cv_key(key)

    # 2) process stdin messages from ui.stdin_listener queue
    try:
        while True:
            msg = stdin_q.get_nowait()
            hk.process_stdin_message(msg)
    except queue.Empty:
        pass

    # 3) tick to commit pending numeric IDs on timeout
    events = hk.tick()  # list[HotkeyEvent]
    for ev in events:
        handle_event(ev)  # your code: lock id, toggle motors, etc.

Design goals:
- One-keystroke ID promotion: type digits '23' and wait ~0.8s (or press ENTER) → emits lock_id=23
- Non-blocking; stateless from caller perspective
- Includes helpful toggles: pause, overlay, motors, E-STOP, gains, detection cadence, target policy
- Manual jog via arrows/WASD

Event schema (HotkeyEvent.type -> payload):
  'quit'                         -> {}
  'clear_target'                 -> {}
  'lock_id'                      -> {'id': int}
  'estop'                        -> {}     # engage
  'estop_clear'                  -> {}     # clear
  'toggle_overlay'               -> {}
  'toggle_motors'                -> {}
  'toggle_pause'                 -> {}
  'gain_delta'                   -> {'d': float}    # relative change
  'set_gain'                     -> {'value': float}
  'detect_every_n_delta'         -> {'d': int}      # +/- 1
  'set_detect_every_n'           -> {'n': int}
  'target_policy_cycle'          -> {}              # cycle 'nearest'/'conf'
  'manual_jog'                   -> {'dx': float, 'dy': float}   # radians or deg (your choice)

You can extend the mapping below to fit your rig.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

# ---------- Event Dataclass ----------

@dataclass
class HotkeyEvent:
    type: str
    payload: Dict = None


# ---------- Handler ----------

class HotkeyHandler:
    def __init__(self,
                 id_commit_timeout_s: float = 0.8,
                 jog_step: float = 0.002,   # radians (example); tune for your rig
                 gain_step: float = 0.10,
                 detect_every_n_choices: List[int] = [1, 2, 3, 5]):
        self._events: List[HotkeyEvent] = []
        self._digits: str = ""
        self._last_digit_t: float = 0.0
        self._id_commit_timeout = float(id_commit_timeout_s)
        self._jog = float(jog_step)
        self._gain_step = float(gain_step)
        self._detect_choices = detect_every_n_choices
        self._detect_idx = 0  # index into detect choices
        self._policy_modes = ["nearest", "conf"]
        self._policy_idx = 0
        self._estopped = False

    # ---- Public API ----

    def process_cv_key(self, key: int | None):
        """Map cv2.waitKey code into events."""
        if key is None or key == -1:
            return

        # Digits accumulate into an ID
        if 48 <= key <= 57:  # '0'..'9'
            self._push_digit(chr(key))
            return

        # ENTER or RETURN commits ID if any
        if key in (10, 13):  # LF/CR
            self._commit_digits()
            return

        # Common letters
        c = chr(key).lower() if 32 <= key < 127 else ""

        if c == 'q':
            self._emit("quit")
            return
        if c == 'n':
            self._emit("clear_target")
            return
        if c == ' ':
            # Space toggles E-STOP
            if self._estopped:
                self._emit("estop_clear"); self._estopped = False
            else:
                self._emit("estop"); self._estopped = True
            return
        if c == 'o':
            self._emit("toggle_overlay"); return
        if c == 'm':
            self._emit("toggle_motors"); return
        if c == 'p':
            self._emit("toggle_pause"); return

        # Gains
        if c in ('+', '='):
            self._emit("gain_delta", {'d': +self._gain_step}); return
        if c == '-':
            self._emit("gain_delta", {'d': -self._gain_step}); return

        # Detection cadence
        if c == 'd':
            self._detect_idx = (self._detect_idx + 1) % len(self._detect_choices)
            self._emit("set_detect_every_n", {'n': self._detect_choices[self._detect_idx]})
            return

        # Target policy cycle
        if c == 'c':
            self._policy_idx = (self._policy_idx + 1) % len(self._policy_modes)
            self._emit("target_policy_cycle", {'mode': self._policy_modes[self._policy_idx]})
            return

        # Manual jog (WASD)
        if c == 'w':
            self._emit("manual_jog", {'dx': 0.0,        'dy': -self._jog}); return
        if c == 's':
            self._emit("manual_jog", {'dx': 0.0,        'dy': +self._jog}); return
        if c == 'a':
            self._emit("manual_jog", {'dx': -self._jog, 'dy': 0.0});        return
        if c == 'd':
            # note: 'd' is used above for detect cadence; prioritize cadence over jog right
            # if you prefer jog-right on 'd', swap these handlers.
            return

        # Arrow keys (often 81..84 on cv2)
        if key in (82,):  # up
            self._emit("manual_jog", {'dx': 0.0,        'dy': -self._jog}); return
        if key in (84,):  # down
            self._emit("manual_jog", {'dx': 0.0,        'dy': +self._jog}); return
        if key in (81,):  # left
            self._emit("manual_jog", {'dx': -self._jog, 'dy': 0.0});        return
        if key in (83,):  # right
            self._emit("manual_jog", {'dx': +self._jog, 'dy': 0.0});        return

        # Ignore other keys

    def process_stdin_message(self, msg: str):
        """
        Parse stdin commands, e.g.:
          '23'        -> lock_id=23
          'n'         -> clear target
          'q'         -> quit
          'stop'      -> estop
          'go'        -> estop_clear
          'gain +0.2' -> gain_delta +0.2
          'gain 0.8'  -> set_gain 0.8
          'det 3'     -> set_detect_every_n 3
          'policy conf' or 'policy nearest'
        """
        m = (msg or "").strip().lower()
        if not m:
            return

        # Pure integer -> lock_id
        if m.isdigit():
            self._emit("lock_id", {'id': int(m)})
            self._clear_digits()
            return

        if m in ("n", "none", "neutral"):
            self._emit("clear_target"); return
        if m in ("q", "quit", "exit"):
            self._emit("quit"); return
        if m in ("stop", "estop", "e-stop"):
            self._emit("estop"); self._estopped = True; return
        if m in ("go", "clear", "resume"):
            self._emit("estop_clear"); self._estopped = False; return

        # gain commands
        if m.startswith("gain "):
            try:
                val = float(m.split()[1])
                if m.split()[1].startswith(('+', '-')):
                    self._emit("gain_delta", {'d': val})
                else:
                    self._emit("set_gain", {'value': val})
            except Exception:
                pass
            return

        # detection cadence
        if m.startswith("det "):
            try:
                n = int(m.split()[1])
                self._emit("set_detect_every_n", {'n': n})
            except Exception:
                pass
            return

        # policy
        if m.startswith("policy"):
            parts = m.split()
            if len(parts) == 2 and parts[1] in ("nearest", "conf"):
                self._emit("target_policy_cycle", {'mode': parts[1]})
            else:
                # cycle if unspecified
                self._policy_idx = (self._policy_idx + 1) % len(self._policy_modes)
                self._emit("target_policy_cycle", {'mode': self._policy_modes[self._policy_idx]})
            return

    def tick(self) -> List[HotkeyEvent]:
        """
        Call once per frame; commits pending numeric ID if timeout elapsed.
        """
        if self._digits and (time.time() - self._last_digit_t) >= self._id_commit_timeout:
            self._commit_digits()
        out, self._events = self._events, []
        return out

    # ---- Internals ----

    def _emit(self, etype: str, payload: Optional[Dict] = None):
        self._events.append(HotkeyEvent(etype, payload or {}))

    def _push_digit(self, d: str):
        self._digits += d
        self._last_digit_t = time.time()

    def _commit_digits(self):
        if not self._digits:
            return
        try:
            tid = int(self._digits)
            self._emit("lock_id", {'id': tid})
        finally:
            self._clear_digits()

    def _clear_digits(self):
        self._digits = ""
        self._last_digit_t = 0.0

