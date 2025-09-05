"""
target_manager.py
Manages promotion of a track ID to 'active target', stickiness, reacquire, and scoring.

Design goals:
- Single source of truth for which ID is 'active target'
- Robust to short occlusions (stickiness window + reacquire logic)
- Deterministic scoring for auto-pick (nearest-center or highest-confidence)
- Small, fast, dependency-light

Inputs:
- Per-frame track table rows: dicts with keys:
  {id:int, bbox:(x,y,w,h), center:(cx,cy), scale:float, conf:float}

Outputs:
- Active target ID (or None)
- Utility helpers: error-to-center, target row, HUD strings

Usage:
    tm = TargetManager(frame_size=(W,H))
    tm.update(tracks, t_now=monotonic_time)
    tm.lock(23)          # user pressed "23"
    tid = tm.active_id() # current target
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import math
import time


@dataclass
class TargetPolicy:
    # How we choose an automatic target when none is locked:
    #   "nearest" → nearest to image center
    #   "conf"    → highest detection confidence
    mode: str = "nearest"

    # Stickiness: how long we keep a missing target before dropping (sec)
    lost_grace_s: float = 1.0

    # Reacquire radius (px): if a matching ID reappears near last predicted center, relock
    reacquire_radius_px: float = 80.0

    # How many history samples to keep for simple velocity estimate
    hist_len: int = 10

    # Minimum conf to consider as candidate when auto-picking
    min_conf: float = 0.15

    # Hysteresis on auto-switching: require better score by this factor to steal lock
    steal_factor: float = 1.25

    # If the user locked a specific id, do not auto-steal unless target is lost
    protect_user_lock: bool = True


@dataclass
class TargetState:
    id: Optional[int] = None
    user_locked: bool = False
    last_seen_t: float = 0.0
    # Small history of (t, cx, cy) for CV prediction
    history: deque = field(default_factory=lambda: deque(maxlen=20))

    def seen_now(self, t: float, cx: float, cy: float):
        self.last_seen_t = t
        self.history.append((t, cx, cy))

    def predict(self, t: float) -> Tuple[float, float]:
        """Constant-velocity prediction from history; falls back to last sample."""
        if len(self.history) < 2:
            return self.history[-1][1], self.history[-1][2] if self.history else (0.0, 0.0)
        (t1, x1, y1), (t2, x2, y2) = self.history[-2], self.history[-1]
        dt = max(1e-3, t2 - t1)
        vx, vy = (x2 - x1) / dt, (y2 - y1) / dt
        dtp = max(0.0, t - t2)
        return (x2 + vx * dtp, y2 + vy * dtp)


class TargetManager:
    def __init__(self, frame_size: Tuple[int, int], policy: TargetPolicy | None = None):
        self.W, self.H = frame_size
        self.cx, self.cy = self.W * 0.5, self.H * 0.5
        self.policy = policy or TargetPolicy()
        self.state = TargetState()
        # Cache of last rows indexed by id (for quick get)
        self._last_rows: Dict[int, Dict] = {}

    # ----------------- Public API -----------------

    def update(self, tracks: List[Dict], t_now: Optional[float] = None):
        """
        Update manager with current frame's tracks; maintain lock, stickiness, and reacquire.
        """
        if t_now is None:
            t_now = time.monotonic()

        # Update last_rows map
        self._last_rows = {int(r["id"]): r for r in tracks}

        # If we have an active id, refresh last_seen and history if it's present
        if self.state.id is not None and self.state.id in self._last_rows:
            r = self._last_rows[self.state.id]
            (cx, cy) = r["center"]
            self.state.seen_now(t_now, cx, cy)
            return  # still locked and visible

        # Not visible this frame → decide whether to keep, drop, or reacquire
        if self.state.id is not None:
            if self._within_grace(t_now):
                # Try to reacquire same ID if it reappeared near predicted position
                pred_x, pred_y = self._have_prediction(t_now)
                # If the same ID is absent, we can optionally pick a *new* ID only if not user-locked
                if (not self.state.user_locked) and self._should_steal(tracks, pred_x, pred_y):
                    self._auto_pick(tracks, t_now)
                # else keep waiting inside grace window
                return
            else:
                # Grace window expired
                self.clear()

        # No active target: consider auto-picking best candidate
        if self.state.id is None:
            self._auto_pick(tracks, t_now)

    def lock(self, target_id: int):
        """User requested a specific ID to follow (promote to active target)."""
        self.state.id = int(target_id)
        self.state.user_locked = True
        self.state.history.clear()
        self._touch_seen(time.monotonic())

    def clear(self):
        """Clear current target."""
        self.state = TargetState()

    def active_id(self) -> Optional[int]:
        return self.state.id

    def active_row(self) -> Optional[Dict]:
        tid = self.state.id
        return self._last_rows.get(tid) if tid is not None else None

    def error_to_center_px(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns (dx, dy, r) from principal point (frame center) to active target center.
        """
        row = self.active_row()
        if not row:
            return None
        cx, cy = row["center"]
        dx, dy = cx - self.cx, cy - self.cy
        r = math.hypot(dx, dy)
        return dx, dy, r

    def hud_string(self) -> str:
        """
        Short status line for overlay.
        """
        tid = self.state.id
        if tid is None:
            return "TARGET: none"
        lock = "USER" if self.state.user_locked else "AUTO"
        row = self.active_row()
        if row:
            (dx, dy, r) = self.error_to_center_px()
            return f"TARGET[{lock}] id={tid} err=({dx:.1f},{dy:.1f}) r={r:.1f}px conf={row.get('conf', 0):.2f}"
        # Missing but within grace:
        remaining = max(0.0, self._grace_remaining(time.monotonic()))
        return f"TARGET[{lock}] id={tid} LOST ({remaining:.2f}s grace)"

    # ----------------- Internals -----------------

    def _within_grace(self, t_now: float) -> bool:
        return (t_now - self.state.last_seen_t) <= self.policy.lost_grace_s

    def _grace_remaining(self, t_now: float) -> float:
        delta = t_now - self.state.last_seen_t
        return max(0.0, self.policy.lost_grace_s - delta)

    def _touch_seen(self, t_now: float):
        row = self.active_row()
        if row:
            cx, cy = row["center"]
            self.state.seen_now(t_now, cx, cy)
        else:
            # No row yet; initialize with center
            self.state.seen_now(t_now, self.cx, self.cy)

    def _have_prediction(self, t_now: float) -> Tuple[float, float]:
        if self.state.history:
            return self.state.predict(t_now)
        return (self.cx, self.cy)

    def _score(self, row: Dict) -> float:
        """Higher is better."""
        mode = self.policy.mode.lower()
        if mode == "conf":
            return float(row.get("conf", 0.0))
        # nearest: inverse distance to center
        cx, cy = row["center"]
        r = math.hypot(cx - self.cx, cy - self.cy) + 1e-6
        return 1.0 / r

    def _best_candidate(self, tracks: List[Dict]) -> Optional[int]:
        best_id, best_score = None, -1.0
        for r in tracks:
            if float(r.get("conf", 0.0)) < self.policy.min_conf:
                continue
            s = self._score(r)
            if s > best_score:
                best_id, best_score = int(r["id"]), s
        return best_id

    def _should_steal(self, tracks: List[Dict], pred_x: float, pred_y: float) -> bool:
        """
        Decide if we may auto-switch while the locked target is temporarily missing.
        Respect user_lock unless protect_user_lock==False.
        """
        if self.state.user_locked and self.policy.protect_user_lock:
            return False
        # If a candidate is *much* better (steal_factor), allow switch
        best_id = self._best_candidate(tracks)
        if best_id is None:
            return False
        # Compare best candidate to hypothetical score of predicted locked target
        fake_row = {"center": (pred_x, pred_y), "conf": 1.0}
        s_best = self._score(self._last_rows[best_id])
        s_pred = self._score(fake_row)
        return s_best > self.policy.steal_factor * s_pred

    def _auto_pick(self, tracks: List[Dict], t_now: float):
        tid = self._best_candidate(tracks)
        if tid is None:
            return
        self.state.id = tid
        self.state.user_locked = False
        self.state.history.clear()
        self._touch_seen(t_now)

    # ------------- Reacquire helpers -------------

    def try_reacquire_by_id(self, tid: int, t_now: Optional[float] = None) -> bool:
        """
        If the specific ID reappears and is close to prediction, relock.
        """
        if self.state.id is None or self.state.id != tid:
            return False
        if t_now is None:
            t_now = time.monotonic()
        row = self._last_rows.get(tid)
        if not row:
            return False
        px, py = self._have_prediction(t_now)
        cx, cy = row["center"]
        if math.hypot(cx - px, cy - py) <= self.policy.reacquire_radius_px:
            self._touch_seen(t_now)
            return True
        return False

