# vision/kalman.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class KFParams:
    """
    Tunables for process/measurement noise.
    State: [cx, cy, s, r, vx, vy, vs]  (r = aspect ratio ~ w/h, assumed constant)
    Measurement: [cx, cy, s, r]
    """
    std_pos: float = 1.0          # base pos noise (px)
    std_scale: float = 0.05       # base scale noise (relative)
    std_aspect: float = 0.01      # aspect ratio noise (absolute)
    std_vel: float = 10.0         # base velocity process noise (px/s)
    std_scale_vel: float = 0.1    # scale velocity process noise (1/s)
    meas_pos: float = 1.0         # measurement noise for cx,cy (px)
    meas_scale: float = 0.1       # measurement noise for s (relative)
    meas_aspect: float = 0.05     # measurement noise for r (absolute)
    max_age: int = 30             # frames without update before considered lost
    min_hits: int = 3             # required hits before considered confirmed


def _motion_mats(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build F (state transition) and Q (process noise) given dt.
    State x = [cx, cy, s, r, vx, vy, vs]^T
      cx' = cx + vx*dt
      cy' = cy + vy*dt
      s'  = s  + vs*dt
      r'  = r  (no dynamics)
      vx' = vx
      vy' = vy
      vs' = vs
    """
    F = np.eye(7, dtype=np.float64)
    F[0, 4] = dt
    F[1, 5] = dt
    F[2, 6] = dt
    return F


def _process_Q(dt: float, p: KFParams) -> np.ndarray:
    """
    Rough constant-velocity process noise.
    """
    q = np.zeros((7, 7), dtype=np.float64)
    # Position / velocity
    q_pos = (p.std_pos**2) * np.array([[dt**4/4, dt**3/2],
                                       [dt**3/2, dt**2]], dtype=np.float64)
    # cx,vx
    q[np.ix_([0,4],[0,4])] = q_pos
    # cy,vy
    q[np.ix_([1,5],[1,5])] = q_pos
    # scale, vs
    q_scale = (p.std_scale**2) * np.array([[dt**4/4, dt**3/2],
                                           [dt**3/2, dt**2]], dtype=np.float64)
    q[np.ix_([2,6],[2,6])] = q_scale
    # aspect r (random walk, small)
    q[3,3] = (p.std_aspect**2) * dt
    return q


def _meas_R(p: KFParams) -> np.ndarray:
    R = np.diag([
        p.meas_pos**2,   # cx
        p.meas_pos**2,   # cy
        p.meas_scale**2, # s
        p.meas_aspect**2 # r
    ]).astype(np.float64)
    return R


class KalmanTrack:
    """
    Single-object Kalman filter for [cx, cy, s, r, vx, vy, vs].
    - s ≈ box scale (area) or height; here we use "scale" as sqrt(area) by default.
    - r = aspect ratio (w/h), modeled as near-constant.
    """
    def __init__(self, init_z: np.ndarray, params: Optional[KFParams] = None, id_hint: Optional[int] = None):
        """
        init_z: measurement vector [cx, cy, s, r]
        """
        self.p = params or KFParams()
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.P = np.eye(7, dtype=np.float64) * 1e1  # init covariance
        self.id = id_hint
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self._last_dt = 1.0/30.0  # default

        self._H = np.zeros((4, 7), dtype=np.float64)  # measurement matrix
        self._H[0,0] = 1.0  # cx
        self._H[1,1] = 1.0  # cy
        self._H[2,2] = 1.0  # s
        self._H[3,3] = 1.0  # r
        self._R = _meas_R(self.p)

        self._init_from_measurement(init_z)

    # ----- initialization -----
    def _init_from_measurement(self, z: np.ndarray) -> None:
        z = z.reshape(4, 1).astype(np.float64)
        self.x[:4, 0] = z[:, 0]  # set cx,cy,s,r
        self.x[4:, 0] = 0.0      # zero velocities
        # Set covariances: confident on measurement dims; larger on velocities
        self.P = np.diag([self.p.meas_pos**2,
                          self.p.meas_pos**2,
                          self.p.meas_scale**2,
                          self.p.meas_aspect**2,
                          (self.p.std_vel**2),
                          (self.p.std_vel**2),
                          (self.p.std_scale_vel**2)]).astype(np.float64)

    # ----- predict/update -----
    def predict(self, dt: float) -> None:
        """
        Time-update. Advances the state by dt.
        """
        self.age += 1
        self.time_since_update += 1
        self._last_dt = float(dt)

        F = _motion_mats(dt)
        Q = _process_Q(dt, self.p)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Constrain aspect ratio to sane range
        self.x[3,0] = float(np.clip(self.x[3,0], 0.2, 5.0))

    def update(self, z: np.ndarray) -> None:
        """
        Measurement-update with z = [cx, cy, s, r].
        """
        z = z.reshape(4, 1).astype(np.float64)
        y = z - (self._H @ self.x)                      # innovation
        S = self._H @ self.P @ self._H.T + self._R      # innovation cov
        K = self.P @ self._H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(7, dtype=np.float64)
        self.P = (I - K @ self._H) @ self.P

        self.time_since_update = 0
        self.hits += 1

        # Re-clamp aspect ratio
        self.x[3,0] = float(np.clip(self.x[3,0], 0.2, 5.0))

    # ----- helpers -----
    def measurement_of_state(self) -> np.ndarray:
        """Return expected measurement h(x) = [cx,cy,s,r]."""
        return (self._H @ self.x).ravel()

    def gating_distance(self, z: np.ndarray) -> float:
        """
        Squared Mahalanobis distance (z - Hx)^T S^{-1} (z - Hx).
        """
        z = z.reshape(4, 1).astype(np.float64)
        y = z - (self._H @ self.x)
        S = self._H @ self.P @ self._H.T + self._R
        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            invS = np.linalg.pinv(S)
        m = float((y.T @ invS @ y)[0,0])
        return m

    def is_confirmed(self) -> bool:
        return self.hits >= self.p.min_hits

    def is_deleted(self) -> bool:
        return self.time_since_update > self.p.max_age

    # ----- bbox conversions -----
    @staticmethod
    def _sr_to_wh(s: float, r: float) -> Tuple[float, float]:
        """
        Interpret s as sqrt(area) (SORT convention). Then:
          area = s^2,  r = w/h  => w = s * sqrt(r), h = s / sqrt(r)
        """
        r = max(r, 1e-6)
        w = s * np.sqrt(r)
        h = s / np.sqrt(r)
        return float(w), float(h)

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """
        Return bbox in (x1,y1,x2,y2) from current state mean.
        """
        cx, cy, s, r = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        w, h = self._sr_to_wh(s, r)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (x1, y1, x2, y2)

    @staticmethod
    def meas_from_bbox(x1y1x2y2: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Compute measurement z = [cx, cy, s, r] from a bbox (x1,y1,x2,y2).
        """
        x1, y1, x2, y2 = x1y1x2y2
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = np.sqrt(w * h)
        r = w / h
        return np.array([cx, cy, s, r], dtype=np.float64)

    # ----- debug -----
    def as_dict(self) -> Dict[str, float]:
        cx, cy, s, r, vx, vy, vs = self.x.ravel().tolist()
        return {
            "cx": float(cx), "cy": float(cy),
            "s": float(s), "r": float(r),
            "vx": float(vx), "vy": float(vy), "vs": float(vs),
            "age": int(self.age), "hits": int(self.hits),
            "tsu": int(self.time_since_update)
        }


# ---------------- convenience multi-track wrapper ----------------

class KalmanBank:
    """
    Manage many KalmanTrack objects (ID → track) with simple lifecycle.
    Association is left to your association.py (Hungarian/IoU/etc.).
    """
    def __init__(self, params: Optional[KFParams] = None):
        self.p = params or KFParams()
        self.tracks: Dict[int, KalmanTrack] = {}
        self._next_id = 1

    def new_track(self, z_meas: np.ndarray, id_hint: Optional[int] = None) -> int:
        tid = id_hint if id_hint is not None else self._next_id
        if id_hint is None:
            self._next_id += 1
        self.tracks[tid] = KalmanTrack(z_meas, self.p, id_hint=tid)
        return tid

    def predict_all(self, dt: float) -> None:
        for t in list(self.tracks.values()):
            t.predict(dt)

    def update_track(self, tid: int, z_meas: np.ndarray) -> None:
        if tid in self.tracks:
            self.tracks[tid].update(z_meas)

    def delete_stale(self) -> None:
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].is_deleted():
                del self.tracks[tid]

    def get_states(self) -> Dict[int, Dict[str, float]]:
        return {tid: t.as_dict() for tid, t in self.tracks.items()}

