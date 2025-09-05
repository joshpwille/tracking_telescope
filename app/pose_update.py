# app/pose_update.py
"""
Pose update from pan/tilt encoders → camera rotation R(θ, φ) (and optional lever-arm t).

Why this exists
- The pipeline must update each camera’s orientation R from encoders every frame,
  and (if the optical center is offset from the pan/tilt pivot) apply a simple lever-arm
  translation model to keep [R|t] correct while slewing. 【turn10file1†L23-L31】

Key features
- Reads radians from hardware.encoder.Encoder (pan, tilt are already smoothed there).
- Configurable axis conventions:
    * pan axis: 'Z' (default) or 'Y'
    * tilt axis: 'X' (default) or 'Y'
    * composition order: 'pan_then_tilt' or 'tilt_then_pan'
    * per-axis sign and zero offsets (fine-tune to your physical zero)
- Optional fixed mount→camera alignment R_cam0_from_mount at zero angles.
- Optional lever arm: camera center C = P_pivot + R_mount · o_cam_from_pivot.
- Returns rotation R_cam (world→camera) and t_cam (camera center in world, if requested).
- Helpers to emit [R|t] and P = K [R|t].

Coordinate notes (defaults; adjust in config if your rig differs)
- World frame: Z up, X right, Y forward (NED/ENU doesn’t matter as long as consistent).
- Camera frame (OpenCV-style): x right, y down, z forward.
- Default rotations: pan = yaw about +Z, tilt = pitch about +X, applied as R = R_tilt * R_pan.
  (Use config to swap axes/order/signs if your gimbal differs.)

Dependencies: numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import math
import time

try:
    # Your encoder module from hardware/
    from hardware.encoder import Encoder
except Exception as e:
    Encoder = object  # type: ignore


# =========================
# Configuration
# =========================

@dataclass
class PoseConfig:
    # Angles read from encoders are radians (continuous).
    # Choose axes for the mechanical pan/tilt:
    pan_axis: str = "Z"   # 'Z' or 'Y' (yaw)
    tilt_axis: str = "X"  # 'X' or 'Y' (pitch)
    # Composition order (which rotation happens first in world frame)
    order: str = "pan_then_tilt"  # or 'tilt_then_pan'

    # Per-axis sign and zero offset corrections (post-encoder)
    pan_sign: float = +1.0
    tilt_sign: float = +1.0
    pan_zero_rad: float = 0.0
    tilt_zero_rad: float = 0.0

    # Fixed mount→camera alignment at zero angles (3x3), if camera is not perfectly aligned
    R_cam0_from_mount: Optional[np.ndarray] = None  # default: identity

    # Lever arm model: camera center relative to pivot expressed in mount frame at zero angles
    # If None, translation is ignored (optical center at pivot).
    o_cam_from_pivot_m: Optional[np.ndarray] = None  # shape (3,), meters

    # Pivot position in world frame (meters); use if you want world-space t
    P_pivot_world_m: Optional[np.ndarray] = None  # shape (3,)

    # Safety: clamp absurd angles (e.g., encoder glitch)
    max_abs_pan_rad: float = math.radians(181.0)
    max_abs_tilt_rad: float = math.radians(95.0)


# =========================
# Math helpers
# =========================

def _Rx(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _Ry(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _Rz(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]], dtype=float)

def _R_axis(axis: str, angle: float) -> np.ndarray:
    axis = axis.upper()
    if axis == "X": return _Rx(angle)
    if axis == "Y": return _Ry(angle)
    if axis == "Z": return _Rz(angle)
    raise ValueError(f"Unsupported axis '{axis}' (must be X/Y/Z)")


# =========================
# Pose updater
# =========================

class PoseUpdater:
    """
    Reads encoders → computes camera rotation R_cam and optional t_cam using a lever arm.
    """
    def __init__(self, encoder: Encoder, cfg: PoseConfig = PoseConfig()):
        self.encoder = encoder
        self.cfg = cfg
        if self.cfg.R_cam0_from_mount is None:
            self.cfg.R_cam0_from_mount = np.eye(3, dtype=float)

        # Sanity checks on vectors if provided
        if self.cfg.o_cam_from_pivot_m is not None:
            self.cfg.o_cam_from_pivot_m = np.asarray(self.cfg.o_cam_from_pivot_m, dtype=float).reshape(3)
        if self.cfg.P_pivot_world_m is not None:
            self.cfg.P_pivot_world_m = np.asarray(self.cfg.P_pivot_world_m, dtype=float).reshape(3)

        self._last_pan = None
        self._last_tilt = None
        self._last_t = 0.0

    def read_angles(self) -> Tuple[float, float]:
        """Read (pan, tilt) radians from the encoder and apply sign/zero and clamps."""
        pan_raw, tilt_raw = self.encoder.read_pan_tilt_rad()  # continuous radians
        pan = self.cfg.pan_sign * pan_raw - self.cfg.pan_zero_rad
        tilt = self.cfg.tilt_sign * tilt_raw - self.cfg.tilt_zero_rad

        # Clamp to reasonable range (avoid wild matrices on glitches)
        pan = float(max(-self.cfg.max_abs_pan_rad, min(self.cfg.max_abs_pan_rad, pan)))
        tilt = float(max(-self.cfg.max_abs_tilt_rad, min(self.cfg.max_abs_tilt_rad, tilt)))
        return pan, tilt

    def rotation_mount(self, pan: float, tilt: float) -> np.ndarray:
        """
        Build the mount rotation R_mount (world→mount’s camera-plate frame) from pan/tilt.
        Order is applied in world frame (left-multiply).
        """
        R_pan  = _R_axis(self.cfg.pan_axis,  pan)
        R_tilt = _R_axis(self.cfg.tilt_axis, tilt)

        if self.cfg.order == "pan_then_tilt":
            R_mount = R_tilt @ R_pan
        elif self.cfg.order == "tilt_then_pan":
            R_mount = R_pan  @ R_tilt
        else:
            raise ValueError("order must be 'pan_then_tilt' or 'tilt_then_pan'")
        return R_mount

    def rotation_camera(self, R_mount: np.ndarray) -> np.ndarray:
        """
        Map mount plate frame to camera optical frame at zero via R_cam0_from_mount.
        Returns R_cam (world→camera).
        """
        return self.cfg.R_cam0_from_mount @ R_mount

    def lever_arm_translation(self, R_mount: np.ndarray) -> Optional[np.ndarray]:
        """
        If lever arm is configured, compute camera center in world: C = P + R_mount · o.
        Returns C (world), NOT t; t depends on your extrinsics convention.
        """
        if self.cfg.o_cam_from_pivot_m is None:
            return None
        o = self.cfg.o_cam_from_pivot_m.reshape(3, 1)
        C_mount = R_mount @ o  # expressed in world since R_mount is world→mount? (see note below)
        # Note: We defined R_mount as world→mount-camera-plate. To map a mount-fixed vector
        # into world, we should use R_world_from_mount = R_mount.T. We want world vector:
        C_world_offset = (R_mount.T @ o).reshape(3)
        P = self.cfg.P_pivot_world_m if self.cfg.P_pivot_world_m is not None else np.zeros(3)
        return P + C_world_offset

    def update(self) -> Dict[str, np.ndarray]:
        """
        Compute current R_cam and (optionally) camera center C_world.
        Returns dict with 'R' (3x3), and maybe 'C_world' (3,).
        """
        pan, tilt = self.read_angles()
        R_mount = self.rotation_mount(pan, tilt)
        R_cam = self.rotation_camera(R_mount)

        out = {"R": R_cam}
        C_world = self.lever_arm_translation(R_mount)
        if C_world is not None:
            out["C_world"] = C_world
        self._last_pan, self._last_tilt, self._last_t = pan, tilt, time.time()
        return out

    # ---------- convenience ----------

    def extrinsics_R_t(self, *, assume_world_t_is_camera_center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (R_cam, t) suitable for P = K [R | t].
        Convention:
          - If assume_world_t_is_camera_center=True (common when world origin is camera),
            we return t = -R * C_world (if lever arm used) or zeros.
          - Otherwise, adapt to your chosen world frame.
        """
        st = self.update()
        R = st["R"]
        if "C_world" in st and assume_world_t_is_camera_center:
            Cw = st["C_world"].reshape(3, 1)
            t = -R @ Cw
        else:
            t = np.zeros((3, 1), dtype=float)
        return R, t

    def projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """If you have K, return P = K [R | t] for current pose."""
        R, t = self.extrinsics_R_t()
        Rt = np.hstack([R, t])
        return K @ Rt


# =========================
# Example usage / smoke test
# =========================

if __name__ == "__main__":
    # Minimal smoke test with a mock encoder (replace with your real Encoder)
    class _MockEnc:
        def __init__(self): self.t0 = time.time()
        def read_pan_tilt_rad(self):
            t = time.time() - self.t0
            return 0.2 * math.sin(0.4 * t), 0.1 * math.sin(0.6 * t)

    enc = _MockEnc()
    cfg = PoseConfig(
        pan_axis="Z", tilt_axis="X",
        order="pan_then_tilt",
        pan_sign=+1.0, tilt_sign=+1.0,
        pan_zero_rad=0.0, tilt_zero_rad=0.0,
        R_cam0_from_mount=np.eye(3),
        o_cam_from_pivot_m=np.array([0.0, 0.0, 0.0]),   # set nonzero if your camera is offset
        P_pivot_world_m=np.array([0.0, 0.0, 0.0])
    )
    updater = PoseUpdater(enc, cfg)

    for _ in range(3):
        pose = updater.update()
        R = pose["R"]
        print("R=\n", R)
        if "C_world" in pose:
            print("C_world=", pose["C_world"])
        time.sleep(0.05)
