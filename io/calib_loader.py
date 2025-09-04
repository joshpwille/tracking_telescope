# io/calib_loader.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2

# ---------- data containers ----------
@dataclass
class MonoCalib:
    K: np.ndarray              # 3x3
    dist: np.ndarray           # (N,) or (1,N)
    image_size: Tuple[int, int]  # (w, h)

@dataclass
class StereoCalib:
    R: np.ndarray              # 3x3 (left->right)
    T: np.ndarray              # 3x1
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q:  Optional[np.ndarray] = None
    valid_roi1: Optional[Tuple[int,int,int,int]] = None
    valid_roi2: Optional[Tuple[int,int,int,int]] = None

@dataclass
class RectifyMaps:
    left_map1: Optional[np.ndarray] = None
    left_map2: Optional[np.ndarray] = None
    right_map1: Optional[np.ndarray] = None
    right_map2: Optional[np.ndarray] = None

# ---------- low-level loaders ----------
def _read_yaml_opencv(path: Path) -> Dict[str, Any]:
    """
    Reads OpenCV-style YAML (cv::FileStorage). Returns dict with numpy arrays
    for common calibration keys if present.
    """
    if not path.exists():
        return {}
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return {}

    def get(name: str):
        node = fs.getNode(name)
        if node.empty():
            return None
        arr = node.mat()
        if arr is None:
            # try raw value (scalars)
            try:
                return node.real()
            except Exception:
                return None
        return np.array(arr)

    keys = [
        "K", "camera_matrix", "dist", "dist_coeffs",
        "R", "T", "R1", "R2", "P1", "P2", "Q",
        "image_width", "image_height",
        "valid_roi1", "valid_roi2"
    ]
    out: Dict[str, Any] = {k: get(k) for k in keys}
    fs.release()

    # Normalize aliases
    if out.get("K") is None and out.get("camera_matrix") is not None:
        out["K"] = out["camera_matrix"]
    if out.get("dist") is None and out.get("dist_coeffs") is not None:
        out["dist"] = out["dist_coeffs"]

    return out

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    # convert lists to np arrays where appropriate
    def as_arr(x):
        return np.array(x, dtype=np.float64) if isinstance(x, (list, tuple)) else x
    return {k: as_arr(v) for k, v in data.items()}

def _try_load(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    if p.suffix.lower() in (".yml", ".yaml"):
        d = _read_yaml_opencv(p)
        return d
    if p.suffix.lower() == ".json":
        return _read_json(p)
    # fallback: attempt YAML first, then JSON
    d = _read_yaml_opencv(p)
    return d if d else _read_json(p)

# ---------- public API ----------
def load_mono_calibration(calib_file: str,
                          image_size_hint: Optional[Tuple[int, int]] = None) -> MonoCalib:
    """
    Load mono intrinsics and distortion from YAML/JSON.
    Expects keys: K (or camera_matrix), dist (or dist_coeffs), and optionally image_width/height.
    image_size_hint=(w,h) is used if file lacks explicit image size.
    """
    d = _try_load(calib_file)
    K = d.get("K")
    dist = d.get("dist")
    if K is None or dist is None:
        raise FileNotFoundError(f"[calib_loader] Missing K/dist in {calib_file}")

    # image size (w,h)
    w = int(d["image_width"]) if d.get("image_width") is not None else None
    h = int(d["image_height"]) if d.get("image_height") is not None else None
    if (w is None or h is None) and image_size_hint is not None:
        w, h = image_size_hint
    if w is None or h is None:
        # final fallback: try to infer from principal point if plausible
        cx, cy = float(K[0, 2]), float(K[1, 2])
        w = int(max(2 * cx, 1))
        h = int(max(2 * cy, 1))

    return MonoCalib(K=np.asarray(K, dtype=np.float64),
                     dist=np.asarray(dist, dtype=np.float64).ravel(),
                     image_size=(w, h))

def load_stereo_extrinsics(stereo_file: str) -> StereoCalib:
    """
    Load stereo extrinsics/rectification from YAML/JSON if available.
    Expects subset of: R, T, R1, R2, P1, P2, Q, valid_roi1, valid_roi2.
    """
    d = _try_load(stereo_file)
    R = d.get("R")
    T = d.get("T")
    if R is None or T is None:
        raise FileNotFoundError(f"[calib_loader] Missing R/T in {stereo_file}")

    def _roi(v):
        if v is None:
            return None
        v = np.array(v).astype(int).ravel()
        if v.size == 4:
            return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        return None

    return StereoCalib(
        R=np.asarray(R, dtype=np.float64),
        T=np.asarray(T, dtype=np.float64).reshape(3, 1),
        R1=d.get("R1"),
        R2=d.get("R2"),
        P1=d.get("P1"),
        P2=d.get("P2"),
        Q=d.get("Q"),
        valid_roi1=_roi(d.get("valid_roi1")),
        valid_roi2=_roi(d.get("valid_roi2")),
    )

def compute_rectification_maps(left: MonoCalib,
                               right: Optional[MonoCalib] = None,
                               stereo: Optional[StereoCalib] = None,
                               alpha: float = 0.0) -> Tuple[StereoCalib, RectifyMaps]:
    """
    If stereo info lacks R1/R2/P1/P2, compute them via cv2.stereoRectify,
    then build undistort/rectify maps for left/right. For mono, produces only left maps.
    alpha: 0..1 (0 = zoomed, no black; 1 = full FOV)
    """
    wL, hL = left.image_size
    rectify = RectifyMaps()

    if right is not None and stereo is not None:
        wR, hR = right.image_size
        if stereo.R1 is None or stereo.R2 is None or stereo.P1 is None or stereo.P2 is None:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                left.K, left.dist, right.K, right.dist,
                (wL, hL), stereo.R, stereo.T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
            )
            stereo.R1, stereo.R2, stereo.P1, stereo.P2, stereo.Q = R1, R2, P1, P2, Q
            stereo.valid_roi1 = roi1
            stereo.valid_roi2 = roi2

        rectify.left_map1, rectify.left_map2 = cv2.initUndistortRectifyMap(
            left.K, left.dist, stereo.R1, stereo.P1, (wL, hL), cv2.CV_16SC2
        )
        rectify.right_map1, rectify.right_map2 = cv2.initUndistortRectifyMap(
            right.K, right.dist, stereo.R2, stereo.P2, (wR, hR), cv2.CV_16SC2
        )
        return stereo, rectify

    # mono only
    R = np.eye(3, dtype=np.float64)
    P = left.K.copy()
    rectify.left_map1, rectify.left_map2 = cv2.initUndistortRectifyMap(
        left.K, left.dist, R, P, (wL, hL), cv2.CV_16SC2
    )
    return StereoCalib(R=np.eye(3), T=np.zeros((3,1))), rectify

def load_rectification_maps(left_map1_path: str,
                            left_map2_path: str,
                            right_map1_path: Optional[str] = None,
                            right_map2_path: Optional[str] = None) -> RectifyMaps:
    rm = RectifyMaps()
    p = Path(left_map1_path)
    if p.exists():
        rm.left_map1 = np.load(str(p))
    p = Path(left_map2_path)
    if p.exists():
        rm.left_map2 = np.load(str(p))
    if right_map1_path:
        p = Path(right_map1_path)
        if p.exists():
            rm.right_map1 = np.load(str(p))
    if right_map2_path:
        p = Path(right_map2_path)
        if p.exists():
            rm.right_map2 = np.load(str(p))
    return rm

def save_rectification_maps(rect: RectifyMaps,
                            left_map1_path: str,
                            left_map2_path: str,
                            right_map1_path: Optional[str] = None,
                            right_map2_path: Optional[str] = None) -> None:
    if rect.left_map1 is not None:
        np.save(left_map1_path, rect.left_map1)
    if rect.left_map2 is not None:
        np.save(left_map2_path, rect.left_map2)
    if right_map1_path and rect.right_map1 is not None:
        np.save(right_map1_path, rect.right_map1)
    if right_map2_path and rect.right_map2 is not None:
        np.save(right_map2_path, rect.right_map2)

# ---------- convenience ops ----------
def undistort(frame: np.ndarray,
              left_maps: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    map1, map2 = left_maps
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

def rectify_pair(left_img: np.ndarray,
                 right_img: np.ndarray,
                 rect: RectifyMaps) -> Tuple[np.ndarray, np.ndarray]:
    if rect.left_map1 is None or rect.right_map1 is None:
        raise RuntimeError("[calib_loader] Rectification maps not loaded/computed.")
    left_rect = cv2.remap(left_img, rect.left_map1, rect.left_map2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, rect.right_map1, rect.right_map2, cv2.INTER_LINEAR)
    return left_rect, right_rect

def pixel_to_ray(px: Tuple[float, float], K: np.ndarray) -> np.ndarray:
    """
    Back-project pixel (x,y) to a unit 3D ray in camera coords.
    """
    x, y = px
    Kinv = np.linalg.inv(K)
    v = Kinv @ np.array([x, y, 1.0], dtype=np.float64)
    v = v / np.linalg.norm(v)
    return v  # (vx, vy, vz)

def pixels_to_angles(px: Tuple[float, float], K: np.ndarray) -> Tuple[float, float]:
    """
    Small-angle model: angle_x = atan((x-cx)/fx), angle_y = atan((y-cy)/fy).
    Useful for turning center-error pixels into pan/tilt angles.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xn = (px[0] - cx) / fx
    yn = (px[1] - cy) / fy
    return float(np.arctan(xn)), float(np.arctan(yn))

# ---------- glue with your Config ----------
def init_from_config(cfg) -> Tuple[MonoCalib, Optional[MonoCalib], Optional[StereoCalib], RectifyMaps]:
    """
    High-level initializer using paths from io/config.py.
    - Loads mono (left) intrinsics + distortion.
    - Optionally loads right intrinsics & stereo extrinsics if cfg.app.use_stereo.
    - Loads or computes rectification maps depending on cfg.calib.use_rect_maps.
    Returns: (left, right_or_None, stereo_or_None, rect_maps)
    """
    # Left mono
    left = load_mono_calibration(
        cfg.calib.mono_calib_file,
        image_size_hint=(cfg.video.width, cfg.video.height)
    )

    right = None
    stereo = None
    rect = RectifyMaps()

    if cfg.app.use_stereo:
        # By convention, if mono_calib_file is left, try to infer right by name; else override in config if needed
        right_path_guess = str(Path(cfg.calib.mono_calib_file).with_name("calib_right.yml"))
        try:
            right = load_mono_calibration(
                right_path_guess,
                image_size_hint=(cfg.video.width, cfg.video.height)
            )
        except Exception:
            # If right isn’t present, we still proceed mono
            right = None

        stereo = load_stereo_extrinsics(cfg.calib.steer o_calib_file) if Path(cfg.calib.stereo_calib_file).exists() else None

    # Maps: load precomputed if available & allowed; else compute on the fly
    if cfg.calib.use_rect_maps:
        rect = load_rectification_maps(
            cfg.calib.rect_map_left_1,
            cfg.calib.rect_map_left_2,
            cfg.calib.rect_map_right_1 if right is not None else None,
            cfg.calib.rect_map_right_2 if right is not None else None
        )

    need_compute_left = rect.left_map1 is None or rect.left_map2 is None
    need_compute_right = (right is not None) and (rect.right_map1 is None or rect.right_map2 is None)

    if need_compute_left or need_compute_right:
        stereo, rect = compute_rectification_maps(left, right, stereo, alpha=0.0)
        # Optionally save for faster next startup
        try:
            save_rectification_maps(
                rect,
                cfg.calib.rect_map_left_1,
                cfg.calib.rect_map_left_2,
                cfg.calib.rect_map_right_1 if right is not None else None,
                cfg.calib.rect_map_right_2 if right is not None else None
            )
        except Exception as e:
            print(f"[calib_loader] Warning: could not save rectification maps: {e}")

    return left, right, stereo, rect

