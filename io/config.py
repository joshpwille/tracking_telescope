# io/config.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# ---------- small YAML shim (optional) ----------
def _try_load_yaml(path: Path) -> Dict[str, Any]:
    """
    Lightweight YAML reader if PyYAML isn't installed:
    - If file ends with .json → use JSON
    - If file ends with .yaml/.yml and PyYAML is present → use yaml.safe_load
    - Else: returns {}
    """
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(path.read_text()) or {}
        except Exception:
            print("[config] PyYAML not available; skipping YAML load.")
            return {}
    return {}

# ---------- sections ----------
@dataclass
class PathsCfg:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    logs_dir: Path = field(default_factory=lambda: Path("outputs/logs"))
    weights_dir: Path = field(default_factory=lambda: Path("vision"))
    calib_dir: Path = field(default_factory=lambda: Path("io"))

@dataclass
class VideoCfg:
    # source can be integer (camera index) or string (rtsp/file path)
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "mp4v"
    # set to a path to record; empty string disables writer
    record_path: str = "outputs/tracked_out.mp4"

@dataclass
class CalibrationCfg:
    # mono intrinsics + distortion
    mono_calib_file: str = "io/calib_left.yml"
    # stereo extrinsics (R, T), rectification (R1,R2,P1,P2,Q) optional
    stereo_calib_file: str = "io/stereo_extrinsics.yml"
    # precomputed undistort/rectify maps (if present, speeds up startup)
    rect_map_left_1: str = "io/rect_map_left_1.npy"
    rect_map_left_2: str = "io/rect_map_left_2.npy"
    rect_map_right_1: str = "io/rect_map_right_1.npy"
    rect_map_right_2: str = "io/rect_map_right_2.npy"
    # zoom LUTs (optional)
    zoom_lut_left: str = "io/zoom_lut_left.json"
    zoom_lut_right: str = "io/zoom_lut_right.json"
    # if true, attempt to load rectification maps; if missing, compute on the fly
    use_rect_maps: bool = True

@dataclass
class DetectorCfg:
    weights: str = "vision/yolov5nu.pt"
    conf_thr: float = 0.25
    iou_thr: float = 0.45
    max_det: int = 100
    device: str = "cuda"  # "cuda" / "cpu"
    classes: Optional[Tuple[int, ...]] = None  # e.g., (0,) to filter

@dataclass
class ByteTrackCfg:
    # Put your Bytetrack YAML here if you use one; else keep simple params
    config_path: str = "vision/bytetrack_planes.yaml"
    track_buffer: int = 30
    match_thresh: float = 0.8
    conf_thr: float = 0.5
    mot20: bool = False

@dataclass
class TrackingCfg:
    max_age: int = 15
    min_hits: int = 3
    # optical flow params (if used for cheap tracking)
    pyr_levels: int = 3
    win_size: int = 21
    max_iter: int = 30
    eps: float = 0.01

@dataclass
class SteeringCfg:
    # px error → angle conversion uses intrinsics; these tune command shaping
    base_gain: float = 0.8          # global gain
    deadband_px: int = 2            # ignore tiny errors
    max_rate_deg_s: float = 18.0    # rate limiter
    accel_deg_s2: float = 120.0     # optional accel limiter
    hysteresis_px: int = 3          # prevents chatter
    axis_split: bool = True         # per-axis gain scaling by error proportion

@dataclass
class HardwarePinsCfg:
    # choose backend: "null" (logs only), "rpi", or "jetson"
    backend: str = "null"
    # Stepper pins (4-phase example)
    stepper_pan: Tuple[int, int, int, int] = (17, 18, 27, 22)
    stepper_tilt: Tuple[int, int, int, int] = (5, 6, 13, 19)
    # Encoders (A,B) per axis (if available)
    enc_pan: Tuple[int, int] = (23, 24)
    enc_tilt: Tuple[int, int] = (25, 12)
    # E-STOP input pin (active-low)
    estop_pin: Optional[int] = 26
    # Optional zoom motor pins
    zoom_motor: Optional[Tuple[int, int, int, int]] = None

@dataclass
class MechanicsCfg:
    # convert commanded degrees to steps
    steps_per_rev: int = 2048            # e.g., 28BYJ-48
    gear_ratio: float = 1.0              # gearbox beyond motor
    deg_per_rev: float = 360.0
    # encoder counts per degree (if using encoders for feedback)
    enc_counts_per_deg_pan: float = 20.0
    enc_counts_per_deg_tilt: float = 20.0

@dataclass
class UIcfg:
    show_overlay: bool = True
    window_name: str = "Tracker"
    # hotkeys (opencv waitKey codes)
    hotkey_estop: str = " "       # space
    hotkey_clear_estop: str = "g" # 'go'
    hotkey_quit: str = "q"

@dataclass
class OverlayCfg:
    draw_boxes: bool = True
    draw_ids: bool = True
    draw_center: bool = True
    draw_target: bool = True
    draw_fps: bool = True

@dataclass
class WriterCfg:
    enable: bool = True
    path: str = "outputs/tracked_out.mp4"
    fps: Optional[float] = None  # None → use input FPS
    fourcc: str = "mp4v"

@dataclass
class AppCfg:
    # high-level toggles
    use_detector: bool = True
    use_bytrack: bool = True
    use_optical_flow_seed: bool = True
    use_motors: bool = False      # start safe (no motion)
    use_stereo: bool = False
    use_voxel_fusion: bool = False

@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    calib: CalibrationCfg = field(default_factory=CalibrationCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    bytetrack: ByteTrackCfg = field(default_factory=ByteTrackCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    steering: SteeringCfg = field(default_factory=SteeringCfg)
    pins: HardwarePinsCfg = field(default_factory=HardwarePinsCfg)
    mechanics: MechanicsCfg = field(default_factory=MechanicsCfg)
    ui: UIcfg = field(default_factory=UIcfg)
    overlay: OverlayCfg = field(default_factory=OverlayCfg)
    writer: WriterCfg = field(default_factory=WriterCfg)
    app: AppCfg = field(default_factory=AppCfg)

    # -------- helpers --------
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ensure_dirs(self) -> None:
        (self.paths.project_root / self.paths.output_dir).mkdir(parents=True, exist_ok=True)
        (self.paths.project_root / self.paths.logs_dir).mkdir(parents=True, exist_ok=True)

# ---------- loader ----------
def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load defaults, then apply overrides from (in order):
      1) config_path (YAML/JSON) if provided
      2) ./config.yaml or ./config.json (if present)
      3) ENV var CFG_JSON (inline JSON string)  ← quick one-off overrides
    """
    cfg = Config()

    # 1) explicit file
    if config_path:
        overrides = _try_load_yaml(Path(config_path))
        _deep_update(cfg.__dict__, _dict_to_namespaced(overrides))

    # 2) local file next to repo root
    for candidate in ("config.yaml", "config.yml", "config.json"):
        p = Path(candidate)
        if p.exists():
            overrides = _try_load_yaml(p)
            _deep_update(cfg.__dict__, _dict_to_namespaced(overrides))
            break

    # 3) ENV inline JSON (e.g., export CFG_JSON='{"video":{"source":"test.mp4"}}')
    env_json = os.environ.get("CFG_JSON", "")
    if env_json.strip():
        try:
            overrides = json.loads(env_json)
            _deep_update(cfg.__dict__, _dict_to_namespaced(overrides))
        except Exception as e:
            print(f"[config] Failed to parse CFG_JSON: {e}")

    cfg.ensure_dirs()
    return cfg

# map flat dicts to dataclass namespaces (e.g., {"video": {...}})
def _dict_to_namespaced(d: Dict[str, Any]) -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    for k, v in d.items():
        ns[k] = v
    return ns

# ---------- convenient singleton ----------
_GLOBAL_CFG: Optional[Config] = None

def get_config(config_path: Optional[str] = None) -> Config:
    global _GLOBAL_CFG
    if _GLOBAL_CFG is None:
        _GLOBAL_CFG = load_config(config_path)
    return _GLOBAL_CFG

# ---------- quick CLI check ----------
if __name__ == "__main__":
    c = get_config()
    print(json.dumps(c.as_dict(), indent=2, default=str))

