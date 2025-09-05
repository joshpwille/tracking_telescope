"""
app/pipeline.py
High-level real-time pipeline:
- Video capture
- YOLO detection + ByteTrack (or fallback)
- Per-frame tracking table
- Encoder-driven geometry (R,t) + zoom LUT intrinsics (K)
- Optional triangulation hook
- Error->motor commands with gain scaling, deadband, rate limits
- UI hotkeys to pick active target ID

This file intentionally has light dependencies on the rest of the tree;
components can be swapped via the config.

Author: you :)
"""

from __future__ import annotations
import time
import math
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2  # ok to import; if headless, ensure QT plugins are not required

# --- Local modules (keep optional; give helpful errors if missing) ---
try:
    from io.video_capture import VideoCapture  # wraps cv2.VideoCapture with sane defaults
except Exception:
    VideoCapture = None

try:
    from io.writer import make_writer  # returns an opened cv2.VideoWriter
except Exception:
    make_writer = None

try:
    from io.calib_loader import load_intrinsics  # returns intrinsics dict or matrix
except Exception:
    load_intrinsics = None

try:
    from io.zoom_lut_loader import load_zoom_lut  # returns {steps: (fx,fy,cx,cy)}
except Exception:
    load_zoom_lut = None

try:
    from ui.stdin_listener import start_stdin_listener  # returns Queue[str]
except Exception:
    start_stdin_listener = None

try:
    from ui.overlay import draw_overlay  # draw boxes/IDs/reticles
except Exception:
    draw_overlay = None

try:
    from vision.yolo_detector import YoloDetector
except Exception:
    YoloDetector = None

try:
    from vision.bytetrack_wrapper import ByteTrack  # thin wrapper around ByteTrack
except Exception:
    ByteTrack = None

try:
    from vision.tracking_table import TrackingTable
except Exception:
    # Minimal internal fallback
    class TrackingTable:
        def __init__(self, cam_id: str):
            self.cam_id = cam_id
            self.tracks: Dict[int, Dict] = {}
        def update_track(self, tid: int, bbox: Tuple[int,int,int,int], conf: float, scale: float=None):
            x,y,w,h = bbox; cx,cy = x+w/2,y+h/2
            self.tracks[tid] = {"id":tid,"bbox":(x,y,w,h),"center":(cx,cy),"scale":float(scale if scale else h),"conf":float(conf)}
        def remove_track(self, tid: int):
            self.tracks.pop(tid, None)
        def get_active_tracks(self) -> List[Dict]:
            return list(self.tracks.values())
        def get_track(self, tid: int) -> Optional[Dict]:
            return self.tracks.get(tid)

try:
    from vision.geometry import (
        CameraIntrinsics,
        interpolate_intrinsics,
        extrinsics_from_mount,
        angles_from_pixel,
    )
except Exception:
    # Minimal intrinsics to keep pipeline importable
    class CameraIntrinsics:
        def __init__(self, fx, fy, cx, cy): self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        @property
        def K(self): return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]], float)
    def interpolate_intrinsics(steps, lut):  # no LUT? just pick first
        (fx,fy,cx,cy) = next(iter(lut.values())) if lut else (1000.,1000.,640.,360.)
        return CameraIntrinsics(fx,fy,cx,cy)
    def extrinsics_from_mount(pivot_world, offset, pan, tilt):
        R = np.eye(3); t = np.zeros(3)  # placeholder
        class E: pass
        E.R, E.t = R, t
        return E
    def angles_from_pixel(K, px):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x,y = px
        xn, yn = (x - cx)/fx, (y - cy)/fy
        return math.atan(xn), math.atan(yn)

# Hardware (optional backends)
try:
    from hardware.stepper import PiStepper
except Exception:
    PiStepper = None

try:
    from hardware.encoder import Encoder  # should expose .read_pan_tilt_rad()
except Exception:
    Encoder = None

try:
    from hardware.zoom_motor import ZoomMotor  # expose .read_steps()
except Exception:
    ZoomMotor = None


# =========================
# Configuration structures
# =========================

@dataclass
class MotorGains:
    base_gain: float = 0.4          # baseline proportional factor
    max_gain: float = 1.0           # clamp
    deadband_px: float = 4.0        # ignore tiny center error
    rate_limit_deg_s: float = 12.0  # clamp commanded slew rate
    split_axes: bool = True         # split proportional weights by |x| vs |y|

@dataclass
class GeometryConfig:
    calib_path: str                   # e.g., io/calib_left.yml
    zoom_lut_path: Optional[str] = None
    pivot_world: Tuple[float,float,float] = (0.0, 0.0, 0.0)
    offset_cam_from_pivot: Tuple[float,float,float] = (0.0, 0.0, 0.0)

@dataclass
class IOConfig:
    video_source: str | int = 0
    output_path: Optional[str] = None
    output_fps: Optional[float] = None

@dataclass
class DetectConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    classes: Optional[List[int]] = None
    detect_every_n: int = 1  # run detector every N frames; trackers fill between

@dataclass
class TrackConfig:
    use_bytetrack: bool = True
    bytetrack_cfg: Optional[str] = None  # path to YAML if your wrapper accepts one

@dataclass
class PipelineConfig:
    cam_id: str = "cam0"
    io: IOConfig = field(default_factory=IOConfig)
    geom: GeometryConfig = field(default_factory=lambda: GeometryConfig(calib_path="io/calib_left.yml"))
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    gains: MotorGains = field(default_factory=MotorGains)
    draw_overlay: bool = True
    show_window: bool = True
    enable_trianguation: bool = False   # hook for app/triangulation if you wire it later
    enable_motors: bool = False         # safe default: visualize only
    target_id: Optional[int] = None     # lock to this ID when set


# =========================
# Pipeline
# =========================

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.cap = None
        self.writer = None
        self.detector = None
        self.tracker = None
        self.tracks = TrackingTable(cfg.cam_id)
        self.ui_q: Optional[queue.Queue] = None

        # geometry state
        self.K: Optional[np.ndarray] = None
        self.zoom_lut = None
        self.extr = None  # will hold R,t if you need them later
        self.frame_size = None
        self.frame_cxcy = (0.0, 0.0)

        # hardware
        self.stepper_x = None
        self.stepper_y = None
        self.encoder = None
        self.zoom = None

        # state
        self.frame_idx = 0
        self.last_time = None
        self.curr_target_id = cfg.target_id

    # ---- setup ----
    def setup(self):
        if VideoCapture is None:
            raise RuntimeError("io.video_capture.VideoCapture not available")
        self.cap = VideoCapture(self.cfg.io.video_source)

        # Detector + tracker
        if YoloDetector is None:
            raise RuntimeError("vision.yolo_detector.YoloDetector not available")
        self.detector = YoloDetector(conf_thres=self.cfg.detect.conf_thres,
                                     iou_thres=self.cfg.detect.iou_thres,
                                     classes=self.cfg.detect.classes)

        if self.cfg.track.use_bytetrack:
            if ByteTrack is None:
                print("[pipeline] ByteTrack not available, falling back to detection only.")
                self.tracker = None
            else:
                self.tracker = ByteTrack(cfg_path=self.cfg.track.bytetrack_cfg) if self.cfg.track.bytetrack_cfg else ByteTrack()
        else:
            self.tracker = None

        # I/O writer
        if self.cfg.io.output_path and make_writer is not None:
            self.writer = None  # will init once we know frame size

        # UI
        if start_stdin_listener is not None:
            self.ui_q = start_stdin_listener()
        else:
            self.ui_q = queue.Queue()

        # Geometry
        if load_intrinsics is None:
            raise RuntimeError("io.calib_loader.load_intrinsics not available")
        intr = load_intrinsics(self.cfg.geom.calib_path)
        if isinstance(intr, dict):
            self.K = np.array([[intr["fx"], 0, intr["cx"]],
                               [0, intr["fy"], intr["cy"]],
                               [0, 0, 1]], dtype=np.float64)
        else:
            self.K = np.asarray(intr, dtype=np.float64)

        if self.cfg.geom.zoom_lut_path and load_zoom_lut is not None:
            self.zoom_lut = load_zoom_lut(self.cfg.geom.zoom_lut_path)

        # Hardware
        if self.cfg.enable_motors and PiStepper is not None:
            self.stepper_x = PiStepper(pins=(4,17,27,22))  # EXAMPLE pins; replace for your rig
            self.stepper_y = PiStepper(pins=(5,6,13,19))
        if Encoder is not None:
            self.encoder = Encoder()
        if ZoomMotor is not None:
            self.zoom = ZoomMotor()

        self.last_time = time.time()

    # ---- main loop ----
    def run(self):
        self.setup()
        print("[pipeline] Running. Type an ID + Enter to follow; 'n' to clear; 'q' to quit.")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[pipeline] End of stream or camera read failed.")
                break

            self.frame_idx += 1
            H, W = frame.shape[:2]
            self.frame_size = (W, H)
            self.frame_cxcy = (W * 0.5, H * 0.5)

            # Init writer once we know size
            if self.writer is None and self.cfg.io.output_path and make_writer is not None:
                self.writer = make_writer(self.cfg.io.output_path,
                                          fps=self.cfg.io.output_fps or self.cap.fps or 30.0,
                                          frame_size=(W, H))

            # Update geometry from encoders + zoom LUT (if present)
            self._update_geometry_runtime()

            # Detection + tracking
            detections = []
            if self._should_detect_this_frame():
                detections = self.detector.detect(frame)  # list of dicts: {'bbox':(x,y,w,h), 'conf':float, 'cls':int}
            tracks = self._update_tracker(detections, frame.shape)

            # Fill per-frame tracking table
            self._refresh_tracking_table(tracks, detections)

            # UI: check for target selection / e-stop
            if self._handle_ui():
                break  # quit

            # Steering for active target (if any)
            if self.curr_target_id is not None:
                self._steer_to_target(frame, self.curr_target_id)

            # Overlay + display/write
            if self.cfg.draw_overlay and draw_overlay is not None:
                draw_overlay(frame, self.tracks.get_active_tracks(), selected_id=self.curr_target_id)
            if self.cfg.show_window:
                cv2.imshow(self.cfg.cam_id, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            if self.writer is not None:
                self.writer.write(frame)

        self._teardown()

    # ---- helpers ----

    def _should_detect_this_frame(self) -> bool:
        n = max(1, self.cfg.detect.detect_every_n)
        return (self.frame_idx % n) == 1

    def _update_tracker(self, detections: List[Dict], img_shape) -> List[Dict]:
        """
        Adapter: if ByteTrack is present, run it; else return detections as "tracks" with ephemeral IDs.
        Output format: list of dicts with keys: id, bbox(x,y,w,h), conf, cls
        """
        if self.tracker is None:
            # Fallback: assign transient IDs
            tracks = []
            for i, d in enumerate(detections):
                bb = d["bbox"]; conf = float(d.get("conf", 1.0)); cls = int(d.get("cls", -1))
                tracks.append({"id": i, "bbox": bb, "conf": conf, "cls": cls})
            return tracks

        return self.tracker.update(detections, img_shape)

    def _refresh_tracking_table(self, tracks: List[Dict], detections: List[Dict]):
        seen = set()
        for t in tracks:
            tid = int(t["id"]); x,y,w,h = map(float, t["bbox"])
            conf = float(t.get("conf", 1.0))
            scale = max(h, 1.0)  # simple per-object scale proxy; replace with your scale estimator if desired
            self.tracks.update_track(tid, (int(x), int(y), int(w), int(h)), conf=conf, scale=scale)
            seen.add(tid)

        # (Optional) you could drop stale tracks here if your tracker doesn't
        # For now, we keep whatever the tracker says is active.

    def _handle_ui(self) -> bool:
        """
        Listen for user commands from stdin:
          - integer: lock to that track id
          - 'n'    : clear target
          - 'q'    : quit
          - 'stop' : (reserved for estop if you wire hardware.e_stop)
        """
        if self.ui_q is None:
            return False
        try:
            while True:
                msg = self.ui_q.get_nowait().strip()
                if msg.lower() == 'q':
                    return True
                if msg.lower() in ('n', 'none', 'neutral'):
                    self.curr_target_id = None
                    print("[pipeline] Cleared target.")
                    continue
                if msg.isdigit():
                    self.curr_target_id = int(msg)
                    print(f"[pipeline] Following target id={self.curr_target_id}")
                # else: ignore unknown
        except queue.Empty:
            pass
        return False

    def _update_geometry_runtime(self):
        """
        Update K from zoom LUT; R,t from encoders and lever arm model.
        """
        # Zoom -> intrinsics
        if self.zoom_lut and self.zoom is not None:
            steps = float(self.zoom.read_steps())
            intr = interpolate_intrinsics(steps, self.zoom_lut)
            self.K = intr.K

        # Encoders -> extrinsics (store if you want world math later)
        if self.encoder is not None:
            pan, tilt = self.encoder.read_pan_tilt_rad()
            self.extr = extrinsics_from_mount(
                self.cfg.geom.pivot_world,
                self.cfg.geom.offset_cam_from_pivot,
                pan, tilt
            )

    def _steer_to_target(self, frame, tid: int):
        """
        Compute centering error in pixels, convert to angular errors via small-angle,
        apply gain scaling/deadband, and command motors (if enabled).
        """
        tr = self.tracks.get_track(tid)
        if not tr:
            return
        cx, cy = tr["center"]
        fx, fy, c0x, c0y = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]

        # Error relative to principal point (or frame center if you prefer)
        err_px = (cx - c0x, cy - c0y)
        r = math.hypot(*err_px)

        # Deadband
        if r < self.cfg.gains.deadband_px:
            return

        # Convert to angular error (small-angle: Δθ≈Δx/f)
        dtheta_x = (err_px[0] / fx)
        dtheta_y = (err_px[1] / fy)

        # Error-scaled gain (larger error => larger gain), split across axes if enabled
        base = self.cfg.gains.base_gain
        k = min(self.cfg.gains.max_gain, base * (1.0 + r / 100.0))  # simple scaling
        if self.cfg.gains.split_axes and (abs(dtheta_x) + abs(dtheta_y)) > 1e-9:
            ax = abs(dtheta_x) / (abs(dtheta_x) + abs(dtheta_y))
            ay = 1.0 - ax
        else:
            ax = ay = 0.5

        cmd_az = k * ax * dtheta_x   # pan command (radians proxy)
        cmd_el = k * ay * dtheta_y   # tilt command (radians proxy)

        # Rate limiting (deg/s)
        dt = max(1e-3, time.time() - self.last_time)
        self.last_time = time.time()
        max_step = math.radians(self.cfg.gains.rate_limit_deg_s) * dt
        cmd_az = float(np.clip(cmd_az, -max_step, max_step))
        cmd_el = float(np.clip(cmd_el, -max_step, max_step))

        # Apply to motors (convert to step counts here if needed)
        if self.cfg.enable_motors and self.stepper_x and self.stepper_y:
            self._apply_motor_command(cmd_az, cmd_el)
        else:
            # Visual reticle to show where we're steering (optional)
            cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 255), 2)

    def _apply_motor_command(self, dpan_rad: float, dtilt_rad: float):
        """
        Convert angular deltas to steps and drive steppers.
        Replace gearing constants with your measured values.
        """
        STEPS_PER_DEG_X = 200.0   # <-- replace for your rig
        STEPS_PER_DEG_Y = 200.0

        steps_x = int(np.clip(math.degrees(dpan_rad) * STEPS_PER_DEG_X, -100, 100))
        steps_y = int(np.clip(math.degrees(dtilt_rad) * STEPS_PER_DEG_Y, -100, 100))

        if steps_x != 0:
            self.stepper_x.step(steps=abs(steps_x), direction=1 if steps_x > 0 else -1)
        if steps_y != 0:
            self.stepper_y.step(steps=abs(steps_y), direction=1 if steps_y > 0 else -1)

    def _teardown(self):
        if self.writer is not None:
            self.writer.release()
        if self.cap is not None:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ================
# Convenience main
# ================

def main():
    cfg = PipelineConfig(
        cam_id="cam0",
        io=IOConfig(video_source=0, output_path=None),
        geom=GeometryConfig(
            calib_path="io/calib_left.yml",
            zoom_lut_path="io/zoom_lut_left.json",
            pivot_world=(0.0,0.0,0.0),
            offset_cam_from_pivot=(0.0,0.0,0.0),
        ),
        detect=DetectConfig(conf_thres=0.25, iou_thres=0.45, detect_every_n=1),
        track=TrackConfig(use_bytetrack=True, bytetrack_cfg="vision/bytetrack_planes.yaml"),
        gains=MotorGains(base_gain=0.4, max_gain=1.2, deadband_px=4.0, rate_limit_deg_s=12.0),
        draw_overlay=True,
        show_window=True,
        enable_trianguation=False,
        enable_motors=False,  # start safe
        target_id=None,
    )
    pipe = Pipeline(cfg)
    pipe.run()


if __name__ == "__main__":
    main()

