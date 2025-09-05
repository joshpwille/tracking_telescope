# ui/overlay.py
"""
Overlay utilities for drawing tracks, target lock, and HUD on video frames.

Key features:
- Per-track box, center dot, and label (id/conf/scale/class).
- Selected target: thicker box + arrow from principal point to target center.
- Principal point reticle (from K) or frame center if K is None.
- Deadband circle around the principal point.
- Multi-line HUD: FPS, target status string, help, etc.
- Auto-scales line thickness and font size by frame size.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Small helpers
# ----------------------------

def _autoscale_sizes(w: int, h: int) -> Dict[str, float]:
    base = min(w, h)
    th = max(1, int(base * 0.003))                 # line thickness
    r_pt = max(2, int(base * 0.004))               # small point radius
    font = cv2.FONT_HERSHEY_SIMPLEX
    fsz = max(0.4, base * 0.0009)                  # font scale
    gap = max(4, int(base * 0.01))                 # text line gap
    pad = max(2, int(base * 0.006))                # label padding
    arr = max(5, int(base * 0.02))                 # arrow head length
    return dict(th=th, r_pt=r_pt, font=font, fsz=fsz, gap=gap, pad=pad, arr=arr)


def _color_for_id(tid: int) -> Tuple[int, int, int]:
    """
    Nice, stable pseudo-random color per ID (BGR).
    """
    # Simple hash -> HSV, then to BGR
    rng = (tid * 2654435761) & 0xFFFFFFFF
    h = rng % 180
    s = 200 + (rng >> 8) % 55
    v = 200 + (rng >> 16) % 55
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_label(img: np.ndarray, x: int, y: int, text: str, font, fsz: float, pad: int,
                fg: Tuple[int, int, int]=(255, 255, 255),
                bg: Tuple[int, int, int]=(0, 0, 0)):
    """
    Draw text with a filled background box anchored at (x,y) above the box.
    """
    (tw, th), bl = cv2.getTextSize(text, font, fsz, 1)
    x2 = x + tw + 2 * pad
    y2 = y - th - 2 * pad
    cv2.rectangle(img, (x, y2), (x2, y), bg, thickness=-1)
    cv2.putText(img, text, (x + pad, y - pad), font, fsz, fg, 1, cv2.LINE_AA)


def _draw_multiline_hud(img: np.ndarray, lines: List[str], org: Tuple[int, int],
                        font, fsz: float, gap: int, color=(230, 230, 230)):
    x, y = org
    for ln in lines:
        cv2.putText(img, ln, (x, y), font, fsz, color, 1, cv2.LINE_AA)
        y += int(gap + fsz * 20)


def _principal_point_from_K(K: Optional[np.ndarray], w: int, h: int) -> Tuple[float, float]:
    if K is not None and K.shape == (3, 3):
        return float(K[0, 2]), float(K[1, 2])
    return w * 0.5, h * 0.5


# ----------------------------
# FPS meter (EMA)
# ----------------------------

@dataclass
class FpsEMA:
    alpha: float = 0.9
    value: float = 0.0
    _t_last: float = 0.0

    def tick(self) -> float:
        t = time.time()
        if self._t_last == 0.0:
            self._t_last = t
            return self.value
        dt = max(1e-6, t - self._t_last)
        inst = 1.0 / dt
        self.value = self.alpha * self.value + (1 - self.alpha) * inst if self.value > 0 else inst
        self._t_last = t
        return self.value


# ----------------------------
# Main overlay function
# ----------------------------

def draw_overlay(frame: np.ndarray,
                 tracks: Iterable[Dict],
                 selected_id: Optional[int] = None,
                 *,
                 K: Optional[np.ndarray] = None,
                 deadband_px: float = 0.0,
                 hud_lines: Optional[List[str]] = None,
                 show_boxes: bool = True,
                 show_centers: bool = True,
                 show_principal_reticle: bool = True,
                 fps_meter: Optional[FpsEMA] = None,
                 show_help: bool = False) -> None:
    """
    Draw overlay elements on a frame (in-place).

    Args:
        frame: BGR image.
        tracks: iterable of dict rows with keys:
            {id:int, bbox:(x,y,w,h), center:(cx,cy), scale:float, conf:float, (optional) cls:int}
        selected_id: currently promoted ID (highlighted).
        K: optional 3x3 intrinsics to place the principal point; falls back to frame center.
        deadband_px: draw a circle of this radius around the principal point.
        hud_lines: extra status lines (e.g., target_manager.hud_string()).
        show_boxes/centers/reticle: toggles for elements.
        fps_meter: optional FpsEMA to show smoothed FPS.
        show_help: draw hotkey help lines.
    """
    H, W = frame.shape[:2]
    S = _autoscale_sizes(W, H)
    cx0, cy0 = _principal_point_from_K(K, W, H)

    # Principal point reticle + deadband
    if show_principal_reticle:
        cv2.drawMarker(frame, (int(cx0), int(cy0)), (180, 180, 180), markerType=cv2.MARKER_CROSS,
                       markerSize=max(10, int(min(W, H) * 0.03)), thickness=S['th'])
    if deadband_px and deadband_px > 0:
        cv2.circle(frame, (int(cx0), int(cy0)), int(deadband_px), (120, 120, 120), S['th'], cv2.LINE_AA)

    # Draw tracks
    for r in tracks:
        tid = int(r.get("id", -1))
        x, y, w, h = map(int, r.get("bbox", (0, 0, 0, 0)))
        ccx, ccy = r.get("center", (x + w / 2.0, y + h / 2.0))
        conf = float(r.get("conf", 0.0))
        scl = float(r.get("scale", h))
        cls = r.get("cls", None)

        is_sel = (selected_id is not None) and (tid == selected_id)
        col = (0, 255, 255) if is_sel else _color_for_id(tid)         # selected: yellow
        th_box = S['th'] * (2 if is_sel else 1)

        if show_boxes and w > 0 and h > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, th_box, cv2.LINE_AA)

        if show_centers:
            cv2.circle(frame, (int(ccx), int(ccy)), S['r_pt'], col, -1, cv2.LINE_AA)

        # Label above box
        tag = f"id={tid}"
        tag += f" conf={conf:.2f}"
        tag += f" s={scl:.0f}"
        if cls is not None:
            tag += f" c={int(cls)}"
        _draw_label(frame, x, y, tag, S['font'], S['fsz'], S['pad'], fg=(255, 255, 255), bg=(0, 0, 0))

        # If selected, draw error arrow from principal point
        if is_sel:
            _draw_error_arrow(frame, (cx0, cy0), (float(ccx), float(ccy)), col, S['th'], S['arr'])

    # HUD (FPS + status lines + help)
    hud: List[str] = []
    if fps_meter is not None:
        fps_val = fps_meter.tick()
        hud.append(f"FPS: {fps_val:5.1f}")
    if hud_lines:
        hud.extend(hud_lines)
    if show_help:
        hud += [
            "keys: [digits]->lock id, 'n' clear, 'q' quit, SPACE E-STOP",
            "      'o' overlay, 'm' motors, '+/-' gain, 'd' detect cadence",
        ]
    if hud:
        _draw_multiline_hud(frame, hud, (10, 20), S['font'], S['fsz'], S['gap'])


def _draw_error_arrow(img: np.ndarray,
                      p0: Tuple[float, float],
                      p1: Tuple[float, float],
                      color: Tuple[int, int, int],
                      thickness: int,
                      head_len: int):
    """
    Draw an arrow from p0 (principal point) to p1 (target center).
    """
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=max(0.2, head_len / 20.0), line_type=cv2.LINE_AA)


# ----------------------------
# Quick self-test
# ----------------------------

if __name__ == "__main__":
    # Create a dummy frame and some tracks to see the overlay quickly.
    f = np.full((720, 1280, 3), 15, np.uint8)
    dummy_tracks = [
        {"id": 1, "bbox": (200, 160, 140, 80), "center": (270, 200), "scale": 80, "conf": 0.87, "cls": 0},
        {"id": 7, "bbox": (860, 300, 200, 130), "center": (960, 365), "scale": 130, "conf": 0.76, "cls": 1},
    ]
    K = np.array([[1000.0, 0.0, 640.0],
                  [0.0, 1000.0, 360.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    fps = FpsEMA()
    for _ in range(5):
        draw_overlay(f, dummy_tracks, selected_id=7, K=K, deadband_px=30,
                     hud_lines=["TARGET[AUTO] id=7 err=(+5.2,-3.1) r=6.1px"],
                     fps_meter=fps, show_help=True)
    cv2.imshow("overlay_demo", f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

