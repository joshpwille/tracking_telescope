# vision/bytetrack_wrapper.py
"""
ByteTrack-style tracker wrapper (dependency-light).

Implements key ByteTrack ideas for telescopic targets:
  1) Split detections by confidence into HIGH (>= high_thresh) and LOW (>= low_thresh).
  2) First match HIGH detections to active tracks via IoU (greedy).
  3) Unmatched HIGH detections spawn new tracks if conf >= new_track_thresh.
  4) Try to recover 'lost' tracks with LOW detections (reacquire window).
  5) Age out tracks that remain unmatched beyond a buffer.

Why this shape?
- Your PDF says: run YOLO and a fast tracker (e.g., ByteTrack’s motion model) to keep
  a per-frame table {id, bbox, center, scale, conf}; keep YOLO+ByteTrack running so
  lost targets can be reacquired and the same ID remains attached. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

Return for each frame: list of dicts
  {"id": int, "bbox": (x, y, w, h), "conf": float, "cls": int}

Notes:
- This is a compact, CPU-friendly implementation meant for live loops.
- If you later install a full ByteTrack package, you can swap the backend under the same API.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
import time


# =========================
# Config
# =========================

@dataclass
class BTConfig:
    high_thresh: float = 0.6       # detections >= high are used for main association
    low_thresh: float = 0.1        # low <= conf < high used only to recover 'lost' tracks
    new_track_thresh: float = 0.7  # min conf to spawn a brand new track
    match_thresh: float = 0.3      # IoU threshold for assignment (tune by target size)
    track_buffer: int = 30         # frames to keep a track 'lost' before dropping
    min_hits: int = 2              # frames required before a track is considered reliable
    max_coast: int = 10            # frames a track may coast (predict without update)
    smoothing: float = 0.6         # EMA on bbox when updated (stabilize jitter)
    use_class_in_match: bool = False  # if True, prefer same-class matches


# =========================
# Utilities
# =========================

def _xywh_to_xyxy(bb: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bb
    return (x, y, x + w, y + h)

def _iou(bb1: Tuple[float, float, float, float], bb2: Tuple[float, float, float, float]) -> float:
    x11, y11, x12, y12 = _xywh_to_xyxy(bb1)
    x21, y21, x22, y22 = _xywh_to_xyxy(bb2)
    xa, ya = max(x11, x21), max(y11, y21)
    xb, yb = min(x12, x22), min(y12, y22)
    iw, ih = max(0.0, xb - xa), max(0.0, yb - ya)
    inter = iw * ih
    a1, a2 = max(0.0, x12 - x11) * max(0.0, y12 - y11), max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = a1 + a2 - inter + 1e-6
    return inter / union

def _center(bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = bb
    return (x + w * 0.5, y + h * 0.5)


# =========================
# Track object
# =========================

class _Track:
    __slots__ = ("id", "bbox", "conf", "cls", "age", "hits", "coast",
                 "last_update_t", "vx", "vy")

    def __init__(self, tid: int, bbox_xywh: Tuple[int, int, int, int],
                 conf: float, cls: int):
        self.id = int(tid)
        self.bbox = (float(bbox_xywh[0]), float(bbox_xywh[1]),
                     float(bbox_xywh[2]), float(bbox_xywh[3]))
        self.conf = float(conf)
        self.cls  = int(cls)
        self.age  = 1           # total frames since birth
        self.hits = 1           # total matched frames
        self.coast = 0          # consecutive unmatched frames
        self.last_update_t = time.time()
        # simple constant-velocity model on center
        self.vx = 0.0
        self.vy = 0.0

    def predict(self):
        """Coast one frame using constant-velocity center; keep size fixed."""
        cx, cy = _center(self.bbox)
        x, y, w, h = self.bbox
        cx_n = cx + self.vx
        cy_n = cy + self.vy
        self.bbox = (cx_n - w * 0.5, cy_n - h * 0.5, w, h)
        self.coast += 1
        self.age += 1

    def update(self, det_bbox: Tuple[int, int, int, int], det_conf: float,
               smoothing: float):
        """
        EMA-smooth bbox; update velocity on center.
        """
        x, y, w, h = self.bbox
        nx, ny, nw, nh = map(float, det_bbox)
        cx, cy = _center((x, y, w, h))
        ncx, ncy = _center((nx, ny, nw, nh))

        # velocity = last center delta (one-step)
        self.vx = ncx - cx
        self.vy = ncy - cy

        # EMA smoothing to stabilize jitter from detectors
        a = float(smoothing)
        self.bbox = (a * x + (1 - a) * nx,
                     a * y + (1 - a) * ny,
                     a * w + (1 - a) * nw,
                     a * h + (1 - a) * nh)

        self.conf = float(det_conf)
        self.coast = 0
        self.hits += 1
        self.age += 1
        self.last_update_t = time.time()


# =========================
# Greedy matcher
# =========================

def _greedy_match(tracks: List[_Track], dets: List[Dict], iou_thresh: float,
                  prefer_same_class: bool = False) -> Tuple[List[Tuple[int, int]],
                                                            List[int], List[int]]:
    """
    Returns:
      matches: list of (track_idx, det_idx)
      unmatched_tracks: list of indices into tracks
      unmatched_dets: list of indices into dets
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    N, M = len(tracks), len(dets)
    iou_mat = np.zeros((N, M), dtype=np.float32)

    for i, tr in enumerate(tracks):
        for j, d in enumerate(dets):
            iou = _iou(tr.bbox, d["bbox"])
            if prefer_same_class and ("cls" in d) and (tr.cls != int(d["cls"])):
                iou *= 0.9  # mild penalty; still allow cross-class if geometry fits
            iou_mat[i, j] = iou

    matches = []
    used_t = set()
    used_d = set()

    # Greedy: repeatedly pick the best remaining IoU above threshold
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best = float(iou_mat[i, j])
        if best < iou_thresh:
            break
        if i in used_t or j in used_d:
            iou_mat[i, j] = -1.0
            continue
        matches.append((i, j))
        used_t.add(i)
        used_d.add(j)
        iou_mat[i, :] = -1.0
        iou_mat[:, j] = -1.0

    unmatched_tracks = [i for i in range(N) if i not in used_t]
    unmatched_dets   = [j for j in range(M) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets


# =========================
# Public Wrapper
# =========================

class ByteTrack:
    """
    Minimal ByteTrack-like wrapper (no external deps).

    API:
        bt = ByteTrack()
        tracks = bt.update(detections, img_shape=(H, W, C))

    'detections' is a list of dicts with keys: bbox(x,y,w,h), conf, cls
    Returns tracks as list of dicts with keys: id, bbox, conf, cls
    """
    def __init__(self, cfg_path: Optional[str] = None, **overrides):
        # cfg_path kept for compatibility; you can parse YAML if you want later.
        self.cfg = BTConfig(**overrides)
        self.next_id = 1
        self.tracked: List[_Track] = []  # currently tracked
        self.lost: List[_Track] = []     # recently lost (candidate for recovery)

    def reset(self):
        self.next_id = 1
        self.tracked.clear()
        self.lost.clear()

    def update(self, detections: List[Dict], img_shape) -> List[Dict]:
        """
        One call per frame.
        """
        cfg = self.cfg
        H, W = img_shape[0], img_shape[1]

        # --- Split detections by confidence ---
        high, low = [], []
        for d in detections or []:
            conf = float(d.get("conf", 0.0))
            row = dict(bbox=d["bbox"], conf=conf, cls=int(d.get("cls", -1)))
            if conf >= cfg.high_thresh:
                high.append(row)
            elif conf >= cfg.low_thresh:
                low.append(row)

        # --- Predict (coast) current tracks before matching ---
        for tr in self.tracked:
            if tr.coast < cfg.max_coast:
                tr.predict()
            else:
                # move to lost pool if over coast limit
                self.lost.append(tr)
        # Keep only tracks that haven't exceeded coast
        self.tracked = [t for t in self.tracked if t.coast <= cfg.max_coast]

        # Age lost tracks and drop stale ones
        for tr in self.lost:
            tr.predict()
        # Drop those lost for too long
        self.lost = [t for t in self.lost if t.coast <= cfg.track_buffer]

        # --- 1) Match HIGH detections to active tracks ---
        m, u_tr, u_det = _greedy_match(self.tracked, high, cfg.match_thresh, cfg.use_class_in_match)

        for (ti, di) in m:
            self.tracked[ti].update(high[di]["bbox"], high[di]["conf"], cfg.smoothing)
            self.tracked[ti].cls = int(high[di]["cls"])

        # Unmatched tracked → move to lost (but keep their state)
        move_to_lost = [self.tracked[i] for i in u_tr]
        if move_to_lost:
            self.lost.extend(move_to_lost)
        self.tracked = [t for i, t in enumerate(self.tracked) if i not in u_tr]

        # --- Spawn new tracks from unmatched HIGH detections ---
        for di in u_det:
            det = high[di]
            if det["conf"] >= cfg.new_track_thresh:
                self._spawn(det["bbox"], det["conf"], det["cls"])

        # --- 2) Try to recover LOST tracks using LOW detections ---
        if self.lost and low:
            m2, u_lost, u_low = _greedy_match(self.lost, low, cfg.match_thresh, cfg.use_class_in_match)
            # Matched lost → return to tracked
            # We reinsert them into tracked and update their state
            recovered_indices = set()
            for (li, di) in m2:
                tr = self.lost[li]
                det = low[di]
                tr.update(det["bbox"], det["conf"], cfg.smoothing)
                tr.cls = int(det["cls"])
                self.tracked.append(tr)
                recovered_indices.add(li)
            # Keep only truly still-lost
            self.lost = [t for i, t in enumerate(self.lost) if i not in recovered_indices]

        # --- Finalize: build outputs from current tracked set ---
        out: List[Dict] = []
        for tr in self.tracked:
            if tr.hits >= cfg.min_hits or tr.coast == 0:
                x, y, w, h = tr.bbox
                # clip & int-cast
                x = int(max(0, min(W - 1, round(x))))
                y = int(max(0, min(H - 1, round(y))))
                w = int(max(0, min(W - x, round(w))))
                h = int(max(0, min(H - y, round(h))))
                out.append({
                    "id": tr.id,
                    "bbox": (x, y, w, h),
                    "conf": float(tr.conf),
                    "cls": int(tr.cls),
                })

        return out

    # -------------------------
    # Internals
    # -------------------------

    def _spawn(self, bbox_xywh: Tuple[int, int, int, int], conf: float, cls: int):
        t = _Track(self.next_id, bbox_xywh, conf, cls)
        self.next_id += 1
        self.tracked.append(t)

