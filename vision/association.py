# vision/association.py
"""
Association utilities for matching detections↔tracks (single cam) and
tracks↔tracks (cross-cam).

Features
- Hybrid cost: (1 - IoU) + normalized center-distance + log-size change
- Hard gates: min IoU, max center distance, max size change, class match, time since seen
- Solver: Hungarian via SciPy if available; else fast greedy fallback
- Two-stage ByteTrack-like recovery: match HIGH first, then try LOW to reacquire
- Cross-camera association helper using per-pixel bearings from intrinsics K

Input shapes
- Tracks: list[dict] with keys: {id:int, bbox:(x,y,w,h), center:(cx,cy), scale:float, conf:float, cls:int, last_seen_t:float?}
- Detections: list[dict] with keys: {bbox:(x,y,w,h), conf:float, cls:int}
Return
- matches: list[(track_index, det_index)] (or (idxA, idxB) for cross-cam)
- unmatched_tracks: list[int]
- unmatched_dets: list[int]

This module aligns with the doc’s loop: keep a per-frame table
{id,bbox,center,scale,conf}; run YOLO + tracker for robust association and
(when stereo) fuse matched tracks across cameras for 3D. 【turn7file2†L9-L12】
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
import time

# Optional Hungarian; fallback to greedy if missing
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Optional geometry bearing helper (used for cross-cam association)
try:
    from vision.geometry import angles_from_pixel  # (K, (x,y)) -> (theta_x, theta_y)
except Exception:
    angles_from_pixel = None


# =========================
# Config
# =========================

@dataclass
class AssocConfig:
    # Cost weights (hybrid):
    w_iou: float = 1.0             # contributes (1 - IoU)
    w_center: float = 1.0          # contributes (dist / norm)
    w_size: float = 0.25           # contributes |log(h2/h1)|
    w_conf: float = 0.0            # optional: -mean(conf) bonus

    # Gates:
    min_iou: float = 0.1           # reject pairs below this IoU
    max_center_px: float = 256.0   # reject pairs beyond this distance
    max_log_size_change: float = 0.7   # reject |log(h2/h1)| beyond this
    class_must_match: bool = False
    max_unseen_time_s: float = 1.5     # drop very stale tracks up front

    # Solver:
    use_hungarian: bool = True      # falls back to greedy if SciPy not present

    # Normalization:
    center_norm_diag_frac: float = 0.5  # center dist normalized by 0.5*image_diag if img_size provided

    # Two-stage:
    enable_two_stage: bool = True
    match_iou_thresh: float = 0.3   # lower bound for a valid match (post-solve)

@dataclass
class AssocOutcome:
    matches: List[Tuple[int, int]]
    unmatched_tracks: List[int]
    unmatched_dets: List[int]


# =========================
# Basic geometry
# =========================

def _xywh_to_xyxy(bb: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bb
    return (x, y, x + w, y + h)

def bbox_iou(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-6
    return inter / union

def center_distance(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ac = (ax + aw * 0.5, ay + ah * 0.5)
    bc = (bx + bw * 0.5, by + bh * 0.5)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])

def log_size_change(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> float:
    # use height as scale proxy (robust for telescopes)
    ha = max(1e-3, a[3])
    hb = max(1e-3, b[3])
    return abs(math.log(hb / ha))


# =========================
# Gating & cost
# =========================

def _gate_pair(track: Dict, det: Dict, cfg: AssocConfig, center_norm: float) -> Tuple[bool, float, float, float]:
    iou = bbox_iou(track["bbox"], det["bbox"])
    if iou < cfg.min_iou:
        return False, iou, 0.0, 0.0
    dist = center_distance(track["bbox"], det["bbox"])
    if dist > cfg.max_center_px:
        return False, iou, dist, 0.0
    lsz = log_size_change(track["bbox"], det["bbox"])
    if lsz > cfg.max_log_size_change:
        return False, iou, dist, lsz
    if cfg.class_must_match and "cls" in track and "cls" in det and int(track["cls"]) != int(det["cls"]):
        return False, iou, dist, lsz
    # Good pair
    return True, iou, dist, lsz

def _hybrid_cost(track: Dict, det: Dict, cfg: AssocConfig, center_norm: float) -> float:
    ok, iou, dist, lsz = _gate_pair(track, det, cfg, center_norm)
    if not ok:
        return float("inf")
    cost = cfg.w_iou * (1.0 - iou)
    cost += cfg.w_center * (dist / max(1e-6, center_norm))
    cost += cfg.w_size * lsz
    if cfg.w_conf != 0.0:
        # bonus: higher mean conf -> lower cost
        tc = float(track.get("conf", 0.0))
        dc = float(det.get("conf", 0.0))
        mean_c = 0.5 * (tc + dc)
        cost -= cfg.w_conf * mean_c
    return float(cost)


# =========================
# Core association (single cam)
# =========================

def associate(tracks: List[Dict],
              detections: List[Dict],
              cfg: Optional[AssocConfig] = None,
              image_size: Optional[Tuple[int, int]] = None) -> AssocOutcome:
    """
    One-shot association for the current frame.
    Drops tracks that have been unseen beyond cfg.max_unseen_time_s (if last_seen_t present).
    """
    cfg = cfg or AssocConfig()

    # Pre-drop stale tracks
    now = time.monotonic()
    active_idx = []
    for i, tr in enumerate(tracks or []):
        last_t = float(tr.get("last_seen_t", now))
        if (now - last_t) <= cfg.max_unseen_time_s:
            active_idx.append(i)

    if not active_idx or not detections:
        return AssocOutcome(matches=[], unmatched_tracks=active_idx, unmatched_dets=list(range(len(detections or []))))

    T = len(active_idx)
    D = len(detections)
    H, W = (image_size[1], image_size[0]) if image_size else (None, None)
    img_diag = math.hypot(W, H) if (W and H) else None
    center_norm = (cfg.center_norm_diag_frac * img_diag) if img_diag else max(32.0, cfg.max_center_px)

    # Build cost matrix
    C = np.full((T, D), float("inf"), dtype=np.float32)
    for ti, i in enumerate(active_idx):
        tr = tracks[i]
        for dj, det in enumerate(detections):
            C[ti, dj] = _hybrid_cost(tr, det, cfg, center_norm)

    # Solve
    matches, ut, ud = _solve_and_filter(C, cfg.match_iou_thresh, tracks, detections, active_idx)
    return AssocOutcome(matches=matches, unmatched_tracks=ut, unmatched_dets=ud)


def associate_two_stage(tracks: List[Dict],
                        dets_high: List[Dict],
                        dets_low: List[Dict],
                        cfg: Optional[AssocConfig] = None,
                        image_size: Optional[Tuple[int, int]] = None) -> Tuple[AssocOutcome, AssocOutcome]:
    """
    ByteTrack-like: first match HIGH detections; spawn/keep from HIGH;
    then attempt to recover remaining tracks with LOW detections.
    Returns (stage1_outcome, stage2_outcome).
    """
    cfg = cfg or AssocConfig()
    s1 = associate(tracks, dets_high, cfg, image_size)

    # Build remaining tracks list from unmatched
    rem_tracks = [tracks[i] for i in s1.unmatched_tracks]
    s2 = associate(rem_tracks, dets_low, cfg, image_size)

    # Remap stage2 track indices to original indices
    s2_matches = []
    for (tidx2, didx) in s2.matches:
        s2_matches.append((s1.unmatched_tracks[tidx2], didx))
    s2_unmatched_tracks = [s1.unmatched_tracks[i] for i in s2.unmatched_tracks]

    return (AssocOutcome(matches=s1.matches, unmatched_tracks=s1.unmatched_tracks, unmatched_dets=s1.unmatched_dets),
            AssocOutcome(matches=s2_matches, unmatched_tracks=s2_unmatched_tracks, unmatched_dets=s2.unmatched_dets))


def _solve_and_filter(C: np.ndarray,
                      iou_thresh: float,
                      tracks: List[Dict],
                      dets: List[Dict],
                      active_idx: List[int]) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Runs Hungarian (if available) or greedy on cost matrix C and filters
    pairs whose IoU falls below iou_thresh after assignment.
    """
    T, D = C.shape
    if T == 0 or D == 0:
        return [], list(active_idx), list(range(D))

    if _HAVE_SCIPY and T > 1 and D > 1:
        row_ind, col_ind = linear_sum_assignment(C)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    else:
        pairs = _greedy_assign(C)

    used_t, used_d, matches = set(), set(), []
    for r, c in pairs:
        if not np.isfinite(C[r, c]):
            continue
        ti = active_idx[r]
        tr = tracks[ti]
        det = dets[c]
        # Post-check with IoU threshold (keeps acceptable geometry)
        if bbox_iou(tr["bbox"], det["bbox"]) < iou_thresh:
            continue
        matches.append((ti, c))
        used_t.add(r)
        used_d.add(c)

    unmatched_tracks = [active_idx[r] for r in range(T) if r not in used_t]
    unmatched_dets = [c for c in range(D) if c not in used_d]
    return matches, unmatched_tracks, unmatched_dets


def _greedy_assign(C: np.ndarray) -> List[Tuple[int, int]]:
    """
    Simple greedy assignment by ascending cost.
    """
    T, D = C.shape
    flat = [(C[r, c], r, c) for r in range(T) for c in range(D)]
    flat.sort(key=lambda x: x[0])
    used_r, used_c, out = set(), set(), []
    for cost, r, c in flat:
        if not np.isfinite(cost):
            continue
        if r in used_r or c in used_c:
            continue
        out.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return out


# =========================
# Cross-camera association
# =========================

def associate_cross_cameras(tracks_A: List[Dict],
                            tracks_B: List[Dict],
                            K_A: Optional[np.ndarray] = None,
                            K_B: Optional[np.ndarray] = None,
                            max_bearing_err_deg: float = 1.5,
                            prefer_same_class: bool = True) -> AssocOutcome:
    """
    Match tracks between two cameras by comparing bearing angles to each target.
    If K is provided and angles_from_pixel is available, we compute (theta_x, theta_y)
    from pixel centers; else we fall back to normalized image coords.

    This supports the “match across cams → triangulate 3D” step in the PDF. 【turn7file4†L15-L23】
    """
    if not tracks_A or not tracks_B:
        return AssocOutcome(matches=[], unmatched_tracks=list(range(len(tracks_A))),
                            unmatched_dets=list(range(len(tracks_B))))

    # Build angle vectors
    angA = [_bearing_from_row(r, K_A) for r in tracks_A]
    angB = [_bearing_from_row(r, K_B) for r in tracks_B]

    # Cost = angular distance (deg), with mild class bonus
    N, M = len(tracks_A), len(tracks_B)
    C = np.full((N, M), float("inf"), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            da = _ang_dist_deg(angA[i], angB[j])
            if prefer_same_class and ("cls" in tracks_A[i]) and ("cls" in tracks_B[j]) and (int(tracks_A[i]["cls"]) != int(tracks_B[j]["cls"])):
                da *= 1.05  # mild penalty
            if da <= max_bearing_err_deg:
                C[i, j] = da

    # Solve with Hungarian/greedy on bearing distance
    pairs = _greedy_assign(C) if (not _HAVE_SCIPY or N <= 1 or M <= 1) else \
            [(int(r), int(c)) for r, c in zip(*linear_sum_assignment(C))]

    used_i, used_j, matches = set(), set(), []
    for r, c in pairs:
        if not np.isfinite(C[r, c]) or C[r, c] > max_bearing_err_deg:
            continue
        matches.append((r, c))
        used_i.add(r)
        used_j.add(c)

    unmatched_A = [i for i in range(N) if i not in used_i]
    unmatched_B = [j for j in range(M) if j not in used_j]
    return AssocOutcome(matches=matches, unmatched_tracks=unmatched_A, unmatched_dets=unmatched_B)


def _bearing_from_row(row: Dict, K: Optional[np.ndarray]) -> Tuple[float, float]:
    cx, cy = row.get("center", (row["bbox"][0] + row["bbox"][2] * 0.5,
                                row["bbox"][1] + row["bbox"][3] * 0.5))
    if K is not None and angles_from_pixel is not None:
        thx, thy = angles_from_pixel(K, (cx, cy))
        return math.degrees(thx), math.degrees(thy)
    # Fallback: normalized coords → small-angle approx
    # Assume principal point ~ image center is already referenced in 'center'.
    return float(cx), float(cy)

def _ang_dist_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

