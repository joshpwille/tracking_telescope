# vision/optical_slow.py
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class SlowOFParams:
    # Shi-Tomasi feature params
    max_corners: int = 800
    quality_level: float = 0.01
    min_distance: int = 7
    block_size: int = 7
    use_harris: bool = False
    k: float = 0.04

    # Pyramidal LK params
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    iters: int = 30
    eps: float = 0.01

    # Reseed / maintenance
    min_valid_ratio: float = 0.35   # reseed if < this fraction survive
    min_points: int = 80            # reseed if < this # valid
    reseed_every: int = 20          # force reseed every N frames

    # Outlier rejection
    ransac_reproj_thresh: float = 3.0
    use_affine_ransac: bool = True  # otherwise just status mask

    # Preproc
    equalize_hist: bool = True
    gaussian_blur_ksize: int = 0  # 0 = disabled


class OpticalFlowSlow:
    """
    Slower but robust optical flow feature tracker using pyramidal LK + RANSAC.
    - Tracks a cloud of Shi-Tomasi corners.
    - Optionally restricts to ROI (x,y,w,h).
    - RANSAC removes outliers; returns flow stats and a bbox around good points.
    """

    def __init__(self, params: Optional[SlowOFParams] = None):
        self.p = params or SlowOFParams()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # Nx1x2 float32
        self.frame_idx: int = 0
        self.roi: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h

    # --------- public API ----------
    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        self.roi = roi

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.frame_idx = 0

    def seed(self, gray: np.ndarray) -> None:
        self.prev_gray = self._preprocess(gray)
        mask = self._roi_mask(self.prev_gray.shape[:2]) if self.roi else None
        self.prev_pts = cv2.goodFeaturesToTrack(
            image=self.prev_gray,
            maxCorners=self.p.max_corners,
            qualityLevel=self.p.quality_level,
            minDistance=self.p.min_distance,
            blockSize=self.p.block_size,
            useHarrisDetector=self.p.use_harris,
            k=self.p.k,
            mask=mask,
        )
        self.frame_idx = 0

    def track(self, gray: np.ndarray) -> Dict[str, object]:
        """
        Track features to new frame.
        Returns a dict with:
          - 'ok': bool
          - 'mean_flow': (dx, dy)
          - 'valid_ratio': float
          - 'n_valid': int
          - 'bbox': (x,y,w,h) or None  (around inlier points)
          - 'M': 2x3 affine matrix or None (if use_affine_ransac)
          - 'prev_pts', 'curr_pts': Nx2 float arrays (inliers only)
          - 'status', 'err': raw LK outputs (before RANSAC)
        """
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 4:
            self.seed(gray)

        curr_gray = self._preprocess(gray)

        lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                       self.p.iters, self.p.eps)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None,
            winSize=self.p.win_size,
            maxLevel=self.p.max_level,
            criteria=lk_criteria,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
        )

        if next_pts is None or status is None:
            # fail → reseed and try next frame
            self.seed(gray)
            return {"ok": False, "mean_flow": (0.0, 0.0), "valid_ratio": 0.0,
                    "n_valid": 0, "bbox": None, "M": None,
                    "prev_pts": np.empty((0, 2), np.float32),
                    "curr_pts": np.empty((0, 2), np.float32),
                    "status": status, "err": err}

        good_prev = self.prev_pts[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]

        # ROI filter (if set)
        if self.roi:
            x, y, w, h = self.roi
            in_roi = (good_next[:, 0, 0] >= x) & (good_next[:, 0, 0] < x + w) & \
                     (good_next[:, 0, 1] >= y) & (good_next[:, 0, 1] < y + h)
            good_prev = good_prev[in_roi]
            good_next = good_next[in_roi]

        # RANSAC affine to remove outliers
        M = None
        if self.p.use_affine_ransac and len(good_prev) >= 3:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                good_prev.reshape(-1, 2),
                good_next.reshape(-1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.p.ransac_reproj_thresh,
                maxIters=2000, confidence=0.99, refineIters=10
            )
            if inlier_mask is not None:
                inlier_mask = inlier_mask.ravel().astype(bool)
                good_prev = good_prev[inlier_mask]
                good_next = good_next[inlier_mask]

        n_total = max(int(self.prev_pts.shape[0]), 1)
        n_valid = int(good_next.shape[0]) if good_next is not None else 0
        valid_ratio = n_valid / float(n_total)

        # Flow summary
        if n_valid > 0:
            flow = (good_next.reshape(-1, 2) - good_prev.reshape(-1, 2))
            mean_flow = tuple(np.mean(flow, axis=0).tolist())  # (dx, dy)
            bbox = self._pts_to_bbox(good_next.reshape(-1, 2))
        else:
            mean_flow = (0.0, 0.0)
            bbox = None

        # Maintenance: reseed if too few
        self.frame_idx += 1
        need_reseed = (
            (valid_ratio < self.p.min_valid_ratio) or
            (n_valid < self.p.min_points) or
            (self.frame_idx % self.p.reseed_every == 0)
        )

        # Prepare for next iteration
        if n_valid >= 4 and not need_reseed:
            self.prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32)
            self.prev_gray = curr_gray
        else:
            # Re-seed on current frame to refresh points
            self.prev_gray = curr_gray
            self.seed(curr_gray)

        return {
            "ok": (n_valid > 0),
            "mean_flow": (float(mean_flow[0]), float(mean_flow[1])),
            "valid_ratio": float(valid_ratio),
            "n_valid": int(n_valid),
            "bbox": bbox,
            "M": M,
            "prev_pts": (good_prev.reshape(-1, 2) if n_valid > 0 else np.empty((0, 2), np.float32)),
            "curr_pts": (good_next.reshape(-1, 2) if n_valid > 0 else np.empty((0, 2), np.float32)),
            "status": status,
            "err": err,
        }

    # --------- helpers ----------
    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Apply optional equalization/blur."""
        im = gray
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if self.p.equalize_hist:
            im = cv2.equalizeHist(im)
        if self.p.gaussian_blur_ksize and self.p.gaussian_blur_ksize > 0:
            k = self.p.gaussian_blur_ksize
            if k % 2 == 0:
                k += 1
            im = cv2.GaussianBlur(im, (k, k), 0)
        return im

    def _roi_mask(self, shape_hw: Tuple[int, int]) -> np.ndarray:
        h, w = shape_hw
        mask = np.zeros((h, w), np.uint8)
        if self.roi is None:
            mask[:, :] = 255
        else:
            x, y, rw, rh = self.roi
            x0 = max(0, x); y0 = max(0, y)
            x1 = min(w, x + rw); y1 = min(h, y + rh)
            mask[y0:y1, x0:x1] = 255
        return mask

    @staticmethod
    def _pts_to_bbox(pts_xy: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if pts_xy.size == 0:
            return None
        x0, y0 = np.min(pts_xy, axis=0)
        x1, y1 = np.max(pts_xy, axis=0)
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        w = max(0, x1 - x0 + 1)
        h = max(0, y1 - y0 + 1)
        return (x0, y0, w, h)


# ------------- quick self-test -------------
if __name__ == "__main__":
    # Demo with webcam 0; draw mean flow vector and bbox
    cap = cv2.VideoCapture(0)
    of = OpticalFlowSlow()

    ok, frame = cap.read()
    if not ok:
        print("No camera.")
        exit(1)
    of.seed(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = of.track(gray)

        # visualize
        if res["ok"]:
            for p in res["curr_pts"]:
                cv2.circle(frame, tuple(p.astype(int)), 2, (0, 255, 0), -1)
            if res["bbox"] is not None:
                x, y, w, h = res["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            dx, dy = res["mean_flow"]
            h2, w2 = frame.shape[:2]
            cx, cy = w2 // 2, h2 // 2
            cv2.arrowedLine(frame, (cx, cy), (int(cx + 10 * dx), int(cy + 10 * dy)), (255, 0, 0), 2)
            cv2.putText(frame, f"valid={res['n_valid']} ({res['valid_ratio']*100:.1f}%)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("optical_slow", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

