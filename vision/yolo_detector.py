# vision/yolo_detector.py
"""
YOLO detector wrapper with a stable, minimal API for the pipeline.

- Primary backend: Ultralytics YOLO (pip install ultralytics)
- Returns: list of dicts per frame:
    {'bbox': (x, y, w, h), 'conf': float, 'cls': int}

Usage:
    det = YoloDetector(model_path="vision/yolov5nu.pt", conf_thres=0.25, iou_thres=0.45, classes=None)
    preds = det.detect(frame_bgr)

Notes:
- 'classes' can be a list of class IDs to keep, e.g., [0] for 'plane'.
- If you pass device='cpu' it forces CPU; otherwise auto-selects CUDA when available.
- img_size controls the inference resolution (square).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
import sys

# -------------------------
# Dataclass config
# -------------------------

@dataclass
class YoloConfig:
    model_path: str = "vision/yolov5nu.pt"  # .pt or .onnx supported by Ultralytics
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    classes: Optional[List[int]] = None     # keep-only filter
    img_size: int = 640
    device: Optional[str] = None            # None -> auto, else 'cpu' or 'cuda:0'
    max_det: int = 300
    half: bool = True                       # fp16 on CUDA-capable devices


# -------------------------
# Public detector wrapper
# -------------------------

class YoloDetector:
    def __init__(self,
                 model_path: Optional[str] = None,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 classes: Optional[List[int]] = None,
                 img_size: int = 640,
                 device: Optional[str] = None,
                 max_det: int = 300,
                 half: bool = True):
        self.cfg = YoloConfig(
            model_path=model_path or "vision/yolov5nu.pt",
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            classes=classes,
            img_size=int(img_size),
            device=device,
            max_det=int(max_det),
            half=bool(half),
        )
        self._backend = _make_backend(self.cfg)

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Run inference on a single BGR frame and return normalized dicts."""
        return self._backend.detect(frame_bgr)


# -------------------------
# Backends
# -------------------------

def _make_backend(cfg: YoloConfig):
    # Prefer Ultralytics YOLO backend
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics not available. Install with: pip install ultralytics\n"
            f"Original import error: {e}"
        )
    return _UltralyticsBackend(cfg)


class _UltralyticsBackend:
    """
    Thin wrapper around ultralytics.YOLO for consistent output shape.
    """
    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        from ultralytics import YOLO  # local import for clarity
        try:
            self.model = YOLO(cfg.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {cfg.model_path}\n{e}")

        # Autoselect device unless forced
        self.device = cfg.device
        # Safety: Ultralytics chooses device automatically; we just pass through if provided.

        # quick warmup (optional)
        self._warm = False

    def _maybe_warmup(self, frame_bgr: np.ndarray):
        if self._warm:
            return
        try:
            _ = self.model.predict(
                source=frame_bgr,
                imgsz=self.cfg.img_size,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                device=self.device,
                half=self.cfg.half,
                max_det=max(1, self.cfg.max_det),
                verbose=False,
                stream=False,
            )
            self._warm = True
        except Exception:
            # If warmup fails, we'll still try regular inference later.
            pass

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        self._maybe_warmup(frame_bgr)

        # Ultralytics expects RGB or handles BGR; it accepts numpy arrays directly.
        try:
            results = self.model.predict(
                source=frame_bgr,
                imgsz=self.cfg.img_size,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                device=self.device,
                half=self.cfg.half,
                max_det=max(1, self.cfg.max_det),
                verbose=False,
                stream=False,
            )
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}")

        out: List[Dict] = []
        if not results:
            return out

        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            return out

        xyxy = r.boxes.xyxy.detach().cpu().numpy()  # (N,4)
        conf = r.boxes.conf.detach().cpu().numpy()  # (N,)
        cls  = r.boxes.cls.detach().cpu().numpy()   # (N,)

        H, W = r.orig_shape if hasattr(r, "orig_shape") else frame_bgr.shape[:2]

        # Filter classes if requested
        keep_mask = np.ones((xyxy.shape[0],), dtype=bool)
        if self.cfg.classes is not None:
            allowed = set(int(c) for c in self.cfg.classes)
            keep_mask &= np.array([int(c) in allowed for c in cls], dtype=bool)

        for i in range(xyxy.shape[0]):
            if not keep_mask[i]:
                continue
            x1, y1, x2, y2 = xyxy[i]
            # Clip to image bounds
            x1 = max(0, min(W - 1, float(x1)))
            y1 = max(0, min(H - 1, float(y1)))
            x2 = max(0, min(W - 1, float(x2)))
            y2 = max(0, min(H - 1, float(y2)))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            # Convert to xywh int
            bbox = (int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))
            out.append({
                "bbox": bbox,
                "conf": float(conf[i]),
                "cls": int(cls[i]),
            })

        return out

