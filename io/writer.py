# io/writer.py
from __future__ import annotations
import cv2
import time
from pathlib import Path
from typing import Optional, Tuple

class VideoWriter:
    """
    Simple wrapper around cv2.VideoWriter for saving frames to disk.
    - Creates parent directories automatically
    - Supports auto fps detection (if not provided)
    - Can overlay timestamp text for debugging
    """

    def __init__(self,
                 path: str = "outputs/out.mp4",
                 size: Optional[Tuple[int, int]] = None,
                 fps: Optional[float] = None,
                 fourcc: str = "mp4v",
                 draw_timestamp: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.size = size      # (width, height)
        self.fps = fps
        self.fourcc = fourcc
        self.draw_timestamp = draw_timestamp

        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        self._t0 = None

    def open(self, frame_shape: Tuple[int, int, int]) -> None:
        """Initialize writer once frame size/fps are known."""
        if self._writer is not None:
            return

        if self.size is None:
            h, w = frame_shape[:2]
            self.size = (w, h)

        if self.fps is None:
            # Fallback: estimate later, but set to 30.0 initially
            self.fps = 30.0

        fourcc_val = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(str(self.path), fourcc_val, self.fps, self.size)
        if not self._writer.isOpened():
            raise RuntimeError(f"[writer] Failed to open {self.path}")

        self._t0 = time.time()
        self._frame_count = 0
        print(f"[writer] Writing {self.size} @ {self.fps:.1f}fps → {self.path}")

    def write(self, frame) -> None:
        """Write one frame (BGR)."""
        if self._writer is None:
            self.open(frame.shape)

        out_frame = frame
        if self.draw_timestamp:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            ms = int((time.time() % 1) * 1000)
            text = f"{ts}.{ms:03d}"
            cv2.putText(out_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self._writer.write(out_frame)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            if self._t0 is not None and self._frame_count > 0:
                dt = time.time() - self._t0
                fps_est = self._frame_count / dt if dt > 0 else 0
                print(f"[writer] Closed {self.path} ({self._frame_count} frames, ~{fps_est:.1f} fps)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

