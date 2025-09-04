# io/video_capture.py
from __future__ import annotations
import cv2
import time
from typing import Optional, Tuple


class VideoCapture:
    """
    Wrapper for Pi HQ camera (or any cv2.VideoCapture source).
    Supports V4L2 (legacy) and libcamera (Bullseye/Bookworm).
    """

    def __init__(self, source: str | int = 0,
                 width: int = 1280, height: int = 720,
                 fps: float = 30.0,
                 fourcc: str = "YUYV",
                 warmup: float = 2.0):
        """
        Args:
            source: camera index (int) or path/rtsp (str).
                    For Pi HQ Camera, usually 0.
                    On libcamera builds, use "libcamerasrc" via GStreamer string if needed.
            width: desired frame width
            height: desired frame height
            fps: target fps
            fourcc: pixel format (default YUYV for Pi HQ V4L2, MJPG also common)
            warmup: seconds to let sensor auto-exposure settle
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap: Optional[cv2.VideoCapture] = None
        self.open_time: float = 0.0

    def open(self) -> None:
        if isinstance(self.source, str) and self.source.endswith("!appsink"):
            # GStreamer pipeline string
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"[video_capture] Failed to open source: {self.source}")

            # Apply settings (best-effort, depends on driver)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if self.fourcc:
                fourcc_val = cv2.VideoWriter_fourcc(*self.fourcc)
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)

        self.open_time = time.time()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"[video_capture] Could not open video source {self.source}")

        if self.open_time > 0:
            print(f"[video_capture] Opened source={self.source} at {self.width}x{self.height}@{self.fps}fps")

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Returns (ok, frame). Call in loop.
        """
        if self.cap is None:
            raise RuntimeError("[video_capture] Call open() first.")
        return self.cap.read()

    def warmup(self, seconds: float = 2.0) -> None:
        """
        Throw away frames for a few seconds so auto-exposure settles.
        """
        if self.cap is None:
            return
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[video_capture] Released camera.")

    def __del__(self):
        self.release()


# --------- example standalone test ----------
if __name__ == "__main__":
    cam = VideoCapture(source=0, width=1920, height=1080, fps=30, fourcc="YUYV")
    cam.open()
    cam.warmup(2.0)

    while True:
        ok, frame = cam.read()
        if not ok:
            print("[video_capture] Frame grab failed.")
            break
        cv2.imshow("PiHQ Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

