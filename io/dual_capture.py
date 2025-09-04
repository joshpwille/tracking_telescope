# io/dual_capture.py
from __future__ import annotations
import cv2
import time
import threading
from collections import deque
from typing import Deque, Optional, Tuple, Union

SrcT = Union[int, str]
FrameT = Tuple[int, "cv2.Mat"]  # (timestamp_ns, frame)


def _open_capture(src: SrcT) -> cv2.VideoCapture:
    if isinstance(src, str) and src.endswith("!appsink"):
        return cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(src)


class _CamThread(threading.Thread):
    """
    Background grab loop. Pushes (ts_ns, frame) into a bounded deque.
    Drops oldest frames to keep latency bounded.
    """
    def __init__(
        self,
        name: str,
        src: SrcT,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        outq: Deque[FrameT],
        maxlen: int = 120,
        warmup_s: float = 1.5,
    ):
        super().__init__(name=name, daemon=True)
        self.src, self.width, self.height, self.fps, self.fourcc = src, width, height, fps, fourcc
        self.outq = outq
        self.maxlen = maxlen
        self.warmup_s = warmup_s
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.cap: Optional[cv2.VideoCapture] = None

    def _configure(self, cap: cv2.VideoCapture):
        # Best-effort property set (may be ignored depending on backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

    def run(self):
        self.cap = _open_capture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Failed to open source: {self.src}")
        self._configure(self.cap)
        # Warm-up (let exposure/awb settle)
        t0 = time.time()
        while not self._stop.is_set() and (time.time() - t0) < self.warmup_s:
            self.cap.read()

        self._ready.set()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                # Give the backend a brief chance to recover
                time.sleep(0.005)
                continue
            ts = time.monotonic_ns()
            if len(self.outq) >= self.maxlen:
                self.outq.popleft()  # drop oldest to bound latency
            self.outq.append((ts, frame))

        # Cleanup
        try:
            self.cap.release()
        except Exception:
            pass

    def stop(self):
        self._stop.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout=timeout)


class DualCapture:
    """
    Open two cameras and read synchronized pairs.

    Pairing strategy:
      - Each stream is timestamped with time.monotonic_ns()
      - When you call read_pair(), we take the newest frame on the left,
        find the closest-timestamp frame on the right within max_dt_ms.
      - Older unmatched frames are dropped to keep latency low.

    Notes:
      - For best results, keep exposure/gain locked on both cameras.
      - If using libcamera via GStreamer, pass pipelines ending with '! appsink'.
    """
    def __init__(
        self,
        src_left: SrcT,
        src_right: SrcT,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fourcc: str = "YUYV",
        queue_len: int = 120,
        warmup_s: float = 1.5,
    ):
        self.left_q: Deque[FrameT] = deque(maxlen=queue_len)
        self.right_q: Deque[FrameT] = deque(maxlen=queue_len)
        self.tL = _CamThread("camL", src_left,  width, height, fps, fourcc, self.left_q,  queue_len, warmup_s)
        self.tR = _CamThread("camR", src_right, width, height, fps, fourcc, self.right_q, queue_len, warmup_s)
        self._started = False

    # ----- lifecycle -----
    def start(self) -> None:
        if self._started:
            return
        self.tL.start()
        self.tR.start()
        if not self.tL.wait_ready() or not self.tR.wait_ready():
            raise RuntimeError("[dual_capture] Cameras not ready.")
        self._started = True
        print("[dual_capture] Both cameras started.")

    def stop(self) -> None:
        if not self._started:
            return
        self.tL.stop()
        self.tR.stop()
        self.tL.join(timeout=2.0)
        self.tR.join(timeout=2.0)
        self._started = False
        print("[dual_capture] Stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ----- helpers -----
    @staticmethod
    def _closest_by_time(target_ts: int, q: Deque[FrameT]) -> Optional[FrameT]:
        """
        Find element in deque with timestamp closest to target_ts.
        Returns the element; does not remove it.
        """
        if not q:
            return None
        # Linear scan is fine for short buffers (<= few hundred)
        best = None
        best_err = None
        for ts, frame in q:
            err = abs(ts - target_ts)
            if best is None or err < best_err:
                best, best_err = (ts, frame), err
        return best

    def read_pair(
        self,
        max_dt_ms: float = 20.0,
        timeout_s: float = 1.0,
        drop_older: bool = True,
    ) -> Optional[Tuple[int, "cv2.Mat", int, "cv2.Mat"]]:
        """
        Attempt to pair frames by nearest timestamps.

        Args:
            max_dt_ms: maximum allowed time difference between paired frames.
            timeout_s: how long to wait for queues to have content before giving up.
            drop_older: if True, drops any frames older than the chosen pair to keep queues fresh.

        Returns:
            (tsL, frameL, tsR, frameR) or None if no pair within thresholds.
        """
        if not self._started:
            raise RuntimeError("[dual_capture] Call start() first.")

        # Wait for both queues to have at least one frame
        deadline = time.time() + timeout_s
        while (not self.left_q or not self.right_q) and time.time() < deadline:
            time.sleep(0.002)
        if not self.left_q or not self.right_q:
            return None

        # Use the newest left frame as reference
        tsL, fL = self.left_q[-1]

        # Find the closest right frame
        candidate = self._closest_by_time(tsL, self.right_q)
        if candidate is None:
            return None
        tsR, fR = candidate

        # Check skew constraint
        if abs(tsR - tsL) > int(max_dt_ms * 1e6):
            # If skew too large, drop older side(s) to catch up
            if drop_older:
                # Drop everything older than min(tsL, tsR)
                threshold = min(tsL, tsR)
                while self.left_q and self.left_q[0][0] < threshold:
                    self.left_q.popleft()
                while self.right_q and self.right_q[0][0] < threshold:
                    self.right_q.popleft()
            return None

        # Optionally drop frames older than the chosen pair to bound latency
        if drop_older:
            while self.left_q and self.left_q[0][0] < tsL:
                self.left_q.popleft()
            while self.right_q and self.right_q[0][0] < tsR:
                self.right_q.popleft()

        return tsL, fL, tsR, fR

    def latest(self) -> Optional[Tuple[int, "cv2.Mat", int, "cv2.Mat"]]:
        """
        Return the most recent frames from both queues without pairing check.
        Useful for preview; may be skewed.
        """
        if not self.left_q or not self.right_q:
            return None
        tsL, fL = self.left_q[-1]
        tsR, fR = self.right_q[-1]
        return tsL, fL, tsR, fR


# ----- quick manual test -----
if __name__ == "__main__":
    # Example: two local cameras 0 and 1 (or GStreamer strings ending with '! appsink')
    left_src = 0
    right_src = 1

    dc = DualCapture(left_src, right_src, width=1280, height=720, fps=30.0, fourcc="YUYV")
    with dc:
        win = "dual_capture"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        shown = 0
        t0 = time.time()
        while True:
            pair = dc.read_pair(max_dt_ms=20.0, timeout_s=0.5)
            if pair is None:
                # no good pair yet; try again
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
            tsL, fL, tsR, fR = pair

            # visualize side-by-side
            h = max(fL.shape[0], fR.shape[0])
            w = fL.shape[1] + fR.shape[1]
            view = cv2.resize(fL, (fL.shape[1], h))
            right_r = cv2.resize(fR, (fR.shape[1], h))
            import numpy as np
            canvas = np.zeros((h, w, 3), dtype=fL.dtype)
            canvas[:view.shape[0], :view.shape[1]] = view
            canvas[:right_r.shape[0], view.shape[1]:] = right_r

            skew_ms = (tsR - tsL) / 1e6
            cv2.putText(canvas, f"skew={skew_ms:.2f} ms", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, canvas)

            shown += 1
            if shown % 60 == 0:
                dt = time.time() - t0
                if dt > 0:
                    print(f"[dual_capture] paired ~{shown/dt:.1f} FPS")
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        cv2.destroyAllWindows()
