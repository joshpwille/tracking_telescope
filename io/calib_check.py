# calib_check.py
from __future__ import annotations
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from io.config import get_config
from io.calib_loader import (
    init_from_config,
    undistort,
    rectify_pair,
    RectifyMaps,
)

def _open_source(src: str | int, width: int | None = None, height: int | None = None, fps: float | None = None):
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except (TypeError, ValueError):
        cap = cv2.VideoCapture(src)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap

def _draw_center(img):
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    return img

def _stack_h(imgL, imgR):
    h = max(imgL.shape[0], imgR.shape[0])
    w = imgL.shape[1] + imgR.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:imgL.shape[0], :imgL.shape[1]] = imgL
    out[:imgR.shape[0], imgL.shape[1]:imgL.shape[1] + imgR.shape[1]] = imgR
    return out

def _draw_epi_lines(pair, num=12, color=(0, 255, 0)):
    h, w = pair.shape[:2]
    left_w = w // 2
    step = max(h // (num + 1), 1)
    for y in range(step, h, step):
        cv2.line(pair, (0, y), (left_w - 1, y), color, 1, cv2.LINE_AA)
        cv2.line(pair, (left_w, y), (w - 1, y), color, 1, cv2.LINE_AA)
    return pair

def main():
    ap = argparse.ArgumentParser(description="Calibration visual check (mono undistort / stereo rectify)")
    ap.add_argument("--config", type=str, default=None, help="Override path to config.yaml/json")
    ap.add_argument("--source", type=str, default=None, help="Left source (index or path). Defaults to cfg.video.source")
    ap.add_argument("--right-source", type=str, default=None, help="Right source (index or path) for stereo")
    ap.add_argument("--force-compute-maps", action="store_true", help="Compute rectification maps even if cached")
    ap.add_argument("--alpha", type=float, default=0.0, help="stereoRectify alpha (0..1)")
    ap.add_argument("--window", type=str, default="calib_check", help="Window name")
    ap.add_argument("--no-epilines", action="store_true", help="Do not draw epipolar lines in stereo view")
    args = ap.parse_args()

    cfg = get_config(args.config)
    left, right, stereo, rect = init_from_config(cfg)

    # Optionally force recompute maps
    if args.force_compute_maps:
        from io.calib_loader import compute_rectification_maps, save_rectification_maps
        stereo, rect = compute_rectification_maps(
            left=left,
            right=right if cfg.app.use_stereo else None,
            stereo=stereo,
            alpha=args.alpha
        )
        # Save back for faster next run
        try:
            save_rectification_maps(
                rect,
                cfg.calib.rect_map_left_1,
                cfg.calib.rect_map_left_2,
                cfg.calib.rect_map_right_1 if cfg.app.use_stereo and right is not None else None,
                cfg.calib.rect_map_right_2 if cfg.app.use_stereo and right is not None else None
            )
            print("[calib_check] Rectification maps recomputed and saved.")
        except Exception as e:
            print(f"[calib_check] Warning: could not save rectification maps: {e}")

    # Print quick summary
    print("----- Calibration summary -----")
    print(f"Left K:\n{left.K}\nDist (len={left.dist.size}): {left.dist.ravel()[:8]}")
    print(f"Left image_size: {left.image_size}")
    if cfg.app.use_stereo and right is not None and stereo is not None:
        print(f"Stereo R:\n{stereo.R}\nT:\n{stereo.T.ravel()}")
        if stereo.P1 is not None and stereo.P2 is not None:
            print("Stereo rectification present (P1/P2).")
    else:
        print("Stereo disabled or not fully available; running mono undistort.")

    # Open sources
    left_src = args.source if args.source is not None else cfg.video.source
    capL = _open_source(left_src, cfg.video.width, cfg.video.height, cfg.video.fps)

    capR = None
    stereo_mode = bool(cfg.app.use_stereo and right is not None and stereo is not None)
    if args.right_source is not None:
        stereo_mode = True
    if stereo_mode:
        right_src = args.right_source if args.right_source is not None else 1  # default to cam index 1 if not provided
        capR = _open_source(right_src, cfg.video.width, cfg.video.height, cfg.video.fps)

    if not capL.isOpened() or (stereo_mode and (capR is None or not capR.isOpened())):
        print("[calib_check] Failed to open camera(s). Use --source/--right-source to set paths or indices.")
        return

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    last_save_idx = 0
    t0 = time.time()
    frames = 0

    try:
        while True:
            okL, frameL = capL.read()
            if not okL:
                print("[calib_check] Left stream ended.")
                break

            if stereo_mode:
                okR, frameR = capR.read()
                if not okR:
                    print("[calib_check] Right stream ended.")
                    break

                left_rect, right_rect = rectify_pair(frameL, frameR, rect)

                # Visuals
                left_rect_v = _draw_center(left_rect.copy())
                right_rect_v = _draw_center(right_rect.copy())
                pair = _stack_h(left_rect_v, right_rect_v)
                if not args.no_epilines:
                    pair = _draw_epi_lines(pair, num=12)

                cv2.imshow(args.window, pair)
            else:
                # Mono undistort
                if rect.left_map1 is None or rect.left_map2 is None:
                    # Fallback: show original if maps not available
                    view = _draw_center(frameL.copy())
                else:
                    und = undistort(frameL, (rect.left_map1, rect.left_map2))
                    # side-by-side original vs undistorted
                    left_v = _draw_center(frameL.copy())
                    und_v = _draw_center(und)
                    view = _stack_h(left_v, und_v)
                    # annotate
                    h, w = view.shape[:2]
                    cv2.putText(view, "LEFT (orig)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(view, "UNDISTORTED", (w // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(args.window, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            if key == ord("s"):
                out_dir = Path(cfg.paths.output_dir) if hasattr(cfg, "paths") else Path("outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"calibcheck_{last_save_idx:03d}.png"
                last_save_idx += 1
                if stereo_mode:
                    cv2.imwrite(str(out_dir / fname), pair)
                else:
                    cv2.imwrite(str(out_dir / fname), view)
                print(f"[calib_check] Saved snapshot to {out_dir / fname}")

            frames += 1
            if frames % 60 == 0:
                dt = time.time() - t0
                if dt > 0:
                    print(f"[calib_check] ~{frames/dt:.1f} FPS")

    finally:
        capL.release()
        if capR is not None:
            capR.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
