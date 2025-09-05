# app/main.py
from app.pipeline import Pipeline, PipelineConfig, IOConfig, GeometryConfig, DetectConfig, TrackConfig, MotorGains

def main():
    cfg = PipelineConfig(
        cam_id="cam0",
        io=IOConfig(video_source=0, output_path=None),
        geom=GeometryConfig(
            calib_path="io/calib_left.yml",
            zoom_lut_path="io/zoom_lut_left.json",
            pivot_world=(0.0, 0.0, 0.0),
            offset_cam_from_pivot=(0.0, 0.0, 0.0),
        ),
        detect=DetectConfig(conf_thres=0.25, iou_thres=0.45, detect_every_n=1),
        track=TrackConfig(use_bytetrack=True, bytetrack_cfg="vision/bytetrack_planes.yaml"),
        gains=MotorGains(base_gain=0.4, max_gain=1.2, deadband_px=4.0, rate_limit_deg_s=12.0),
        draw_overlay=True,
        show_window=True,       # press 'q' to quit
        enable_trianguation=False,
        enable_motors=False,    # start safe; turn on after verifying
        target_id=None,
    )
    Pipeline(cfg).run()

if __name__ == "__main__":
    main()

