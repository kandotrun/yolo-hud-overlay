from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from .theme import load_theme, parse_classes


def overlay_video(input_path: Path, raw_output_path: Path, theme: dict, model_name: str, conf: float, imgsz: int, classes: list[int], stride: int, track: bool = True, track_iou: float = 0.25, track_max_age: int = 10) -> None:
    import cv2
    from ultralytics import YOLO

    from .render import render_frame
    from .tracker import AnonymousTracker, detections_from_yolo

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output video: {raw_output_path}")
    stride = max(1, stride)
    tracker = AnonymousTracker(iou_threshold=track_iou, max_age=track_max_age)
    last_boxes = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                result = model.predict(frame, imgsz=imgsz, conf=conf, classes=classes, verbose=False, device="cpu")[0]
                detections = detections_from_yolo(result.boxes)
                last_boxes = tracker.update(detections) if track else detections
            writer.write(render_frame(frame, last_boxes, theme, model.names, frame_index=idx, fps=fps))
            idx += 1
    finally:
        writer.release()
        cap.release()


def h264_faststart(raw_output_path: Path, output_path: Path, crf: int, preset: str) -> None:
    tmp_h264 = output_path.with_suffix(".h264.tmp.mp4")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(raw_output_path), "-an", "-c:v", "libx264", "-preset", preset,
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(tmp_h264),
        ],
        check=True,
    )
    tmp_h264.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a themeable YOLO HUD overlay to a video.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--conf", default=0.2, type=float)
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--classes", default=",".join(str(index) for index in range(80)))
    parser.add_argument("--stride", default=5, type=int)
    parser.add_argument("--crf", default=18, type=int)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--theme", default=None, help="Path to a theme JSON. Defaults to YOLO_HUD_THEME or themes/neon.json.")
    parser.set_defaults(track=True)
    parser.add_argument("--track", dest="track", action="store_true", help="Enable anonymous per-video tracking IDs, e.g. PERSON-001.")
    parser.add_argument("--no-track", dest="track", action="store_false", help="Disable anonymous tracking IDs.")
    parser.add_argument("--track-iou", default=0.25, type=float)
    parser.add_argument("--track-max-age", default=10, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    theme = load_theme(args.theme)
    raw_output = args.output.with_suffix(".raw.tmp.mp4")
    overlay_video(args.input, raw_output, theme, args.model, args.conf, args.imgsz, parse_classes(args.classes), args.stride, args.track, args.track_iou, args.track_max_age)
    h264_faststart(raw_output, args.output, args.crf, args.preset)
    raw_output.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
