from yolo_hud_overlay.cli import build_parser


def test_cli_keeps_tapo_watcher_compatible_args():
    args = build_parser().parse_args([
        "--input", "in.mp4",
        "--output", "out.mp4",
        "--model", "yolo11s.pt",
        "--conf", "0.25",
        "--imgsz", "960",
        "--classes", "0,1,2",
        "--stride", "5",
        "--crf", "16",
        "--preset", "slow",
        "--theme", "themes/neon.json",
    ])
    assert args.input.name == "in.mp4"
    assert args.output.name == "out.mp4"
    assert args.model == "yolo11s.pt"
    assert args.theme == "themes/neon.json"
