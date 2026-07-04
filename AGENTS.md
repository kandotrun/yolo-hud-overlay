# Agent guidance

This repo is a public, standalone display-layer tool for YOLO video overlays.

Rules:
- Do not commit real camera videos, credentials, `.env`, YOLO weights (`*.pt`), or local secret paths.
- Keep the CLI compatible with `scripts/yolo_overlay_video.py --input --output --model --conf --imgsz --classes --stride --crf --preset`.
- Visual design should be changed via JSON themes when possible, not hardcoded.
- Run `uv run pytest -q` after code changes.
- For real-video smoke tests, use local files outside the repo and do not commit outputs.
