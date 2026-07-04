# yolo-hud-overlay

Themeable YOLO HUD overlay for videos. Built for quick iteration on detection-box visual design without editing Python.

## Quick start

```bash
python scripts/yolo_overlay_video.py \
  --input input.mp4 \
  --output output.mp4 \
  --model yolo11s.pt \
  --conf 0.25 \
  --imgsz 960 \
  --classes 0,1,2,3,5,7,16 \
  --stride 5 \
  --theme themes/neon.json \
  --track
```

If `--theme` is omitted, the CLI uses `YOLO_HUD_THEME` when set, then falls back to `themes/neon.json`.

## Edit the design

Edit:

```txt
themes/neon.json
```

Useful knobs:

Label format variables: `{name}`, `{track_label}`, `{track_id}`, `{class_name}`, `{class_id}`, `{confidence}`, `{confidence_percent}`. By default tracking is anonymous per-video only, e.g. `PERSON-001`; it does not identify real people or compare faces across videos.

- `box.style`: `corners` or `rect`
- `box.glow`: neon glow on/off
- `label.format`: e.g. `{name} {confidence_percent:.0f}%`
- `confidence_bar.enabled`
- `effects.vignette`, `effects.scanline`, `effects.contrast`, `effects.brightness`
- `colors`: per-class/group colors
- `priority_classes`
- `max_boxes`

## Anonymous tracking IDs

Tracking is enabled by default. It assigns temporary per-video IDs by class using bounding-box overlap:

```txt
PERSON-001
DOG-001
CAR-001
```

These IDs reset per run and are not biometric identity recognition. Use `--no-track` to disable.

## Cron integration example

```env
TAPO_WATCH_YOLO_SCRIPT=/home/kan/yolo-hud-overlay/scripts/yolo_overlay_video.py
YOLO_HUD_THEME=/home/kan/yolo-hud-overlay/themes/neon.json
```

The script stays compatible with the existing Tapo watcher arguments.

## Safety

Do not commit personal camera clips, `.env`, YOLO weights, or Tapo credentials.
