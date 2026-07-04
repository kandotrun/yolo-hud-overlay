# yolo-hud-overlay

Themeable YOLO HUD overlay for videos. Built for quick iteration on detection-box visual design without editing Python.

Ships with a **CCTV / surveillance-editorial** look: a color-graded feed, one accent color, technical type, and a single "hero" target with a redacted zoom inset — inspired by streetwear camera-analysis edits, kept honest for an anonymous tool.

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
  --theme themes/recon.json \
  --track
```

If `--theme` is omitted, the CLI uses `YOLO_HUD_THEME` when set, then falls back to `themes/trace.json`.

## Themes

| Theme | Look |
| --- | --- |
| `themes/trace.json` | Red on a desaturated cool grade — high-alert "subject of interest" feed (default). |
| `themes/recon.json` | Lime on a cold blue grade — the flagship analysis-panel look. |
| `themes/surveil.json` | Amber on a warm grade — clean municipal-CCTV monitor. |
| `themes/neon.json` | The original neon corner-bracket theme. |
| `themes/minimal.json` | Plain rectangles, no effects. |

## Edit the design

Edit the active theme:

```txt
themes/trace.json
```

Or switch the cron/env to `themes/recon.json` / `themes/surveil.json` if you prefer that look.

Useful knobs:

Label format variables: `{name}`, `{track_label}`, `{track_id}`, `{class_name}`, `{class_id}`, `{confidence}`, `{confidence_percent}`. By default tracking is anonymous per-video only, e.g. `PERSON-001`; it does not identify real people or compare faces across videos.

- `box.style`: `corners` or `rect`
- `box.glow`: neon glow on/off
- `label.format`: e.g. `{name} {confidence_percent:.0f}%`
- `confidence_bar.enabled`
- `effects.vignette`, `effects.scanline`, `effects.contrast`, `effects.brightness`
- `effects.saturation`, `effects.tint`, `effects.tint_strength`: cinematic color grade (desaturate + color cast)
- `colors`: per-class/group colors
- `priority_classes`
- `max_boxes`

### CCTV-editorial mode (`hud` block)

Set `hud.enabled` to switch on the surveillance-editorial renderer (see `themes/trace.json`). Knobs:

- `hud.accent`: the single accent color that carries the whole HUD
- `hud.accent_classes`: classes drawn in the accent (others render muted grey)
- `hud.chrome`: whole-frame overlay — corner brackets, `top_left`/`bottom_left`/`bottom_right` status strings (support `{cam}`, `{timecode}`, `{count}`, `{frame}`), `rec` blinking dot, `margin`, `bracket`
- `hud.primary`: the single hero target's treatment — `inset` (zoom panel), `pixelate` / `pixelate_classes` (mosaic redaction, on for `person`), `card` (ID / class / confidence), `leader` (connector line), `inset_label`
- `label.font` (`plain`/`simplex`/`duplex`/`triplex`), `label.tracking`, `label.text_color`: technical tag typography
- `box.confidence_tick`: per-box confidence bar (off in editorial themes; confidence lives in the hero card)

Only the top-priority detection gets the zoom + card; the rest stay quiet. The zoom inset is mosaiced for people, so the overlay never up-resolves a recognizable face — it stays consistent with the anonymous tracking guarantee below.

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
YOLO_HUD_THEME=/home/kan/yolo-hud-overlay/themes/trace.json
```

The script stays compatible with the existing Tapo watcher arguments.

## Safety

Do not commit personal camera clips, `.env`, YOLO weights, or Tapo credentials.
