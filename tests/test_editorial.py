"""Tests for the CCTV/surveillance-editorial HUD path and grading.

These import cv2/numpy (project runtime deps) but not ultralytics/torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from yolo_hud_overlay import hud
from yolo_hud_overlay.render import render_frame, style_frame
from yolo_hud_overlay.tracker import Detection

FRAME = np.full((360, 640, 3), 120, np.uint8)
DETS = [
    Detection(0, 0.91, (60, 80, 160, 320), track_id=1),
    Detection(2, 0.80, (400, 200, 600, 300), track_id=1),
]

EDITORIAL = {
    "box": {"style": "rect", "thickness": 2},
    "label": {"format": "{track_label}", "font": "plain", "font_scale": 1.0, "text_color": "#0a0f0d"},
    "effects": {"saturation": 0.3, "tint": "#11212e", "tint_strength": 0.2},
    "colors": {"person": "#39ff14", "default": "#4f5f57"},
    "priority_classes": ["person"],
    "max_boxes": 8,
    "hud": {
        "enabled": True,
        "accent": "#39ff14",
        "chrome": {"enabled": True, "top_left": "CAM {cam} · {timecode}", "cam": "01"},
        "primary": {"enabled": True, "inset": True, "card": True, "leader": True},
    },
}


def test_timecode_formats_frames_and_seconds():
    assert hud.timecode(0, 30.0) == "00:00:00:00"
    assert hud.timecode(90, 30.0) == "00:00:03:00"
    assert hud.timecode(3630, 30.0) == "00:02:01:00"


def test_font_id_falls_back_to_plain():
    import cv2
    assert hud.font_id("nope") == cv2.FONT_HERSHEY_PLAIN
    assert hud.font_id("duplex") == cv2.FONT_HERSHEY_DUPLEX


def test_saturation_desaturates_colored_pixels():
    colored = np.zeros((4, 4, 3), np.uint8)
    colored[:] = (0, 0, 255)  # pure red (BGR)
    out = style_frame(colored, {"effects": {"saturation": 0.0}})
    # fully desaturated -> all channels roughly equal (gray)
    assert abs(int(out[0, 0, 0]) - int(out[0, 0, 2])) < 5


def test_editorial_render_returns_same_shape_without_error():
    out = render_frame(FRAME.copy(), DETS, EDITORIAL, model_names=None, frame_index=812, fps=30.0)
    assert out.shape == FRAME.shape
    assert out.dtype == np.uint8
    # the graded/annotated frame must differ from the flat input
    assert not np.array_equal(out, FRAME)


def test_legacy_theme_still_renders():
    legacy = {
        "box": {"style": "corners", "glow": True},
        "label": {"format": "{name} {confidence_percent:.0f}%"},
        "colors": {"person": "#ff5050", "default": "#40ff80"},
        "effects": {"vignette": 0.2, "scanline": True},
    }
    out = render_frame(FRAME.copy(), DETS, legacy, model_names=None)
    assert out.shape == FRAME.shape


def test_editorial_ignores_when_hud_disabled():
    theme = {**EDITORIAL, "hud": {"enabled": False}}
    out = render_frame(FRAME.copy(), DETS, theme, model_names=None)
    assert out.shape == FRAME.shape
