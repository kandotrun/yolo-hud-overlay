import json
from pathlib import Path

import pytest

from yolo_hud_overlay.theme import color_for_class, hex_to_bgr, load_theme, parse_classes, resolve_theme_path


def test_hex_to_bgr_converts_css_hex_to_opencv_bgr():
    assert hex_to_bgr("#ff8000") == (0, 128, 255)


def test_hex_to_bgr_rejects_invalid_color():
    with pytest.raises(ValueError):
        hex_to_bgr("#fff")


def test_theme_env_resolution(monkeypatch, tmp_path):
    theme = tmp_path / "theme.json"
    theme.write_text(json.dumps({"colors": {"default": "#123456"}}))
    monkeypatch.setenv("YOLO_HUD_THEME", str(theme))
    assert resolve_theme_path(None) == theme
    assert load_theme()["colors"]["default"] == "#123456"


def test_theme_arg_beats_env(monkeypatch, tmp_path):
    env_theme = tmp_path / "env.json"
    arg_theme = tmp_path / "arg.json"
    env_theme.write_text("{}")
    arg_theme.write_text(json.dumps({"max_boxes": 7}))
    monkeypatch.setenv("YOLO_HUD_THEME", str(env_theme))
    assert load_theme(str(arg_theme))["max_boxes"] == 7


def test_parse_classes_accepts_commas_and_spaces():
    assert parse_classes("0, 1 2,3") == [0, 1, 2, 3]


def test_color_for_vehicle_uses_vehicle_group():
    theme = {"colors": {"vehicle": "#00ccff", "default": "#40ff80"}}
    assert color_for_class(2, theme) == (255, 204, 0)
