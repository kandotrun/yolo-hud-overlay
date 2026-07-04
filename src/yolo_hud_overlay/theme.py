from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

COCO_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
    9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 16: "dog",
    17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}

DEFAULT_THEME_PATH = Path(__file__).resolve().parents[2] / "themes" / "neon.json"


def resolve_theme_path(theme_arg: str | None) -> Path:
    if theme_arg:
        return Path(theme_arg).expanduser()
    env_theme = os.environ.get("YOLO_HUD_THEME")
    if env_theme:
        return Path(env_theme).expanduser()
    return DEFAULT_THEME_PATH


def load_theme(theme_arg: str | None = None) -> dict[str, Any]:
    path = resolve_theme_path(theme_arg)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    return (b, g, r)


def class_name(class_id: int, model_names: dict[int, str] | None = None) -> str:
    if model_names and class_id in model_names:
        return str(model_names[class_id])
    return COCO_NAMES.get(class_id, f"class-{class_id}")


def display_name(class_id: int, model_names: dict[int, str] | None = None) -> str:
    return class_name(class_id, model_names).upper()


def color_for_class(class_id: int, theme: dict[str, Any], model_names: dict[int, str] | None = None) -> tuple[int, int, int]:
    name = class_name(class_id, model_names)
    colors = theme.get("colors", {})
    key = name.replace(" ", "_")
    if name in colors:
        return hex_to_bgr(colors[name])
    if key in colors:
        return hex_to_bgr(colors[key])
    if name in VEHICLE_CLASSES and "vehicle" in colors:
        return hex_to_bgr(colors["vehicle"])
    return hex_to_bgr(colors.get("default", "#40ff80"))


def parse_classes(value: str) -> list[int]:
    return [int(part) for part in value.replace(" ", ",").split(",") if part]
