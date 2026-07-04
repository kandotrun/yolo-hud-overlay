from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .theme import class_name, color_for_class, display_name
from .tracker import Detection, track_label


def style_frame(frame: np.ndarray, theme: dict[str, Any]) -> np.ndarray:
    effects = theme.get("effects", {})
    contrast = float(effects.get("contrast", 1.0))
    brightness = float(effects.get("brightness", 0))
    out = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    vignette = float(effects.get("vignette", 0.0))
    if vignette > 0:
        h, w = out.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        mask = np.clip(1 - vignette * dist, 1 - vignette, 1.0).astype(np.float32)
        out = (out.astype(np.float32) * mask[..., None]).astype(np.uint8)
    if effects.get("scanline", False):
        out[::4] = (out[::4].astype(np.float32) * 0.93).astype(np.uint8)
    return out


def draw_corner_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], box: dict[str, Any]) -> None:
    thickness = int(box.get("thickness", 2))
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    ratio = float(box.get("corner_ratio", 0.22))
    length = max(int(box.get("corner_min", 12)), min(int(box.get("corner_max", 34)), int(min(w, h) * ratio)))
    segments = [
        ((x1, y1), (x1 + length, y1)), ((x1, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1)), ((x2, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2)), ((x1, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2)), ((x2, y2), (x2, y2 - length)),
    ]
    if box.get("glow", True):
        glow_alpha = float(box.get("glow_alpha", 0.18))
        glow = img.copy()
        for a, b in segments:
            cv2.line(glow, a, b, color, thickness + 5, cv2.LINE_AA)
        cv2.addWeighted(glow, glow_alpha, img, 1 - glow_alpha, 0, img)
    for a, b in segments:
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_rect_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], box: dict[str, Any]) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, int(box.get("thickness", 2)), cv2.LINE_AA)


def draw_label(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, class_id: int, conf: float, color: tuple[int, int, int], theme: dict[str, Any], model_names: dict[int, str] | None, track_id: int | None = None) -> None:
    label_cfg = theme.get("label", {})
    if not label_cfg.get("enabled", True):
        return
    text = str(label_cfg.get("format", "{name} {confidence_percent:.0f}%")).format(
        name=display_name(class_id, model_names),
        track_label=track_label(display_name(class_id, model_names), track_id),
        track_id=track_id or "",
        class_name=class_name(class_id, model_names),
        class_id=class_id,
        confidence=conf,
        confidence_percent=conf * 100,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(label_cfg.get("font_scale", 0.55))
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    pad_x, pad_y = 9, 6
    lx = max(3, min(x1, img.shape[1] - tw - pad_x * 2 - 3))
    ly = y1 - th - base - pad_y * 2 - 6
    if ly < 3:
        ly = min(img.shape[0] - th - base - pad_y * 2 - 3, y1 + 6)
    overlay = img.copy()
    cv2.rectangle(overlay, (lx, ly), (lx + tw + pad_x * 2, ly + th + base + pad_y * 2), (8, 10, 12), -1)
    alpha = float(label_cfg.get("background_alpha", 0.62))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (lx, ly), (lx + tw + pad_x * 2, ly + th + base + pad_y * 2), color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (lx + pad_x, ly + pad_y + th), font, scale, (245, 250, 255), 1, cv2.LINE_AA)


def draw_confidence_bar(img: np.ndarray, x1: int, y2: int, x2: int, conf: float, color: tuple[int, int, int], theme: dict[str, Any]) -> None:
    cfg = theme.get("confidence_bar", {})
    if not cfg.get("enabled", True):
        return
    height = int(cfg.get("height", 3))
    y = min(img.shape[0] - height - 1, y2 + 5)
    cv2.line(img, (x1, y), (x2, y), (25, 25, 25), height, cv2.LINE_AA)
    cv2.line(img, (x1, y), (int(x1 + (x2 - x1) * conf), y), color, height, cv2.LINE_AA)


def render_frame(frame: np.ndarray, boxes: Any, theme: dict[str, Any], model_names: dict[int, str] | None = None) -> np.ndarray:
    out = style_frame(frame, theme)
    h, w = out.shape[:2]
    priority = {name.replace("_", " ") for name in theme.get("priority_classes", [])}
    items = []
    for box_item in boxes:
        if isinstance(box_item, Detection):
            class_id = box_item.class_id
            conf = box_item.confidence
            xyxy = list(box_item.xyxy)
            track_id = box_item.track_id
        else:
            class_id = int(box_item.cls[0])
            conf = float(box_item.conf[0])
            xyxy = list(map(int, box_item.xyxy[0].cpu().numpy().tolist()))
            track_id = None
        area = max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])
        pri = 0 if class_name(class_id, model_names) in priority else 1
        items.append((pri, -conf, -area, class_id, conf, xyxy, track_id))
    items.sort()
    max_boxes = int(theme.get("max_boxes", 28))
    box_cfg = theme.get("box", {})
    style = box_cfg.get("style", "corners")
    for _, _, _, class_id, conf, (x1, y1, x2, y2), track_id in items[:max_boxes]:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_for_class(class_id, theme, model_names)
        if style == "rect":
            draw_rect_box(out, x1, y1, x2, y2, color, box_cfg)
        else:
            draw_corner_box(out, x1, y1, x2, y2, color, box_cfg)
        draw_label(out, x1, y1, x2, y2, class_id, conf, color, theme, model_names, track_id)
        draw_confidence_bar(out, x1, y2, x2, conf, color, theme)
    return out
