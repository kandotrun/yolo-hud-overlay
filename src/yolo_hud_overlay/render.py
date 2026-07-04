from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import hud
from .theme import class_name, color_for_class, display_name, hex_to_bgr
from .tracker import Detection, track_label


def style_frame(frame: np.ndarray, theme: dict[str, Any]) -> np.ndarray:
    effects = theme.get("effects", {})
    contrast = float(effects.get("contrast", 1.0))
    brightness = float(effects.get("brightness", 0))
    out = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    saturation = float(effects.get("saturation", 1.0))
    if saturation != 1.0:
        gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, saturation, gray, 1.0 - saturation, 0)
    tint = effects.get("tint")
    if tint:
        strength = float(effects.get("tint_strength", 0.15))
        tcol = np.array(hex_to_bgr(tint), np.float32)
        out = np.clip(out.astype(np.float32) * (1 - strength) + tcol * strength, 0, 255).astype(np.uint8)
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


def _label_text(fmt: str, class_id: int, conf: float, model_names: dict[int, str] | None, track_id: int | None) -> str:
    try:
        return str(fmt).format(
            name=display_name(class_id, model_names),
            track_label=track_label(display_name(class_id, model_names), track_id),
            track_id=track_id or "",
            class_name=class_name(class_id, model_names),
            class_id=class_id,
            confidence=conf,
            confidence_percent=conf * 100,
        )
    except (KeyError, IndexError, ValueError):
        return str(fmt)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _sorted_items(boxes: Any, theme: dict[str, Any], model_names: dict[int, str] | None) -> list[tuple]:
    priority = {name.replace("_", " ") for name in theme.get("priority_classes", [])}
    items: list[tuple] = []
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
    return items


def draw_progress(img: np.ndarray, x: int, y: int, w: int, h: int, frac: float,
                  color: tuple[int, int, int], track_color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), track_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1, cv2.LINE_AA)


def _panel_geometry(item: tuple, w: int, hud_cfg: dict[str, Any]) -> dict[str, Any]:
    """Place the primary panel on the side opposite the hero subject."""
    _, _, _, _, _, (x1, y1, x2, y2), _ = item
    prim = hud_cfg.get("primary", {})
    pw = int(prim.get("inset_w", 210))
    ph = int(prim.get("inset_h", 210))
    margin = int(hud_cfg.get("chrome", {}).get("margin", 26))
    card_h = 122
    if (x1 + x2) / 2 > w / 2:
        ax1 = margin + 22
        side, attach, box_pt = "left", "right", (x1, y1)
    else:
        ax1 = w - margin - 22 - pw
        side, attach, box_pt = "right", "left", (x2, y1)
    ay1 = margin + 96
    inset = (ax1, ay1, ax1 + pw, ay1 + ph)
    region = (ax1, ay1, ax1 + pw, ay1 + ph + 10 + card_h)
    return {"inset": inset, "attach": attach, "side": side, "box_pt": box_pt, "region": region, "pw": pw}


def _draw_primary(out: np.ndarray, source: np.ndarray, item: tuple, theme: dict[str, Any],
                  model_names: dict[int, str] | None, accent: tuple[int, int, int], font: int,
                  geom: dict[str, Any]) -> None:
    """Redacted zoom inset + data card + leader line for the single lead target."""
    _, _, _, class_id, conf, (x1, y1, x2, y2), track_id = item
    hud_cfg = theme["hud"]
    prim = hud_cfg.get("primary", {})
    label_cfg = theme.get("label", {})
    tracking = float(label_cfg.get("tracking", 1.0))
    text_color = hex_to_bgr(hud_cfg.get("chrome", {}).get("color", "#e8f0ee"))
    dark = hex_to_bgr(label_cfg.get("text_color", "#0a0f0d"))
    fs0 = float(label_cfg.get("font_scale", 1.1))
    lthick = int(label_cfg.get("thickness", 1))

    ax1, ay1, ax2, ay2 = geom["inset"]
    pw = geom["pw"]

    if prim.get("inset", True):
        name = class_name(class_id, model_names)
        pixelate = int(prim.get("pixelate", 14)) if name in set(prim.get("pixelate_classes", ["person"])) else 0
        attach = hud.draw_inset(out, source, (x1, y1, x2, y2), (ax1, ay1, ax2, ay2), accent,
                                thickness=2, pixelate=pixelate, attach=geom["attach"])
        if prim.get("leader", True):
            hud.draw_leader(out, geom["box_pt"], attach, accent, thickness=1, node=True)
        hud.draw_tag(out, (ax1, ay1 - 6),
                     _label_text(prim.get("inset_label", "OBJ // {class_name}"), class_id, conf, model_names, track_id).upper(),
                     accent, dark, font, fs0, lthick, tracking, anchor="bl")

    if prim.get("card", True):
        cx1, cy1 = ax1, ay2 + 10
        cw = pw
        rows = [
            ("ID", track_label(display_name(class_id, model_names), track_id)),
            ("CLASS", display_name(class_id, model_names)),
            ("CONF", f"{conf * 100:.1f}%"),
        ]
        row_h = 26
        ch = row_h * len(rows) + 34
        overlay = out.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + cw, cy1 + ch), (10, 13, 12), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
        cv2.rectangle(out, (cx1, cy1), (cx1 + cw, cy1 + ch), accent, 1, cv2.LINE_AA)
        cv2.rectangle(out, (cx1, cy1), (cx1 + cw, cy1 + 6), accent, -1, cv2.LINE_AA)
        fs = float(label_cfg.get("font_scale", 1.1)) * 0.85
        for i, (k, v) in enumerate(rows):
            ry = cy1 + 30 + i * row_h
            hud.put_text_tracked(out, k, (cx1 + 12, ry), font, fs, text_color, 1, tracking)
            vw, _ = hud.text_size_tracked(v, font, fs, 1, tracking)
            hud.put_text_tracked(out, v, (cx1 + cw - 12 - vw, ry), font, fs, accent, 1, tracking)
        draw_progress(out, cx1 + 12, cy1 + ch - 12, cw - 24, 5, conf, accent, (40, 46, 44))


def render_editorial(out: np.ndarray, source: np.ndarray, items: list[tuple], theme: dict[str, Any],
                     model_names: dict[int, str] | None, frame_index: int, fps: float) -> np.ndarray:
    h, w = out.shape[:2]
    hud_cfg = theme["hud"]
    accent = hex_to_bgr(hud_cfg.get("accent", "#39ff14"))
    dim_accent = tuple(int(c * 0.5) for c in accent)
    accent_classes = set(hud_cfg.get("accent_classes", ["person", "dog", "cat"]))
    box_cfg = theme.get("box", {})
    label_cfg = theme.get("label", {})
    font = hud.font_id(label_cfg.get("font", "plain"))
    tracking = float(label_cfg.get("tracking", 1.0))
    fs = float(label_cfg.get("font_scale", 1.1))
    lthick = int(label_cfg.get("thickness", 1))
    dark = hex_to_bgr(label_cfg.get("text_color", "#0a0f0d"))
    style = box_cfg.get("style", "rect")
    thick = int(box_cfg.get("thickness", 2))
    max_boxes = int(theme.get("max_boxes", 12))
    shown = items[:max_boxes]

    chrome = hud_cfg.get("chrome", {})
    if chrome.get("enabled", True):
        ctx = {
            "frame": frame_index, "count": len(items),
            "timecode": hud.timecode(frame_index, fps),
            "cam": chrome.get("cam", "01"), "blink": chrome.get("blink", 12),
        }
        hud.draw_frame_chrome(out, chrome, font, tracking, ctx)

    primary_on = bool(shown) and hud_cfg.get("primary", {}).get("enabled", True)
    geom = _panel_geometry(shown[0], w, hud_cfg) if primary_on else None

    for rank, item in enumerate(shown):
        _, _, _, class_id, conf, (x1, y1, x2, y2), track_id = item
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        is_primary = rank == 0
        name = class_name(class_id, model_names)
        # secondary boxes occluded by the primary panel are dropped (clean hierarchy)
        if not is_primary and geom and _boxes_intersect((x1, y1 - 30, x2, y2), geom["region"]):
            continue
        if is_primary:
            color = accent
        elif name in accent_classes:
            color = dim_accent
        else:
            color = color_for_class(class_id, theme, model_names)
        bt = thick if is_primary else max(1, thick - 1)
        if style == "corners":
            draw_corner_box(out, x1, y1, x2, y2, color, {**box_cfg, "glow": False, "thickness": bt})
        else:
            draw_rect_box(out, x1, y1, x2, y2, color, {**box_cfg, "thickness": bt})
        if is_primary:
            hud.draw_corner_ticks(out, x1, y1, x2, y2, accent, max(10, (x2 - x1) // 5), thick + 1)
        if label_cfg.get("enabled", True):
            text = _label_text(label_cfg.get("format", "{track_label}"), class_id, conf, model_names, track_id).upper()
            hud.draw_tag(out, (x1, y1 - 3), text, color, dark, font, fs, lthick, tracking,
                         anchor="bl", style=("fill" if is_primary else "outline"))
        if is_primary and box_cfg.get("confidence_tick", False):
            draw_progress(out, x1, min(h - 6, y2 + 5), max(24, x2 - x1), 4, conf, color, (30, 34, 33))

    if geom:
        _draw_primary(out, source, shown[0], theme, model_names, accent, font, geom)
    return out


def render_frame(frame: np.ndarray, boxes: Any, theme: dict[str, Any], model_names: dict[int, str] | None = None,
                 frame_index: int = 0, fps: float = 30.0) -> np.ndarray:
    out = style_frame(frame, theme)
    items = _sorted_items(boxes, theme, model_names)
    if theme.get("hud", {}).get("enabled"):
        return render_editorial(out, frame, items, theme, model_names, frame_index, fps)
    h, w = out.shape[:2]
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
