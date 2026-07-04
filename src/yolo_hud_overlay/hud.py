"""Drawing primitives for the CCTV / surveillance-editorial HUD look.

Kept separate from `render.py` so the editorial vocabulary (tags, leader lines,
zoom insets, frame chrome, technical type) stays isolated from the core box
renderer. Everything here is opt-in via theme keys; themes without a `hud`
block never reach this module.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

FONTS: dict[str, int] = {
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "mono": cv2.FONT_HERSHEY_PLAIN,
}


def font_id(name: str) -> int:
    return FONTS.get(str(name).lower(), cv2.FONT_HERSHEY_PLAIN)


def text_size_tracked(text: str, font: int, scale: float, thickness: int, tracking: float) -> tuple[int, int]:
    """Width/height of text drawn char-by-char with extra letter spacing."""
    if not text:
        return (0, 0)
    width = 0.0
    height = 0
    for ch in text:
        (cw, chh), base = cv2.getTextSize(ch if ch != " " else "M", font, scale, thickness)
        # keep spaces narrow-ish but real
        width += (cw if ch != " " else int(cw * 0.6)) + tracking
        height = max(height, chh + base)
    width -= tracking
    return (int(round(width)), height)


def put_text_tracked(img: np.ndarray, text: str, org: tuple[int, int], font: int, scale: float,
                     color: tuple[int, int, int], thickness: int, tracking: float = 0.0) -> int:
    """Draw uppercase-friendly technical text with manual letter spacing.

    Returns the x coordinate just past the last glyph.
    """
    x, y = float(org[0]), org[1]
    for ch in text:
        if ch == " ":
            (cw, _), _ = cv2.getTextSize("M", font, scale, thickness)
            x += int(cw * 0.6) + tracking
            continue
        cv2.putText(img, ch, (int(round(x)), y), font, scale, color, thickness, cv2.LINE_AA)
        (cw, _), _ = cv2.getTextSize(ch, font, scale, thickness)
        x += cw + tracking
    return int(round(x))


def draw_tag(img: np.ndarray, org: tuple[int, int], text: str, fill: tuple[int, int, int],
             text_color: tuple[int, int, int], font: int, scale: float, thickness: int,
             tracking: float, pad: tuple[int, int] = (7, 5), anchor: str = "tl",
             style: str = "fill") -> tuple[int, int, int, int]:
    """Technical label tag. Returns the tag rect (x1,y1,x2,y2).

    anchor: which corner `org` refers to — tl, bl, tr, br.
    style: "fill" = solid accent plate + dark text (hero); "outline" = dark
    backing + accent border + accent text (quiet secondary).
    """
    h, w = img.shape[:2]
    tw, th = text_size_tracked(text, font, scale, thickness, tracking)
    bw = tw + pad[0] * 2
    bh = th + pad[1] * 2
    x, y = org
    if "r" in anchor:
        x -= bw
    if "b" in anchor:
        y -= bh
    # keep the tag inside the frame so edge subjects never lose their label
    x = max(0, min(x, w - bw))
    y = max(0, min(y, h - bh))
    if style == "outline":
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (12, 15, 14), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), fill, 1, cv2.LINE_AA)
        put_text_tracked(img, text, (x + pad[0], y + bh - pad[1]), font, scale, fill, thickness, tracking)
    else:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), fill, -1, cv2.LINE_AA)
        put_text_tracked(img, text, (x + pad[0], y + bh - pad[1]), font, scale, text_color, thickness, tracking)
    return (x, y, x + bw, y + bh)


def draw_leader(img: np.ndarray, p_from: tuple[int, int], p_to: tuple[int, int],
                color: tuple[int, int, int], thickness: int = 1, node: bool = True) -> None:
    """Connector line from a detection anchor to a callout, with a small node dot."""
    cv2.line(img, p_from, p_to, color, thickness, cv2.LINE_AA)
    if node:
        cv2.circle(img, p_from, max(2, thickness + 1), color, -1, cv2.LINE_AA)


def draw_corner_ticks(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                      color: tuple[int, int, int], length: int, thickness: int) -> None:
    """Thin L brackets at the 4 corners of a rect (tracking-lock look)."""
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)


def draw_inset(img: np.ndarray, source: np.ndarray, bbox: tuple[int, int, int, int],
               anchor_rect: tuple[int, int, int, int], accent: tuple[int, int, int],
               thickness: int = 2, pixelate: int = 0, attach: str = "left") -> tuple[int, int]:
    """Crop `bbox` from `source`, scale into `anchor_rect`, paste with an accent bracket.

    `pixelate` (>0) mosaics the crop to that many blocks across — a redacted,
    de-identified zoom that reads as surveillance chrome while never up-ressing
    a recognizable subject. Returns the leader attach point on side `attach`.
    """
    ax1, ay1, ax2, ay2 = anchor_rect
    aw, ah = ax2 - ax1, ay2 - ay1
    bx1, by1, bx2, by2 = bbox
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2 = min(source.shape[1], max(bx1 + 1, bx2))
    by2 = min(source.shape[0], max(by1 + 1, by2))
    crop = source[by1:by2, bx1:bx2]
    mid_y = (ay1 + ay2) // 2
    attach_pt = (ax1, mid_y) if attach == "left" else (ax2, mid_y)
    if crop.size == 0:
        return attach_pt
    ch, cw = crop.shape[:2]
    scale = max(aw / cw, ah / ch)
    rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
    ox, oy = (rw - aw) // 2, (rh - ah) // 2
    panel = resized[oy:oy + ah, ox:ox + aw]
    if panel.shape[:2] != (ah, aw):
        panel = cv2.resize(panel, (aw, ah), interpolation=cv2.INTER_LINEAR)
    if pixelate > 0:
        blocks = max(3, int(pixelate))
        small = cv2.resize(panel, (blocks, max(3, blocks * ah // max(1, aw))), interpolation=cv2.INTER_LINEAR)
        panel = cv2.resize(small, (aw, ah), interpolation=cv2.INTER_NEAREST)
    img[ay1:ay2, ax1:ax2] = panel
    cv2.rectangle(img, (ax1, ay1), (ax2, ay2), accent, thickness, cv2.LINE_AA)
    draw_corner_ticks(img, ax1, ay1, ax2, ay2, accent, max(8, aw // 6), thickness)
    return attach_pt


def timecode(frame_index: int, fps: float) -> str:
    fps = fps or 30.0
    total = frame_index / fps
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)
    ff = int(frame_index % max(1, round(fps)))
    return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"


def draw_frame_chrome(img: np.ndarray, chrome: dict[str, Any], font: int, tracking: float,
                      context: dict[str, Any]) -> None:
    """Whole-frame HUD: corner brackets, top/bottom status strings, REC dot."""
    h, w = img.shape[:2]
    from .theme import hex_to_bgr  # local import to avoid cycle

    color = hex_to_bgr(chrome.get("color", "#e8f0ee"))
    margin = int(chrome.get("margin", 26))
    blen = int(chrome.get("bracket", 46))
    th = int(chrome.get("thickness", 2))
    scale = float(chrome.get("font_scale", 1.1))
    ftk = float(chrome.get("tracking", tracking))

    m, b = margin, blen
    for (cx, cy, dx, dy) in (
        (m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1),
    ):
        cv2.line(img, (cx, cy), (cx + dx * b, cy), color, th, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * b), color, th, cv2.LINE_AA)

    def fmt(s: str) -> str:
        try:
            return str(s).format(**context)
        except (KeyError, IndexError, ValueError):
            return str(s)

    pad = margin + 14
    tly = margin + 26
    if chrome.get("rec", True):
        # blinking REC dot + label, top-left
        blink = (context.get("frame", 0) // max(1, int(context.get("blink", 12)))) % 2 == 0
        dot = hex_to_bgr(chrome.get("rec_color", "#ff3b30"))
        if blink:
            cv2.circle(img, (pad + 6, tly - 5), 6, dot, -1, cv2.LINE_AA)
        put_text_tracked(img, "REC", (pad + 20, tly), font, scale, color, th, ftk)
    tl = chrome.get("top_left")
    if tl:
        yy = tly + (26 if chrome.get("rec", True) else 0)
        put_text_tracked(img, fmt(tl), (pad, yy), font, scale, color, th, ftk)
    tr = chrome.get("top_right")
    if tr:
        s = fmt(tr)
        tw, _ = text_size_tracked(s, font, scale, th, ftk)
        put_text_tracked(img, s, (w - pad - tw, tly), font, scale, color, th, ftk)
    bl = chrome.get("bottom_left")
    if bl:
        put_text_tracked(img, fmt(bl), (pad, h - margin - 14), font, scale, color, th, ftk)
    br = chrome.get("bottom_right")
    if br:
        s = fmt(br)
        tw, _ = text_size_tracked(s, font, scale, th, ftk)
        put_text_tracked(img, s, (w - pad - tw, h - margin - 14), font, scale, color, th, ftk)
