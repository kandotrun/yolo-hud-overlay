from __future__ import annotations

from dataclasses import dataclass, replace

from .theme import display_name


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None


@dataclass
class _Track:
    class_id: int
    xyxy: tuple[int, int, int, int]
    track_id: int
    age: int = 0


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


class AnonymousTracker:
    """Simple per-video, per-class IoU tracker.

    This intentionally does not do face recognition, cross-video re-ID, or names.
    IDs are anonymous and reset when a new tracker instance is created.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: list[_Track] = []
        self._next_by_class: dict[int, int] = {}

    def _next_id(self, class_id: int) -> int:
        value = self._next_by_class.get(class_id, 1)
        self._next_by_class[class_id] = value + 1
        return value

    def update(self, detections: list[Detection]) -> list[Detection]:
        for track in self._tracks:
            track.age += 1

        result: list[Detection] = []
        used_tracks: set[int] = set()
        for detection in detections:
            best_index = None
            best_iou = 0.0
            for index, track in enumerate(self._tracks):
                if index in used_tracks or track.class_id != detection.class_id:
                    continue
                score = iou(track.xyxy, detection.xyxy)
                if score > best_iou:
                    best_iou = score
                    best_index = index
            if best_index is not None and best_iou >= self.iou_threshold:
                track = self._tracks[best_index]
                track.xyxy = detection.xyxy
                track.age = 0
                used_tracks.add(best_index)
                result.append(replace(detection, track_id=track.track_id))
            else:
                track_id = self._next_id(detection.class_id)
                self._tracks.append(_Track(detection.class_id, detection.xyxy, track_id, age=0))
                result.append(replace(detection, track_id=track_id))

        self._tracks = [track for track in self._tracks if track.age <= self.max_age]
        return result


def track_label(name: str, track_id: int | None) -> str:
    return f"{name}-{track_id:03d}" if track_id is not None else name


def track_label_for_detection(detection: Detection, model_names: dict[int, str] | None = None) -> str:
    return track_label(display_name(detection.class_id, model_names), detection.track_id)


def detections_from_yolo(boxes) -> list[Detection]:
    if boxes is None:
        return []
    detections: list[Detection] = []
    for box in boxes:
        detections.append(
            Detection(
                class_id=int(box.cls[0]),
                confidence=float(box.conf[0]),
                xyxy=tuple(map(int, box.xyxy[0].cpu().numpy().tolist())),
            )
        )
    return detections
