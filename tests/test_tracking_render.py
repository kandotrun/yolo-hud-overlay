from yolo_hud_overlay.tracker import Detection, track_label_for_detection


def test_track_label_for_detection_uses_model_name():
    detection = Detection(0, 0.9, (0, 0, 10, 10), track_id=3)
    assert track_label_for_detection(detection, {0: "person"}) == "PERSON-003"
