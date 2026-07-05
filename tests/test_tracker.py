from yolo_hud_overlay.tracker import AnonymousTracker, Detection, track_label


def test_tracker_reuses_id_for_overlapping_same_class():
    tracker = AnonymousTracker(iou_threshold=0.2, max_age=2)
    first = tracker.update([Detection(0, 0.9, (10, 10, 50, 50))])
    second = tracker.update([Detection(0, 0.8, (12, 12, 52, 52))])
    assert first[0].track_id == 1
    assert second[0].track_id == 1


def test_tracker_assigns_new_id_for_far_box():
    tracker = AnonymousTracker(iou_threshold=0.2, max_age=2)
    tracker.update([Detection(0, 0.9, (10, 10, 50, 50))])
    second = tracker.update([Detection(0, 0.8, (200, 200, 260, 260))])
    assert second[0].track_id == 2


def test_tracker_numbers_are_per_class():
    tracker = AnonymousTracker(iou_threshold=0.2, max_age=2)
    tracked = tracker.update([
        Detection(0, 0.9, (10, 10, 50, 50)),
        Detection(16, 0.8, (100, 100, 140, 140)),
    ])
    assert [d.track_id for d in tracked] == [1, 1]


def test_tracker_drops_stale_tracks_after_max_age():
    tracker = AnonymousTracker(iou_threshold=0.2, max_age=1)
    tracker.update([Detection(0, 0.9, (10, 10, 50, 50))])
    tracker.update([])
    tracker.update([])
    tracked = tracker.update([Detection(0, 0.9, (12, 12, 52, 52))])
    assert tracked[0].track_id == 2


def test_track_label_is_anonymous_class_id():
    assert track_label("PERSON", 7) == "PERSON-007"
    assert track_label("DOG", None) == "DOG"


def test_tracker_scores_motion_for_reused_tracks():
    tracker = AnonymousTracker(iou_threshold=0.2, max_age=2)
    tracker.update([
        Detection(2, 0.95, (100, 100, 300, 220)),
        Detection(0, 0.80, (20, 80, 80, 220)),
    ])

    tracked = tracker.update([
        Detection(2, 0.95, (100, 100, 300, 220)),  # parked/static car
        Detection(0, 0.80, (45, 80, 105, 220)),    # moving person
    ])

    by_class = {d.class_id: d for d in tracked}
    assert by_class[2].motion_score == 0.0
    assert by_class[0].motion_score > by_class[2].motion_score
