import torch

from scc.faces_localization import (
    SearchGeometry,
    box_iou,
    generate_candidate_views,
    object_box_from_view,
    select_evidence_candidate,
)


def _geometry() -> SearchGeometry:
    return SearchGeometry(
        view_height_fractions=(0.8, 1.0),
        view_aspect=0.75,
        object_width_in_view=0.5,
        object_height_in_view=0.75,
        prior_box_normalized=(0.3, 0.1, 0.7, 0.9),
    )


def test_candidate_views_are_unique_and_inside_image():
    boxes = generate_candidate_views(120, 80, _geometry(), x_positions=5)
    assert len(boxes) == len(set(boxes))
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 120
        assert 0 <= y1 < y2 <= 80


def test_object_box_is_centered_inside_view():
    box = object_box_from_view((20, 10, 100, 90), _geometry(), (120, 100))
    assert box == (40.0, 20.0, 80.0, 80.0)
    assert box_iou(box, box) == 1.0


def test_spike_margin_has_priority_over_synaptic_margin():
    boxes = [(0, 0, 20, 20), (40, 0, 60, 20)]
    selected = select_evidence_candidate(
        torch.tensor([2.0, 1.0]),
        torch.tensor([-10.0, 10.0]),
        boxes,
        (0.5, 0.5),
        (60, 20),
    )
    assert selected == 0


def test_synaptic_margin_breaks_decision_spike_tie():
    boxes = [(0, 0, 20, 20), (40, 0, 60, 20)]
    selected = select_evidence_candidate(
        torch.tensor([2.0, 2.0]),
        torch.tensor([-1.0, 1.0]),
        boxes,
        (0.5, 0.5),
        (60, 20),
    )
    assert selected == 1


def test_location_prior_breaks_complete_neural_tie():
    boxes = [(0, 0, 20, 20), (40, 0, 60, 20)]
    selected = select_evidence_candidate(
        torch.zeros(2),
        torch.zeros(2),
        boxes,
        (0.8, 0.5),
        (60, 20),
    )
    assert selected == 1
