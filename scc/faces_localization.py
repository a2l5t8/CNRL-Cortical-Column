"""Geometry and evaluation helpers for Caltech Faces saccadic search.

The learned cortical column scores candidate foveal views. This module keeps
the view geometry, box conversion, and evidence selection separate from the
CoNeX/PyMoNNtorch runtime so that the localization protocol is easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SearchGeometry:
    """Training-side geometry used to define the zero-shot search."""

    view_height_fractions: tuple[float, ...]
    view_aspect: float
    object_width_in_view: float
    object_height_in_view: float
    prior_box_normalized: Box


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return intersection over union for two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def generate_candidate_views(
    width: int,
    height: int,
    geometry: SearchGeometry,
    *,
    x_positions: int = 13,
    y_positions: int = 3,
) -> list[tuple[int, int, int, int]]:
    """Generate a deterministic multi-scale grid of valid foveal views."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if x_positions <= 0 or y_positions <= 0:
        raise ValueError("grid dimensions must be positive")

    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for fraction in geometry.view_height_fractions:
        view_height = min(height, max(1, int(round(height * fraction))))
        view_width = min(
            width,
            max(1, int(round(view_height * geometry.view_aspect))),
        )
        x_values = np.linspace(0, width - view_width, x_positions)
        y_values = np.linspace(0, height - view_height, y_positions)
        for y_value in y_values:
            for x_value in x_values:
                left = int(round(float(x_value)))
                top = int(round(float(y_value)))
                box = (
                    left,
                    top,
                    left + view_width,
                    top + view_height,
                )
                if box not in seen:
                    seen.add(box)
                    boxes.append(box)
    return boxes


def object_box_from_view(
    view: Sequence[float],
    geometry: SearchGeometry,
    image_size: tuple[int, int],
) -> Box:
    """Map a selected Face_easy-like view to a predicted object box."""
    x1, y1, x2, y2 = (float(value) for value in view)
    image_width, image_height = image_size
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    object_width = (x2 - x1) * geometry.object_width_in_view
    object_height = (y2 - y1) * geometry.object_height_in_view
    return (
        max(0.0, center_x - 0.5 * object_width),
        max(0.0, center_y - 0.5 * object_height),
        min(float(image_width), center_x + 0.5 * object_width),
        min(float(image_height), center_y + 0.5 * object_height),
    )


def prior_object_box(
    geometry: SearchGeometry,
    image_size: tuple[int, int],
) -> Box:
    """Scale the training-side median location prior to an image."""
    width, height = image_size
    x1, y1, x2, y2 = geometry.prior_box_normalized
    return (
        float(x1 * width),
        float(y1 * height),
        float(x2 * width),
        float(y2 * height),
    )


def normalized_center_distance(
    first: Sequence[float],
    second: Sequence[float],
    image_size: tuple[int, int],
) -> float:
    """Return box-center distance normalized independently by image axes."""
    width, height = image_size
    first_x = 0.5 * (float(first[0]) + float(first[2]))
    first_y = 0.5 * (float(first[1]) + float(first[3]))
    second_x = 0.5 * (float(second[0]) + float(second[2]))
    second_y = 0.5 * (float(second[1]) + float(second[3]))
    return float(
        np.hypot(
            (first_x - second_x) / max(width, 1),
            (first_y - second_y) / max(height, 1),
        )
    )


def center_inside(
    prediction: Sequence[float],
    target: Sequence[float],
) -> bool:
    """Return whether the predicted center falls inside the target box."""
    center_x = 0.5 * (float(prediction[0]) + float(prediction[2]))
    center_y = 0.5 * (float(prediction[1]) + float(prediction[3]))
    return bool(
        float(target[0]) <= center_x <= float(target[2])
        and float(target[1]) <= center_y <= float(target[3])
    )


def select_evidence_candidate(
    decision_margin: torch.Tensor,
    synaptic_margin: torch.Tensor,
    candidate_views: Sequence[Sequence[float]],
    prior_center_normalized: tuple[float, float],
    image_size: tuple[int, int],
) -> int:
    """Select a view by spike margin, then synaptic evidence, then prior.

    Decision spike-count margin is the primary signal. Integrated
    L5/6-gated synaptic evidence resolves exact spike-count ties. The
    training-side location prior is used only when both neural signals tie,
    which also defines a non-arbitrary fallback for the silent-L5/6 control.
    """
    decision_margin = torch.as_tensor(decision_margin).flatten().float().cpu()
    synaptic_margin = torch.as_tensor(synaptic_margin).flatten().float().cpu()
    if len(decision_margin) != len(candidate_views):
        raise ValueError("one decision margin is required per candidate")
    if len(synaptic_margin) != len(candidate_views):
        raise ValueError("one synaptic margin is required per candidate")
    if len(candidate_views) == 0:
        raise ValueError("at least one candidate view is required")

    primary = torch.where(decision_margin == decision_margin.max())[0]
    secondary_values = synaptic_margin[primary]
    secondary = primary[
        torch.where(secondary_values == secondary_values.max())[0]
    ]
    if len(secondary) == 1:
        return int(secondary.item())

    width, height = image_size
    prior_x = float(prior_center_normalized[0]) * width
    prior_y = float(prior_center_normalized[1]) * height
    distances = []
    for index in secondary.tolist():
        x1, y1, x2, y2 = candidate_views[index]
        distances.append(
            ((0.5 * (x1 + x2) - prior_x) / max(width, 1)) ** 2
            + ((0.5 * (y1 + y2) - prior_y) / max(height, 1)) ** 2
        )
    return int(secondary[int(np.argmin(distances))])


def candidate_object_boxes(
    candidate_views: Iterable[Sequence[float]],
    geometry: SearchGeometry,
    image_size: tuple[int, int],
) -> list[Box]:
    """Convert all candidate views to their corresponding object boxes."""
    return [
        object_box_from_view(view, geometry, image_size)
        for view in candidate_views
    ]
