"""Zero-shot saccadic localization in the uncropped Caltech ``Faces`` set.

The production Face_easy-versus-Motorbikes fully spiking checkpoint is kept
immutable. Search geometry is calibrated only from source indices 1--200, the
same source pool available when the checkpoint was built. Localization is then
evaluated on the disjoint Faces source indices 201--435 using official Caltech
bounding boxes. Every candidate view is processed by the CoNeX/PyMoNNtorch
cortical column; no conventional detector or derivative-trained head is added.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat
from scipy.stats import beta, binomtest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scc.config import Config
from scc.data.pipeline import build_transform
from scc.faces_localization import (
    SearchGeometry,
    box_iou,
    candidate_object_boxes,
    center_inside,
    generate_candidate_views,
    normalized_center_distance,
    object_box_from_view,
    prior_object_box,
    select_evidence_candidate,
)
from scc.fully_spiking import load_fully_spiking_column
from scc.utils import seed_everything


COLORS = {
    "navy": "#164863",
    "blue": "#2878B5",
    "cyan": "#2A9D8F",
    "gold": "#E9A23B",
    "red": "#C94845",
    "purple": "#745296",
    "gray": "#7A7A7A",
    "light": "#E8EEF2",
    "black": "#222222",
}
CALTECH_URL = (
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
)
CALTECH_MD5 = "3138e1922a9193bfa496528edbbc45d0"
METHOD_LABELS = {
    "normal": "Ordered L5/6",
    "reversed": "Reversed L5/6",
    "silent": "Silent L5/6",
    "prior": "Location prior",
    "oracle": "Candidate oracle",
}
METHOD_COLORS = {
    "normal": COLORS["cyan"],
    "reversed": COLORS["purple"],
    "silent": COLORS["red"],
    "prior": COLORS["gray"],
    "oracle": COLORS["gold"],
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CNRL-CorticalColumn/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    partial.replace(destination)


def _extract_face_annotations(zip_path: Path, dataset_root: Path) -> None:
    """Extract only official Faces and Faces_easy annotation files."""
    wanted = ("Annotations/Faces_2/", "Annotations/Faces_3/")
    with zipfile.ZipFile(zip_path) as archive:
        member = "caltech-101/Annotations.tar"
        if member not in archive.namelist():
            raise RuntimeError(f"{member} is absent from {zip_path}")
        with archive.open(member) as compressed:
            tar_bytes = io.BytesIO(compressed.read())
    with tarfile.open(fileobj=tar_bytes, mode="r:") as archive:
        for member in archive:
            normalized = member.name.replace("\\", "/")
            if not member.isfile() or not normalized.startswith(wanted):
                continue
            relative = Path(normalized)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe annotation member: {member.name}")
            destination = dataset_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read {member.name}")
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def _ensure_annotations(dataset_root: Path, cache_dir: Path) -> Path:
    target = dataset_root / "Annotations" / "Faces_2"
    if len(list(target.glob("annotation_*.mat"))) == 435:
        return target

    archive = cache_dir / "caltech-101.zip"
    if not archive.exists():
        print(f"Downloading official Caltech annotations from {CALTECH_URL}")
        _download(CALTECH_URL, archive)
    if _md5(archive) != CALTECH_MD5:
        raise RuntimeError(f"MD5 mismatch for {archive}")
    _extract_face_annotations(archive, dataset_root)
    if len(list(target.glob("annotation_*.mat"))) != 435:
        raise RuntimeError("official Faces annotations were not extracted")
    return target


def _load_box(annotation_dir: Path, index: int) -> tuple[float, ...]:
    values = loadmat(
        annotation_dir / f"annotation_{index:04d}.mat"
    )["box_coord"][0].astype(float)
    y1, y2, x1, x2 = values
    return (float(x1), float(y1), float(x2), float(y2))


def _calibrate_geometry(
    category_dir: Path,
    annotation_dir: Path,
    calibration_indices: Sequence[int],
) -> tuple[SearchGeometry, dict[str, object]]:
    height_fractions = []
    view_aspects = []
    object_widths = []
    object_heights = []
    normalized_boxes = []
    for index in calibration_indices:
        faces_path = category_dir / "Faces" / f"image_{index:04d}.jpg"
        easy_path = category_dir / "Faces_easy" / f"image_{index:04d}.jpg"
        with Image.open(faces_path) as image:
            width, height = image.size
        with Image.open(easy_path) as image:
            easy_width, easy_height = image.size
        target = _load_box(annotation_dir, index)
        height_fractions.append(easy_height / height)
        view_aspects.append(easy_width / easy_height)
        object_widths.append((target[2] - target[0]) / easy_width)
        object_heights.append((target[3] - target[1]) / easy_height)
        normalized_boxes.append(
            (
                target[0] / width,
                target[1] / height,
                target[2] / width,
                target[3] / height,
            )
        )

    geometry = SearchGeometry(
        view_height_fractions=tuple(
            float(value)
            for value in np.quantile(height_fractions, [0.15, 0.5, 0.85])
        ),
        view_aspect=float(np.median(view_aspects)),
        object_width_in_view=float(np.median(object_widths)),
        object_height_in_view=float(np.median(object_heights)),
        prior_box_normalized=tuple(
            float(value)
            for value in np.median(normalized_boxes, axis=0)
        ),
    )
    details = {
        "indices": [int(index) for index in calibration_indices],
        "statistics": {
            "view_height_fraction_quantiles_15_50_85": (
                geometry.view_height_fractions
            ),
            "median_view_aspect_width_over_height": geometry.view_aspect,
            "median_object_width_in_view": geometry.object_width_in_view,
            "median_object_height_in_view": geometry.object_height_in_view,
            "median_full_image_box_normalized": (
                geometry.prior_box_normalized
            ),
        },
    }
    return geometry, details


@torch.no_grad()
def _classify_paths(
    model,
    paths: Sequence[Path],
    transform,
    centers: Sequence[Sequence[int]],
    batch_size: int,
) -> dict[str, object]:
    predictions = []
    counts = []
    for start in range(0, len(paths), batch_size):
        images = torch.stack(
            [
                transform(Image.open(path).convert("RGB"))
                for path in paths[start:start + batch_size]
            ]
        )
        result = model.run_images(images, centers, track_eligibility=False)
        predictions.append(result.predictions)
        counts.append(result.decision_spike_counts)
    return {
        "predictions": torch.cat(predictions),
        "decision_spike_counts": torch.cat(counts),
    }


def _method_result(
    box: Sequence[float],
    target: Sequence[float],
    image_size: tuple[int, int],
) -> dict[str, object]:
    return {
        "box": [float(value) for value in box],
        "iou": box_iou(box, target),
        "center_inside_target": center_inside(box, target),
        "normalized_center_error": normalized_center_distance(
            box,
            target,
            image_size,
        ),
    }


@torch.no_grad()
def _scan_faces(
    model,
    test_indices: Sequence[int],
    category_dir: Path,
    annotation_dir: Path,
    transform,
    centers: Sequence[Sequence[int]],
    geometry: SearchGeometry,
    *,
    batch_size: int,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    weights = model.effective_decision_weights.detach().cpu()
    prior_center = (
        0.5 * (
            geometry.prior_box_normalized[0]
            + geometry.prior_box_normalized[2]
        ),
        0.5 * (
            geometry.prior_box_normalized[1]
            + geometry.prior_box_normalized[3]
        ),
    )
    records: list[dict[str, object]] = []
    evidence_cache: dict[int, dict[str, object]] = {}
    started = time.time()

    for position, index in enumerate(test_indices, start=1):
        image_path = category_dir / "Faces" / f"image_{index:04d}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        target = _load_box(annotation_dir, index)
        candidate_views = generate_candidate_views(
            width,
            height,
            geometry,
            x_positions=13,
            y_positions=3,
        )

        normal_counts = []
        normal_synaptic = []
        reversed_counts = []
        reversed_synaptic = []
        silent_counts = []
        normal_traces = []
        reversed_traces = []
        silent_traces = []
        normal_l23 = []
        normal_l56 = []
        for start in range(0, len(candidate_views), batch_size):
            batch_views = candidate_views[start:start + batch_size]
            images = torch.stack(
                [transform(image.crop(view)) for view in batch_views]
            )
            live = model.run_images(
                images,
                centers,
                track_eligibility=False,
            )
            reversed_result = model.run_spike_events(
                live.l23_spike_events.flip(1),
                track_eligibility=False,
            )
            silent_result = model.run_spike_events(
                live.l23_spike_events,
                track_eligibility=False,
                silence_l56=True,
            )

            normal_counts.append(live.decision_spike_counts)
            normal_synaptic.append(
                torch.einsum(
                    "blf,clf->bc",
                    live.l23_spike_events.float(),
                    weights,
                )
            )
            reversed_counts.append(reversed_result.decision_spike_counts)
            reversed_synaptic.append(
                torch.einsum(
                    "blf,clf->bc",
                    live.l23_spike_events.flip(1).float(),
                    weights,
                )
            )
            silent_counts.append(silent_result.decision_spike_counts)
            normal_traces.append(live.decision_trace)
            reversed_traces.append(reversed_result.decision_trace)
            silent_traces.append(silent_result.decision_trace)
            normal_l23.append(live.l23_spike_events)
            normal_l56.append(live.l56_spike_events)

        normal_counts_tensor = torch.cat(normal_counts)
        normal_synaptic_tensor = torch.cat(normal_synaptic)
        reversed_counts_tensor = torch.cat(reversed_counts)
        reversed_synaptic_tensor = torch.cat(reversed_synaptic)
        silent_counts_tensor = torch.cat(silent_counts)
        normal_traces_tensor = torch.cat(normal_traces)
        reversed_traces_tensor = torch.cat(reversed_traces)
        silent_traces_tensor = torch.cat(silent_traces)
        normal_l23_tensor = torch.cat(normal_l23)
        normal_l56_tensor = torch.cat(normal_l56)

        normal_margin = (
            normal_counts_tensor[:, 0] - normal_counts_tensor[:, 1]
        )
        normal_synaptic_margin = (
            normal_synaptic_tensor[:, 0] - normal_synaptic_tensor[:, 1]
        )
        reversed_margin = (
            reversed_counts_tensor[:, 0] - reversed_counts_tensor[:, 1]
        )
        reversed_synaptic_margin = (
            reversed_synaptic_tensor[:, 0]
            - reversed_synaptic_tensor[:, 1]
        )
        silent_margin = (
            silent_counts_tensor[:, 0] - silent_counts_tensor[:, 1]
        )

        selected = {
            "normal": select_evidence_candidate(
                normal_margin,
                normal_synaptic_margin,
                candidate_views,
                prior_center,
                (width, height),
            ),
            "reversed": select_evidence_candidate(
                reversed_margin,
                reversed_synaptic_margin,
                candidate_views,
                prior_center,
                (width, height),
            ),
            "silent": select_evidence_candidate(
                silent_margin,
                torch.zeros_like(silent_margin),
                candidate_views,
                prior_center,
                (width, height),
            ),
        }
        candidate_boxes = candidate_object_boxes(
            candidate_views,
            geometry,
            (width, height),
        )
        oracle_index = int(
            np.argmax([box_iou(box, target) for box in candidate_boxes])
        )
        selected["oracle"] = oracle_index

        methods = {}
        for method in ("normal", "reversed", "silent"):
            methods[method] = _method_result(
                candidate_boxes[selected[method]],
                target,
                (width, height),
            )
            methods[method]["candidate_view"] = [
                int(value) for value in candidate_views[selected[method]]
            ]
            methods[method]["candidate_index"] = int(selected[method])
        methods["prior"] = _method_result(
            prior_object_box(geometry, (width, height)),
            target,
            (width, height),
        )
        methods["oracle"] = _method_result(
            candidate_boxes[oracle_index],
            target,
            (width, height),
        )
        methods["oracle"]["candidate_view"] = [
            int(value) for value in candidate_views[oracle_index]
        ]
        methods["oracle"]["candidate_index"] = oracle_index

        image_box = (0.0, 0.0, float(width), float(height))
        displacement = normalized_center_distance(
            target,
            image_box,
            (width, height),
        )
        normal_index = selected["normal"]
        record = {
            "image_index": int(index),
            "image_path": str(image_path.relative_to(ROOT)),
            "image_width": int(width),
            "image_height": int(height),
            "ground_truth_box": [float(value) for value in target],
            "ground_truth_displacement_from_image_center": displacement,
            "n_candidate_views": len(candidate_views),
            "normal_selected_prediction": int(
                normal_counts_tensor[normal_index].argmax()
            ),
            "normal_selected_decision_spike_counts": (
                normal_counts_tensor[normal_index].tolist()
            ),
            "methods": methods,
        }
        records.append(record)
        evidence_cache[index] = {
            "candidate_views": torch.tensor(candidate_views),
            "normal_decision_counts": normal_counts_tensor,
            "normal_synaptic_margin": normal_synaptic_margin,
            "reversed_decision_counts": reversed_counts_tensor,
            "reversed_synaptic_margin": reversed_synaptic_margin,
            "silent_decision_counts": silent_counts_tensor,
            "selected_indices": selected,
            "normal_selected_trace": (
                normal_traces_tensor[normal_index].clone()
            ),
            "reversed_selected_trace": (
                reversed_traces_tensor[selected["reversed"]].clone()
            ),
            "silent_selected_trace": (
                silent_traces_tensor[selected["silent"]].clone()
            ),
            "normal_selected_l23": (
                normal_l23_tensor[normal_index].clone()
            ),
            "normal_selected_l56": (
                normal_l56_tensor[normal_index].clone()
            ),
        }

        elapsed = time.time() - started
        rate = elapsed / position
        remaining = rate * (len(test_indices) - position)
        print(
            f"[{position:3d}/{len(test_indices)}] Faces {index:04d}: "
            f"IoU={methods['normal']['iou']:.3f}, "
            f"prior={methods['prior']['iou']:.3f}, "
            f"{remaining / 60:.1f} min remaining",
            flush=True,
        )
    return records, evidence_cache


def _interval(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    low = 0.0 if successes == 0 else float(
        beta.ppf(0.025, successes, trials - successes + 1)
    )
    high = 1.0 if successes == trials else float(
        beta.ppf(0.975, successes + 1, trials - successes)
    )
    return [low, high]


def _subset_metrics(
    records: Sequence[dict[str, object]],
    method: str,
) -> dict[str, object]:
    ious = np.array(
        [record["methods"][method]["iou"] for record in records],
        dtype=float,
    )
    pointing = np.array(
        [
            record["methods"][method]["center_inside_target"]
            for record in records
        ],
        dtype=bool,
    )
    center_errors = np.array(
        [
            record["methods"][method]["normalized_center_error"]
            for record in records
        ],
        dtype=float,
    )
    localized = ious >= 0.5
    return {
        "n": len(records),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "corloc_iou_at_least_0_5": float(localized.mean()),
        "corloc_count": int(localized.sum()),
        "corloc_exact_95_ci": _interval(int(localized.sum()), len(records)),
        "pointing_accuracy": float(pointing.mean()),
        "mean_normalized_center_error": float(center_errors.mean()),
    }


def _paired_statistics(
    records: Sequence[dict[str, object]],
    first: str,
    second: str,
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, object]:
    first_iou = torch.tensor(
        [record["methods"][first]["iou"] for record in records]
    )
    second_iou = torch.tensor(
        [record["methods"][second]["iou"] for record in records]
    )
    difference = first_iou - second_iou
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(difference),
        (draws, len(difference)),
        generator=generator,
    )
    means = difference[indices].mean(dim=1)

    first_correct = first_iou >= 0.5
    second_correct = second_iou >= 0.5
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((~first_correct & second_correct).sum())
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(
            min(first_only, second_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    return {
        "first": first,
        "second": second,
        "mean_paired_iou_difference": float(difference.mean()),
        "bootstrap_95_ci": [
            float(torch.quantile(means, 0.025)),
            float(torch.quantile(means, 0.975)),
        ],
        "corloc_first_only": first_only,
        "corloc_second_only": second_only,
        "mcnemar_exact_two_sided_p": p_value,
    }


def _build_metrics(
    records: Sequence[dict[str, object]],
    *,
    seed: int,
) -> dict[str, object]:
    subsets = {
        "all_unseen": list(records),
        "displacement_at_least_0_10": [
            record for record in records
            if record["ground_truth_displacement_from_image_center"] >= 0.10
        ],
        "displacement_at_least_0_15": [
            record for record in records
            if record["ground_truth_displacement_from_image_center"] >= 0.15
        ],
    }
    result = {"subsets": {}, "paired_tests": {}}
    for subset_name, subset_records in subsets.items():
        result["subsets"][subset_name] = {
            method: _subset_metrics(subset_records, method)
            for method in METHOD_LABELS
        }
        result["paired_tests"][subset_name] = {
            f"normal_vs_{control}": _paired_statistics(
                subset_records,
                "normal",
                control,
                seed=seed,
            )
            for control in ("reversed", "silent", "prior")
        }
    return result


def _draw_boxes(
    ax: plt.Axes,
    record: dict[str, object],
    *,
    show_view: bool = True,
    legend: bool = False,
) -> None:
    boxes = [
        (
            record["ground_truth_box"],
            COLORS["gold"],
            "-",
            1.8,
            "Ground truth",
        ),
        (
            record["methods"]["normal"]["box"],
            COLORS["cyan"],
            "-",
            1.8,
            "Spiking search",
        ),
        (
            record["methods"]["prior"]["box"],
            COLORS["gray"],
            "--",
            1.2,
            "Location prior",
        ),
    ]
    if show_view:
        boxes.append(
            (
                record["methods"]["normal"]["candidate_view"],
                COLORS["blue"],
                ":",
                1.2,
                "Selected view",
            )
        )
    for box, color, linestyle, linewidth, label in boxes:
        x1, y1, x2, y2 = box
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
        )
    if legend:
        ax.legend(
            loc="lower center",
            ncol=2,
            fontsize=7,
            frameon=True,
            facecolor="white",
            framealpha=0.88,
        )


def _montage(paths: Sequence[Path], tile_size: tuple[int, int]) -> Image.Image:
    tiles = []
    for path in paths:
        with Image.open(path) as image:
            tile = ImageOps.fit(image.convert("RGB"), tile_size)
        tiles.append(tile)
    canvas = Image.new(
        "RGB",
        (tile_size[0] * len(tiles), tile_size[1]),
        "white",
    )
    for index, tile in enumerate(tiles):
        canvas.paste(tile, (index * tile_size[0], 0))
    return canvas


def _training_montage(category_dir: Path) -> Image.Image:
    face_row = _montage(
        [
            category_dir / "Faces_easy" / f"image_{index:04d}.jpg"
            for index in (21, 78, 154)
        ],
        (120, 110),
    )
    motor_row = _montage(
        [
            category_dir / "Motorbikes" / f"image_{index:04d}.jpg"
            for index in (33, 171)
        ],
        (180, 110),
    )
    canvas = Image.new("RGB", (360, 220), "white")
    canvas.paste(face_row, (0, 0))
    canvas.paste(motor_row, (0, 110))
    return canvas


def _choose_protocol_case(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    eligible = [
        record for record in records
        if record["ground_truth_displacement_from_image_center"] >= 0.10
        and record["methods"]["normal"]["iou"] >= 0.5
    ]
    return max(
        eligible,
        key=lambda record: (
            record["methods"]["normal"]["iou"]
            - record["methods"]["prior"]["iou"],
            record["ground_truth_displacement_from_image_center"],
        ),
    )


def _plot_protocol(
    path: Path,
    record: dict[str, object],
    evidence: dict[str, object],
    category_dir: Path,
    centers: Sequence[Sequence[int]],
    patch_size: int,
) -> None:
    index = record["image_index"]
    image_path = category_dir / "Faces" / f"image_{index:04d}.jpg"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    montage = _training_montage(category_dir)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2))
    ax = axes[0, 0]
    ax.imshow(montage)
    ax.axhline(110, color="white", lw=3)
    ax.text(
        180,
        8,
        "Face_easy",
        ha="center",
        va="top",
        color="white",
        weight="bold",
        fontsize=9,
        bbox={"facecolor": COLORS["navy"], "edgecolor": "none", "pad": 2},
    )
    ax.text(
        180,
        118,
        "Motorbikes",
        ha="center",
        va="top",
        color="white",
        weight="bold",
        fontsize=9,
        bbox={"facecolor": COLORS["red"], "edgecolor": "none", "pad": 2},
    )
    ax.set_title(
        "A  Training views build feature-location evidence",
        loc="left",
        weight="bold",
    )
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(image)
    _draw_boxes(ax, record, legend=True)
    ax.set_title(
        f"B  Unseen full image {index}: the face is displaced",
        loc="left",
        weight="bold",
    )
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(image, alpha=0.28)
    views = evidence["candidate_views"].numpy()
    counts = evidence["normal_decision_counts"]
    margins = (counts[:, 0] - counts[:, 1]).numpy()
    centers_x = 0.5 * (views[:, 0] + views[:, 2])
    centers_y = 0.5 * (views[:, 1] + views[:, 3])
    scatter = ax.scatter(
        centers_x,
        centers_y,
        c=margins,
        cmap="RdYlBu_r",
        vmin=-max(abs(margins.min()), abs(margins.max()), 1),
        vmax=max(abs(margins.min()), abs(margins.max()), 1),
        s=20,
        edgecolors="white",
        linewidths=0.25,
    )
    _draw_boxes(ax, record, show_view=False)
    fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.02,
        label="Face - motorbike decision spikes",
    )
    ax.set_title(
        "C  Candidate centers carry face evidence",
        loc="left",
        weight="bold",
    )
    ax.axis("off")

    ax = axes[1, 0]
    view = record["methods"]["normal"]["candidate_view"]
    crop = image.crop(tuple(int(value) for value in view)).resize((80, 80))
    ax.imshow(crop)
    half = patch_size // 2
    for location, (row, col) in enumerate(centers):
        ax.add_patch(
            Rectangle(
                (col - half, row - half),
                patch_size,
                patch_size,
                fill=False,
                linewidth=0.75,
                edgecolor=plt.cm.viridis(
                    location / max(len(centers) - 1, 1)
                ),
            )
        )
        ax.text(
            col,
            row,
            str(location + 1),
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            weight="bold",
        )
    ax.set_title(
        "D  Winning view: nine internal saccades",
        loc="left",
        weight="bold",
    )
    ax.axis("off")

    ax = axes[1, 1]
    for name, linestyle, alpha in [
        ("normal", "-", 1.0),
        ("reversed", "--", 0.9),
        ("silent", ":", 0.9),
    ]:
        trace = evidence[f"{name}_selected_trace"]
        margin = trace[:, 0] - trace[:, 1]
        ax.plot(
            np.arange(1, len(margin) + 1),
            margin.numpy(),
            marker="o",
            ms=3,
            lw=1.5,
            linestyle=linestyle,
            alpha=alpha,
            color=METHOD_COLORS[name],
            label=METHOD_LABELS[name],
        )
    ax.axhline(0, color="#B8B8B8", lw=0.8)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Internal saccade")
    ax.set_ylabel("Cumulative face - motorbike spikes")
    ax.set_title(
        "E  Ordered frame evidence reaches the decision",
        loc="left",
        weight="bold",
    )
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    l23 = evidence["normal_selected_l23"].bool()
    l56 = evidence["normal_selected_l56"].bool()
    locations, features = torch.where(l23)
    ax.scatter(
        locations.numpy() + 1,
        features.numpy(),
        s=8,
        color=COLORS["cyan"],
        linewidths=0,
        label="L2/3 feature spikes",
    )
    l56_location, l56_neuron = torch.where(l56)
    ax.scatter(
        l56_location.numpy() + 1,
        np.full(len(l56_location), l23.shape[1] + 35)
        + l56_neuron.numpy() * 3,
        marker="s",
        s=10,
        color=COLORS["purple"],
        linewidths=0,
        label="L5/6 location spikes",
    )
    ax.set_xlim(0.5, 9.5)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Internal saccade")
    ax.set_ylabel("Neuron index")
    ax.set_title(
        "F  Feature spikes bind to one-hot L5/6 states",
        loc="left",
        weight="bold",
    )
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "A face reference frame learned from centered crops guides saccadic "
        "search in an uncropped image",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _save(fig, path)


def _choose_example_records(
    records: Sequence[dict[str, object]],
) -> list[tuple[str, dict[str, object]]]:
    gains = sorted(
        records,
        key=lambda record: (
            record["methods"]["normal"]["iou"]
            - record["methods"]["prior"]["iou"]
        ),
        reverse=True,
    )
    selected: list[tuple[str, dict[str, object]]] = [
        ("Largest prior gain", record) for record in gains[:4]
    ]
    remaining = [
        record for record in records
        if record["image_index"] not in {
            item["image_index"] for _, item in selected
        }
    ]
    ordered = sorted(
        remaining,
        key=lambda record: record["methods"]["normal"]["iou"],
    )
    selected.extend(
        [
            ("Lowest IoU", ordered[0]),
            ("Lowest IoU", ordered[1]),
            ("Median IoU", ordered[len(ordered) // 2]),
            ("Median IoU", ordered[len(ordered) // 2 + 1]),
        ]
    )
    return selected


def _plot_examples(
    path: Path,
    records: Sequence[dict[str, object]],
    category_dir: Path,
) -> list[dict[str, object]]:
    selected = _choose_example_records(records)
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.0))
    selection_report = []
    for ax, (selection, record) in zip(axes.flat, selected):
        image_path = (
            category_dir
            / "Faces"
            / f"image_{record['image_index']:04d}.jpg"
        )
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
        _draw_boxes(ax, record, legend=False)
        ax.set_title(
            f"{selection}: image {record['image_index']}\n"
            f"IoU {record['methods']['normal']['iou']:.2f}; "
            f"prior {record['methods']['prior']['iou']:.2f}; "
            f"offset {record['ground_truth_displacement_from_image_center']:.2f}",
            loc="left",
            weight="bold",
            fontsize=8.5,
        )
        ax.axis("off")
        selection_report.append(
            {
                "selection_stratum": selection,
                "image_index": record["image_index"],
                "normal_iou": record["methods"]["normal"]["iou"],
                "prior_iou": record["methods"]["prior"]["iou"],
            }
        )
    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS["gold"], lw=2),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS["cyan"], lw=2),
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor=COLORS["gray"],
            lw=1.5,
            linestyle="--",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor=COLORS["blue"],
            lw=1.5,
            linestyle=":",
        ),
    ]
    fig.legend(
        handles,
        ["Ground truth", "Spiking search", "Location prior", "Selected view"],
        loc="lower center",
        ncol=4,
    )
    fig.suptitle(
        "Zero-shot localization examples include largest gains, median cases, "
        "and the two lowest-IoU failures",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, path)
    return selection_report


def _plot_metrics(
    path: Path,
    records: Sequence[dict[str, object]],
    metrics: dict[str, object],
    classification: dict[str, object],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))

    ax = axes[0, 0]
    names = [
        "Face_easy\ncrop",
        "Full Faces\nimage",
        "Selected view\n(search endpoint)",
    ]
    values = [
        classification["face_easy_accuracy"],
        classification["full_faces_accuracy"],
        classification["selected_view_face_rate"],
    ]
    bars = ax.bar(
        names,
        values,
        color=[COLORS["navy"], COLORS["gray"], COLORS["cyan"]],
        width=0.64,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{100 * value:.1f}%",
            ha="center",
            weight="bold",
        )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Face decision rate")
    ax.set_title(
        "A  Saccadic search recovers a face-like view",
        loc="left",
        weight="bold",
    )

    ax = axes[0, 1]
    subset_names = [
        "all_unseen",
        "displacement_at_least_0_10",
        "displacement_at_least_0_15",
    ]
    subset_labels = [
        (
            "All unseen\n"
            f"n={metrics['subsets']['all_unseen']['normal']['n']}"
        ),
        (
            "Offset >= 0.10\n"
            f"n={metrics['subsets']['displacement_at_least_0_10']['normal']['n']}"
        ),
        (
            "Offset >= 0.15\n"
            f"n={metrics['subsets']['displacement_at_least_0_15']['normal']['n']}"
        ),
    ]
    methods = ["normal", "reversed", "silent", "prior", "oracle"]
    x = np.arange(len(subset_names))
    width = 0.15
    for method_index, method in enumerate(methods):
        values = [
            metrics["subsets"][subset][method][
                "corloc_iou_at_least_0_5"
            ]
            for subset in subset_names
        ]
        ax.bar(
            x + (method_index - 2) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(x, subset_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("CorLoc (IoU >= 0.5)")
    ax.set_title(
        "B  Localization remains measurable away from image center",
        loc="left",
        weight="bold",
    )
    ax.legend(ncol=2, fontsize=7)

    ax = axes[1, 0]
    distributions = [
        [record["methods"][method]["iou"] for record in records]
        for method in ("normal", "reversed", "silent", "prior", "oracle")
    ]
    boxplot = ax.boxplot(
        distributions,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": COLORS["black"], "linewidth": 1.2},
    )
    for patch, method in zip(
        boxplot["boxes"],
        ("normal", "reversed", "silent", "prior", "oracle"),
    ):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.85)
    ax.set_xticks(
        range(1, 6),
        ["Ordered", "Reversed", "Silent", "Prior", "Oracle"],
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Intersection over union")
    ax.set_title(
        "C  Paired localization quality across all unseen images",
        loc="left",
        weight="bold",
    )

    ax = axes[1, 1]
    displacement = np.array(
        [
            record["ground_truth_displacement_from_image_center"]
            for record in records
        ]
    )
    gain = np.array(
        [
            record["methods"]["normal"]["iou"]
            - record["methods"]["prior"]["iou"]
            for record in records
        ]
    )
    localized = np.array(
        [record["methods"]["normal"]["iou"] >= 0.5 for record in records]
    )
    ax.scatter(
        displacement[~localized],
        gain[~localized],
        s=17,
        color=COLORS["red"],
        alpha=0.65,
        label="IoU < 0.5",
    )
    ax.scatter(
        displacement[localized],
        gain[localized],
        s=17,
        color=COLORS["cyan"],
        alpha=0.65,
        label="IoU >= 0.5",
    )
    edges = np.quantile(displacement, np.linspace(0, 1, 7))
    bin_x = []
    bin_y = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (displacement >= low) & (displacement <= high)
        bin_x.append(float(displacement[mask].mean()))
        bin_y.append(float(gain[mask].mean()))
    ax.plot(
        bin_x,
        bin_y,
        color=COLORS["navy"],
        marker="o",
        lw=2,
        label="Six-bin mean",
    )
    ax.axhline(0, color="#B8B8B8", lw=0.8)
    ax.axvline(0.10, color="#B8B8B8", lw=0.8, linestyle="--")
    ax.set_xlabel("Ground-truth center offset from image center")
    ax.set_ylabel("IoU gain over location prior")
    ax.set_title(
        "D  Search adds value as the face moves off center",
        loc="left",
        weight="bold",
    )
    ax.legend(fontsize=7)

    fig.suptitle(
        "The fully spiking column localizes unseen uncropped Caltech faces",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _save(fig, path)


def _write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "image_index",
        "image_width",
        "image_height",
        "ground_truth_displacement",
        "method",
        "iou",
        "corloc",
        "center_inside_target",
        "normalized_center_error",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for method in METHOD_LABELS:
                result = record["methods"][method]
                box = result["box"]
                writer.writerow(
                    {
                        "image_index": record["image_index"],
                        "image_width": record["image_width"],
                        "image_height": record["image_height"],
                        "ground_truth_displacement": (
                            record[
                                "ground_truth_displacement_from_image_center"
                            ]
                        ),
                        "method": method,
                        "iou": result["iou"],
                        "corloc": int(result["iou"] >= 0.5),
                        "center_inside_target": int(
                            result["center_inside_target"]
                        ),
                        "normalized_center_error": (
                            result["normalized_center_error"]
                        ),
                        "box_x1": box[0],
                        "box_y1": box[1],
                        "box_x2": box[2],
                        "box_y2": box[3],
                    }
                )


def _write_markdown(
    path: Path,
    report: dict[str, object],
) -> None:
    all_metrics = report["localization"]["subsets"]["all_unseen"]
    far_metrics = report["localization"]["subsets"][
        "displacement_at_least_0_10"
    ]
    normal = all_metrics["normal"]
    prior = all_metrics["prior"]
    reversed_result = all_metrics["reversed"]
    far_normal = far_metrics["normal"]
    far_prior = far_metrics["prior"]
    classification = report["classification_transfer"]
    comparison = report["localization"]["paired_tests"]["all_unseen"][
        "normal_vs_prior"
    ]
    text = f"""# Caltech Faces zero-shot localization experiment

## Question

Can the immutable fully spiking column trained on `Faces_easy` versus
`Motorbikes` find the corresponding face inside the larger, uncropped
Caltech `Faces` image?

## Leakage boundary

`Faces` and `Faces_easy` are paired views of the same 435 source images.
The checkpoint was built from the first 200 files in each training category.
Geometry calibration therefore uses only source indices 1--200, and every
reported localization endpoint uses the disjoint source indices
{report['protocol']['test_indices'][0]}--{report['protocol']['test_indices'][-1]}
(n={len(report['protocol']['test_indices'])}). No test annotation, test
`Faces_easy` crop, or test localization result changes model weights, search
geometry, or candidate selection.

## Method

The search generates {report['protocol']['candidate_grid']['x_positions']}
horizontal by {report['protocol']['candidate_grid']['y_positions']} vertical
views at three training-calibrated scales. Each view is resized and passed
through the unmodified CoNeX/PyMoNNtorch path:

`sensory spikes -> L4 spikes -> L2/3 spikes -> L5/6-gated decision spikes`.

The face-minus-motorbike output spike margin ranks candidate views. Exact
spike-count ties are resolved by the integrated L5/6-gated synaptic margin;
the training-side median location prior is used only if both neural signals
tie. The selected view is converted to an object box with training-side
median object-to-view ratios.

## Results

- Unseen `Faces_easy` face classification:
  **{100 * classification['face_easy_accuracy']:.2f}%**.
- Uncropped whole-image face classification:
  **{100 * classification['full_faces_accuracy']:.2f}%**.
- Selected-view face decision rate:
  **{100 * classification['selected_view_face_rate']:.2f}%**. This is the
  search endpoint, not an independently sampled classification accuracy:
  candidate selection explicitly maximizes face evidence.
- Ordered L5/6 localization: mean IoU **{normal['mean_iou']:.3f}**,
  CorLoc **{normal['corloc_count']}/{normal['n']} =
  {100 * normal['corloc_iou_at_least_0_5']:.2f}%**
  (exact 95% CI {100 * normal['corloc_exact_95_ci'][0]:.2f}--
  {100 * normal['corloc_exact_95_ci'][1]:.2f}%).
- Reversed L5/6 localization: mean IoU
  **{reversed_result['mean_iou']:.3f}**, CorLoc
  **{100 * reversed_result['corloc_iou_at_least_0_5']:.2f}%**.
- Training-side location prior: mean IoU **{prior['mean_iou']:.3f}**,
  CorLoc **{100 * prior['corloc_iou_at_least_0_5']:.2f}%**.
- For faces displaced by at least 0.10 image units (n={far_normal['n']}),
  ordered L5/6 CorLoc is
  **{100 * far_normal['corloc_iou_at_least_0_5']:.2f}%** versus
  **{100 * far_prior['corloc_iou_at_least_0_5']:.2f}%** for the prior.
- Paired ordered-minus-prior mean IoU difference:
  **{comparison['mean_paired_iou_difference']:.3f}**
  (bootstrap 95% CI {comparison['bootstrap_95_ci'][0]:.3f}--
  {comparison['bootstrap_95_ci'][1]:.3f}; CorLoc McNemar p =
  {comparison['mcnemar_exact_two_sided_p']:.3g}).

## Interpretation

This is a category-level, single-object localization assay, not a general
face detector. It shows whether a reference frame learned from centered
training crops can guide search over larger unseen source images. The
candidate oracle reports the limit imposed by the fixed view grid, and the
reversed/silent L5/6 conditions test whether ordered feature-location binding
contributes beyond image-center bias. The source-index split excludes the
exact paired crop corresponding to every test image, but Caltech Faces is not
an identity-disjoint benchmark and may repeat people or environments.

## Artifacts

- `caltech_faces_localization_results.json`: complete protocol and statistics.
- `caltech_faces_localization_predictions.csv`: per-image boxes and metrics.
- `caltech_faces_localization_evidence.pt`: candidate evidence and selected
  L2/3, L5/6, and decision spike traces.
- `figures/08_caltech_faces_search_protocol.pdf`
- `figures/09_caltech_faces_localization_examples.pdf`
- `figures/10_caltech_faces_localization_metrics.pdf`
"""
    path.write_text(text, encoding="utf-8")


def _write_tex(path: Path, report: dict[str, object]) -> None:
    all_metrics = report["localization"]["subsets"]["all_unseen"]
    far_metrics = report["localization"]["subsets"][
        "displacement_at_least_0_10"
    ]
    normal = all_metrics["normal"]
    reversed_result = all_metrics["reversed"]
    prior = all_metrics["prior"]
    comparison = report["localization"]["paired_tests"]["all_unseen"][
        "normal_vs_prior"
    ]
    mantissa, exponent = (
        f"{comparison['mcnemar_exact_two_sided_p']:.2e}".split("e")
    )
    p_value_tex = (
        f"{float(mantissa):.2f}\\times 10^{{{int(exponent)}}}"
    )
    text = rf"""\subsection{{Zero-shot localization in uncropped Caltech Faces}}
\label{{sec:faces-localization}}

We asked whether the face reference frame learned from centered
\dataset{{Faces\_easy}} crops could guide search in the paired, uncropped
\dataset{{Faces}} images. Because the two categories contain paired views of
the same 435 source indices, source indices 1--200 were used only for
training-side geometric calibration and indices 201--435 were reserved for
zero-shot localization. The production checkpoint was not retrained. At each
of three calibrated scales, a fixed grid of candidate views was passed through
the complete CoNeX/PyMoNNtorch network. Candidate rank was determined first by
the face-minus-motorbike decision-spike margin and then, only for exact ties,
by integrated \Lfive{{}}-gated synaptic evidence.

Across {normal['n']} unseen source images, ordered \Lfive{{}} search obtained a
mean IoU of {normal['mean_iou']:.3f} and CorLoc of
{100 * normal['corloc_iou_at_least_0_5']:.2f}\%
({normal['corloc_count']}/{normal['n']}; exact 95\% CI
{100 * normal['corloc_exact_95_ci'][0]:.2f}--{100 * normal['corloc_exact_95_ci'][1]:.2f}\%).
Reversing the \Lfive{{}} alignment yielded mean IoU
{reversed_result['mean_iou']:.3f} and CorLoc
{100 * reversed_result['corloc_iou_at_least_0_5']:.2f}\%; the training-side
location prior yielded {prior['mean_iou']:.3f} and
{100 * prior['corloc_iou_at_least_0_5']:.2f}\%, respectively. For the
{far_metrics['normal']['n']} images whose annotated center was displaced by at
least 0.10 normalized image units, ordered \Lfive{{}} CorLoc was
{100 * far_metrics['normal']['corloc_iou_at_least_0_5']:.2f}\% compared with
{100 * far_metrics['prior']['corloc_iou_at_least_0_5']:.2f}\% for the prior.
The paired ordered-minus-prior mean IoU difference was
{comparison['mean_paired_iou_difference']:.3f}
(bootstrap 95\% CI {comparison['bootstrap_95_ci'][0]:.3f}--
{comparison['bootstrap_95_ci'][1]:.3f}; exact McNemar
$p={p_value_tex}$).
The source-index split excludes each test image's exact paired crop, but the
Caltech category is not an identity-disjoint face benchmark.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{08_caltech_faces_search_protocol.pdf}}
\caption{{\textbf{{A centered face reference frame guides search in an
uncropped image.}} Training exemplars, candidate-view decision evidence, the
selected foveal view, cumulative output spikes, and the corresponding
\Ltwo{{}}/\Lfive{{}} events are shown for a prespecified displaced success.}}
\label{{fig:faces-search-protocol}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{09_caltech_faces_localization_examples.pdf}}
\caption{{\textbf{{Qualitative zero-shot localization examples.}} Displayed
cases are selected by deterministic strata: the four largest IoU gains over
the location prior, two median cases, and the two lowest-IoU failures.}}
\label{{fig:faces-localization-examples}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{10_caltech_faces_localization_metrics.pdf}}
\caption{{\textbf{{Quantitative localization and causal controls.}} CorLoc is
IoU $\geq 0.5$. Reversal and silencing alter only the alignment or presence of
\Lfive{{}} spikes while holding recorded \Ltwo{{}} events fixed.}}
\label{{fig:faces-localization-metrics}}
\end{{figure}}
"""
    path.write_text(text, encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data" / "caltech" / "caltech101",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "joint_training"
            / "caltech_all_acquisition_fixation_gated"
            / "best_fully_spiking_column.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "manuscript_fully_spiking"
            / "caltech_faces_localization_no_replay"
        ),
    )
    parser.add_argument("--first-test-index", type=int, default=201)
    parser.add_argument("--last-test-index", type=int, default=435)
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="Use all indices by default; positive values are for diagnostics.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    _style()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    category_dir = args.dataset_root / "101_ObjectCategories"
    annotation_dir = _ensure_annotations(
        args.dataset_root,
        args.output_dir / "download_cache",
    )
    for category in ("Faces", "Faces_easy", "Motorbikes"):
        if not (category_dir / category).exists():
            raise FileNotFoundError(category_dir / category)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("backend") != "CoNeX/PyMoNNtorch":
        raise RuntimeError("checkpoint is not the CoNeX/PyMoNNtorch model")
    if checkpoint.get("gradient_free") is not True:
        raise RuntimeError("checkpoint is not marked gradient-free")
    centers = checkpoint["metadata"]["centers"]
    model = load_fully_spiking_column(args.checkpoint, device=args.device)
    transform = build_transform(Config.caltech())

    calibration_indices = list(range(1, 201))
    test_indices = list(
        range(args.first_test_index, args.last_test_index + 1)
    )
    if args.max_test_images > 0:
        test_indices = test_indices[: args.max_test_images]
    if set(calibration_indices) & set(test_indices):
        raise RuntimeError("calibration and test source indices overlap")
    geometry, calibration = _calibrate_geometry(
        category_dir,
        annotation_dir,
        calibration_indices,
    )
    print("Search geometry:", asdict(geometry), flush=True)

    experiment_started = time.time()
    face_easy_paths = [
        category_dir / "Faces_easy" / f"image_{index:04d}.jpg"
        for index in test_indices
    ]
    full_faces_paths = [
        category_dir / "Faces" / f"image_{index:04d}.jpg"
        for index in test_indices
    ]
    easy_result = _classify_paths(
        model,
        face_easy_paths,
        transform,
        centers,
        args.batch_size,
    )
    full_result = _classify_paths(
        model,
        full_faces_paths,
        transform,
        centers,
        args.batch_size,
    )
    records, evidence = _scan_faces(
        model,
        test_indices,
        category_dir,
        annotation_dir,
        transform,
        centers,
        geometry,
        batch_size=args.batch_size,
    )
    classification = {
        "face_easy_accuracy": float(
            easy_result["predictions"].eq(0).float().mean()
        ),
        "full_faces_accuracy": float(
            full_result["predictions"].eq(0).float().mean()
        ),
        "selected_view_face_rate": float(
            np.mean(
                [
                    record["normal_selected_prediction"] == 0
                    for record in records
                ]
            )
        ),
        "face_easy_decision_spike_counts_mean": (
            easy_result["decision_spike_counts"].float().mean(0).tolist()
        ),
        "full_faces_decision_spike_counts_mean": (
            full_result["decision_spike_counts"].float().mean(0).tolist()
        ),
    }
    localization = _build_metrics(records, seed=args.seed)

    protocol_record = _choose_protocol_case(records)
    protocol_index = protocol_record["image_index"]
    _plot_protocol(
        figures_dir / "08_caltech_faces_search_protocol",
        protocol_record,
        evidence[protocol_index],
        category_dir,
        centers,
        int(checkpoint["cfg"]["patch_size"]),
    )
    example_selection = _plot_examples(
        figures_dir / "09_caltech_faces_localization_examples",
        records,
        category_dir,
    )
    _plot_metrics(
        figures_dir / "10_caltech_faces_localization_metrics",
        records,
        localization,
        classification,
    )

    report = {
        "experiment": "caltech_faces_zero_shot_saccadic_localization",
        "backend": checkpoint["backend"],
        "gradient_free": checkpoint["gradient_free"],
        "surrogate_derivatives": checkpoint["surrogate_derivatives"],
        "checkpoint": {
            "path": str(args.checkpoint.relative_to(ROOT)),
            "sha256": _sha256(args.checkpoint),
            "kind": checkpoint["kind"],
        },
        "data": {
            "source": CALTECH_URL,
            "official_archive_md5": CALTECH_MD5,
            "category_directory": str(category_dir.relative_to(ROOT)),
            "annotation_directory": str(annotation_dir.relative_to(ROOT)),
            "faces_count": len(list((category_dir / "Faces").glob("*.jpg"))),
            "faces_easy_count": len(
                list((category_dir / "Faces_easy").glob("*.jpg"))
            ),
            "paired_source_index_warning": (
                "Faces and Faces_easy are paired views sharing source indices; "
                "only indices above 200 are reported."
            ),
            "identity_disjoint_not_established": True,
        },
        "protocol": {
            "seed": args.seed,
            "calibration_indices": calibration_indices,
            "test_indices": test_indices,
            "calibration": calibration,
            "search_geometry": asdict(geometry),
            "candidate_grid": {
                "x_positions": 13,
                "y_positions": 3,
                "scales": list(geometry.view_height_fractions),
            },
            "candidate_ranking": [
                "face-minus-motorbike decision spike-count margin",
                "L5/6-gated synaptic margin for exact spike-count ties",
                "training-side median location prior for complete ties",
            ],
            "test_annotations_used_only_for_evaluation": True,
            "checkpoint_updated": False,
            "protocol_figure_case": {
                "selection_rule": (
                    "largest IoU gain over prior among displacement >= 0.10 "
                    "and ordered-L5/6 IoU >= 0.5"
                ),
                "image_index": protocol_index,
            },
            "qualitative_example_selection": example_selection,
        },
        "classification_transfer": classification,
        "localization": localization,
        "records": records,
        "runtime_seconds": time.time() - experiment_started,
        "artifacts": {
            "predictions_csv": "caltech_faces_localization_predictions.csv",
            "evidence": "caltech_faces_localization_evidence.pt",
            "figures": [
                "figures/08_caltech_faces_search_protocol.pdf",
                "figures/09_caltech_faces_localization_examples.pdf",
                "figures/10_caltech_faces_localization_metrics.pdf",
            ],
        },
    }

    _write_csv(
        args.output_dir / "caltech_faces_localization_predictions.csv",
        records,
    )
    torch.save(
        {
            "kind": "caltech_faces_localization_evidence",
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "test_indices": test_indices,
            "geometry": asdict(geometry),
            "evidence": evidence,
        },
        args.output_dir / "caltech_faces_localization_evidence.pt",
    )
    (args.output_dir / "caltech_faces_localization_results.json").write_text(
        json.dumps(_json_value(report), indent=2),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "RESULTS.md", report)
    _write_tex(
        args.output_dir / "caltech_faces_localization_section.tex",
        report,
    )

    normal = localization["subsets"]["all_unseen"]["normal"]
    prior = localization["subsets"]["all_unseen"]["prior"]
    print(
        "\nCompleted zero-shot localization:\n"
        f"  n={normal['n']}\n"
        f"  ordered L5/6 mean IoU={normal['mean_iou']:.4f}, "
        f"CorLoc={normal['corloc_iou_at_least_0_5']:.4f}\n"
        f"  prior mean IoU={prior['mean_iou']:.4f}, "
        f"CorLoc={prior['corloc_iou_at_least_0_5']:.4f}\n"
        f"  outputs={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
