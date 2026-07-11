"""Turn the Caltech Faces localization assay into a true detection test.

The frozen face-vs-motorbike cortical column scans every image with the same
multi-scale candidate grid used by the localization experiment. A development
set selects one of five prespecified spike-evidence summaries and one operating
threshold. Final evaluation includes source-disjoint Faces, source-disjoint
Motorbikes, novel background images, and face-like object distractors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scc.config import Config
from scc.data.pipeline import build_transform
from scc.faces_localization import (
    SearchGeometry,
    candidate_object_boxes,
    generate_candidate_views,
)
from scc.fully_spiking import load_fully_spiking_column


DISTRACTOR_CATEGORIES = (
    "cougar_face",
    "brain",
    "watch",
    "binocular",
    "camera",
    "cellphone",
    "headphone",
)
RULE_NAMES = (
    "normal_max",
    "normal_top5",
    "ordered_advantage",
    "normal_plus_advantage",
    "coherent_plus_advantage",
)
RULE_LABELS = {
    "normal_max": "Maximum ordered evidence",
    "normal_top5": "Top-5 ordered coherence",
    "ordered_advantage": "Ordered - reversed",
    "normal_plus_advantage": "Ordered + frame advantage",
    "coherent_plus_advantage": "Coherence + frame advantage",
}
COLORS = {
    "selected": "#0072B2",
    "normal_max": "#009E73",
    "reversed_max": "#D55E00",
    "silent": "#777777",
    "Faces": "#0072B2",
    "Motorbikes": "#D55E00",
    "BACKGROUND_Google": "#7A7A7A",
    "hard_distractors": "#CC79A7",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_from_report(report: Dict[str, object]) -> SearchGeometry:
    record = report["protocol"]["search_geometry"]
    allowed = {field.name for field in fields(SearchGeometry)}
    return SearchGeometry(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in record.items()
            if key in allowed
        }
    )


def _lexicographic_score(
    counts: torch.Tensor,
    synaptic_margin: torch.Tensor,
) -> torch.Tensor:
    spike_margin = counts[:, 0].float() - counts[:, 1].float()
    # The bounded term is always smaller than one spike-count unit, so it can
    # break ties but can never overturn a difference in emitted spike count.
    return spike_margin + 0.49 * torch.tanh(synaptic_margin.float())


def _summaries(
    normal_scores: torch.Tensor,
    reversed_scores: torch.Tensor,
) -> Dict[str, float]:
    normal_sorted = normal_scores.sort(descending=True).values
    reversed_sorted = reversed_scores.sort(descending=True).values
    normal_max = float(normal_sorted[0])
    reversed_max = float(reversed_sorted[0])
    normal_top5 = float(normal_sorted[:5].mean())
    reversed_top5 = float(reversed_sorted[:5].mean())
    advantage = normal_max - reversed_max
    return {
        "normal_max": normal_max,
        "normal_top5": normal_top5,
        "reversed_max": reversed_max,
        "reversed_top5": reversed_top5,
        "ordered_advantage": advantage,
        "normal_plus_advantage": normal_max + advantage,
        "coherent_plus_advantage": (
            normal_top5 + normal_top5 - reversed_top5
        ),
        "silent": 0.0,
    }


@torch.no_grad()
def _scan_image(
    model,
    image_path: Path,
    transform,
    centers: Sequence[Sequence[int]],
    geometry: SearchGeometry,
    *,
    batch_size: int,
) -> Dict[str, object]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    views = generate_candidate_views(
        width,
        height,
        geometry,
        x_positions=13,
        y_positions=3,
    )
    weights = model.effective_decision_weights.detach().cpu()
    normal_scores: List[torch.Tensor] = []
    reversed_scores: List[torch.Tensor] = []
    normal_counts_all: List[torch.Tensor] = []
    reversed_counts_all: List[torch.Tensor] = []
    for start in range(0, len(views), batch_size):
        batch_views = views[start:start + batch_size]
        images = torch.stack(
            [transform(image.crop(view)) for view in batch_views]
        )
        normal = model.run_images(
            images,
            centers,
            track_eligibility=False,
        )
        reversed_result = model.run_spike_events(
            normal.l23_spike_events.flip(1),
            track_eligibility=False,
        )
        normal_synaptic = torch.einsum(
            "blf,clf->bc",
            normal.l23_spike_events.float(),
            weights,
        )
        reversed_synaptic = torch.einsum(
            "blf,clf->bc",
            normal.l23_spike_events.flip(1).float(),
            weights,
        )
        normal_margin = normal_synaptic[:, 0] - normal_synaptic[:, 1]
        reversed_margin = reversed_synaptic[:, 0] - reversed_synaptic[:, 1]
        normal_scores.append(
            _lexicographic_score(
                normal.decision_spike_counts,
                normal_margin,
            )
        )
        reversed_scores.append(
            _lexicographic_score(
                reversed_result.decision_spike_counts,
                reversed_margin,
            )
        )
        normal_counts_all.append(normal.decision_spike_counts)
        reversed_counts_all.append(reversed_result.decision_spike_counts)

    normal_score = torch.cat(normal_scores)
    reversed_score = torch.cat(reversed_scores)
    normal_counts = torch.cat(normal_counts_all)
    reversed_counts = torch.cat(reversed_counts_all)
    selected = int(normal_score.argmax())
    boxes = candidate_object_boxes(views, geometry, (width, height))
    return {
        "image_path": str(image_path.relative_to(ROOT)),
        "image_width": width,
        "image_height": height,
        "n_candidate_views": len(views),
        "summaries": _summaries(normal_score, reversed_score),
        "selected_index": selected,
        "selected_view": [int(value) for value in views[selected]],
        "selected_box": [float(value) for value in boxes[selected]],
        "selected_decision_counts": normal_counts[selected].tolist(),
        "normal_candidate_scores": normal_score,
        "reversed_candidate_scores": reversed_score,
        "normal_candidate_counts": normal_counts,
        "reversed_candidate_counts": reversed_counts,
    }


def _scan_paths(
    model,
    paths: Sequence[Path],
    category: str,
    transform,
    centers: Sequence[Sequence[int]],
    geometry: SearchGeometry,
    *,
    batch_size: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    records: List[Dict[str, object]] = []
    evidence: Dict[str, object] = {}
    started = time.time()
    for index, path in enumerate(paths, start=1):
        result = _scan_image(
            model,
            path,
            transform,
            centers,
            geometry,
            batch_size=batch_size,
        )
        result["category"] = category
        records.append(
            {
                key: value
                for key, value in result.items()
                if not torch.is_tensor(value)
            }
        )
        evidence[str(path.relative_to(ROOT))] = {
            key: value
            for key, value in result.items()
            if torch.is_tensor(value)
        }
        elapsed = time.time() - started
        remaining = elapsed / index * (len(paths) - index)
        print(
            f"[{category} {index:3d}/{len(paths)}] "
            f"score={result['summaries']['normal_max']:.2f}, "
            f"{remaining / 60:.1f} min remaining",
            flush=True,
        )
    return records, evidence


def _positive_records(
    localization_report: Dict[str, object],
    localization_evidence: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    records: List[Dict[str, object]] = []
    evidence: Dict[str, object] = {}
    by_index = {
        int(record["image_index"]): record
        for record in localization_report["records"]
    }
    for index, item in localization_evidence.items():
        index = int(index)
        source = by_index[index]
        normal_counts = item["normal_decision_counts"]
        reversed_counts = item["reversed_decision_counts"]
        normal_score = _lexicographic_score(
            normal_counts,
            item["normal_synaptic_margin"],
        )
        reversed_score = _lexicographic_score(
            reversed_counts,
            item["reversed_synaptic_margin"],
        )
        selected = int(normal_score.argmax())
        normal_method = source["methods"]["normal"]
        record = {
            "image_path": source["image_path"],
            "image_width": source["image_width"],
            "image_height": source["image_height"],
            "category": "Faces",
            "label": 1,
            "n_candidate_views": source["n_candidate_views"],
            "summaries": _summaries(normal_score, reversed_score),
            "selected_index": selected,
            "selected_view": normal_method["candidate_view"],
            "selected_box": normal_method["box"],
            "ground_truth_box": source["ground_truth_box"],
            "selected_iou": normal_method["iou"],
            "localized": bool(normal_method["iou"] >= 0.5),
            "selected_decision_counts": (
                normal_counts[selected].tolist()
            ),
        }
        records.append(record)
        evidence[record["image_path"]] = {
            "normal_candidate_scores": normal_score,
            "reversed_candidate_scores": reversed_score,
            "normal_candidate_counts": normal_counts,
            "reversed_candidate_counts": reversed_counts,
        }
    return records, evidence


def _balanced_distractors(
    category_dir: Path,
    count: int,
) -> List[Path]:
    category_paths = {
        category: sorted((category_dir / category).glob("*.jpg"))
        for category in DISTRACTOR_CATEGORIES
    }
    selected: List[Path] = []
    offset = 0
    while len(selected) < count:
        added = False
        for category in DISTRACTOR_CATEGORIES:
            paths = category_paths[category]
            if offset < len(paths):
                selected.append(paths[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    return selected


def _development_paths(
    category_dir: Path,
    count_per_class: int,
) -> Tuple[List[Path], List[Path]]:
    faces = [
        category_dir / "Faces" / f"image_{index:04d}.jpg"
        for index in range(1, count_per_class + 1)
    ]
    motors = [
        category_dir / "Motorbikes" / f"image_{index:04d}.jpg"
        for index in range(1, count_per_class + 1)
    ]
    return faces, motors


def _arrays(
    records: Sequence[Dict[str, object]],
    rule: str,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(record["label"]) for record in records])
    scores = np.asarray(
        [float(record["summaries"][rule]) for record in records]
    )
    return labels, scores


def _select_rule(
    development: Sequence[Dict[str, object]],
) -> Tuple[str, Dict[str, float]]:
    metrics = {}
    for rule in RULE_NAMES:
        labels, scores = _arrays(development, rule)
        metrics[rule] = float(average_precision_score(labels, scores))
    selected = max(RULE_NAMES, key=lambda name: (metrics[name], -RULE_NAMES.index(name)))
    return selected, metrics


def _select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    unique = np.unique(scores)
    candidates = np.concatenate(
        [
            [np.nextafter(unique.max(), np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique.min(), -np.inf)],
        ]
    )
    best = None
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        balanced = float(balanced_accuracy_score(labels, predictions))
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        candidate = (balanced, f1, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[2], {
        "balanced_accuracy": best[0],
        "f1": best[1],
    }


def _bootstrap_ap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    draws: int = 5000,
) -> List[float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(draws):
        indices = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        values.append(average_precision_score(labels[indices], scores[indices]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _detection_average_precision(
    records: Sequence[Dict[str, object]],
    scores: np.ndarray,
) -> float:
    order = np.argsort(-scores, kind="stable")
    true_positive = np.asarray(
        [
            int(
                records[index]["label"] == 1
                and records[index].get("localized", False)
            )
            for index in order
        ]
    )
    false_positive = 1 - true_positive
    cumulative_tp = np.cumsum(true_positive)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + np.cumsum(false_positive),
        1,
    )
    n_positive_images = sum(record["label"] == 1 for record in records)
    return float((precision * true_positive).sum() / max(n_positive_images, 1))


def _froc(
    records: Sequence[Dict[str, object]],
    scores: np.ndarray,
) -> Dict[str, List[float]]:
    thresholds = np.r_[
        np.nextafter(scores.max(), np.inf),
        np.unique(scores)[::-1],
        np.nextafter(scores.min(), -np.inf),
    ]
    n_positive = sum(record["label"] == 1 for record in records)
    n_images = len(records)
    sensitivity = []
    false_positives_per_image = []
    for threshold in thresholds:
        active = scores >= threshold
        tp = 0
        fp = 0
        for enabled, record in zip(active, records):
            if not enabled:
                continue
            if record["label"] == 1 and record.get("localized", False):
                tp += 1
            else:
                fp += 1
        sensitivity.append(tp / max(n_positive, 1))
        false_positives_per_image.append(fp / max(n_images, 1))
    return {
        "thresholds": thresholds.tolist(),
        "sensitivity": sensitivity,
        "false_positives_per_image": false_positives_per_image,
    }


def _evaluate(
    records: Sequence[Dict[str, object]],
    selected_rule: str,
    threshold: float,
    *,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    labels, scores = _arrays(records, selected_rule)
    predictions = scores >= threshold
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    normal_scores = _arrays(records, "normal_max")[1]
    reversed_scores = _arrays(records, "reversed_max")[1]
    silent_scores = _arrays(records, "silent")[1]
    controls = {}
    for name, values in (
        ("selected", scores),
        ("normal_max", normal_scores),
        ("reversed_max", reversed_scores),
        ("silent", silent_scores),
    ):
        controls[name] = {
            "average_precision": float(average_precision_score(labels, values)),
            "roc_auc": (
                float(roc_auc_score(labels, values))
                if len(np.unique(values)) > 1
                else 0.5
            ),
        }
    category_false_positive_rate = {}
    for category in sorted({record["category"] for record in records}):
        if category == "Faces":
            continue
        mask = np.asarray([record["category"] == category for record in records])
        category_false_positive_rate[category] = float(predictions[mask].mean())
    report = {
        "selected_rule": selected_rule,
        "operating_threshold": threshold,
        "n_images": len(records),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "image_level_average_precision": controls["selected"]["average_precision"],
        "image_level_average_precision_95_stratified_bootstrap_ci": (
            _bootstrap_ap(labels, scores, seed=seed)
        ),
        "image_level_roc_auc": controls["selected"]["roc_auc"],
        "localization_aware_detection_average_precision": (
            _detection_average_precision(records, scores)
        ),
        "sensitivity_at_development_threshold": tp / max(tp + fn, 1),
        "specificity_at_development_threshold": tn / max(tn + fp, 1),
        "precision_at_development_threshold": tp / max(tp + fp, 1),
        "balanced_accuracy_at_development_threshold": (
            0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        ),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "false_positive_rate_by_negative_group": category_false_positive_rate,
        "causal_score_controls": controls,
        "froc": _froc(records, scores),
    }
    return report, {
        "labels": labels,
        "scores": scores,
        "normal_scores": normal_scores,
        "reversed_scores": reversed_scores,
        "silent_scores": silent_scores,
        "predictions": predictions,
    }


def _plot_results(
    records: Sequence[Dict[str, object]],
    report: Dict[str, object],
    arrays: Dict[str, np.ndarray],
    output_base: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    labels = arrays["labels"]
    for name in ("selected", "normal_max", "reversed_max", "silent"):
        values = (
            arrays["scores"] if name == "selected"
            else arrays[f"{name.split('_max')[0]}_scores"]
        )
        precision, recall, _ = precision_recall_curve(labels, values)
        ap = report["causal_score_controls"][name]["average_precision"]
        axes[0, 0].plot(
            recall,
            precision,
            color=COLORS[name],
            lw=2,
            label=f"{name.replace('_', ' ')} (AP={ap:.3f})",
        )
    axes[0, 0].axhline(labels.mean(), color="#999999", ls=":", label="Prevalence")
    axes[0, 0].set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.03),
        title="A  Face-presence precision-recall",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    for name in ("selected", "normal_max", "reversed_max"):
        values = (
            arrays["scores"] if name == "selected"
            else arrays[f"{name.split('_max')[0]}_scores"]
        )
        fpr, tpr, _ = roc_curve(labels, values)
        auc = report["causal_score_controls"][name]["roc_auc"]
        axes[0, 1].plot(
            fpr,
            tpr,
            color=COLORS[name],
            lw=2,
            label=f"{name.replace('_', ' ')} (AUC={auc:.3f})",
        )
    axes[0, 1].plot([0, 1], [0, 1], color="#999999", ls=":")
    axes[0, 1].set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1.03),
        title="B  Receiver operating characteristic",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    categories = ["Faces", "Motorbikes", "BACKGROUND_Google", "hard_distractors"]
    category_scores = [
        [
            arrays["scores"][index]
            for index, record in enumerate(records)
            if record["category"] == category
        ]
        for category in categories
    ]
    parts = axes[1, 0].violinplot(
        category_scores,
        showmedians=True,
        showextrema=False,
    )
    for body, category in zip(parts["bodies"], categories):
        body.set_facecolor(COLORS[category])
        body.set_alpha(0.75)
    axes[1, 0].axhline(
        report["operating_threshold"],
        color="#111111",
        ls="--",
        label="Development threshold",
    )
    axes[1, 0].set_xticks(
        range(1, len(categories) + 1),
        ["Faces", "Motorbikes", "Background", "Hard distractors"],
        rotation=18,
        ha="right",
    )
    axes[1, 0].set(
        ylabel="Frozen detector score",
        title="C  Score distributions under category shift",
    )
    axes[1, 0].legend(frameon=False)

    froc = report["froc"]
    axes[1, 1].plot(
        froc["false_positives_per_image"],
        froc["sensitivity"],
        color=COLORS["selected"],
        lw=2.2,
    )
    axes[1, 1].set(
        xlabel="False positive detections per image",
        ylabel="Localized-face sensitivity (IoU >= 0.5)",
        xlim=(0, max(froc["false_positives_per_image"]) * 1.02),
        ylim=(0, 1.0),
        title="D  Localization-aware FROC",
    )
    fig.suptitle(
        "Caltech Faces detection with unseen negatives and causal L5/6 controls",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_record(record: Dict[str, object]) -> Image.Image:
    image = Image.open(ROOT / record["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    box = tuple(record["selected_box"])
    draw.rectangle(box, outline=(215, 48, 39), width=max(2, image.width // 200))
    if "ground_truth_box" in record:
        draw.rectangle(
            tuple(record["ground_truth_box"]),
            outline=(0, 145, 110),
            width=max(2, image.width // 200),
        )
    return image


def _plot_examples(
    records: Sequence[Dict[str, object]],
    arrays: Dict[str, np.ndarray],
    output_base: Path,
) -> List[int]:
    labels = arrays["labels"]
    scores = arrays["scores"]
    localized = np.asarray(
        [bool(record.get("localized", False)) for record in records]
    )
    positive = np.flatnonzero((labels == 1) & localized)
    missed = np.flatnonzero((labels == 1) & ~localized)
    negative = np.flatnonzero(labels == 0)
    selection = [
        int(positive[np.argmax(scores[positive])]),
        int(missed[np.argmax(scores[missed])]) if len(missed) else int(positive[0]),
        int(negative[np.argmax(scores[negative])]),
        int(negative[np.argmin(scores[negative])]),
    ]
    titles = [
        "Localized true positive",
        "Localization failure",
        "Hard false positive",
        "Easy true negative",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    for ax, index, title in zip(axes, selection, titles):
        image = _draw_record(records[index])
        ax.imshow(image)
        ax.set_title(
            f"{title}\n{records[index]['category']}, score={scores[index]:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle(
        "Frozen detector examples (green: ground truth, red: prediction)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selection


def _write_csv(
    records: Sequence[Dict[str, object]],
    arrays: Dict[str, np.ndarray],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "category",
                "label",
                "score",
                "prediction",
                "selected_iou",
                "localized",
                "normal_max",
                "reversed_max",
            ]
        )
        for index, record in enumerate(records):
            writer.writerow(
                [
                    record["image_path"],
                    record["category"],
                    record["label"],
                    arrays["scores"][index],
                    int(arrays["predictions"][index]),
                    record.get("selected_iou", ""),
                    int(record.get("localized", False)),
                    record["summaries"]["normal_max"],
                    record["summaries"]["reversed_max"],
                ]
            )


def run(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category_dir = (
        Path(args.dataset_root).resolve() / "101_ObjectCategories"
    )
    localization_dir = Path(args.localization_dir).resolve()
    localization_report_path = (
        localization_dir / "caltech_faces_localization_results.json"
    )
    localization_evidence_path = (
        localization_dir / "caltech_faces_localization_evidence.pt"
    )
    localization_report = json.loads(
        localization_report_path.read_text(encoding="utf-8")
    )
    localization_payload = torch.load(
        localization_evidence_path,
        map_location="cpu",
        weights_only=False,
    )
    geometry = _geometry_from_report(localization_report)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    centers = checkpoint_payload["metadata"]["centers"]
    model = load_fully_spiking_column(str(checkpoint), device=args.device)
    transform = build_transform(Config.caltech())

    positive_records, positive_evidence = _positive_records(
        localization_report,
        localization_payload["evidence"],
    )

    development_faces, development_motors = _development_paths(
        category_dir,
        args.development_per_class,
    )
    dev_face_records, dev_face_evidence = _scan_paths(
        model,
        development_faces,
        "Faces",
        transform,
        centers,
        geometry,
        batch_size=args.batch_size,
    )
    dev_motor_records, dev_motor_evidence = _scan_paths(
        model,
        development_motors,
        "Motorbikes",
        transform,
        centers,
        geometry,
        batch_size=args.batch_size,
    )
    for record in dev_face_records:
        record["label"] = 1
    for record in dev_motor_records:
        record["label"] = 0
    development = dev_face_records + dev_motor_records
    selected_rule, development_rule_ap = _select_rule(development)
    dev_labels, dev_scores = _arrays(development, selected_rule)
    threshold, development_threshold_metrics = _select_threshold(
        dev_labels,
        dev_scores,
    )

    motor_paths = [
        category_dir / "Motorbikes" / f"image_{index:04d}.jpg"
        for index in range(201, 201 + args.motorbike_test_count)
    ]
    background_paths = sorted(
        (category_dir / "BACKGROUND_Google").glob("*.jpg")
    )[:args.background_test_count]
    distractor_paths = _balanced_distractors(
        category_dir,
        args.distractor_test_count,
    )
    motor_records, motor_evidence = _scan_paths(
        model,
        motor_paths,
        "Motorbikes",
        transform,
        centers,
        geometry,
        batch_size=args.batch_size,
    )
    background_records, background_evidence = _scan_paths(
        model,
        background_paths,
        "BACKGROUND_Google",
        transform,
        centers,
        geometry,
        batch_size=args.batch_size,
    )
    distractor_records, distractor_evidence = _scan_paths(
        model,
        distractor_paths,
        "hard_distractors",
        transform,
        centers,
        geometry,
        batch_size=args.batch_size,
    )
    for record in motor_records + background_records + distractor_records:
        record["label"] = 0
    test_records = (
        positive_records
        + motor_records
        + background_records
        + distractor_records
    )
    evaluation, arrays = _evaluate(
        test_records,
        selected_rule,
        threshold,
        seed=args.seed,
    )

    figure_dir = output_dir / "figures"
    _plot_results(
        test_records,
        evaluation,
        arrays,
        figure_dir / "11_caltech_faces_detection",
    )
    example_indices = _plot_examples(
        test_records,
        arrays,
        figure_dir / "12_caltech_faces_detection_examples",
    )
    _write_csv(
        test_records,
        arrays,
        output_dir / "caltech_faces_detection_predictions.csv",
    )
    torch.save(
        {
            "kind": "caltech_faces_detection_evidence",
            "development": {
                **dev_face_evidence,
                **dev_motor_evidence,
            },
            "test": {
                **positive_evidence,
                **motor_evidence,
                **background_evidence,
                **distractor_evidence,
            },
        },
        output_dir / "caltech_faces_detection_evidence.pt",
    )
    report = {
        "experiment": "caltech_faces_source_disjoint_detection",
        "backend": checkpoint_payload["backend"],
        "gradient_free": checkpoint_payload["gradient_free"],
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": _sha256(checkpoint),
            "decision_learning_rule": checkpoint_payload.get(
                "decision_learning_rule"
            ),
            "learning_replays": checkpoint_payload.get("learning_replays"),
        },
        "protocol": {
            "candidate_views_per_image": 91,
            "score_definition": (
                "decision spike margin + 0.49*tanh(gated synaptic margin); "
                "the synaptic term breaks ties but cannot overturn one spike"
            ),
            "candidate_rules": {
                name: RULE_LABELS[name] for name in RULE_NAMES
            },
            "development_source_indices": list(
                range(1, args.development_per_class + 1)
            ),
            "selected_rule": selected_rule,
            "development_rule_average_precision": development_rule_ap,
            "development_threshold": threshold,
            "development_threshold_metrics": development_threshold_metrics,
            "test_sources": {
                "Faces": "indices 201-435; source-disjoint from model input",
                "Motorbikes": (
                    f"indices 201-{200 + args.motorbike_test_count}; "
                    "source-disjoint from model input"
                ),
                "BACKGROUND_Google": (
                    f"first {args.background_test_count}; unseen category"
                ),
                "hard_distractors": {
                    "count": args.distractor_test_count,
                    "categories": list(DISTRACTOR_CATEGORIES),
                    "model_training_exposure": 0,
                },
            },
            "test_data_used_for_rule_or_threshold_selection": False,
        },
        "evaluation": evaluation,
        "test_group_counts": {
            category: sum(
                record["category"] == category for record in test_records
            )
            for category in (
                "Faces",
                "Motorbikes",
                "BACKGROUND_Google",
                "hard_distractors",
            )
        },
        "qualitative_example_indices": example_indices,
        "artifacts": {
            "predictions": "caltech_faces_detection_predictions.csv",
            "evidence": "caltech_faces_detection_evidence.pt",
            "figures": [
                "figures/11_caltech_faces_detection.pdf",
                "figures/12_caltech_faces_detection_examples.pdf",
            ],
        },
    }
    (output_dir / "caltech_faces_detection_results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="data/caltech/caltech101",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/joint_training/caltech_all_acquisition_fixation_gated/"
            "best_fully_spiking_column.pt"
        ),
    )
    parser.add_argument(
        "--localization-dir",
        default=(
            "outputs/manuscript_fully_spiking/"
            "caltech_faces_localization_no_replay"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/manuscript_fully_spiking/"
            "caltech_faces_detection"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--development-per-class", type=int, default=60)
    parser.add_argument("--motorbike-test-count", type=int, default=235)
    parser.add_argument("--background-test-count", type=int, default=150)
    parser.add_argument("--distractor-test-count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
