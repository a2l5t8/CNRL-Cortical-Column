"""Reproduce manuscript experiments for the fully spiking cortical column.

The script reloads the immutable Caltech and MNIST checkpoints, reruns raw-image
inference, reconstructs all event-level controls, records population spikes,
and performs causal assays in both directions of reference-frame binding.
Every model used here is scheduled by CoNeX/PyMoNNtorch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scc.config import Config
from scc.data.pipeline import make_caltech_loaders, make_mnist_loaders
from scc.fully_spiking import FullySpikingCorticalColumn, load_fully_spiking_column
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
DATASET_LABEL = {"caltech": "Caltech Faces/Motorbikes", "mnist": "MNIST"}
CLASS_NAMES = {
    "caltech": ["Face", "Motorbike"],
    "mnist": [str(index) for index in range(10)],
}
STAGES = ["initial", "early", "middle", "final"]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _interval(successes: int, trials: int, alpha: float = 0.05) -> List[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    low = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, trials - successes + 1)
    )
    high = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return [low, high]


def _mcnemar(
    reference_predictions: torch.Tensor,
    control_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, object]:
    reference_correct = reference_predictions.eq(targets)
    control_correct = control_predictions.eq(targets)
    reference_only = int((reference_correct & ~control_correct).sum())
    control_only = int((~reference_correct & control_correct).sum())
    discordant = reference_only + control_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(
            min(reference_only, control_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    return {
        "reference_correct_control_wrong": reference_only,
        "reference_wrong_control_correct": control_only,
        "exact_two_sided_p": p_value,
    }


def _bootstrap_mean_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    seed: int,
    draws: int = 10000,
) -> List[float]:
    difference = (first - second).float().cpu()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(difference),
        (draws, len(difference)),
        generator=generator,
    )
    means = difference[indices].mean(dim=1)
    return [
        float(torch.quantile(means, 0.025)),
        float(torch.quantile(means, 0.975)),
    ]


def _paths(dataset: str) -> Dict[str, Path]:
    checkpoint_directory = (
        ROOT
        / "outputs"
        / "joint_training"
        / f"{dataset}_all_acquisition_fixation_gated"
    )
    event_directory = ROOT / "outputs" / "best_models" / (
        f"{dataset}_fully_spiking_column_no_replay"
    )
    return {
        "directory": checkpoint_directory,
        "checkpoint": checkpoint_directory / "best_fully_spiking_column.pt",
        "events": event_directory / "cortical_spike_events.pt",
        "report": checkpoint_directory / "fully_spiking_report.json",
        "audit": checkpoint_directory / "fully_spiking_audit.json",
        "l4_stages": (
            ROOT
            / "outputs"
            / "l4_kernel_study"
            / dataset
            / "l4_stage_weights.pt"
        ),
        "l4_report": (
            ROOT
            / "outputs"
            / "l4_kernel_study"
            / dataset
            / "l4_kernel_study.json"
        ),
    }


def _validation_dataset(dataset: str, seed: int = 42):
    cfg = Config.caltech() if dataset == "caltech" else Config.mnist()
    if dataset == "caltech":
        _, loader = make_caltech_loaders(
            cfg,
            batch_size=64,
            seed=seed,
            download=True,
        )
    else:
        _, loader = make_mnist_loaders(cfg, batch_size=64, seed=seed)
    return cfg, loader.dataset


def _run_raw_validation(
    model,
    dataset,
    centers: Sequence[Tuple[int, int]],
    expected_targets: torch.Tensor,
    *,
    batch_size: int,
) -> Dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sensory_spikes = 0.0
    l4_spikes = 0.0
    start_time = time.time()
    n_expected = len(expected_targets)
    seen = 0
    for images, labels in loader:
        if seen >= n_expected:
            break
        take = min(len(images), n_expected - seen)
        images = images[:take]
        labels = labels[:take]
        result = model.run_images(
            images,
            centers,
            learn_apical=False,
            track_eligibility=False,
        )
        predictions.append(result.predictions)
        targets.append(labels.long().cpu())
        sensory_spikes += float(result.sensory_spike_count.sum())
        l4_spikes += float(result.l4_spike_count.sum())
        seen += take
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    if not torch.equal(target, expected_targets):
        raise RuntimeError("validation order differs from the cortical event cache")
    successes = int(prediction.eq(target).sum())
    return {
        "predictions": prediction,
        "targets": target,
        "accuracy": successes / len(target),
        "accuracy_95_ci": _interval(successes, len(target)),
        "correct": successes,
        "n": len(target),
        "mean_sensory_spikes": sensory_spikes / len(target),
        "mean_l4_spikes": l4_spikes / len(target),
        "seconds": time.time() - start_time,
    }


def _event_condition(
    model,
    events: torch.Tensor,
    *,
    mode: str,
    batch_size: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    predictions: List[torch.Tensor] = []
    traces: List[torch.Tensor] = []
    counts: List[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size]
        silence_l56 = mode == "silent"
        if mode == "reversed":
            batch = batch.flip(1)
        elif mode == "shuffled":
            shuffled = torch.empty_like(batch)
            for index in range(len(batch)):
                order = torch.randperm(
                    batch.shape[1],
                    generator=generator,
                )
                shuffled[index] = batch[index, order]
            batch = shuffled
        result = model.run_spike_events(
            batch,
            track_eligibility=False,
            silence_l56=silence_l56,
        )
        predictions.append(result.predictions)
        traces.append(result.decision_trace)
        counts.append(result.decision_spike_counts)
    return {
        "predictions": torch.cat(predictions),
        "decision_trace": torch.cat(traces),
        "decision_spike_counts": torch.cat(counts),
    }


def _control_statistics(
    model,
    events: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, torch.Tensor]]]:
    outputs: Dict[str, Dict[str, torch.Tensor]] = {}
    for offset, mode in enumerate(["normal", "reversed", "shuffled", "silent"]):
        outputs[mode] = _event_condition(
            model,
            events,
            mode=mode,
            batch_size=batch_size,
            seed=seed + 100 * offset,
        )
    report: Dict[str, object] = {}
    reference = outputs["normal"]["predictions"]
    for mode, output in outputs.items():
        prediction = output["predictions"]
        correct = int(prediction.eq(targets).sum())
        per_saccade = (
            output["decision_trace"]
            .argmax(dim=2)
            .eq(targets[:, None])
            .float()
            .mean(dim=0)
        )
        report[mode] = {
            "accuracy": correct / len(targets),
            "accuracy_95_ci": _interval(correct, len(targets)),
            "correct": correct,
            "n": len(targets),
            "accuracy_by_saccade": per_saccade,
            "confusion": confusion_matrix(
                targets.numpy(),
                prediction.numpy(),
                labels=list(range(model.n_classes)),
            ),
        }
        if mode != "normal":
            report[mode]["mcnemar_vs_normal"] = _mcnemar(
                reference,
                prediction,
                targets,
            )
    return report, outputs


def _reciprocal_assay(
    model,
    dataset: str,
    train_events: torch.Tensor,
    val_events: torch.Tensor,
    *,
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    if float(model.l23_l56_sg.weights.abs().sum().detach().cpu()) == 0.0:
        model.fit_reciprocal_spike_events(
            train_events,
            epochs=1,
            batch_size=256,
            seed=seed,
        )
    cues = val_events.reshape(-1, val_events.shape[2])
    targets = torch.arange(model.n_locations).repeat(len(val_events))
    output = model.predict_l56_from_l23_events(
        cues,
        batch_size=512,
        gain=1.0,
        record_first=True,
    )
    spike_ranks = output["spike_counts"].argsort(dim=1, descending=True)
    current_ranks = output["currents"].argsort(dim=1, descending=True)

    shuffled = FullySpikingCorticalColumn(
        model.n_classes,
        model.n_locations,
        model.l4_kernels,
        copy.deepcopy(model.cfg),
        device,
    )
    shuffled.fit_reciprocal_spike_events(
        train_events,
        epochs=1,
        batch_size=256,
        shuffle_targets=True,
        seed=seed + 71,
    )
    shuffled_output = shuffled.predict_l56_from_l23_events(
        cues,
        batch_size=512,
        gain=1.0,
    )

    matched = output["currents"].gather(1, targets[:, None]).squeeze(1)
    other = (
        output["currents"].sum(dim=1) - matched
    ) / float(model.n_locations - 1)
    correct = output["predictions"].eq(targets)
    shuffled_correct = shuffled_output["predictions"].eq(targets)
    top_k = {
        f"top_{k}_accuracy": float(
            (spike_ranks[:, :k] == targets[:, None]).any(dim=1).float().mean()
        )
        for k in (1, 2, 3, 5)
    }
    current_top_k = {
        f"current_rank_top_{k}_accuracy": float(
            (current_ranks[:, :k] == targets[:, None]).any(dim=1).float().mean()
        )
        for k in (1, 2, 3, 5)
    }
    report = {
        "backend": model.backend,
        "gradient_free": True,
        "integrated_in_fully_spiking_column": True,
        "plasticity": "local L2/3-to-L5/6 pre/post coactivity with synaptic scaling",
        "decision_loop_reciprocal_gain": float(model.cfg.reciprocal_gain),
        "query_gain": 1.0,
        "n_trials": len(targets),
        "top_1_accuracy": float(correct.float().mean()),
        "top_1_95_ci": _interval(int(correct.sum()), len(correct)),
        **top_k,
        **current_top_k,
        "shuffled_binding_accuracy": float(shuffled_correct.float().mean()),
        "shuffled_binding_95_ci": _interval(
            int(shuffled_correct.sum()),
            len(shuffled_correct),
        ),
        "chance_accuracy": 1.0 / model.n_locations,
        "matched_current": float(matched.mean()),
        "other_location_current": float(other.mean()),
        "matched_minus_other": float((matched - other).mean()),
        "matched_minus_other_95_bootstrap_ci": _bootstrap_mean_difference(
            matched,
            other,
            seed=seed + 17,
        ),
        "hard_spikes_per_trial": float(
            output["spike_counts"].sum(dim=1).float().mean()
        ),
        "structural_checks": {
            "l23_is_spiking": "Spiking" in model.l23_ng.tags,
            "l56_is_spiking": "Spiking" in model.l56_ng.tags,
            "physical_l23_to_l56_synapse": "Reciprocal" in model.l23_l56_sg.tags,
            "weights_require_grad": bool(model.l23_l56_sg.weights.requires_grad),
            "cue_events_are_binary": cues.dtype == torch.bool,
        },
    }
    checkpoint_path = output_dir / f"{dataset}_reciprocal_l23_l56.pt"
    torch.save(
        {
            "kind": "integrated_l23_l56_reciprocal_binding",
            "backend": model.backend,
            "weights": model.l23_l56_sg.weights.detach().cpu(),
            "cfg": dict(vars(model.cfg)),
        },
        checkpoint_path,
    )
    tensors = {
        "predictions": output["predictions"],
        "spike_counts": output["spike_counts"],
        "currents": output["currents"],
        "targets": targets,
        "shuffled_predictions": shuffled_output["predictions"],
        "first_raster": output["first_raster"],
        "weights": model.l23_l56_sg.weights.detach().cpu(),
    }
    return report, tensors


def _apical_assay(
    model,
    observed_events: torch.Tensor,
    *,
    seed: int,
    settle_steps: int = 8,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    observed = observed_events.float().mean(dim=0).to(model.device)
    primed_membrane: List[torch.Tensor] = []
    primed_spikes: List[torch.Tensor] = []
    for location in range(model.n_locations):
        model.reset_trial(1)
        model.net.apical_learning_enabled = False
        model.net.eligibility_enabled = False
        model._set_motor_location(location, pulse=True)
        model.net.cortical_phase = "l4_release"
        model._simulate(1)
        model._set_motor_location(location, pulse=False)
        model._simulate(settle_steps - 1)
        primed_membrane.append(model.l23_ng.v[0].detach().cpu())
        model.net.cortical_phase = "l23_competition"
        model._simulate(1)
        primed_spikes.append(model.l23_ng.spikes[0].detach().cpu())
    primed = torch.stack(primed_membrane).clamp(min=0)
    prime_spikes = torch.stack(primed_spikes)
    observed = observed.cpu()
    cosine = F.normalize(primed, dim=1) @ F.normalize(observed, dim=1).T
    matched = cosine.diag()
    reversed_match = cosine[
        torch.arange(model.n_locations),
        torch.arange(model.n_locations - 1, -1, -1),
    ]
    k = min(model.cfg.l23_k, model.n_l23_features)
    primed_top = primed.topk(k, dim=1).indices
    observed_top = observed.topk(k, dim=1).indices
    recall = (
        primed_top.unsqueeze(2) == observed_top.unsqueeze(1)
    ).any(dim=2).float().mean(dim=1)
    report = {
        "intervention": (
            "L5/6 motor spike with visual/proximal input clamped to zero"
        ),
        "settle_steps": settle_steps,
        "matched_cosine": float(matched.mean()),
        "reversed_location_cosine": float(reversed_match.mean()),
        "matched_minus_reversed": float((matched - reversed_match).mean()),
        "matched_minus_reversed_95_bootstrap_ci": _bootstrap_mean_difference(
            matched,
            reversed_match,
            seed=seed + 29,
        ),
        f"top_{k}_feature_recall": float(recall.mean()),
        f"top_{k}_feature_recall_by_location": recall,
        "primed_spikes_per_location": prime_spikes.sum(dim=1),
        "mean_primed_membrane": float(primed.mean()),
        "max_primed_membrane": float(primed.max()),
    }
    tensors = {
        "primed_membrane": primed,
        "primed_spikes": prime_spikes,
        "observed_probability": observed,
        "cosine": cosine,
        "matched": matched,
        "reversed": reversed_match,
    }
    return report, tensors


def _target_margin(trace: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_trace = trace.gather(
        2,
        targets[:, None, None].expand(-1, trace.shape[1], 1),
    ).squeeze(2)
    alternatives = trace.clone()
    alternatives.scatter_(
        2,
        targets[:, None, None].expand(-1, trace.shape[1], 1),
        -torch.inf,
    )
    return target_trace - alternatives.max(dim=2).values


def _select_ambiguity(
    controls: Dict[str, Dict[str, torch.Tensor]],
    targets: torch.Tensor,
) -> int:
    normal = controls["normal"]
    predictions = normal["decision_trace"].argmax(dim=2)
    candidates = torch.where(
        predictions[:, 0].ne(targets)
        & predictions[:, -1].eq(targets)
    )[0]
    if len(candidates) == 0:
        candidates = torch.where(predictions[:, -1].eq(targets))[0]
    if len(candidates) == 0:
        return 0

    target = targets[candidates]
    normal_margin = _target_margin(normal["decision_trace"][candidates], target)
    reversed_margin = _target_margin(
        controls["reversed"]["decision_trace"][candidates],
        target,
    )
    shuffled_margin = _target_margin(
        controls["shuffled"]["decision_trace"][candidates],
        target,
    )
    silent_margin = _target_margin(
        controls["silent"]["decision_trace"][candidates],
        target,
    )

    first_positive = torch.full(
        (len(candidates),),
        normal_margin.shape[1],
        dtype=torch.float32,
        device=normal_margin.device,
    )
    positive = normal_margin.gt(0)
    for col in range(normal_margin.shape[1]):
        first_positive = torch.where(
            positive[:, col] & first_positive.eq(normal_margin.shape[1]),
            torch.full_like(first_positive, float(col)),
            first_positive,
        )
    stable_positive = torch.full_like(first_positive, float(normal_margin.shape[1]))
    for col in range(normal_margin.shape[1]):
        stable_from_col = positive[:, col:].all(dim=1)
        stable_positive = torch.where(
            stable_from_col & stable_positive.eq(normal_margin.shape[1]),
            torch.full_like(stable_positive, float(col)),
            stable_positive,
        )

    early_depth = (-normal_margin[:, :3].min(dim=1).values).clamp(min=0)
    final_margin = normal_margin[:, -1]
    improvement = final_margin - normal_margin[:, 0]
    crossing_bonus = first_positive.clamp(max=5.0)
    stable_crossing_bonus = stable_positive.clamp(max=normal_margin.shape[1] - 1)
    control_gap = final_margin - torch.stack(
        [
            reversed_margin[:, -1],
            shuffled_margin[:, -1],
            silent_margin[:, -1],
        ],
        dim=1,
    ).max(dim=1).values
    monotone_gain = normal_margin.diff(dim=1).clamp(min=0).sum(dim=1)

    score = (
        2.5 * early_depth
        + 2.0 * improvement
        + 1.5 * final_margin
        + 1.0 * control_gap
        + 0.7 * crossing_bonus
        + 1.3 * stable_crossing_bonus
        + 0.35 * monotone_gain
    )
    score = torch.where(final_margin.gt(0), score, score - 1000.0)
    return int(candidates[score.argmax()])


def _activity_summary(record: Dict[str, object]) -> Dict[str, object]:
    report = {}
    for layer in ["sensory", "l4", "l23", "l56", "decision"]:
        spikes = record[f"{layer}_spikes"]
        report[layer] = {
            "neurons": int(spikes.shape[1]),
            "timesteps": int(spikes.shape[0]),
            "total_spikes": int(spikes.sum()),
            "mean_spikes_per_neuron_per_step": float(spikes.float().mean()),
            "active_neurons": int(spikes.any(dim=0).sum()),
        }
    return report


def _cross_image_selection(
    dataset: str,
    val_events: torch.Tensor,
    val_y: torch.Tensor,
    reciprocal: Dict[str, torch.Tensor],
    val_dataset,
) -> Dict[str, object]:
    n_samples, n_locations, _ = val_events.shape
    predictions = reciprocal["predictions"].reshape(n_samples, n_locations)
    currents = reciprocal["currents"].reshape(
        n_samples,
        n_locations,
        n_locations,
    )
    location_candidates = (
        [0, 2, 6]
        if dataset == "mnist"
        else list(range(n_locations))
    )
    best = None
    for cls in torch.unique(val_y).tolist():
        class_indices = torch.where(val_y == int(cls))[0]
        if len(class_indices) < 5:
            continue
        for location in location_candidates:
            accuracy = float(
                predictions[class_indices, location]
                .eq(location)
                .float()
                .mean()
            )
            candidate = (accuracy, len(class_indices), int(cls), location)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    _, _, cls, location = best
    candidates = torch.where(
        val_y.eq(cls) & predictions[:, location].eq(location)
    )[0]
    if len(candidates) < 5:
        candidates = torch.where(val_y.eq(cls))[0]
    location_current = currents[candidates, location]
    competitor = location_current.clone()
    competitor[:, location] = -torch.inf
    confidence = (
        location_current[:, location] - competitor.max(dim=1).values
    )
    anchor = int(candidates[confidence.argmax()])
    cue = F.normalize(val_events[anchor, location].float(), dim=0)
    similarities = (
        F.normalize(val_events[candidates, location].float(), dim=1) @ cue
    )
    order = similarities.argsort(descending=True)
    selected = [int(candidates[index]) for index in order[:6]]
    images = [val_dataset[index][0].float() for index in selected]
    labels = [int(val_dataset[index][1]) for index in selected]
    if labels != [cls] * len(labels):
        raise RuntimeError("cross-image class selection is inconsistent")
    return {
        "class": cls,
        "class_name": CLASS_NAMES[dataset][cls],
        "location": location,
        "indices": selected,
        "anchor_index": anchor,
        "similarity_to_anchor": [
            float(
                F.cosine_similarity(
                    val_events[index, location].float(),
                    val_events[anchor, location].float(),
                    dim=0,
                )
            )
            for index in selected
        ],
        "predicted_locations": [
            int(predictions[index, location]) for index in selected
        ],
        "currents": currents[selected, location],
        "images": images,
    }


def _l4_reproduction_audit(dataset: str) -> Dict[str, object]:
    return {
        "available": False,
        "reason": (
            "The legacy standalone L4 reproduction runner was removed during "
            "the fully-spiking-only refactor. Reported L4 evolution metrics "
            "are read from the archived manuscript artifact bundle."
        ),
    }


def _run_dataset(
    dataset: str,
    output_dir: Path,
    *,
    device: torch.device,
    seed: int,
    raw_batch_size: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    paths = _paths(dataset)
    payload = torch.load(paths["events"], map_location="cpu", weights_only=False)
    checkpoint_payload = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    model = load_fully_spiking_column(str(paths["checkpoint"]), device=device)
    centers = [tuple(center) for center in payload["centers"]]
    cfg, val_dataset = _validation_dataset(dataset, seed=seed)
    val_y = payload["val_y"].long()

    raw = _run_raw_validation(
        model,
        val_dataset,
        centers,
        val_y,
        batch_size=raw_batch_size,
    )
    control_report, control_outputs = _control_statistics(
        model,
        payload["val_events"].bool(),
        val_y,
        batch_size=512,
        seed=seed,
    )
    reciprocal_report, reciprocal_tensors = _reciprocal_assay(
        model,
        dataset,
        payload["train_events"].bool(),
        payload["val_events"].bool(),
        device=device,
        output_dir=output_dir,
        seed=seed,
    )
    apical_report, apical_tensors = _apical_assay(
        model,
        payload["train_events"].bool(),
        seed=seed,
    )
    ambiguity_index = _select_ambiguity(control_outputs, val_y)
    ambiguity_image, ambiguity_label = val_dataset[ambiguity_index]
    if int(ambiguity_label) != int(val_y[ambiguity_index]):
        raise RuntimeError("ambiguous raw image label is inconsistent")
    raster = model.record_image_dynamics(
        ambiguity_image,
        centers,
    )
    cross_image = _cross_image_selection(
        dataset,
        payload["val_events"].bool(),
        val_y,
        reciprocal_tensors,
        val_dataset,
    )
    l4_report = json.loads(paths["l4_report"].read_text(encoding="utf-8"))
    report = {
        "dataset": dataset,
        "label": DATASET_LABEL[dataset],
        "backend": model.backend,
        "checkpoint": paths["checkpoint"],
        "cortical_event_cache": paths["events"],
        "n_train_events": len(payload["train_events"]),
        "n_validation_events": len(payload["val_events"]),
        "n_classes": model.n_classes,
        "n_locations": model.n_locations,
        "n_l4_features": model.n_l4_features,
        "n_l23_features": model.n_l23_features,
        "centers": centers,
        "spiking_config": dict(vars(model.cfg)),
        "training_protocol": {
            "mode": checkpoint_payload.get("metadata", {}).get(
                "training_mode", "joint"
            ),
            "binding_epochs": checkpoint_payload.get("metadata", {}).get(
                "joint_binding_epochs"
            ),
            "binding_update_steps": checkpoint_payload.get("metadata", {}).get(
                "joint_binding_update_steps"
            ),
            "learning_replays": False,
            "forced_output_spikes": False,
            "label_modulates_binding": False,
        },
        "training_history": list(checkpoint_payload.get("training_history", [])),
        "raw_image_reproduction": {
            key: value
            for key, value in raw.items()
            if key not in {"predictions", "targets"}
        },
        "event_controls": control_report,
        "raw_event_prediction_agreement": float(
            raw["predictions"]
            .eq(control_outputs["normal"]["predictions"])
            .float()
            .mean()
        ),
        "reciprocal_feature_to_frame": reciprocal_report,
        "causal_frame_to_feature": apical_report,
        "ambiguity_case": {
            "validation_index": ambiguity_index,
            "target": int(val_y[ambiguity_index]),
            "target_name": CLASS_NAMES[dataset][int(val_y[ambiguity_index])],
            "early_prediction": int(
                control_outputs["normal"]["decision_trace"][
                    ambiguity_index, 0
                ].argmax()
            ),
            "final_prediction": int(
                control_outputs["normal"]["decision_trace"][
                    ambiguity_index, -1
                ].argmax()
            ),
            "normal_trace": control_outputs["normal"]["decision_trace"][
                ambiguity_index
            ],
            "reversed_trace": control_outputs["reversed"]["decision_trace"][
                ambiguity_index
            ],
            "shuffled_trace": control_outputs["shuffled"]["decision_trace"][
                ambiguity_index
            ],
            "silent_trace": control_outputs["silent"]["decision_trace"][
                ambiguity_index
            ],
        },
        "raster_activity": _activity_summary(raster),
        "cross_image_binding": {
            key: value
            for key, value in cross_image.items()
            if key not in {"images", "currents"}
        },
        "l4_kernel_study": l4_report,
        "l4_independent_reproduction": _l4_reproduction_audit(dataset),
        "structural_checks": {
            "sensory_spiking": "Spiking" in model.sensory_ng.tags,
            "l4_spiking": "Spiking" in model.l4_ng.tags,
            "l23_spiking": "Spiking" in model.l23_ng.tags,
            "l56_spiking": "Spiking" in model.l56_ng.tags,
            "decision_spiking": "Spiking" in model.decision_ng.tags,
            "physical_apical_synapse": "Apical" in model.l56_l23_sg.tags,
            "physical_reciprocal_synapse": "Reciprocal" in model.l23_l56_sg.tags,
            "decision_location_gated": "L56Gated" in model.l23_decision_sg.tags,
            "decision_weights_gradient_free": not bool(
                model.effective_decision_weights.requires_grad
            ),
            "reciprocal_weights_gradient_free": not bool(
                model.l23_l56_sg.weights.requires_grad
            ),
            "decision_loop_reciprocal_gain": float(model.cfg.reciprocal_gain),
            "learning_replays": False,
            "forced_output_spikes": False,
        },
    }
    tensors = {
        "model": model,
        "payload": payload,
        "val_dataset": val_dataset,
        "raw": raw,
        "controls": control_outputs,
        "reciprocal": reciprocal_tensors,
        "apical": apical_tensors,
        "raster": raster,
        "ambiguity_image": ambiguity_image,
        "ambiguity_index": ambiguity_index,
        "cross_image": cross_image,
        "l4_stages": torch.load(
            paths["l4_stages"],
            map_location="cpu",
            weights_only=False,
        ),
        "l4_report": l4_report,
    }
    return report, tensors


def _box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    size: Tuple[float, float],
    text: str,
    color: str,
) -> None:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=color,
        edgecolor="white",
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        weight="bold",
        fontsize=9,
    )


def _arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    color: str = COLORS["black"],
    style: str = "-|>",
    connectionstyle: str = "arc3",
    linewidth: float = 1.7,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
        )
    )


def _plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, height = 0.145, 0.2
    _box(ax, (0.02, 0.55), (width, height), "Retinal\nspikes", COLORS["gray"])
    _box(ax, (0.21, 0.55), (width, height), "L4\nburst spikes", COLORS["blue"])
    _box(ax, (0.42, 0.55), (width, height), "L2/3\nsparse spikes", COLORS["cyan"])
    _box(ax, (0.79, 0.55), (width, height), "Decision\nLIF spikes", COLORS["red"])
    _box(ax, (0.42, 0.13), (width, height), "L5/6\nlocation spikes", COLORS["purple"])
    _box(ax, (0.68, 0.13), (0.18, height), "Eligibility trace\n+/- dopamine", COLORS["gold"])
    _arrow(ax, (0.165, 0.65), (0.21, 0.65))
    _arrow(ax, (0.355, 0.65), (0.42, 0.65))
    _arrow(ax, (0.565, 0.65), (0.79, 0.65))
    _arrow(ax, (0.49, 0.33), (0.49, 0.55), color=COLORS["purple"])
    _arrow(
        ax,
        (0.565, 0.23),
        (0.79, 0.58),
        color=COLORS["purple"],
        connectionstyle="arc3,rad=-0.18",
    )
    _arrow(
        ax,
        (0.79, 0.55),
        (0.77, 0.33),
        color=COLORS["gold"],
        connectionstyle="arc3,rad=0.12",
    )
    _arrow(
        ax,
        (0.68, 0.23),
        (0.57, 0.56),
        color=COLORS["gold"],
        connectionstyle="arc3,rad=-0.2",
    )
    _arrow(
        ax,
        (0.42, 0.2),
        (0.35, 0.2),
        color=COLORS["cyan"],
        connectionstyle="arc3,rad=0.2",
    )
    ax.text(0.275, 0.79, "STDP kernels", ha="center", color=COLORS["blue"])
    ax.text(0.61, 0.71, "L5/6-gated synapses", ha="center")
    ax.text(0.505, 0.43, "physical apical priming", ha="left", color=COLORS["purple"])
    ax.text(0.23, 0.11, "reciprocal causal assay", ha="center", color=COLORS["cyan"])
    ax.text(
        0.5,
        0.93,
        "One event-driven CoNeX/PyMoNNtorch cortical-column runtime",
        ha="center",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        0.5,
        0.87,
        "Binary spikes communicate between populations; continuous state is confined to membranes, synapses, and traces",
        ha="center",
        fontsize=9,
    )
    _save(fig, path)


def _top_neurons(spikes: torch.Tensor, limit: int) -> torch.Tensor:
    counts = spikes.sum(dim=0)
    active = torch.where(counts > 0)[0]
    if len(active) == 0:
        return torch.arange(min(limit, spikes.shape[1]))
    order = active[counts[active].argsort(descending=True)]
    return order[:limit]


def _raster(
    ax: plt.Axes,
    spikes: torch.Tensor,
    *,
    color: str,
    limit: int,
    title: str,
    locations: torch.Tensor,
) -> None:
    selected = _top_neurons(spikes, limit)
    visible = spikes[:, selected]
    time_index, neuron_index = torch.where(visible)
    ax.scatter(
        time_index.numpy(),
        neuron_index.numpy(),
        s=1.0 if len(time_index) > 10000 else 3.0,
        color=color,
        linewidths=0,
        rasterized=True,
    )
    changes = torch.where(locations[1:] != locations[:-1])[0] + 1
    for change in changes:
        ax.axvline(int(change), color="#B8B8B8", lw=0.45, zorder=0)
    ax.set_ylim(-1, max(len(selected), 1))
    ax.set_ylabel("Neuron")
    ax.set_title(title, loc="left", weight="bold", fontsize=9)


def _plot_rasters(
    path: Path,
    reports: Dict[str, Dict[str, object]],
    data: Dict[str, Dict[str, object]],
) -> None:
    layers = [
        ("sensory", COLORS["gray"], 120),
        ("l4", COLORS["blue"], 160),
        ("l23", COLORS["cyan"], 120),
        ("l56", COLORS["purple"], 20),
        ("decision", COLORS["red"], 20),
    ]
    fig, axes = plt.subplots(
        6,
        2,
        figsize=(13.2, 13.6),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1, 1, 0.75, 0.75, 1.0]},
    )
    for column, dataset in enumerate(["caltech", "mnist"]):
        record = data[dataset]["raster"]
        label = reports[dataset]["ambiguity_case"]["target_name"]
        index = reports[dataset]["ambiguity_case"]["validation_index"]
        for row, (layer, color, limit) in enumerate(layers):
            _raster(
                axes[row, column],
                record[f"{layer}_spikes"],
                color=color,
                limit=limit,
                title=(
                        f"{layer.upper()} spikes"
                    if row > 0
                    else (
                        f"{DATASET_LABEL[dataset]}: held-out sample {index}, "
                        f"target {label}"
                    )
                ),
                locations=record["location"],
            )
        ax = axes[5, column]
        for layer, color, _ in layers:
            spikes = record[f"{layer}_spikes"].float()
            rate = spikes.mean(dim=1)
            window = 5
            smooth = F.avg_pool1d(
                rate[None, None],
                kernel_size=window,
                stride=1,
                padding=window // 2,
            )[0, 0, : len(rate)]
            normalized = smooth / smooth.max().clamp(min=1e-12)
            ax.plot(normalized.numpy(), lw=1.0, color=color, label=layer.upper())
        changes = torch.where(
            record["location"][1:] != record["location"][:-1]
        )[0] + 1
        for change in changes:
            ax.axvline(int(change), color="#B8B8B8", lw=0.45)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("Normalized\nactivity")
        ax.set_xlabel("Simulation step (1 ms)")
        ax.legend(ncol=5, fontsize=7, loc="upper center")
    fig.suptitle(
        "Population spiking activity during raw-image inference",
        fontsize=14,
        weight="bold",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, path)


def _image_with_saccades(
    ax: plt.Axes,
    image: torch.Tensor,
    centers: Sequence[Sequence[int]],
    patch_size: int,
) -> None:
    ax.imshow(image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    half = patch_size // 2
    for index, (row, col) in enumerate(centers):
        ax.add_patch(
            Rectangle(
                (col - half, row - half),
                patch_size,
                patch_size,
                fill=False,
                edgecolor=plt.cm.viridis(index / max(len(centers) - 1, 1)),
                linewidth=0.8,
            )
        )
        ax.text(
            col,
            row,
            str(index + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            weight="bold",
            bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0},
        )
    ax.axis("off")


def _plot_ambiguity(
    path: Path,
    reports: Dict[str, Dict[str, object]],
    data: Dict[str, Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.5))
    for row, dataset in enumerate(["caltech", "mnist"]):
        report = reports[dataset]
        tensors = data[dataset]
        model = tensors["model"]
        target = int(report["ambiguity_case"]["target"])
        early = int(report["ambiguity_case"]["early_prediction"])
        index = int(report["ambiguity_case"]["validation_index"])
        centers = report["centers"]
        _image_with_saccades(
            axes[row, 0],
            tensors["ambiguity_image"],
            centers,
            model.cfg.patch_size,
        )
        axes[row, 0].set_title(
            f"{DATASET_LABEL[dataset]}\nheld-out sample {index}",
            weight="bold",
        )

        trace = tensors["controls"]["normal"]["decision_trace"][index]
        x = np.arange(1, model.n_locations + 1)
        for cls in range(model.n_classes):
            highlighted = cls in {target, early}
            axes[row, 1].plot(
                x,
                trace[:, cls].numpy(),
                lw=2.2 if highlighted else 0.7,
                alpha=1.0 if highlighted else 0.32,
                color=(
                    COLORS["cyan"]
                    if cls == target
                    else COLORS["red"]
                    if cls == early
                    else COLORS["gray"]
                ),
                label=(
                    f"target {CLASS_NAMES[dataset][cls]}"
                    if cls == target
                    else f"early competitor {CLASS_NAMES[dataset][cls]}"
                    if cls == early
                    else None
                ),
            )
        axes[row, 1].set(
            xlabel="Accumulated saccades",
            ylabel="Cumulative decision spikes",
            xticks=x,
            title=f"Ambiguity resolves: {early} -> {target}",
        )
        axes[row, 1].legend(fontsize=7)

        normal = tensors["controls"]["normal"]["decision_trace"][index]
        reversed_trace = tensors["controls"]["reversed"]["decision_trace"][index]
        for name, selected, color in [
            ("Ordered L5/6", normal, COLORS["cyan"]),
            ("Reversed L5/6", reversed_trace, COLORS["red"]),
        ]:
            target_count = selected[:, target]
            competitors = selected.clone()
            competitors[:, target] = -torch.inf
            margin = target_count - competitors.max(dim=1).values
            axes[row, 2].plot(
                x,
                margin.numpy(),
                marker="o",
                ms=3,
                color=color,
                label=name,
            )
        axes[row, 2].axhline(0, color="black", lw=0.8, ls=":")
        axes[row, 2].set(
            xlabel="Accumulated saccades",
            ylabel="Target - strongest competitor",
            xticks=x,
            title="Reference-frame order changes evidence",
        )
        axes[row, 2].legend(fontsize=7)

        for mode, color in [
            ("normal", COLORS["cyan"]),
            ("reversed", COLORS["red"]),
            ("shuffled", COLORS["gold"]),
            ("silent", COLORS["gray"]),
        ]:
            curve = report["event_controls"][mode]["accuracy_by_saccade"]
            axes[row, 3].plot(
                x,
                torch.as_tensor(curve).numpy(),
                marker="o",
                ms=3,
                color=color,
                label=mode.capitalize(),
            )
        axes[row, 3].axhline(
            1.0 / model.n_classes,
            color="black",
            ls=":",
            lw=0.8,
        )
        axes[row, 3].set(
            xlabel="Accumulated saccades",
            ylabel="Held-out accuracy",
            ylim=(0, 1.03),
            xticks=x,
            title="Population-level evidence accumulation",
        )
        axes[row, 3].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Ordered L5/6 location spikes disambiguate objects across saccades",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


def _prime_montage(values: torch.Tensor, n_l4: int) -> np.ndarray:
    maps = values.reshape(values.shape[0], n_l4, 5, 5).sum(dim=1)
    maps = maps / maps.amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    canvas = np.ones((17, 17), dtype=np.float32) * np.nan
    for location in range(9):
        row, col = divmod(location, 3)
        top, left = row * 6, col * 6
        canvas[top:top + 5, left:left + 5] = maps[location].numpy()
    return canvas


def _plot_bidirectional(
    path: Path,
    reports: Dict[str, Dict[str, object]],
    data: Dict[str, Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.3, 7.7))
    for row, dataset in enumerate(["caltech", "mnist"]):
        report = reports[dataset]
        tensors = data[dataset]
        reciprocal = tensors["reciprocal"]
        targets = reciprocal["targets"]
        confusion = confusion_matrix(
            targets.numpy(),
            reciprocal["predictions"].numpy(),
            labels=list(range(9)),
            normalize="true",
        )
        image = axes[row, 0].imshow(
            confusion,
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        axes[row, 0].set(
            xlabel="Retrieved L5/6 location",
            ylabel="Cued location",
            title=f"{DATASET_LABEL[dataset]}\nL2/3 cue -> L5/6 spikes",
            xticks=range(9),
            yticks=range(9),
        )
        fig.colorbar(image, ax=axes[row, 0], fraction=0.046)

        reciprocal_report = report["reciprocal_feature_to_frame"]
        values = [
            reciprocal_report["top_1_accuracy"],
            reciprocal_report["top_3_accuracy"],
            reciprocal_report["shuffled_binding_accuracy"],
            reciprocal_report["chance_accuracy"],
        ]
        bars = axes[row, 1].bar(
            range(4),
            values,
            color=[
                COLORS["cyan"],
                COLORS["blue"],
                COLORS["gold"],
                COLORS["gray"],
            ],
        )
        axes[row, 1].set(
            ylabel="Held-out retrieval fraction",
            ylim=(0, 1.03),
            xticks=range(4),
            xticklabels=["Top 1", "Top 3", "Shuffled", "Chance"],
            title="Reverse association is learned",
        )
        for bar, value in zip(bars, values):
            axes[row, 1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                fontsize=7,
            )

        cosine = tensors["apical"]["cosine"]
        image = axes[row, 2].imshow(
            cosine.numpy(),
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        axes[row, 2].plot(
            np.arange(9),
            np.arange(9),
            "s",
            ms=4,
            markerfacecolor="none",
            markeredgecolor="white",
        )
        axes[row, 2].set(
            xlabel="Observed sensory location",
            ylabel="Stimulated L5/6 location",
            title="L5/6 spike -> L2/3 prime similarity",
            xticks=range(9),
            yticks=range(9),
        )
        fig.colorbar(image, ax=axes[row, 2], fraction=0.046)

        montage = _prime_montage(
            tensors["apical"]["primed_membrane"],
            report["n_l4_features"],
        )
        axes[row, 3].imshow(montage, cmap="magma")
        axes[row, 3].set_title("Location-specific apical prime maps")
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])
        axes[row, 3].set_facecolor("white")
    fig.suptitle(
        "Causal bidirectional binding between L2/3 features and L5/6 locations",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


def _crop(
    image: torch.Tensor,
    center: Sequence[int],
    patch_size: int,
) -> torch.Tensor:
    half = patch_size // 2
    row, col = int(center[0]), int(center[1])
    return image[
        row - half:row + half + 1,
        col - half:col + half + 1,
    ]


def _plot_cross_image(
    path: Path,
    reports: Dict[str, Dict[str, object]],
    data: Dict[str, Dict[str, object]],
) -> None:
    fig = plt.figure(figsize=(13.4, 7.8))
    outer = fig.add_gridspec(2, 1, hspace=0.52)
    for row, dataset in enumerate(["caltech", "mnist"]):
        selected = data[dataset]["cross_image"]
        report = reports[dataset]
        location = int(selected["location"])
        inner = outer[row].subgridspec(
            2,
            7,
            width_ratios=[1, 1, 1, 1, 1, 1, 1.45],
            hspace=0.08,
            wspace=0.18,
        )
        for column, (image, index, similarity) in enumerate(
            zip(
                selected["images"],
                selected["indices"],
                selected["similarity_to_anchor"],
            )
        ):
            patch = _crop(
                image,
                report["centers"][location],
                report["spiking_config"]["patch_size"],
            )
            ax = fig.add_subplot(inner[0, column])
            ax.imshow(patch.numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                ("Anchor" if column == 0 else f"Image {index}")
                + f"\ncos={similarity:.2f}",
                fontsize=8,
                weight="bold" if column == 0 else "normal",
            )
            ax.axis("off")
            ax = fig.add_subplot(inner[1, column])
            current = selected["currents"][column]
            ax.bar(
                np.arange(9),
                current.numpy(),
                color=[
                    COLORS["cyan"] if value == location else COLORS["light"]
                    for value in range(9)
                ],
            )
            ax.set_xticks(range(9))
            ax.set_xticklabels([str(value + 1) for value in range(9)], fontsize=6)
            ax.axvline(location, color=COLORS["cyan"], lw=0.8)
            if column == 0:
                ax.set_ylabel("L5/6 current")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Location", fontsize=7)
        ax = fig.add_subplot(inner[:, 6])
        matrix = selected["currents"].numpy()
        matrix = matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-12)
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.axvline(location, color="white", ls="--", lw=1.0)
        ax.set(
            xlabel="L5/6 location",
            ylabel="Independent held-out image",
            title=(
                f"{DATASET_LABEL[dataset]}\n"
                f"{selected['class_name']} cue, bound location {location + 1}"
            ),
            xticks=range(9),
            xticklabels=[str(value + 1) for value in range(9)],
            yticks=range(len(selected["indices"])),
            yticklabels=[str(index) for index in selected["indices"]],
        )
        fig.colorbar(image, ax=ax, fraction=0.045, label="Normalized current")
    fig.suptitle(
        "Class- and location-matched feature-spike cues retrieve the same L5/6 representation",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    _save(fig, path)


def _kernel_mosaic(
    weights: torch.Tensor,
    selected: Sequence[int],
    columns: int = 4,
) -> np.ndarray:
    kernels = weights[list(selected), 0].float()
    rows = math.ceil(len(kernels) / columns)
    size = kernels.shape[-1]
    canvas = np.full(
        (rows * size + rows - 1, columns * size + columns - 1),
        np.nan,
        dtype=np.float32,
    )
    for index, kernel in enumerate(kernels):
        row, col = divmod(index, columns)
        top = row * (size + 1)
        left = col * (size + 1)
        canvas[top:top + size, left:left + size] = kernel.numpy()
    return canvas


def _plot_kernels(
    path: Path,
    reports: Dict[str, Dict[str, object]],
    data: Dict[str, Dict[str, object]],
) -> None:
    fig = plt.figure(figsize=(14.0, 7.0))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 1.35],
        hspace=0.42,
        wspace=0.22,
    )
    for row, dataset in enumerate(["caltech", "mnist"]):
        stages = data[dataset]["l4_stages"]
        l4_report = data[dataset]["l4_report"]
        selected = l4_report["selected_features"][:12]
        for column, stage in enumerate(STAGES):
            ax = fig.add_subplot(grid[row, column])
            ax.imshow(
                _kernel_mosaic(stages[stage], selected),
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            count = l4_report["stage_counts"][stage]
            ax.set_title(f"{stage.capitalize()} ({count} images)")
            ax.axis("off")
            if column == 0:
                ax.text(
                    -0.08,
                    0.5,
                    DATASET_LABEL[dataset],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    weight="bold",
                    fontsize=10,
                )
        ax = fig.add_subplot(grid[row, 4])
        x = np.arange(4)
        rms = [l4_report["stages"][stage]["rms_change"] for stage in STAGES]
        autocorrelation = [
            l4_report["stages"][stage]["spatial_autocorrelation"]
            for stage in STAGES
        ]
        pairwise_correlation = [
            l4_report["stages"][stage]["mean_pairwise_abs_correlation"]
            for stage in STAGES
        ]
        ax.plot(x, rms, "o-", color=COLORS["blue"], label="RMS change")
        ax.plot(
            x,
            autocorrelation,
            "o-",
            color=COLORS["cyan"],
            label="Spatial autocorrelation",
        )
        ax.plot(
            x,
            pairwise_correlation,
            "o-",
            color=COLORS["gold"],
            label="Mean filter correlation",
        )
        ax.set(
            xticks=x,
            xticklabels=["Init.", "Early", "Middle", "Final"],
            ylim=(-0.05, 1.03),
            ylabel="Diagnostic value",
            title="Quantified kernel learning",
        )
        ax.legend(fontsize=7)
    fig.suptitle(
        "L4 kernels change substantially and organize during spike-timing plasticity",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


def _plot_performance(
    path: Path,
    reports: Dict[str, Dict[str, object]],
    data: Dict[str, Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6))
    for row, dataset in enumerate(["caltech", "mnist"]):
        report = reports[dataset]
        model = data[dataset]["model"]
        raw = data[dataset]["raw"]
        confusion = confusion_matrix(
            raw["targets"].numpy(),
            raw["predictions"].numpy(),
            labels=list(range(model.n_classes)),
            normalize="true",
        )
        image = axes[row, 0].imshow(
            confusion,
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        axes[row, 0].set(
            xlabel="Predicted class",
            ylabel="True class",
            title=(
                f"{DATASET_LABEL[dataset]} raw images\n"
                f"accuracy={raw['accuracy']:.4f}"
            ),
        )
        axes[row, 0].set_xticks(range(model.n_classes))
        axes[row, 0].set_yticks(range(model.n_classes))
        axes[row, 0].set_xticklabels(CLASS_NAMES[dataset], rotation=45)
        axes[row, 0].set_yticklabels(CLASS_NAMES[dataset])
        fig.colorbar(image, ax=axes[row, 0], fraction=0.046)

        modes = ["normal", "reversed", "shuffled", "silent"]
        values = [
            report["event_controls"][mode]["accuracy"] for mode in modes
        ]
        bars = axes[row, 1].bar(
            range(4),
            values,
            color=[
                COLORS["cyan"],
                COLORS["red"],
                COLORS["gold"],
                COLORS["gray"],
            ],
        )
        axes[row, 1].axhline(
            1 / model.n_classes,
            color="black",
            ls=":",
            lw=0.8,
        )
        axes[row, 1].set(
            xticks=range(4),
            xticklabels=["Ordered", "Reversed", "Shuffled", "Silent"],
            ylabel="Held-out event accuracy",
            ylim=(0, 1.05),
            title="Reference-frame causal controls",
        )
        for bar, value in zip(bars, values):
            axes[row, 1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                fontsize=7,
            )

        history = model.training_history
        epochs = [entry["epoch"] for entry in history]
        axes[row, 2].plot(
            epochs,
            [entry["policy_accuracy"] for entry in history],
            "o-",
            color=COLORS["cyan"],
            label="Free-spike policy accuracy",
        )
        axes[row, 2].plot(
            epochs,
            [entry["update_fraction"] for entry in history],
            "o-",
            color=COLORS["red"],
            label="Dopamine update fraction",
        )
        axes[row, 2].set(
            xlabel="Training epoch",
            ylabel="Fraction",
            ylim=(0, 1.03),
            xticks=epochs,
            title="Three-factor decision plasticity",
        )
        axes[row, 2].legend(fontsize=7)
    fig.suptitle(
        "The fully spiking decision population generalizes and depends on ordered L5/6 gating",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/manuscript_fully_spiking",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    args = parser.parse_args()

    _style()
    seed_everything(args.seed)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = ROOT / args.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Dict[str, object]] = {}
    data: Dict[str, Dict[str, object]] = {}
    for dataset in ["caltech", "mnist"]:
        print(f"[manuscript] running {dataset} on {device}", flush=True)
        reports[dataset], data[dataset] = _run_dataset(
            dataset,
            output_dir,
            device=device,
            seed=args.seed,
            raw_batch_size=args.raw_batch_size,
        )
        print(
            f"[manuscript] {dataset} raw accuracy "
            f"{reports[dataset]['raw_image_reproduction']['accuracy']:.4f}",
            flush=True,
        )

    _plot_architecture(figure_dir / "01_fully_spiking_architecture")
    _plot_ambiguity(
        figure_dir / "03_ambiguity_resolution",
        reports,
        data,
    )
    _plot_bidirectional(
        figure_dir / "04_bidirectional_binding",
        reports,
        data,
    )
    _plot_cross_image(
        figure_dir / "05_cross_image_feature_retrieval",
        reports,
        data,
    )
    _plot_kernels(
        figure_dir / "06_l4_kernel_evolution",
        reports,
        data,
    )
    _plot_performance(
        figure_dir / "07_performance_and_controls",
        reports,
        data,
    )

    figure_hashes = {
        path.name: _sha256(path)
        for path in sorted(figure_dir.glob("*.pdf"))
    }
    source_paths = {
        "experiment_script": Path(__file__).resolve(),
        "fully_spiking_model": ROOT / "scc" / "fully_spiking.py",
    }
    manifest = {
        "kind": "fully_spiking_manuscript_experiments",
        "seed": args.seed,
        "device": str(device),
        "framework": "CoNeX/PyMoNNtorch",
        "gradient_free": True,
        "datasets": reports,
        "hashes": {
            "sources": {
                name: _sha256(path) for name, path in source_paths.items()
            },
            "checkpoints": {
                dataset: _sha256(_paths(dataset)["checkpoint"])
                for dataset in reports
            },
            "cortical_event_caches": {
                dataset: _sha256(_paths(dataset)["events"])
                for dataset in reports
            },
            "figures": figure_hashes,
        },
        "figures": [
            str(path.relative_to(ROOT))
            for path in sorted(figure_dir.glob("*.pdf"))
        ],
    }
    report_path = output_dir / "manuscript_experiments.json"
    report_path.write_text(
        json.dumps(_json_value(manifest), indent=2),
        encoding="utf-8",
    )
    print(f"[manuscript] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
