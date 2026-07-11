"""Held-out prospective and bidirectional reference-frame experiments.

The forward assay injects only a one-hot location current into the L5/6 LIF
population, clamps visual input to zero, and asks whether the resulting L2/3
apical membrane pattern predicts the spikes evoked later by a held-out patch.
The reverse assay asks whether a held-out L2/3 feature pattern retrieves its
location from the same learned binding matrix. No validation event updates any
weight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scc.fully_spiking import load_fully_spiking_column


COLORS = {
    "matched": "#0072B2",
    "reversed": "#D55E00",
    "shuffled": "#CC79A7",
    "global_prior": "#7A7A7A",
    "random": "#B8B8B8",
    "silent": "#1B1B1B",
}
CONDITIONS = ["matched", "reversed", "shuffled", "global_prior", "random"]
LABELS = {
    "matched": "Correct L5/6",
    "reversed": "Reversed L5/6",
    "shuffled": "Shuffled binding",
    "global_prior": "Feature prior",
    "random": "Random",
    "silent": "Silent L5/6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_interval(
    values: torch.Tensor,
    *,
    seed: int,
    draws: int = 10000,
) -> List[float]:
    values = values.float().cpu().flatten()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(values),
        (draws, len(values)),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    return [
        float(torch.quantile(means, 0.025)),
        float(torch.quantile(means, 0.975)),
    ]


def _paired_statistics(
    matched: torch.Tensor,
    control: torch.Tensor,
    *,
    seed: int,
) -> Dict[str, object]:
    difference = matched.float().cpu() - control.float().cpu()
    try:
        test = wilcoxon(
            matched.cpu().numpy(),
            control.cpu().numpy(),
            alternative="greater",
            zero_method="wilcox",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    except ValueError:
        statistic = 0.0
        p_value = 1.0
    return {
        "mean_paired_difference": float(difference.mean()),
        "difference_95_bootstrap_ci": _bootstrap_interval(
            difference,
            seed=seed,
        ),
        "one_sided_wilcoxon_statistic": statistic,
        "one_sided_wilcoxon_p": p_value,
        "unit_of_analysis": "held-out image",
    }


def _wilson_interval(successes: int, trials: int) -> List[float]:
    if trials == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [float(center - half), float(center + half)]


@torch.no_grad()
def _causal_l56_primes(
    model,
    settle_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    membranes: List[torch.Tensor] = []
    spikes: List[torch.Tensor] = []
    for location in range(model.n_locations):
        model.reset_trial(1)
        model.net.apical_learning_enabled = False
        model.net.eligibility_enabled = False
        model._set_motor_location(location, pulse=True)
        model.net.cortical_phase = "l4_release"
        model._simulate(1)
        model._set_motor_location(location, pulse=False)
        model._simulate(max(int(settle_steps) - 1, 0))
        membranes.append(model.l23_ng.v[0].detach().cpu().clamp(min=0))
        model.net.cortical_phase = "l23_competition"
        model._simulate(1)
        spikes.append(model.l23_ng.spikes[0].detach().cpu())
    return torch.stack(membranes), torch.stack(spikes)


def _average_precision_and_recall(
    scores: torch.Tensor,
    events: torch.Tensor,
    *,
    k: int,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = scores.argsort(dim=1, descending=True)
    top = order[:, :k]
    ranks = torch.arange(1, scores.shape[1] + 1, dtype=torch.float32)
    ap_batches: List[torch.Tensor] = []
    recall_batches: List[torch.Tensor] = []
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size].bool()
        ranked = batch.gather(
            2,
            order.unsqueeze(0).expand(len(batch), -1, -1),
        ).float()
        precision = ranked.cumsum(dim=2) / ranks[None, None, :]
        ap = (precision * ranked).sum(dim=2) / ranked.sum(dim=2).clamp(min=1)
        predicted = torch.zeros_like(batch)
        predicted.scatter_(
            2,
            top.unsqueeze(0).expand(len(batch), -1, -1),
            True,
        )
        recall = (predicted & batch).sum(dim=2).float() / float(k)
        ap_batches.append(ap)
        recall_batches.append(recall)
    return torch.cat(ap_batches), torch.cat(recall_batches)


def _condition_report(
    scores: torch.Tensor,
    events: torch.Tensor,
    *,
    k: int,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    average_precision, recall = _average_precision_and_recall(
        scores,
        events,
        k=k,
    )
    image_ap = average_precision.mean(dim=1)
    image_recall = recall.mean(dim=1)
    report = {
        "mean_average_precision": float(average_precision.mean()),
        "mean_average_precision_95_bootstrap_ci": _bootstrap_interval(
            image_ap,
            seed=seed,
        ),
        f"recall_at_{k}": float(recall.mean()),
        f"recall_at_{k}_95_bootstrap_ci": _bootstrap_interval(
            image_recall,
            seed=seed + 1,
        ),
        "average_precision_by_location": average_precision.mean(dim=0).tolist(),
        f"recall_at_{k}_by_location": recall.mean(dim=0).tolist(),
    }
    return report, {
        "average_precision": average_precision,
        "recall": recall,
        "image_average_precision": image_ap,
        "image_recall": image_recall,
    }


def _feature_to_frame(
    prime: torch.Tensor,
    events: torch.Tensor,
    *,
    batch_size: int = 256,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    normalized_prime = F.normalize(prime.float(), dim=1)
    similarities: List[torch.Tensor] = []
    for start in range(0, len(events), batch_size):
        batch = F.normalize(
            events[start:start + batch_size].float(),
            dim=2,
        )
        similarities.append(torch.einsum("nlf,qf->nlq", batch, normalized_prime))
    similarity = torch.cat(similarities)
    targets = torch.arange(prime.shape[0])[None, :].expand(len(events), -1)
    predictions = similarity.argmax(dim=2)
    top3 = similarity.topk(min(3, prime.shape[0]), dim=2).indices
    correct = predictions.eq(targets)
    top3_correct = top3.eq(targets.unsqueeze(2)).any(dim=2)
    confusion = torch.zeros(
        prime.shape[0],
        prime.shape[0],
        dtype=torch.long,
    )
    for target, prediction in zip(targets.flatten(), predictions.flatten()):
        confusion[target, prediction] += 1
    successes = int(correct.sum())
    trials = int(correct.numel())
    chance = 1.0 / prime.shape[0]
    p_value = float(
        binomtest(
            successes,
            trials,
            p=chance,
            alternative="greater",
        ).pvalue
    )
    return {
        "top_1_accuracy": successes / trials,
        "top_1_accuracy_95_wilson_ci": _wilson_interval(successes, trials),
        "top_3_accuracy": float(top3_correct.float().mean()),
        "chance_accuracy": chance,
        "exact_binomial_p_vs_chance": p_value,
        "confusion": confusion.tolist(),
    }, {
        "similarity": similarity,
        "predictions": predictions,
        "targets": targets,
        "correct": correct,
    }


@torch.no_grad()
def _decision_condition(
    model,
    events: torch.Tensor,
    labels: torch.Tensor,
    *,
    silence_l56: bool = False,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    traces: List[torch.Tensor] = []
    predictions: List[torch.Tensor] = []
    for start in range(0, len(events), batch_size):
        result = model.run_spike_events(
            events[start:start + batch_size],
            track_eligibility=False,
            silence_l56=silence_l56,
        )
        traces.append(result.decision_trace)
        predictions.append(result.predictions)
    trace = torch.cat(traces)
    accuracy = trace.argmax(dim=2).eq(labels[:, None]).float().mean(dim=0)
    return accuracy, torch.cat(predictions)


def _evidence_accumulation(
    model,
    events: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(model.n_locations, generator=generator) for _ in events]
    )
    shuffled = events.gather(
        1,
        permutations[:, :, None].expand_as(events),
    )
    conditions = {
        "matched": (events, False),
        "reversed": (events.flip(1), False),
        "shuffled": (shuffled, False),
        "silent": (events, True),
    }
    report: Dict[str, object] = {}
    tensors: Dict[str, torch.Tensor] = {}
    for name, (condition_events, silence) in conditions.items():
        accuracy, predictions = _decision_condition(
            model,
            condition_events,
            labels,
            silence_l56=silence,
        )
        report[name] = {
            "accuracy_by_accumulated_saccade": accuracy.tolist(),
            "final_accuracy": float(predictions.eq(labels).float().mean()),
        }
        tensors[name] = accuracy
    return report, tensors


def _run_dataset(
    dataset: str,
    checkpoint: Path,
    event_cache: Path,
    *,
    device: str,
    seed: int,
    settle_steps: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    model = load_fully_spiking_column(str(checkpoint), device=device)
    payload = torch.load(event_cache, map_location="cpu", weights_only=False)
    train_events = payload["train_events"].bool()
    val_events = payload["val_events"].bool()
    val_y = payload["val_y"].long()
    prime, prime_spikes = _causal_l56_primes(model, settle_steps)
    k = int(model.cfg.l23_k)

    generator = torch.Generator().manual_seed(seed + 17)
    permutation = torch.randperm(model.n_locations, generator=generator)
    global_prior = train_events.float().mean(dim=(0, 1))
    random_scores = torch.rand(
        model.n_locations,
        model.n_l23_features,
        generator=generator,
    )
    score_conditions = {
        "matched": prime,
        "reversed": prime.flip(0),
        "shuffled": prime[permutation],
        "global_prior": global_prior[None].expand(model.n_locations, -1),
        "random": random_scores,
    }
    condition_reports: Dict[str, object] = {}
    condition_tensors: Dict[str, object] = {}
    for index, (name, scores) in enumerate(score_conditions.items()):
        condition_reports[name], condition_tensors[name] = _condition_report(
            scores,
            val_events,
            k=k,
            seed=seed + 100 * index,
        )

    paired_tests: Dict[str, object] = {}
    matched_metrics = condition_tensors["matched"]
    for index, name in enumerate(CONDITIONS[1:]):
        paired_tests[name] = {
            "average_precision": _paired_statistics(
                matched_metrics["image_average_precision"],
                condition_tensors[name]["image_average_precision"],
                seed=seed + 1000 + index,
            ),
            f"recall_at_{k}": _paired_statistics(
                matched_metrics["image_recall"],
                condition_tensors[name]["image_recall"],
                seed=seed + 1100 + index,
            ),
        }

    frame_report, frame_tensors = _feature_to_frame(prime, val_events)
    shuffled_frame_report, _ = _feature_to_frame(prime[permutation], val_events)
    evidence_report, evidence_tensors = _evidence_accumulation(
        model,
        val_events,
        val_y,
        seed=seed + 23,
    )

    mean_observed = train_events.float().mean(dim=0)
    prime_observed_cosine = (
        F.normalize(prime, dim=1) @ F.normalize(mean_observed, dim=1).T
    )
    report = {
        "dataset": dataset,
        "backend": model.backend,
        "n_training_images": len(train_events),
        "n_held_out_images": len(val_events),
        "n_locations": model.n_locations,
        "n_l23_features": model.n_l23_features,
        "l23_winners_per_patch": k,
        "intervention": {
            "visual_input": "clamped to zero",
            "l56_input": "one-hot location vector injected as LIF current",
            "l56_spikes_forced": False,
            "settle_steps": settle_steps,
            "validation_weight_updates": 0,
        },
        "prospective_feature_prediction": condition_reports,
        "paired_tests": paired_tests,
        "feature_to_reference_frame": {
            "learned_binding": frame_report,
            "shuffled_binding": shuffled_frame_report,
        },
        "decision_evidence_accumulation": evidence_report,
        "prime_observed_location_cosine": prime_observed_cosine.tolist(),
        "hashes": {
            "checkpoint_sha256": _sha256(checkpoint),
            "event_cache_sha256": _sha256(event_cache),
        },
    }
    tensors = {
        "prime": prime,
        "prime_spikes": prime_spikes,
        "mean_observed": mean_observed,
        "prime_observed_cosine": prime_observed_cosine,
        "conditions": condition_tensors,
        "frame": frame_tensors,
        "evidence": evidence_tensors,
        "permutation": permutation,
    }
    return report, tensors


def _mean_and_ci(
    report: Dict[str, object],
    metric: str,
) -> Tuple[float, Sequence[float]]:
    return (
        float(report[metric]),
        report[f"{metric}_95_bootstrap_ci"],
    )


def _plot_summary(
    reports: Dict[str, object],
    tensors: Dict[str, object],
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
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 12.0))
    datasets = ["caltech", "mnist"]
    dataset_labels = ["Caltech-101", "MNIST"]
    width = 0.35

    for panel_index, (metric, ylabel) in enumerate(
        [
            ("mean_average_precision", "Average precision"),
            (
                None,
                "Recall of actual L2/3 winners",
            ),
        ]
    ):
        ax = axes[0, panel_index]
        x = np.arange(len(CONDITIONS))
        for ds_index, dataset in enumerate(datasets):
            condition_report = reports[dataset][
                "prospective_feature_prediction"
            ]
            if metric is None:
                k = reports[dataset]["l23_winners_per_patch"]
                metric_name = f"recall_at_{k}"
            else:
                metric_name = metric
            means = []
            low = []
            high = []
            for condition in CONDITIONS:
                value, ci = _mean_and_ci(
                    condition_report[condition],
                    metric_name,
                )
                means.append(value)
                low.append(value - float(ci[0]))
                high.append(float(ci[1]) - value)
            offset = (ds_index - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=np.asarray([low, high]),
                capsize=2,
                label=dataset_labels[ds_index],
                color=("#0072B2" if ds_index == 0 else "#009E73"),
                alpha=(0.95 if ds_index == 0 else 0.72),
            )
        ax.set_xticks(x, [LABELS[name] for name in CONDITIONS], rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.set_title(
            f"{chr(65 + panel_index)}  Prospective held-out prediction",
            loc="left",
            fontweight="bold",
        )

    for column, dataset in enumerate(datasets):
        ax = axes[1, column]
        matrix = np.asarray(
            reports[dataset]["feature_to_reference_frame"][
                "learned_binding"
            ]["confusion"]
        )
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        accuracy = reports[dataset]["feature_to_reference_frame"][
            "learned_binding"
        ]["top_1_accuracy"]
        top3 = reports[dataset]["feature_to_reference_frame"][
            "learned_binding"
        ]["top_3_accuracy"]
        ax.set(
            xlabel="Retrieved L5/6 location",
            ylabel="True patch location",
            title=(
                f"{chr(67 + column)}  {dataset_labels[column]} reverse retrieval\n"
                f"top-1={accuracy:.3f}, top-3={top3:.3f}"
            ),
        )
        ax.set_xticks(range(9), range(1, 10))
        ax.set_yticks(range(9), range(1, 10))
        fig.colorbar(image, ax=ax, fraction=0.046, label="Row probability")

    for column, dataset in enumerate(datasets):
        ax = axes[2, column]
        for condition in ["matched", "reversed", "shuffled", "silent"]:
            values = tensors[dataset]["evidence"][condition].numpy()
            ax.plot(
                np.arange(1, len(values) + 1),
                values,
                marker="o",
                ms=3,
                lw=1.8,
                color=COLORS[condition],
                label=LABELS[condition],
            )
        chance = 1.0 / (2 if dataset == "caltech" else 10)
        ax.axhline(chance, color="#777777", ls=":", lw=1, label="Chance")
        ax.set(
            xlabel="Accumulated saccades",
            ylabel="Held-out accuracy",
            ylim=(0, 1.04),
            xticks=range(1, 10),
            title=(
                f"{chr(69 + column)}  {dataset_labels[column]} evidence accumulation"
            ),
        )
        ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "Bidirectional location-feature binding and prospective prediction",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_binding_matrices(
    reports: Dict[str, object],
    tensors: Dict[str, object],
    output_base: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    for row, dataset in enumerate(["caltech", "mnist"]):
        label = "Caltech-101" if dataset == "caltech" else "MNIST"
        prime = tensors[dataset]["prime"]
        observed = tensors[dataset]["mean_observed"]
        cosine = tensors[dataset]["prime_observed_cosine"]
        for column, (matrix, title, cmap) in enumerate(
            [
                (prime, "Causal apical prime", "magma"),
                (observed, "Training feature probability", "magma"),
                (cosine, "Prime-to-observed cosine", "viridis"),
            ]
        ):
            ax = axes[row, column]
            image = ax.imshow(
                matrix.numpy(),
                aspect="auto",
                cmap=cmap,
                vmin=(0 if column == 2 else None),
                vmax=(1 if column == 2 else None),
            )
            ax.set_ylabel(f"{label}\nL5/6 location")
            ax.set_xlabel(
                "Observed location" if column == 2 else "L2/3 neuron"
            )
            if row == 0:
                ax.set_title(
                    f"{chr(65 + column)}  {title}",
                    loc="left",
                    fontweight="bold",
                )
            fig.colorbar(image, ax=ax, fraction=0.035)
    fig.suptitle(
        "Physical L5/6-to-L2/3 synapses predict location-specific features",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, object] = {}
    tensors: Dict[str, object] = {}
    for dataset in ("caltech", "mnist"):
        model_base = (
            Path(args.models_root)
            / f"{dataset}_all_acquisition_fixation_gated"
        )
        event_base = (
            Path(args.events_root)
            / f"{dataset}_fully_spiking_column_no_replay"
        )
        report, dataset_tensors = _run_dataset(
            dataset,
            model_base / "best_fully_spiking_column.pt",
            event_base / "cortical_spike_events.pt",
            device=args.device,
            seed=args.seed,
            settle_steps=args.settle_steps,
        )
        reports[dataset] = report
        tensors[dataset] = dataset_tensors

    figure_dir = output_dir / "figures"
    _plot_summary(
        reports,
        tensors,
        figure_dir / "07_prospective_bidirectional_binding",
    )
    _plot_binding_matrices(
        reports,
        tensors,
        figure_dir / "08_reference_frame_binding_matrices",
    )
    torch.save(
        {
            "kind": "prospective_reference_frame_evidence",
            "caltech": tensors["caltech"],
            "mnist": tensors["mnist"],
        },
        output_dir / "prospective_reference_frame_evidence.pt",
    )
    result = {
        "kind": "held_out_prospective_reference_frame_experiment",
        "seed": args.seed,
        "datasets": reports,
        "artifacts": {
            "evidence": "prospective_reference_frame_evidence.pt",
            "summary_figure_pdf": (
                "figures/07_prospective_bidirectional_binding.pdf"
            ),
            "binding_figure_pdf": (
                "figures/08_reference_frame_binding_matrices.pdf"
            ),
        },
    }
    result_path = output_dir / "prospective_reference_frame_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root",
        default="outputs/joint_training",
    )
    parser.add_argument(
        "--events-root",
        default="outputs/best_models",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/manuscript_fully_spiking/"
            "prospective_reference_frame"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--settle-steps", type=int, default=8)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
