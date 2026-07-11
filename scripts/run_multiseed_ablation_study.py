"""Training-seed, causal-ablation, and matched-readout study.

All proposed models are the CoNeX/PyMoNNtorch fully spiking column. Sparse
logistic readouts are included only as explicitly nonbiological comparators on
the exact same frozen L2/3 events.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from scipy.stats import binomtest, t
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scc.config import Config
from scc.data.pipeline import make_caltech_loaders, make_mnist_loaders
from scc.fully_spiking import (
    FullySpikingConfig,
    FullySpikingCorticalColumn,
    load_fully_spiking_column,
)
from scc.utils import seed_everything


DATASET_LABEL = {"caltech": "Caltech-101", "mnist": "MNIST"}
COLORS = {
    "caltech": "#0072B2",
    "mnist": "#009E73",
    "full": "#0072B2",
    "reversed": "#D55E00",
    "shuffled": "#CC79A7",
    "silent": "#222222",
    "nonspatial": "#E69F00",
    "untrained": "#999999",
}


def _paths(
    models_root: Path,
    events_root: Path,
    dataset: str,
) -> Dict[str, Path]:
    model_base = models_root / f"{dataset}_all_acquisition_fixation_gated"
    event_base = events_root / f"{dataset}_fully_spiking_column_no_replay"
    return {
        "base": model_base,
        "checkpoint": model_base / "best_fully_spiking_column.pt",
        "events": event_base / "cortical_spike_events.pt",
        "report": model_base / "fully_spiking_report.json",
    }


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    pred = predictions.long().cpu().numpy()
    target = targets.long().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(target, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro")),
    }


def _mean_interval(values: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        half = float(
            t.ppf(0.975, len(values) - 1)
            * standard_deviation
            / math.sqrt(len(values))
        )
    else:
        half = 0.0
    return {
        "n_seeds": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_95_t_interval": [mean - half, mean + half],
    }


def _validation_loader(dataset: str) -> Tuple[DataLoader, Sequence[Tuple[int, int]]]:
    cfg = Config.caltech() if dataset == "caltech" else Config.mnist()
    if dataset == "caltech":
        _, validation = make_caltech_loaders(
            cfg,
            batch_size=128,
            seed=42,
            download=True,
        )
    else:
        _, validation = make_mnist_loaders(
            cfg,
            batch_size=128,
            seed=42,
        )
    centers = [
        (20, 20), (20, 40), (20, 60),
        (40, 20), (40, 40), (40, 60),
        (60, 20), (60, 40), (60, 60),
    ]
    return (
        DataLoader(validation.dataset, batch_size=128, shuffle=False),
        centers,
    )


def _new_model(
    canonical: Dict[str, object],
    seed: int,
    device: str,
) -> FullySpikingCorticalColumn:
    cfg_values = dict(canonical["cfg"])
    cfg_values["random_state"] = int(seed)
    cfg = FullySpikingConfig(**cfg_values)
    return FullySpikingCorticalColumn(
        n_classes=int(canonical["n_classes"]),
        n_locations=int(canonical["n_locations"]),
        l4_kernels=canonical["l4_kernels"],
        cfg=cfg,
        device=device,
    )


def _train_seed(
    canonical: Dict[str, object],
    train_events: torch.Tensor,
    train_y: torch.Tensor,
    dataset: str,
    seed: int,
    device: str,
) -> FullySpikingCorticalColumn:
    seed_everything(seed)
    model = _new_model(canonical, seed, device)
    model.fit_apical_spike_events(
        train_events,
        epochs=1,
        batch_size=1024,
    )
    acquisition = 3 if dataset == "caltech" else 8
    consolidation = 0 if dataset == "caltech" else 2
    model.fit_spike_events(
        train_events,
        train_y,
        epochs=acquisition,
        batch_size=1024,
        seed=seed,
    )
    history = list(model.training_history)
    if consolidation:
        model.fit_spike_events(
            train_events,
            train_y,
            epochs=consolidation,
            batch_size=1024,
            seed=seed + 100,
        )
        second = []
        for item in model.training_history:
            adjusted = dict(item)
            adjusted["epoch"] = float(acquisition + int(item["epoch"]))
            second.append(adjusted)
        model.training_history = history + second
    return model


@torch.no_grad()
def _event_predictions(
    model,
    events: torch.Tensor,
    *,
    silence_l56: bool = False,
    batch_size: int = 512,
) -> torch.Tensor:
    predictions = []
    for start in range(0, len(events), batch_size):
        result = model.run_spike_events(
            events[start:start + batch_size],
            track_eligibility=False,
            silence_l56=silence_l56,
        )
        predictions.append(result.predictions)
    return torch.cat(predictions)


def _raw_metrics(
    model,
    dataset: str,
    expected_targets: torch.Tensor,
) -> Dict[str, float]:
    loader, centers = _validation_loader(dataset)
    result = model.evaluate_images(
        loader,
        centers,
        n_samples=len(expected_targets),
    )
    if not torch.equal(result["targets"], expected_targets):
        raise RuntimeError("raw validation order differs from event cache")
    return _metrics(result["predictions"], result["targets"])


def _mcnemar(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, object]:
    reference_correct = reference.eq(targets)
    comparison_correct = comparison.eq(targets)
    reference_only = int((reference_correct & ~comparison_correct).sum())
    comparison_only = int((~reference_correct & comparison_correct).sum())
    discordant = reference_only + comparison_only
    p_value = (
        float(
            binomtest(
                min(reference_only, comparison_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "reference_only_correct": reference_only,
        "comparison_only_correct": comparison_only,
        "exact_two_sided_p": p_value,
    }


def _causal_ablations(
    checkpoint: Path,
    events: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(events.shape[1], generator=generator) for _ in events]
    )
    shuffled_events = events.gather(
        1,
        permutations[:, :, None].expand_as(events),
    )
    full_model = load_fully_spiking_column(str(checkpoint), device=device)
    full = _event_predictions(full_model, events)
    reversed_prediction = _event_predictions(full_model, events.flip(1))
    shuffled_prediction = _event_predictions(full_model, shuffled_events)
    silent_prediction = _event_predictions(
        full_model,
        events,
        silence_l56=True,
    )

    nonspatial_model = load_fully_spiking_column(str(checkpoint), device=device)
    with torch.no_grad():
        excitatory = nonspatial_model.l23_decision_sg.excitatory_weights
        inhibitory = nonspatial_model.l23_decision_sg.inhibitory_weights
        excitatory.copy_(
            excitatory.mean(dim=1, keepdim=True).expand_as(excitatory)
        )
        inhibitory.copy_(
            inhibitory.mean(dim=1, keepdim=True).expand_as(inhibitory)
        )
    nonspatial_prediction = _event_predictions(nonspatial_model, events)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    untrained_model = _new_model(
        checkpoint_payload,
        seed + 5000,
        device,
    )
    untrained_prediction = _event_predictions(untrained_model, events)
    predictions = {
        "full": full,
        "reversed_l56": reversed_prediction,
        "shuffled_l56": shuffled_prediction,
        "silent_l56": silent_prediction,
        "nonspatial_decision_weights": nonspatial_prediction,
        "no_dopamine_untrained_decision": untrained_prediction,
    }
    report = {}
    for name, prediction in predictions.items():
        report[name] = {
            **_metrics(prediction, targets),
            "mcnemar_vs_full": (
                None
                if name == "full"
                else _mcnemar(full, prediction, targets)
            ),
        }
    return report, predictions


def _no_apical_raw_ablation(
    checkpoint: Path,
    dataset: str,
    targets: torch.Tensor,
    *,
    device: str,
) -> Dict[str, object]:
    model = load_fully_spiking_column(str(checkpoint), device=device)
    normal = _raw_metrics(model, dataset, targets)
    with torch.no_grad():
        model.l56_l23_sg.weights.zero_()
    no_apical = _raw_metrics(model, dataset, targets)
    return {
        "normal": normal,
        "no_apical_feedback": no_apical,
        "accuracy_difference": normal["accuracy"] - no_apical["accuracy"],
    }


def _event_csr(
    events: torch.Tensor,
    *,
    preserve_location: bool,
) -> sparse.csr_matrix:
    coordinates = torch.nonzero(events, as_tuple=False)
    rows = coordinates[:, 0].numpy()
    if preserve_location:
        columns = (
            coordinates[:, 1] * events.shape[2] + coordinates[:, 2]
        ).numpy()
        n_columns = events.shape[1] * events.shape[2]
    else:
        columns = coordinates[:, 2].numpy()
        n_columns = events.shape[2]
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(events), n_columns),
    )


def _readout_baselines(
    train_events: torch.Tensor,
    train_y: torch.Tensor,
    val_events: torch.Tensor,
    val_y: torch.Tensor,
    *,
    seed: int,
) -> Dict[str, object]:
    report = {}
    for preserve_location, name in (
        (True, "spatial_sparse_logistic_readout"),
        (False, "location_agnostic_rate_readout"),
    ):
        train_x = _event_csr(
            train_events,
            preserve_location=preserve_location,
        )
        val_x = _event_csr(
            val_events,
            preserve_location=preserve_location,
        )
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-4,
            random_state=seed,
            average=True,
        )
        classifier.fit(train_x, train_y.numpy())
        predictions = classifier.predict(val_x)
        report[name] = {
            "accuracy": float(accuracy_score(val_y.numpy(), predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(val_y.numpy(), predictions)
            ),
            "macro_f1": float(
                f1_score(val_y.numpy(), predictions, average="macro")
            ),
            "derivative_based": True,
            "part_of_proposed_model": False,
            "preserves_location": preserve_location,
        }
        if preserve_location:
            generator = torch.Generator().manual_seed(seed)
            shuffled_y = train_y[
                torch.randperm(len(train_y), generator=generator)
            ]
            shuffled_classifier = SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-4,
                random_state=seed,
                average=True,
            )
            shuffled_classifier.fit(train_x, shuffled_y.numpy())
            shuffled_prediction = shuffled_classifier.predict(val_x)
            report["spatial_readout_shuffled_labels"] = {
                "accuracy": float(
                    accuracy_score(val_y.numpy(), shuffled_prediction)
                ),
                "derivative_based": True,
                "part_of_proposed_model": False,
            }
    return report


def _run_dataset(
    dataset: str,
    paths: Dict[str, Path],
    seeds: Sequence[int],
    *,
    device: str,
    evaluate_raw: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    canonical = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    payload = torch.load(
        paths["events"],
        map_location="cpu",
        weights_only=False,
    )
    train_events = payload["train_events"].bool()
    train_y = payload["train_y"].long()
    val_events = payload["val_events"].bool()
    val_y = payload["val_y"].long()
    canonical_report = json.loads(paths["report"].read_text(encoding="utf-8"))

    seed_records = []
    seed_predictions = {}
    for seed in seeds:
        print(f"[{dataset}] training seed {seed}", flush=True)
        if seed == 42:
            model = load_fully_spiking_column(str(paths["checkpoint"]), device=device)
        else:
            model = _train_seed(
                canonical,
                train_events,
                train_y,
                dataset,
                seed,
                device,
            )
        prediction = _event_predictions(model, val_events)
        seed_predictions[str(seed)] = prediction
        event_metrics = _metrics(prediction, val_y)
        if evaluate_raw:
            raw_metrics = (
                canonical_report["raw_image_live_metrics"]
                if seed == 42
                else _raw_metrics(model, dataset, val_y)
            )
        else:
            raw_metrics = None
        seed_records.append(
            {
                "seed": seed,
                "event_metrics": event_metrics,
                "raw_image_metrics": raw_metrics,
                "learning_replays": False,
                "forced_output_spikes": False,
            }
        )
        print(
            f"[{dataset}] seed {seed}: event accuracy "
            f"{event_metrics['accuracy']:.4f}"
            + (
                f", raw {raw_metrics['accuracy']:.4f}"
                if raw_metrics is not None
                else ""
            ),
            flush=True,
        )

    event_accuracies = [
        record["event_metrics"]["accuracy"] for record in seed_records
    ]
    raw_accuracies = [
        record["raw_image_metrics"]["accuracy"]
        for record in seed_records
        if record["raw_image_metrics"] is not None
    ]
    ablations, ablation_predictions = _causal_ablations(
        paths["checkpoint"],
        val_events,
        val_y,
        device=device,
        seed=seeds[0] + 700,
    )
    apical_ablation = (
        _no_apical_raw_ablation(
            paths["checkpoint"],
            dataset,
            val_y,
            device=device,
        )
        if evaluate_raw
        else None
    )
    readouts = _readout_baselines(
        train_events,
        train_y,
        val_events,
        val_y,
        seed=seeds[0],
    )
    report = {
        "seed_runs": seed_records,
        "event_accuracy_across_seeds": _mean_interval(event_accuracies),
        "raw_image_accuracy_across_seeds": (
            _mean_interval(raw_accuracies) if raw_accuracies else None
        ),
        "causal_ablations": ablations,
        "raw_apical_ablation": apical_ablation,
        "matched_nonspiking_readouts": readouts,
    }
    tensors = {
        "targets": val_y,
        "seed_predictions": seed_predictions,
        "ablation_predictions": ablation_predictions,
    }
    return report, tensors


def _plot(
    reports: Dict[str, object],
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
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for index, dataset in enumerate(("caltech", "mnist")):
        values = [
            item["raw_image_metrics"]["accuracy"]
            for item in reports[dataset]["seed_runs"]
        ]
        x = np.full(len(values), index)
        axes[0, 0].scatter(
            x,
            values,
            s=42,
            color=COLORS[dataset],
            alpha=0.85,
            edgecolor="white",
            zorder=3,
        )
        axes[0, 0].errorbar(
            index,
            np.mean(values),
            yerr=np.std(values, ddof=1),
            fmt="_",
            ms=18,
            color="#111111",
            capsize=4,
            lw=1.5,
        )
    axes[0, 0].set_xticks([0, 1], ["Caltech-101", "MNIST"])
    axes[0, 0].set(
        ylabel="Raw-image accuracy",
        ylim=(0, 1.03),
        title="A  Training-seed stability (mean +/- SD)",
    )

    names = [
        "full",
        "reversed_l56",
        "shuffled_l56",
        "silent_l56",
        "nonspatial_decision_weights",
        "no_dopamine_untrained_decision",
    ]
    labels = [
        "Full",
        "Reversed",
        "Shuffled",
        "Silent",
        "Nonspatial",
        "Untrained",
    ]
    x = np.arange(len(names))
    width = 0.36
    for index, dataset in enumerate(("caltech", "mnist")):
        values = [
            reports[dataset]["causal_ablations"][name]["accuracy"]
            for name in names
        ]
        axes[0, 1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=COLORS[dataset],
            alpha=(0.95 if index == 0 else 0.72),
            label=DATASET_LABEL[dataset],
        )
    axes[0, 1].set_xticks(x, labels, rotation=24, ha="right")
    axes[0, 1].set(
        ylabel="Held-out event accuracy",
        ylim=(0, 1.03),
        title="B  Causal spatial ablations",
    )
    axes[0, 1].legend(frameon=False)

    baseline_names = [
        "spatial_sparse_logistic_readout",
        "location_agnostic_rate_readout",
        "spatial_readout_shuffled_labels",
    ]
    baseline_labels = ["Spatial logistic", "Pooled rate", "Shuffled labels"]
    for index, dataset in enumerate(("caltech", "mnist")):
        full = reports[dataset]["causal_ablations"]["full"]["accuracy"]
        values = [full] + [
            reports[dataset]["matched_nonspiking_readouts"][name]["accuracy"]
            for name in baseline_names
        ]
        axes[1, index].bar(
            range(4),
            values,
            color=[
                COLORS[dataset],
                "#56B4E9",
                "#E69F00",
                "#999999",
            ],
        )
        axes[1, index].set_xticks(
            range(4),
            ["Spiking\ncolumn"] + baseline_labels,
            rotation=20,
            ha="right",
        )
        axes[1, index].set(
            ylabel="Held-out accuracy",
            ylim=(0, 1.03),
            title=(
                f"{chr(67 + index)}  {DATASET_LABEL[dataset]} matched readouts"
            ),
        )
    fig.suptitle(
        "Robustness, causal ablations, and matched nonspiking comparators",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = Path(args.models_root).resolve()
    events_root = Path(args.events_root).resolve()
    reports = {}
    tensors = {}
    for dataset in ("caltech", "mnist"):
        reports[dataset], tensors[dataset] = _run_dataset(
            dataset,
            _paths(models_root, events_root, dataset),
            args.seeds,
            device=args.device,
            evaluate_raw=not args.skip_raw,
        )
    _plot(
        reports,
        output_dir / "figures" / "13_multiseed_ablation_readouts",
    )
    torch.save(
        {
            "kind": "multiseed_ablation_predictions",
            **tensors,
        },
        output_dir / "multiseed_ablation_predictions.pt",
    )
    result = {
        "kind": "fully_spiking_multiseed_ablation_study",
        "backend": "CoNeX/PyMoNNtorch",
        "decision_learning_rule": "online_free_trial_three_factor",
        "learning_replays": False,
        "forced_output_spikes": False,
        "seeds": list(args.seeds),
        "datasets": reports,
        "artifacts": {
            "predictions": "multiseed_ablation_predictions.pt",
            "figure": "figures/13_multiseed_ablation_readouts.pdf",
        },
    }
    (output_dir / "multiseed_ablation_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", default="outputs/joint_training")
    parser.add_argument("--events-root", default="outputs/best_models")
    parser.add_argument(
        "--output-dir",
        default="outputs/manuscript_fully_spiking/multiseed_ablation",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 142, 242, 342, 442],
    )
    parser.add_argument("--skip-raw", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
