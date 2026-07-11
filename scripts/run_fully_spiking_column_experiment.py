"""Train and evaluate the end-to-end CoNeX/PyMoNNtorch cortical column."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scc.config import Config
from scc.data.pipeline import make_caltech_loaders, make_mnist_loaders
from scc.fully_spiking import FullySpikingConfig, FullySpikingCorticalColumn
from scc.utils import seed_everything


def _dataset_spec(
    dataset: str,
) -> Tuple[Config, int, List[Tuple[int, int]], Path]:
    if dataset == "caltech":
        return (
            Config.caltech(),
            2,
            [
                (20, 20), (20, 40), (20, 60),
                (40, 20), (40, 40), (40, 60),
                (60, 20), (60, 40), (60, 60),
            ],
            Path("outputs/l4_kernel_study/caltech/l4_best_weights.pt"),
        )
    return (
        Config.mnist(),
        10,
        [
            (20, 20), (20, 40), (20, 60),
            (40, 20), (40, 40), (40, 60),
            (60, 20), (60, 40), (60, 60),
        ],
        Path("outputs/l4_kernel_study/mnist/l4_best_weights.pt"),
    )


def _loaders(
    cfg: Config,
    dataset: str,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    if dataset == "caltech":
        train, val = make_caltech_loaders(
            cfg, batch_size=batch_size, seed=seed
        )
    else:
        train, val = make_mnist_loaders(
            cfg, batch_size=batch_size, seed=seed
        )
    # The plasticity loop performs its own deterministic permutation. Keep
    # event extraction sequential so the cortical event cache is reproducible.
    return (
        DataLoader(train.dataset, batch_size=batch_size, shuffle=False),
        DataLoader(val.dataset, batch_size=batch_size, shuffle=False),
    )


def _collect_events(
    model: FullySpikingCorticalColumn,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    centers: Sequence[Tuple[int, int]],
    n_samples: int,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    events: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    sensory_spikes = 0.0
    l4_spikes = 0.0
    seen = 0
    start_time = time.time()
    total = (n_samples + int(loader.batch_size) - 1) // int(loader.batch_size)
    for images, batch_y in tqdm(loader, desc=desc, total=total):
        if seen >= n_samples:
            break
        take = min(len(images), n_samples - seen)
        result = model.run_images(
            images[:take],
            centers,
            learn_apical=False,
            track_eligibility=False,
        )
        if result.l23_spike_events.dtype != torch.bool:
            raise RuntimeError("L2/3 event cache is not binary")
        if not torch.all(result.l56_spike_events.sum(dim=2) == 1):
            raise RuntimeError("L5/6 did not emit exactly one location spike")
        events.append(result.l23_spike_events)
        labels.append(batch_y[:take].long().cpu())
        sensory_spikes += float(result.sensory_spike_count.sum())
        l4_spikes += float(result.l4_spike_count.sum())
        seen += take
    return (
        torch.cat(events),
        torch.cat(labels),
        {
            "samples": seen,
            "seconds": time.time() - start_time,
            "mean_sensory_spikes": sensory_spikes / max(seen, 1),
            "mean_l4_spikes": l4_spikes / max(seen, 1),
        },
    )


def _metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> Dict[str, object]:
    pred = predictions.long().cpu().numpy()
    true = targets.long().cpu().numpy()
    return {
        "accuracy": float((pred == true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "confusion": confusion_matrix(
            true, pred, labels=list(range(n_classes))
        ).tolist(),
    }


def _evaluate_event_controls(
    model: FullySpikingCorticalColumn,
    val_events: torch.Tensor,
    val_y: torch.Tensor,
    n_classes: int,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    normal = model.evaluate_spike_events(val_events, val_y, batch_size)
    reversed_result = model.evaluate_spike_events(
        val_events.flip(1), val_y, batch_size
    )
    generator = torch.Generator().manual_seed(seed + 700)
    shuffled_events = torch.empty_like(val_events)
    for idx in range(len(val_events)):
        permutation = torch.randperm(
            val_events.shape[1], generator=generator
        )
        shuffled_events[idx] = val_events[idx, permutation]
    shuffled_result = model.evaluate_spike_events(
        shuffled_events, val_y, batch_size
    )

    silent_predictions: List[torch.Tensor] = []
    for start in range(0, len(val_events), batch_size):
        result = model.run_spike_events(
            val_events[start:start + batch_size],
            track_eligibility=False,
            silence_l56=True,
        )
        silent_predictions.append(result.predictions)
    silent_pred = torch.cat(silent_predictions)
    majority = int(torch.bincount(val_y).argmax())
    majority_pred = torch.full_like(val_y, majority)
    return {
        "normal": _metrics(normal["predictions"], val_y, n_classes),
        "reversed_l56": _metrics(
            reversed_result["predictions"], val_y, n_classes
        ),
        "shuffled_l56": _metrics(
            shuffled_result["predictions"], val_y, n_classes
        ),
        "silent_l56": _metrics(silent_pred, val_y, n_classes),
        "majority": _metrics(majority_pred, val_y, n_classes),
    }


def _train_shuffled_label_control(
    source: FullySpikingCorticalColumn,
    train_events: torch.Tensor,
    train_y: torch.Tensor,
    val_events: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    consolidation_epochs: int,
    batch_size: int,
    seed: int,
) -> float:
    model = FullySpikingCorticalColumn(
        source.n_classes,
        source.n_locations,
        source.l4_kernels,
        source.cfg,
        source.device,
    )
    with torch.no_grad():
        model.l56_l23_sg.weights.copy_(source.l56_l23_sg.weights)
        model.l23_l56_sg.weights.copy_(source.l23_l56_sg.weights)
    generator = torch.Generator().manual_seed(seed + 991)
    shuffled = train_y[torch.randperm(len(train_y), generator=generator)]
    model.fit_spike_events(
        train_events,
        shuffled,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    if consolidation_epochs > 0:
        model.fit_spike_events(
            train_events,
            shuffled,
            epochs=consolidation_epochs,
            batch_size=batch_size,
            seed=seed + 100,
        )
    return float(
        model.evaluate_spike_events(
            val_events, val_y, batch_size=min(batch_size, 512)
        )["accuracy"]
    )


def _evaluate_reciprocal_binding(
    model: FullySpikingCorticalColumn,
    val_events: torch.Tensor,
    *,
    batch_size: int,
    gain: float,
) -> Dict[str, object]:
    cues = val_events.reshape(-1, val_events.shape[-1])
    targets = torch.arange(model.n_locations).repeat(len(val_events))
    output = model.predict_l56_from_l23_events(
        cues,
        batch_size=batch_size,
        gain=gain,
        record_first=True,
    )
    predictions = output["predictions"]
    currents = output["currents"]
    spike_ranks = output["spike_counts"].argsort(dim=1, descending=True)
    current_ranks = currents.argsort(dim=1, descending=True)
    matched = currents.gather(1, targets[:, None]).squeeze(1)
    other = (
        currents.sum(dim=1) - matched
    ) / max(model.n_locations - 1, 1)
    return {
        "gain": float(gain),
        "n_trials": int(len(targets)),
        "top_1_accuracy": float(predictions.eq(targets).float().mean()),
        "top_2_accuracy": float(
            (spike_ranks[:, : min(2, model.n_locations)] == targets[:, None])
            .any(dim=1)
            .float()
            .mean()
        ),
        "top_3_accuracy": float(
            (spike_ranks[:, : min(3, model.n_locations)] == targets[:, None])
            .any(dim=1)
            .float()
            .mean()
        ),
        "current_rank_top_1_accuracy": float(
            (current_ranks[:, 0] == targets).float().mean()
        ),
        "current_rank_top_3_accuracy": float(
            (current_ranks[:, : min(3, model.n_locations)] == targets[:, None])
            .any(dim=1)
            .float()
            .mean()
        ),
        "chance_accuracy": 1.0 / model.n_locations,
        "matched_current": float(matched.mean()),
        "other_location_current": float(other.mean()),
        "matched_minus_other": float((matched - other).mean()),
        "hard_spikes_per_trial": float(
            output["spike_counts"].sum(dim=1).float().mean()
        ),
        "first_raster_shape": list(output.get("first_raster", torch.empty(0)).shape),
    }


def _plot_reports(
    output_dir: Path,
    model: FullySpikingCorticalColumn,
    controls: Dict[str, object],
    val_events: torch.Tensor,
    val_y: torch.Tensor,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    names = ["normal", "reversed_l56", "shuffled_l56", "silent_l56", "majority"]
    values = [float(controls[name]["accuracy"]) for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(
        ["Held out", "Reversed\nL5/6", "Shuffled\nL5/6", "Silent\nL5/6", "Majority"],
        values,
        color=["#176B87", "#64CCC5", "#DAA520", "#B94A48", "#777777"],
    )
    ax.axhline(
        1.0 / model.n_classes,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Chance",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Fully spiking reference-frame controls")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_accuracy_controls.png", dpi=220)
    plt.close(fig)

    history = model.training_history
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epoch = [item["epoch"] for item in history]
        axes[0].plot(epoch, [item["policy_accuracy"] for item in history], marker="o")
        axes[0].set(xlabel="Epoch", ylabel="Free-spike policy accuracy")
        axes[0].set_ylim(0, 1.02)
        axes[1].plot(
            epoch,
            [item["update_fraction"] for item in history],
            marker="o",
            color="#B94A48",
        )
        axes[1].set(xlabel="Epoch", ylabel="Trials receiving dopamine")
        fig.suptitle("Online dopamine-gated free-trial eligibility learning")
        fig.tight_layout()
        fig.savefig(figure_dir / "02_learning_dynamics.png", dpi=220)
        plt.close(fig)

    sample = val_events[:1]
    result = model.run_spike_events(sample, track_eligibility=False)
    decision = result.decision_trace[0].numpy()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].imshow(
        result.l56_spike_events[0].T.numpy(),
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    axes[0].set_ylabel("L5/6 unit")
    axes[0].set_title("Live L5/6 spikes gate every visual saccade")
    axes[1].plot(sample[0].sum(dim=1).numpy(), marker="o", color="#176B87")
    axes[1].set_ylabel("L2/3 spikes")
    axes[2].plot(decision, marker="o")
    axes[2].set_ylabel("Cumulative\ndecision spikes")
    axes[2].set_xlabel("Saccade / L5/6 location")
    axes[2].legend(
        [f"class {idx}" for idx in range(model.n_classes)],
        ncol=min(model.n_classes, 5),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_spike_dynamics.png", dpi=220)
    plt.close(fig)

    apical = model.l56_l23_sg.weights.detach().cpu()
    reciprocal = model.l23_l56_sg.weights.detach().cpu()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    image = axes[0].imshow(apical.numpy(), aspect="auto", cmap="magma")
    axes[0].set(
        xlabel="L2/3 neuron",
        ylabel="L5/6 location neuron",
        title="Learned physical apical synapses",
    )
    fig.colorbar(image, ax=axes[0], label="Synaptic efficacy")
    overlap = F_cosine(apical)
    axes[1].imshow(overlap.numpy(), vmin=0, vmax=1, cmap="viridis")
    axes[1].set(
        xlabel="L5/6 location",
        ylabel="L5/6 location",
        title="Similarity of primed L2/3 patterns",
    )
    image = axes[2].imshow(reciprocal.numpy(), aspect="auto", cmap="viridis")
    axes[2].set(
        xlabel="L2/3 neuron",
        ylabel="L5/6 location neuron",
        title="Learned reciprocal feature-to-frame synapses",
    )
    fig.colorbar(image, ax=axes[2], label="Synaptic efficacy")
    fig.tight_layout()
    fig.savefig(figure_dir / "04_apical_priming.png", dpi=220)
    plt.close(fig)

    cm = np.asarray(controls["normal"]["confusion"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    ax.set(xlabel="Predicted", ylabel="True", title="Held-out confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "05_confusion_matrix.png", dpi=220)
    plt.close(fig)


def F_cosine(weights: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(weights.float(), dim=1)
    return normalized @ normalized.T


def run(args: argparse.Namespace) -> Dict[str, object]:
    seed_everything(args.seed)
    cfg, n_classes, centers, default_l4 = _dataset_spec(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    l4_path = Path(args.l4_weights_path) if args.l4_weights_path else default_l4
    kernels = torch.load(l4_path, map_location="cpu")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    spiking_cfg = FullySpikingConfig(
        l23_k=args.l23_k,
        sensory_steps=args.sensory_steps,
        l4_release_steps=args.l4_release_steps,
        decision_steps=args.decision_steps,
        replay_steps=args.replay_steps,
        decision_synaptic_gain=args.decision_gain,
        decision_tonic_current=args.tonic_current,
        decision_learning_rate=args.learning_rate,
        decision_weight_decay=args.weight_decay,
        decision_margin=args.margin,
        eligibility_tau_pre=args.tau_pre,
        eligibility_tau_post=args.tau_post,
        eligibility_tau=args.tau_eligibility,
        eligibility_threshold=args.eligibility_threshold,
        apical_gain=args.apical_gain,
        apical_learning_rate=args.apical_learning_rate,
        apical_decay=args.apical_decay,
        reciprocal_gain=args.reciprocal_gain,
        reciprocal_learning_rate=args.reciprocal_learning_rate,
        reciprocal_decay=args.reciprocal_decay,
        reciprocal_max=args.reciprocal_max,
        reciprocal_trace_tau=args.reciprocal_trace_tau,
        random_state=args.seed,
    )
    model = FullySpikingCorticalColumn(
        n_classes,
        len(centers),
        kernels,
        spiking_cfg,
        device,
    )
    train_loader, val_loader = _loaders(
        cfg, args.dataset, args.feature_batch_size, args.seed
    )

    event_path = output_dir / "cortical_spike_events.pt"
    if args.events_path:
        event_path = Path(args.events_path)
    if event_path.exists() and not args.regenerate_events:
        payload = torch.load(event_path, map_location="cpu")
        train_events = payload["train_events"].bool()
        train_y = payload["train_y"].long()
        val_events = payload["val_events"].bool()
        val_y = payload["val_y"].long()
        event_stats = payload.get("event_stats", {})
    else:
        train_events, train_y, train_stats = _collect_events(
            model,
            train_loader,
            centers,
            args.train_samples,
            "train cortical spikes",
        )
        val_events, val_y, val_stats = _collect_events(
            model,
            val_loader,
            centers,
            args.eval_samples,
            "held-out cortical spikes",
        )
        event_stats = {"train": train_stats, "validation": val_stats}
        torch.save(
            {
                "kind": "binary_cortical_spike_events",
                "train_events": train_events,
                "train_y": train_y,
                "val_events": val_events,
                "val_y": val_y,
                "event_stats": event_stats,
                "centers": centers,
                "cfg": spiking_cfg.__dict__,
            },
            event_path,
        )

    if args.training_mode == "joint":
        binding_epochs = int(args.joint_binding_epochs)
        model.fit_joint_spike_events(
            train_events,
            train_y,
            epochs=args.epochs,
            batch_size=args.plasticity_batch_size,
            binding_epochs=min(binding_epochs, int(args.epochs)),
            binding_update_steps=args.joint_binding_update_steps,
            seed=args.seed,
        )
        acquisition_history = list(model.training_history)
        for item in acquisition_history:
            item["training_stage"] = "acquisition"
        if args.consolidation_epochs > 0:
            remaining_binding_epochs = max(
                binding_epochs - int(args.epochs),
                0,
            )
            if remaining_binding_epochs > 0:
                model.fit_joint_spike_events(
                    train_events,
                    train_y,
                    epochs=args.consolidation_epochs,
                    batch_size=args.plasticity_batch_size,
                    binding_epochs=remaining_binding_epochs,
                    binding_update_steps=args.joint_binding_update_steps,
                    seed=args.seed + 100,
                )
            else:
                model.fit_spike_events(
                    train_events,
                    train_y,
                    epochs=args.consolidation_epochs,
                    batch_size=args.plasticity_batch_size,
                    seed=args.seed + 100,
                )
            consolidation_history = []
            for item in model.training_history:
                adjusted = dict(item)
                adjusted["epoch"] = float(
                    int(args.epochs) + int(item["epoch"])
                )
                adjusted["training_stage"] = "consolidation"
                adjusted.setdefault("joint_binding_active", 0.0)
                adjusted.setdefault(
                    "apical_weight_norm",
                    float(model.l56_l23_sg.weights.norm().detach().cpu()),
                )
                adjusted.setdefault("apical_norm_change", 0.0)
                adjusted.setdefault(
                    "reciprocal_weight_norm",
                    float(model.l23_l56_sg.weights.norm().detach().cpu()),
                )
                adjusted.setdefault("reciprocal_spike_updates", 0.0)
                consolidation_history.append(adjusted)
            model.training_history = acquisition_history + consolidation_history
    else:
        model.fit_apical_spike_events(
            train_events,
            epochs=args.apical_epochs,
            batch_size=args.plasticity_batch_size,
        )
        model.fit_reciprocal_spike_events(
            train_events,
            epochs=args.reciprocal_epochs,
            batch_size=args.plasticity_batch_size,
            seed=args.seed,
        )
        model.fit_spike_events(
            train_events,
            train_y,
            epochs=args.epochs,
            batch_size=args.plasticity_batch_size,
            seed=args.seed,
        )
        initial_history = list(model.training_history)
        if args.consolidation_epochs > 0:
            model.fit_spike_events(
                train_events,
                train_y,
                epochs=args.consolidation_epochs,
                batch_size=args.plasticity_batch_size,
                seed=args.seed + 100,
            )
            consolidation_history = []
            for item in model.training_history:
                adjusted = dict(item)
                adjusted["epoch"] = float(
                    int(args.epochs) + int(item["epoch"])
                )
                adjusted["training_stage"] = "consolidation"
                consolidation_history.append(adjusted)
            for item in initial_history:
                item["training_stage"] = "acquisition"
            model.training_history = initial_history + consolidation_history
    event_controls = _evaluate_event_controls(
        model,
        val_events,
        val_y,
        n_classes,
        min(args.plasticity_batch_size, 512),
        args.seed,
    )
    reciprocal_binding = _evaluate_reciprocal_binding(
        model,
        val_events,
        batch_size=min(args.plasticity_batch_size, 512),
        gain=args.reciprocal_eval_gain,
    )

    if args.shuffled_label_control:
        event_controls["shuffled_training_labels_accuracy"] = (
            _train_shuffled_label_control(
                model,
                train_events,
                train_y,
                val_events,
                val_y,
                args.epochs,
                args.consolidation_epochs,
                args.plasticity_batch_size,
                args.seed,
            )
        )

    # This is the acceptance metric: raw held-out images traverse every
    # spiking population. The binary replay cache is not used here.
    live_result = model.evaluate_images(
        val_loader,
        centers,
        args.eval_samples,
    )
    live_metrics = _metrics(
        live_result["predictions"],
        live_result["targets"],
        n_classes,
    )
    event_predictions = model.evaluate_spike_events(
        val_events,
        val_y,
        min(args.plasticity_batch_size, 512),
    )["predictions"]
    event_live_agreement = float(
        (
            event_predictions[:len(live_result["predictions"])]
            == live_result["predictions"]
        ).float().mean()
    )

    checkpoint_path = output_dir / "best_fully_spiking_column.pt"
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "centers": centers,
        "l4_weights_path": str(l4_path),
        "event_cache_path": str(event_path),
        "training_mode": args.training_mode,
        "joint_binding_epochs": (
            int(args.joint_binding_epochs)
            if args.training_mode == "joint"
            else 0
        ),
        "joint_binding_update_steps": (
            int(args.joint_binding_update_steps)
            if args.training_mode == "joint"
            else 0
        ),
        "decision_learning_rule": "online_free_trial_three_factor",
        "reciprocal_binding_rule": "local_l23_l56_pre_post_coactivity",
        "learning_replays": False,
        "forced_output_spikes": False,
        "live_metrics": live_metrics,
        "event_controls": event_controls,
        "reciprocal_feature_to_frame": reciprocal_binding,
        "event_live_agreement": event_live_agreement,
    }
    torch.save(model.checkpoint(metadata), checkpoint_path)
    _plot_reports(output_dir, model, event_controls, val_events, val_y)

    report = {
        "kind": "fully_spiking_cortical_column_experiment",
        "dataset": args.dataset,
        "backend": model.backend,
        "gradient_free": True,
        "surrogate_derivatives": False,
        "training_mode": args.training_mode,
        "raw_image_live_metrics": live_metrics,
        "event_controls": event_controls,
        "reciprocal_feature_to_frame": reciprocal_binding,
        "event_cache_live_agreement": event_live_agreement,
        "event_stats": event_stats,
        "training_history": model.training_history,
        "spiking_config": spiking_cfg.__dict__,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "event_cache": str(event_path),
            "figures": str(output_dir / "figures"),
        },
    }
    with (output_dir / "fully_spiking_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["caltech", "mnist"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--events-path")
    parser.add_argument("--l4-weights-path")
    parser.add_argument(
        "--training-mode",
        choices=["staged", "joint"],
        default="joint",
    )
    parser.add_argument("--joint-binding-epochs", type=int)
    parser.add_argument("--joint-binding-update-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--plasticity-batch-size", type=int, default=1024)
    parser.add_argument("--l23-k", type=int)
    parser.add_argument("--sensory-steps", type=int)
    parser.add_argument("--l4-release-steps", type=int)
    parser.add_argument("--decision-steps", type=int, default=8)
    parser.add_argument("--replay-steps", type=int, default=0)
    parser.add_argument("--decision-gain", type=float, default=0.3)
    parser.add_argument("--tonic-current", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--tau-pre", type=float, default=0.1)
    parser.add_argument("--tau-post", type=float, default=8.0)
    parser.add_argument("--tau-eligibility", type=float, default=128.0)
    parser.add_argument("--eligibility-threshold", type=float, default=0.1)
    parser.add_argument("--apical-gain", type=float, default=0.01)
    parser.add_argument("--apical-learning-rate", type=float, default=0.002)
    parser.add_argument("--apical-decay", type=float, default=0.001)
    parser.add_argument("--apical-epochs", type=int, default=1)
    parser.add_argument("--reciprocal-gain", type=float, default=0.0)
    parser.add_argument("--reciprocal-learning-rate", type=float, default=0.2)
    parser.add_argument("--reciprocal-decay", type=float, default=0.0)
    parser.add_argument("--reciprocal-max", type=float, default=1.0)
    parser.add_argument("--reciprocal-trace-tau", type=float, default=8.0)
    parser.add_argument("--reciprocal-epochs", type=int, default=1)
    parser.add_argument("--reciprocal-eval-gain", type=float, default=1.0)
    parser.add_argument("--consolidation-epochs", type=int)
    parser.add_argument("--regenerate-events", action="store_true")
    parser.add_argument("--shuffled-label-control", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    defaults = {
        "caltech": {
            "train_samples": 320,
            "eval_samples": 80,
            "l23_k": 5,
            "sensory_steps": 32,
            "l4_release_steps": 32,
            "learning_rate": 0.02,
            "margin": 2.0,
            "epochs": 3,
            "consolidation_epochs": 0,
        },
        "mnist": {
            "train_samples": 20000,
            "eval_samples": 5000,
            "l23_k": 16,
            "sensory_steps": 48,
            "l4_release_steps": 48,
            "learning_rate": 0.05,
            "margin": 6.0,
            "epochs": 8,
            "consolidation_epochs": 2,
        },
    }[args.dataset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.joint_binding_epochs is None:
        args.joint_binding_epochs = int(args.epochs)
    run(args)


if __name__ == "__main__":
    main()
