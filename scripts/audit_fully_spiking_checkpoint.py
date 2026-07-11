"""Structural and causal audit for a saved fully-spiking column checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scc.fully_spiking as fully_spiking_module
from scc.fully_spiking import load_fully_spiking_column


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apical_interventions(
    model,
    observed_events: torch.Tensor,
    settle_steps: int = 8,
) -> Dict[str, object]:
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
        model._simulate(max(settle_steps - 1, 0))
        primed_membrane.append(model.l23_ng.v[0].detach().cpu())
        model.net.cortical_phase = "l23_competition"
        model._simulate(1)
        primed_spikes.append(model.l23_ng.spikes[0].detach().cpu())

    primed = torch.stack(primed_membrane).clamp(min=0)
    spike_prime = torch.stack(primed_spikes)
    observed_cpu = observed.cpu()
    cosine = F.normalize(primed, dim=1) @ F.normalize(observed_cpu, dim=1).T
    matched = cosine.diag()
    reversed_match = cosine[
        torch.arange(model.n_locations),
        torch.arange(model.n_locations - 1, -1, -1),
    ]
    k = min(int(model.cfg.l23_k), model.n_l23_features)
    predicted_top = primed.topk(k, dim=1).indices
    observed_top = observed_cpu.topk(k, dim=1).indices
    top_k_recall = (
        predicted_top.unsqueeze(2) == observed_top.unsqueeze(1)
    ).any(dim=2).float().mean()
    observed_mask = F.one_hot(
        observed_top,
        num_classes=model.n_l23_features,
    ).any(dim=1)
    spike_overlap = (
        (spike_prime & observed_mask).sum(dim=1).float() / float(k)
    ).mean()
    return {
        "settle_steps": settle_steps,
        "matched_cosine": float(matched.mean()),
        "reversed_location_cosine": float(reversed_match.mean()),
        "matched_minus_reversed": float((matched - reversed_match).mean()),
        f"top_{k}_feature_recall": float(top_k_recall),
        "primed_neurons_per_location": spike_prime.sum(dim=1).tolist(),
        "mean_primed_membrane": float(primed.mean()),
        "max_primed_membrane": float(primed.max()),
        "spike_overlap_fraction": float(spike_overlap),
        "cosine_matrix": cosine.tolist(),
        "primed_membrane": primed,
        "observed_rates": observed_cpu,
    }


def _plot_apical(
    metrics: Dict[str, object],
    output: Path,
) -> None:
    primed = metrics["primed_membrane"]
    observed = metrics["observed_rates"]
    cosine = torch.tensor(metrics["cosine_matrix"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    image0 = axes[0].imshow(primed.numpy(), aspect="auto", cmap="magma")
    axes[0].set(
        xlabel="L2/3 neuron",
        ylabel="Stimulated L5/6 neuron",
        title="Causal apical intervention",
    )
    fig.colorbar(image0, ax=axes[0], label="L2/3 membrane priming")
    image1 = axes[1].imshow(observed.numpy(), aspect="auto", cmap="magma")
    axes[1].set(
        xlabel="L2/3 neuron",
        ylabel="Observed location",
        title="Observed sensory spike probability",
    )
    fig.colorbar(image1, ax=axes[1], label="Spike probability")
    image2 = axes[2].imshow(cosine.numpy(), vmin=0, vmax=1, cmap="viridis")
    axes[2].set(
        xlabel="Observed location",
        ylabel="Stimulated L5/6 neuron",
        title="Prime-to-sensory cosine",
    )
    fig.colorbar(image2, ax=axes[2], label="Cosine similarity")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _learning_mechanism_audit(
    model,
    events: torch.Tensor,
) -> Dict[str, object]:
    """Verify free-trial eligibility and the absence of replay/clamping paths."""
    module_source = inspect.getsource(fully_spiking_module)
    class_source = inspect.getsource(type(model))
    sample = events[:min(len(events), 64)]
    weights_before = model.effective_decision_weights.detach().clone()
    result = model.run_spike_events(sample, track_eligibility=True)
    weights_after = model.effective_decision_weights.detach().clone()
    pre_eligibility_observed = bool(
        model.l23_decision_sg.eligibility.sum() > 0
    )
    post_eligibility_observed = bool(
        model.l23_decision_sg.post_eligibility.sum() > 0
    )

    model.reset_trial(1)
    model._set_motor_location(0, pulse=True)
    model.net.cortical_phase = "sensory"
    model._simulate(1)
    selected_current = model.net.active_location_current[0].detach().cpu()
    selected_spikes = model.l56_ng.spikes[0].detach().cpu()

    return {
        "no_class_replay_method": not hasattr(model, "_class_replay"),
        "no_replay_phase_in_runtime_source": not (
            'cortical_phase = "replay"' in module_source
            or "cortical_phase == \"replay\"" in module_source
        ),
        "no_forced_class_path_in_runtime_source": (
            "forced_class" not in module_source
        ),
        "fit_source_calls_single_free_presentation": (
            class_source.count(
                "result = self.run_spike_events(batch_x, track_eligibility=True)"
            )
            == 1
        ),
        "natural_decision_spikes_observed": bool(
            result.decision_spike_counts.sum() > 0
        ),
        "natural_presynaptic_eligibility_observed": pre_eligibility_observed,
        "natural_postsynaptic_eligibility_observed": post_eligibility_observed,
        "inference_did_not_change_decision_weights": bool(
            torch.equal(weights_before, weights_after)
        ),
        "l56_lif_tag": "LIF" in model.l56_ng.tags,
        "l56_selected_current_is_one_hot": bool(
            (selected_current > 0).sum() == 1
            and selected_current[0] == float(model.cfg.l56_motor_gain)
        ),
        "l56_selected_spike_is_one_hot": bool(
            selected_spikes.sum() == 1 and selected_spikes[0]
        ),
        "l56_membrane_reset_after_spike": bool(
            torch.count_nonzero(model.l56_ng.v) == 0
        ),
    }


def run(checkpoint_path: Path, events_path: Path, output_path: Path) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fully_spiking_column(str(checkpoint_path), device=device)
    payload = torch.load(events_path, map_location="cpu")
    val_events = payload["val_events"].bool()
    val_y = payload["val_y"].long()
    evaluation = model.evaluate_spike_events(val_events, val_y, batch_size=512)

    reloaded = load_fully_spiking_column(str(checkpoint_path), device=device)
    repeated = reloaded.evaluate_spike_events(val_events, val_y, batch_size=512)
    apical = _apical_interventions(model, payload["train_events"].bool())
    mechanism = _learning_mechanism_audit(model, payload["train_events"].bool())
    figure_path = output_path.parent / "figures" / "06_causal_apical_audit.png"
    _plot_apical(apical, figure_path)

    serializable_apical = {
        key: value
        for key, value in apical.items()
        if not torch.is_tensor(value)
    }
    report = {
        "checkpoint": str(checkpoint_path),
        "cortical_event_cache": str(events_path),
        "backend": model.backend,
        "structural_checks": {
            "sensory_group_spiking": "Spiking" in model.sensory_ng.tags,
            "l4_group_spiking": "Spiking" in model.l4_ng.tags,
            "l23_group_spiking": "Spiking" in model.l23_ng.tags,
            "l56_group_spiking": "Spiking" in model.l56_ng.tags,
            "l56_group_lif": "LIF" in model.l56_ng.tags,
            "decision_group_spiking": "Spiking" in model.decision_ng.tags,
            "physical_apical_synapse": "Apical" in model.l56_l23_sg.tags,
            "location_gated_decision_synapse": (
                "L56Gated" in model.l23_decision_sg.tags
            ),
            "decision_weights_gradient_free": not bool(
                model.effective_decision_weights.requires_grad
            ),
            "cached_events_are_binary": val_events.dtype == torch.bool,
            "checkpoint_online_free_trial_learning": (
                model.checkpoint()["decision_learning_rule"]
                == "online_free_trial_three_factor"
            ),
            "checkpoint_learning_replays_disabled": (
                model.checkpoint()["learning_replays"] is False
            ),
            "checkpoint_forced_output_spikes_disabled": (
                model.checkpoint()["forced_output_spikes"] is False
            ),
            "configured_replay_steps_zero": int(model.cfg.replay_steps) == 0,
        },
        "learning_mechanism_audit": mechanism,
        "event_accuracy": float(evaluation["accuracy"]),
        "reload_predictions_identical": bool(
            torch.equal(evaluation["predictions"], repeated["predictions"])
        ),
        "causal_apical_priming": serializable_apical,
        "hashes": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "cortical_event_cache_sha256": _sha256(events_path),
        },
        "figure": str(figure_path),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.checkpoint), Path(args.events), Path(args.output))


if __name__ == "__main__":
    main()
