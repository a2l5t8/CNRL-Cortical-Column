"""Tests for the stateful fully spiking cortical-column runtime."""
from __future__ import annotations

import torch

from scc.fully_spiking import (
    FullySpikingConfig,
    FullySpikingCorticalColumn,
    load_fully_spiking_column,
)


def _tiny_column(n_classes: int = 3) -> FullySpikingCorticalColumn:
    kernels = torch.tensor(
        [
            [[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]],
            [[[0.0, 0.5, 1.0], [0.5, 1.0, 0.5], [1.0, 0.5, 0.0]]],
        ]
    )
    cfg = FullySpikingConfig(
        patch_size=7,
        feature_map_size=5,
        pool_grid_size=1,
        l23_k=1,
        sensory_steps=4,
        l4_release_steps=4,
        decision_steps=3,
        replay_steps=0,
        decision_learning_rate=0.2,
        eligibility_tau_pre=0.1,
        eligibility_tau=1e6,
        apical_gain=0.2,
    )
    return FullySpikingCorticalColumn(
        n_classes=n_classes,
        n_locations=2,
        l4_kernels=kernels,
        cfg=cfg,
        device="cpu",
    )


def test_complete_path_emits_binary_spikes_and_live_l56_path():
    model = _tiny_column()
    images = torch.rand(4, 1, 9, 9)
    result = model.run_images(images, [(3, 3), (5, 5)])

    assert result.l23_spike_events.dtype == torch.bool
    assert result.l56_spike_events.dtype == torch.bool
    assert result.l23_spike_events.shape == (4, 2, 2)
    assert result.l56_spike_events.shape == (4, 2, 2)
    assert torch.equal(
        result.l56_spike_events[0],
        torch.tensor([[True, False], [False, True]]),
    )
    assert torch.all(result.l23_spike_events.sum(dim=2) == 1)
    assert model.sensory_ng.spikes.dtype == torch.bool
    assert model.l4_ng.spikes.dtype == torch.bool
    assert model.decision_ng.spikes.dtype == torch.bool


def test_l56_is_current_driven_lif_with_no_forced_spike_assignment():
    model = _tiny_column()
    model.reset_trial(batch_size=1)

    model._set_motor_location(1, pulse=True)
    model.net.cortical_phase = "sensory"
    model._simulate(1)

    assert torch.equal(
        model.net.active_location_current,
        torch.tensor([[0.0, model.cfg.l56_motor_gain]]),
    )
    assert torch.equal(
        model.l56_ng.spikes,
        torch.tensor([[False, True]]),
    )
    assert torch.equal(model.l56_ng.v, torch.zeros_like(model.l56_ng.v))
    assert "LIF" in model.l56_ng.tags

    model._set_motor_location(1, pulse=False)
    model._simulate(1)
    assert torch.equal(
        model.l56_ng.spikes,
        torch.tensor([[False, True]]),
    )


def test_l56_spikes_select_location_specific_decision_synapses():
    model = _tiny_column(n_classes=2)
    events = torch.tensor([[[True, False], [True, False]]])
    with torch.no_grad():
        model.l23_decision_sg.excitatory_weights.zero_()
        model.l23_decision_sg.inhibitory_weights.zero_()
        model.l23_decision_sg.excitatory_weights[0, 0, 0] = 4.0
        model.l23_decision_sg.excitatory_weights[1, 1, 0] = 4.0

    normal = model.run_spike_events(events).decision_spike_counts
    reversed_events = model.run_spike_events(events.flip(1)).decision_spike_counts

    assert normal[0, 0] > 0
    assert normal[0, 1] > 0
    # Removing the second-location event removes only class 1 evidence.
    ablated = events.clone()
    ablated[:, 1] = False
    ablated_counts = model.run_spike_events(ablated).decision_spike_counts
    assert ablated_counts[0, 0] > ablated_counts[0, 1]
    assert reversed_events.shape == normal.shape


def test_online_outcome_modulation_uses_only_free_trial_eligibility():
    model = _tiny_column(n_classes=3)
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    targets = torch.tensor([1, 2])
    competitors = torch.tensor([0, 0])
    mask = torch.tensor([True, True])
    result = model.run_spike_events(events, track_eligibility=True)
    assert result.decision_spike_counts.sum() > 0
    assert model.l23_decision_sg.eligibility.shape == (2, 2, 2)
    assert model.l23_decision_sg.post_eligibility.shape == (2, 3)
    assert model.l23_decision_sg.eligibility.sum() > 0
    assert model.l23_decision_sg.post_eligibility.sum() > 0

    before_e = model.l23_decision_sg.excitatory_weights.clone()
    before_i = model.l23_decision_sg.inhibitory_weights.clone()
    model._apply_outcome_modulation(targets, competitors, mask)
    after_e = model.l23_decision_sg.excitatory_weights
    after_i = model.l23_decision_sg.inhibitory_weights

    assert torch.equal(after_e[0], before_e[0])
    assert (after_e[1] > before_e[1]).any()
    assert (after_e[2] > before_e[2]).any()
    assert (after_i[0] > before_i[0]).any()
    assert torch.equal(after_i[1], before_i[1])
    assert torch.equal(after_i[2], before_i[2])
    assert not model.effective_decision_weights.requires_grad
    assert not hasattr(model, "_class_replay")


def test_actual_apical_synapse_primes_location_specific_l23_neurons():
    model = _tiny_column(n_classes=2)
    model.reset_trial(batch_size=1)
    with torch.no_grad():
        model.l56_l23_sg.weights.zero_()
        model.l56_l23_sg.weights[0, 0] = 1.0
        model.l56_l23_sg.weights[1, 1] = 1.0
    model.net.apical_learning_enabled = False

    model._set_motor_location(0, pulse=True)
    model.net.cortical_phase = "l4_release"
    model._simulate(1)
    location_zero_v = model.l23_ng.v.clone()

    model.l23_ng.v.zero_()
    model._set_motor_location(1, pulse=True)
    model._simulate(1)
    location_one_v = model.l23_ng.v.clone()

    assert location_zero_v[0, 0] > location_zero_v[0, 1]
    assert location_one_v[0, 1] > location_one_v[0, 0]
    assert "Apical" in model.l56_l23_sg.tags


def test_reciprocal_l23_to_l56_synapse_retrieves_locations():
    model = _tiny_column(n_classes=2)
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[True, False], [False, True]],
        ]
    )
    model.cfg.reciprocal_gain = 1.0
    model.fit_reciprocal_spike_events(events, epochs=1, batch_size=4)
    result = model.predict_l56_from_l23_events(
        events.reshape(-1, 2),
        steps=3,
        gain=1.0,
        record_first=True,
    )

    assert torch.equal(result["predictions"], torch.tensor([0, 1, 0, 1]))
    assert result["first_raster"].dtype == torch.bool
    assert result["first_raster"].shape == (3, 2)
    assert (model.l23_l56_sg.weights > 0).any()
    assert not model.l23_l56_sg.weights.requires_grad
    assert "Reciprocal" in model.l23_l56_sg.tags


def test_joint_free_trial_updates_both_bindings_and_decision_synapses():
    model = _tiny_column(n_classes=2)
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    labels = torch.tensor([0, 1])
    decision_before = model.effective_decision_weights.clone()

    model.fit_joint_spike_events(
        events,
        labels,
        epochs=1,
        batch_size=2,
        seed=3,
    )

    assert (model.l56_l23_sg.weights > 0).any()
    assert (model.l23_l56_sg.weights > 0).any()
    assert not torch.equal(model.effective_decision_weights, decision_before)
    assert model.training_history[0]["joint_binding_active"] == 1.0
    assert model.training_history[0]["learning_replays"] == 0.0
    assert model.training_history[0]["forced_output_spikes"] == 0.0


def test_joint_binding_plasticity_can_close_while_decision_learning_continues():
    model = _tiny_column(n_classes=2)
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    labels = torch.tensor([0, 1])

    model.fit_joint_spike_events(
        events,
        labels,
        epochs=2,
        binding_epochs=1,
        batch_size=2,
        seed=3,
    )

    assert model.training_history[0]["joint_binding_active"] == 1.0
    assert model.training_history[1]["joint_binding_active"] == 0.0
    assert model.training_history[1]["apical_norm_change"] == 0.0
    assert model.training_history[1]["reciprocal_spike_updates"] == 0.0


def test_joint_binding_weights_are_independent_of_class_labels():
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
            [[True, False], [True, False]],
            [[False, True], [False, True]],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1])
    shuffled = labels.flip(0)
    normal_model = _tiny_column(n_classes=2)
    shuffled_model = _tiny_column(n_classes=2)

    normal_model.fit_joint_spike_events(
        events,
        labels,
        epochs=1,
        batch_size=2,
        seed=7,
    )
    shuffled_model.fit_joint_spike_events(
        events,
        shuffled,
        epochs=1,
        batch_size=2,
        seed=7,
    )

    assert torch.equal(
        normal_model.l56_l23_sg.weights,
        shuffled_model.l56_l23_sg.weights,
    )
    assert torch.equal(
        normal_model.l23_l56_sg.weights,
        shuffled_model.l23_l56_sg.weights,
    )


def test_joint_and_decision_only_rules_produce_identical_decision_weights():
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
            [[True, False], [True, False]],
            [[False, True], [False, True]],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1])
    decision_only = _tiny_column(n_classes=2)
    joint = _tiny_column(n_classes=2)

    decision_only.fit_spike_events(
        events,
        labels,
        epochs=2,
        batch_size=2,
        seed=11,
    )
    joint.fit_joint_spike_events(
        events,
        labels,
        epochs=2,
        batch_size=2,
        binding_epochs=1,
        seed=11,
    )

    assert torch.equal(
        decision_only.l23_decision_sg.excitatory_weights,
        joint.l23_decision_sg.excitatory_weights,
    )
    assert torch.equal(
        decision_only.l23_decision_sg.inhibitory_weights,
        joint.l23_decision_sg.inhibitory_weights,
    )


def test_raw_image_dynamics_records_each_scheduler_step_without_learning():
    model = _tiny_column(n_classes=2)
    image = torch.rand(1, 1, 9, 9)
    decision_before = model.effective_decision_weights.clone()
    apical_before = model.l56_l23_sg.weights.clone()

    record = model.record_image_dynamics(image, [(3, 3), (5, 5)])

    steps_per_location = (
        model.cfg.sensory_steps
        + model.cfg.l4_release_steps
        + 1
        + model.cfg.decision_steps
    )
    expected_steps = model.n_locations * steps_per_location
    assert record["sensory_spikes"].shape == (
        expected_steps,
        model.sensory_ng.size,
    )
    assert record["l4_spikes"].shape == (
        expected_steps,
        model.l4_ng.size,
    )
    assert record["l23_spikes"].dtype == torch.bool
    assert record["l56_spikes"].dtype == torch.bool
    assert record["decision_spikes"].dtype == torch.bool
    assert record["patches"].shape == (2, 1, 7, 7)
    assert len(record["phase"]) == expected_steps
    assert torch.equal(decision_before, model.effective_decision_weights)
    assert torch.equal(apical_before, model.l56_l23_sg.weights)


def test_checkpoint_round_trip_preserves_spike_decisions(tmp_path):
    model = _tiny_column(n_classes=2)
    events = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    labels = torch.tensor([0, 1])
    model.fit_spike_events(events, labels, epochs=2, batch_size=2)
    before = model.evaluate_spike_events(events, labels)["predictions"]
    path = tmp_path / "fully_spiking.pt"
    torch.save(model.checkpoint(), path)

    restored = load_fully_spiking_column(str(path), device="cpu")
    after = restored.evaluate_spike_events(events, labels)["predictions"]

    assert torch.equal(before, after)
    assert torch.equal(
        model.effective_decision_weights,
        restored.effective_decision_weights,
    )
    assert torch.equal(
        model.l23_l56_sg.weights,
        restored.l23_l56_sg.weights,
    )
