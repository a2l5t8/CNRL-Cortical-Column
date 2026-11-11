import torch
import pytest
from column import Column, Config
from column.DecisionLayer import DecisionLayer


@pytest.fixture
def model():
    torch.set_num_threads(2)
    m = Column(
        Config(
            maps=2, kernel_size=5, pool=2, winners=2, sensory_steps=8, decision_steps=8
        )
    )
    m.net.initialize(info=False, warnings=False)
    return m


def weights(model):
    return [
        s.clone()
        for s in (
            model.l4.conv.raw_weights,
            model.l23.apical.raw,
            model.l56.apical.raw,
            model.decision.w,
        )
    ]


def test_all_circuits_share_native_network(model):
    from conex import Neocortex
    from pymonntorch import NeuronGroup, SynapseGroup

    assert isinstance(model.net, Neocortex)
    assert (
        model.l4.net
        is model.l23.net
        is model.l56.net
        is model.decision.net
        is model.net
    )
    assert all(isinstance(n, NeuronGroup) for n in model.net.NeuronGroups)
    assert all(isinstance(s, SynapseGroup) for s in model.net.SynapseGroups)


@pytest.mark.parametrize("classes", [2, 10])
def test_reference_frame_spikes_reach_decision_through_fixed_synapses(classes):
    model = Column(
        Config(
            classes=classes,
            maps=2,
            kernel_size=5,
            pool=2,
            winners=2,
            sensory_steps=8,
            decision_steps=8,
        )
    )
    model.net.initialize(info=False, warnings=False)
    model.predict(torch.rand(2, 80, 80))
    frames = model.l56.neurons.held_spikes.reshape(2, 9, classes, 26)
    object_spikes = frames[:, :, :, model.l56.object_locations]
    expected = torch.eye(9, dtype=torch.bool, device="cuda")[None, :, None, :].expand(
        2, 9, classes, 9
    )
    assert torch.equal(object_spikes, expected)
    assert torch.equal(model.decision.locations.external, object_spikes[:, -1].any(1))


def test_every_plastic_pathway_updates_and_evaluation_is_immutable(model):
    x = torch.rand(2, 80, 80)
    before = weights(model)
    model.net.baseline = 0.5
    model.train(x, torch.tensor([0, 1]), shuffle=False)
    after = weights(model)
    assert all(not torch.equal(a, b) for a, b in zip(before, after))
    first = model.predict(x).clone()
    counts = model.decision.counts.clone()
    model.predict(torch.rand_like(x))
    assert torch.equal(first, model.predict(x))
    assert torch.equal(counts, model.decision.counts)
    assert all(torch.equal(a, b) for a, b in zip(after, weights(model)))
    with pytest.raises(TypeError):
        model.predict(x, torch.tensor([0, 1]))


def test_selective_plasticity(model):
    before = weights(model)
    model.plasticity(("l4",))
    model.train(torch.rand(2, 80, 80), torch.tensor([0, 1]), shuffle=False)
    after = weights(model)
    assert not torch.equal(before[0], after[0])
    assert all(torch.equal(a, b) for a, b in zip(before[1:], after[1:]))


def test_checkpoint_roundtrip(model, tmp_path):
    x = torch.rand(2, 80, 80)
    model.train(x, torch.tensor([0, 1]), shuffle=False)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = Column.load(path)
    assert all(torch.equal(a, b) for a, b in zip(weights(model), weights(restored)))
    assert torch.equal(model.predict(x), restored.predict(x))
    assert torch.equal(model.decision.counts, restored.decision.counts)


def test_event_advance_matches_explicit_local_tick_equations():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    from conex import Neocortex
    from column.behavior.network import RewardPulse

    net = Neocortex(device="cpu", behavior={10: RewardPulse()})
    m = DecisionLayer(net, 8, 2, 2)
    net.initialize(info=False, warnings=False)
    m.reset(3, learning=True)
    w = m.w.clone()
    v = torch.zeros(3, 2)
    counts = torch.zeros_like(v)
    # Double-precision oracle avoids cancellation of long post/expected traces.
    pre_trace = torch.zeros(3, 72, dtype=torch.float64)
    post_trace = torch.zeros_like(v, dtype=torch.float64)
    expected_trace = torch.zeros_like(v, dtype=torch.float64)
    eligibility = torch.zeros(3, 72, 2, dtype=torch.float64)
    for q in range(9):
        feature = (torch.rand(3, 8) > 0.7).float()
        pre = torch.zeros(3, 72)
        pre[:, q * 8 : (q + 1) * 8] = feature
        base = 0.5 + m.net.gain * (pre @ w - 0.5 * pre.sum(1, keepdim=True))
        current = (base + m.neurons.bias_noise).clamp(0, 1)
        for tick in range(64):
            v += current
            post = (v >= 1).float()
            v -= post
            counts += post
            pre_trace = 0.999 * (pre_trace + pre)
            post_trace = 0.999 * (post_trace + post)
            expected_trace = 0.999 * (expected_trace + base.clamp(0, 1))
            eligibility = (
                0.999 * eligibility
                + pre_trace[:, :, None] * (post - base.clamp(0, 1))[:, None, :]
            )
            eligibility -= (
                0.1 * pre[:, :, None] * (post_trace - expected_trace)[:, None, :]
            )
        m.present(feature, q, 64)
    assert torch.equal(counts, m.counts)
    # Accumulation-order roundoff in the event integral is bounded explicitly.
    assert torch.allclose(
        eligibility.float(), m.synapse.eligibility, atol=0.003, rtol=0.0001
    )
    reward = torch.tensor([1.0, 0.0, 1.0])
    expected = (
        (
            w
            + 0.003
            * torch.einsum("bic,b->ic", eligibility * 0.999, reward.double())
            / 3
        )
        .clamp(0, 1)
        .float()
    )
    m.feedback(reward)
    assert torch.allclose(expected, m.w, atol=4e-6, rtol=0)
    with pytest.raises(RuntimeError):
        m.feedback(reward)


def test_training_resumes_exactly_after_save(model, tmp_path):
    x = torch.rand(4, 80, 80)
    labels = torch.tensor([0, 1, 0, 1])
    model.train(x, labels, batch_size=2, validation_data=(x, labels))
    model.plasticity(("bindings", "decision"))
    path = tmp_path / "resume.pt"
    model.save(path)
    restored = Column.load(path)
    assert restored.stages == model.stages
    assert restored.history == model.history
    expected = model.train(x, labels, batch_size=2)
    actual = restored.train(x, labels, batch_size=2)
    assert actual == expected
    assert all(torch.equal(a, b) for a, b in zip(weights(model), weights(restored)))
    assert torch.equal(model.predict(x), restored.predict(x))


def test_canonical_api_rejects_invalid_inputs(model):
    with pytest.raises(ValueError):
        model.predict(torch.rand(2, 35, 35))
    with pytest.raises(ValueError):
        model.train(torch.rand(2, 80, 80), [0, 2])
    with pytest.raises(ValueError):
        model.predict(torch.rand(2, 80, 80), batch_size=0)
    assert model.predict(torch.empty(0, 80, 80)).numel() == 0
