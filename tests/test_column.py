"""Sensory pathway and cortical column built from CoNeX behaviors."""
import torch
import torch.nn.functional as F
from conex import (Fire, Neocortex, NeuronAxon, SimpleDendriteComputation, SimpleDendriteStructure, SynapseInit,
                   prioritize_behaviors)
from pymonntorch import NeuronGroup, SynapseGroup

from scc.behaviors import IF, ReplayDendriticInput
from scc.column import CENTRE, INNER, LATTICE, OFFSETS, ColumnConfig, CorticalColumn
from scc.sensory import SensoryConfig, SensoryPathway, extract_patches, phase_code
from scc.utils import set_batch, simulate


def _codes(n, t, f, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, t, f)
    for i in range(n):
        for j in range(t):
            x[i, j, torch.randperm(f, generator=g)[:k]] = 100.0  # strong drive: these k neurons win
    return x


def test_if_neuron_spike_count_is_the_floor_of_integrated_input():
    net = Neocortex(dt=1)
    net.batch_size = 1
    src = NeuronGroup(net=net, size=4, behavior={})
    ng = NeuronGroup(net=net, size=4, behavior=prioritize_behaviors([
        SimpleDendriteStructure(), SimpleDendriteComputation(), IF(), Fire(), NeuronAxon()]))
    SynapseGroup(net=net, src=src, dst=ng, tag="Proximal",
                 behavior=prioritize_behaviors([SynapseInit(), ReplayDendriticInput(source="drive")]))
    net.initialize(info=False)
    set_batch(net, 2)
    drive = torch.tensor([0.0, 0.25, 0.5, 0.875, 1.0, 0.625, 0.125, 0.75])  # at most one spike per step
    net.drive, count = drive, torch.zeros(8)
    for _ in range(8):
        simulate(net)
        count += ng.spikes
    assert torch.equal(count, torch.floor(8 * drive))


def test_sensory_drive_pools_the_l4_burst_of_the_integrated_retinal_spikes():
    g = torch.Generator().manual_seed(0)
    kernels, image = torch.rand(3, 1, 11, 11, generator=g), torch.rand(2, 80, 80, generator=g)
    centers = [(20, 20), (40, 60)]
    drive = SensoryPathway(kernels, SensoryConfig(steps=8)).drive(image, centers)
    patches = extract_patches(image, centers, 35).clamp(0, 1)
    k = kernels / kernels.sum((1, 2, 3), keepdim=True)
    for i in range(len(centers)):
        spikes = phase_code(patches[:, i].flatten(1), 8).sum(0).float().view(2, 1, 35, 35)
        l4 = F.conv2d(spikes, k).floor().clamp(max=8)  # integrate silently, then release with subtraction
        assert torch.allclose(drive[:, i], F.adaptive_avg_pool2d(l4, 5).flatten(1), atol=1e-4)


def test_location_table_implements_path_integration():
    m = CorticalColumn(2, 10, offsets=OFFSETS)
    f = m.frames
    for p, (r, c) in enumerate(LATTICE):
        for d, (dy, dx) in enumerate(OFFSETS):
            assert int(f.table[p, d]) == f.index(r - dy, c - dx)
    assert int(f.table[0, OFFSETS.index((2, 2))]) == f.background


def test_known_position_reduces_to_the_quantised_linear_readout():
    """Proposition 2: with non-negative input, n_c = floor(sum_t tonic + gain * W_c . x_t)."""
    C, n_features, cfg = 3, 40, ColumnConfig(k=5, decision_gain=0.25, tonic=0.5)
    m = CorticalColumn(C, n_features, cfg, [CENTRE])
    W = torch.randint(0, 17, (C, 9, n_features)).float() / 64  # dyadic, input below one spike per step
    m.synapses.weights = W
    x = _codes(6, 9, n_features, 5)
    out = m.infer(x, INNER, known_anchor=CENTRE)
    code = (x > 0).float()
    expected = torch.floor((cfg.tonic + cfg.decision_gain * torch.einsum("cqf,nqf->nqc", W, code))
                           .cumsum(1) * cfg.decision_steps)
    assert torch.equal(out["code"], x > 0)
    assert torch.equal(out["prediction"], expected)


def test_outstar_bindings_learn_object_specific_frequencies():
    n_features, C = 20, 2
    x = _codes(40, 25, n_features, 4)
    y = torch.tensor([0, 1] * 20)
    m = CorticalColumn(C, n_features, ColumnConfig(k=4), [CENTRE]).learn_bindings(x, y, list(range(25)), batch_size=7)
    for c in range(C):
        for p, (r, col) in enumerate(LATTICE):
            expected = (x[y == c, p] > 0).float().mean(0)
            assert torch.allclose(m.binding[c, m.frames.index(r - 1, col - 1)], expected, atol=1e-5)


def test_three_factor_update_is_the_margin_perceptron_step():
    C, n_features, cfg = 3, 30, ColumnConfig(k=4, lr=0.5, margin=100.0)  # every trial is insecure
    m = CorticalColumn(C, n_features, cfg, [CENTRE])
    w0 = m.decision_weights.clone()
    x, y = _codes(8, 9, n_features, 4, seed=3), torch.arange(8) % C
    counts = m.infer(x, INNER, known_anchor=CENTRE)["prediction"][:, -1]
    m.learn_decisions(x, y, INNER, batch_size=8)
    rival = counts.scatter(1, y[:, None], -torch.inf).argmax(1)
    tag = F.one_hot(y, C).float() - F.one_hot(rival, C).float()
    e = (x > 0).float()
    e = e / e.flatten(1).norm(dim=1).clamp(min=1)[:, None, None]
    assert torch.allclose(m.decision_weights - w0, cfg.lr * torch.einsum("bc,bqf->cqf", tag, e), atol=1e-6)


def test_frames_eliminate_hypotheses_and_the_true_object_survives():
    n_features, C = 30, 3
    g = torch.Generator().manual_seed(1)
    protos = (torch.rand(C, 25, n_features, generator=g) < 0.15).float() * 100
    m = CorticalColumn(C, n_features, ColumnConfig(k=30, elimination_margin=5.0), OFFSETS)
    m.learn_bindings(protos[torch.arange(C).repeat(30)], torch.arange(C).repeat(30), list(range(25)))
    test = torch.zeros(1, 25, n_features)  # object 2 at offset (0, 2): canvas codes follow l = p - d
    for p, (r, col) in enumerate(LATTICE):
        ly, lx = r, col - 2
        if -1 <= ly <= 3 and -1 <= lx <= 3:
            test[0, p] = protos[2, (ly + 1) * 5 + (lx + 1)]
    out = m.infer(test, list(range(25)))
    active, truth = out["active"][0], OFFSETS.index((0, 2))
    assert active[0].sum() == C * len(OFFSETS)  # everything possible before the first sensation
    assert active[-1].sum() < active[0].sum()  # ambiguity narrows
    assert bool(active[-1, 2, truth])  # the true (object, anchor) survives
    assert int(out["evidence"][0, -1].flatten().argmax()) == 2 * len(OFFSETS) + truth
