"""Synapse behaviors the cortical column adds to CoNeX.

Trials are simulated in batches: every population holds ``batch_size`` copies
of its neurons, flattened trial-major, so CoNeX's element-wise, convolutional
and pooling behaviors run unchanged. The behaviors below reshape to
(batch, neurons) where they mix neurons across a connection.
"""
import torch
import torch.nn.functional as F
from conex.behaviors.synapses import BaseDendriticInput, BaseLearning

__all__ = [
    "ReplayDendriticInput",
    "DenseDendriticInput",
    "PathIntegrationDendriticInput",
    "FrameDendriticInput",
    "FrameGatedDendriticInput",
    "Conv2dWTASTDP",
    "Outstar",
    "GatedRSTDP",
]


def _position(motor, b):
    """Sensor position of each trial: the index of the spiking motor unit."""
    return motor.spikes.view(b, -1).float().argmax(1)


class ReplayDendriticInput(BaseDendriticInput):
    """Replay a recorded feed-forward current, e.g. the pooled L4 spikes of
    one fixation recorded once from the sensory pathway (which receives no
    feedback, so recording and replaying it is exact).

    Args:
        source (str): Network attribute holding the current; None is silence.
    """

    def __init__(self, *args, source, current_coef=1, **kwargs):
        super().__init__(*args, source=source, current_coef=current_coef, **kwargs)

    def initialize(self, synapse):
        super().initialize(synapse)
        self.source = self.parameter("source", None, required=True)

    def calculate_input(self, synapse):
        current = getattr(synapse.network, self.source, None)
        return synapse.I.new_zeros(synapse.dst.size) if current is None else current


class DenseDendriticInput(BaseDendriticInput):
    """Fully connected dendrite: I = s @ W in each trial, W of shape (src, dst)."""

    def calculate_input(self, synapse):
        spikes = synapse.pre_spike.view(synapse.network.batch_size, -1)
        return (spikes.to(synapse.weights.dtype) @ synapse.weights).view(-1)


class _FrameInput(BaseDendriticInput):
    """Base of the reference-frame dendrites.

    Args:
        frames (ReferenceFrames): Frame geometry (location table, object slots).
        motor (NeuronGroup): Motor population; its spike gives the sensor position.
    """

    def __init__(self, *args, frames, motor, current_coef=1, **kwargs):
        super().__init__(*args, frames=frames, motor=motor, current_coef=current_coef, **kwargs)

    def initialize(self, synapse):
        super().initialize(synapse)
        self.frames = self.parameter("frames", None, required=True)
        self.motor = self.parameter("motor", None, required=True)

    def locations(self, synapse):
        """Location index l = p - d of every anchor d at the current position p."""
        b = synapse.network.batch_size
        return self.frames.table[_position(self.motor, b)]  # (b, D)


class PathIntegrationDendriticInput(_FrameInput):
    """Hypotheses -> location units. Hypothesis (c, d) and sensor position p
    drive location l = p - d of frame c (fixed conjunctive wiring): the anchor
    stays put while movement changes the location."""

    def calculate_input(self, synapse):
        f, b = self.frames, synapse.network.batch_size
        hyp = synapse.pre_spike.view(b, f.C, f.D).to(self.def_dtype)
        loc = self.locations(synapse)[:, None, :].expand(-1, f.C, -1)
        return hyp.new_zeros(b, f.C, f.L).scatter_add_(2, loc, hyp).view(-1)


class FrameDendriticInput(_FrameInput):
    """Reciprocal L2/3 -> hypothesis synapses. Hypothesis (c, d) receives the
    log-likelihood that code x arose at location l = p - d of object c,
        I = sum_f x_f log(A / (1 - A)) + sum_f log(1 - A),
    with the log-odds in ``weights`` (C, L, F) and the second term in ``bias`` (C, L)."""

    def calculate_input(self, synapse):
        f, b = self.frames, synapse.network.batch_size
        x = synapse.pre_spike.view(b, -1).to(self.def_dtype)
        ll = torch.einsum("bf,clf->bcl", x, synapse.weights) + synapse.bias
        return ll.gather(2, self.locations(synapse)[:, None, :].expand(-1, f.C, -1)).view(-1)


class FrameGatedDendriticInput(_FrameInput):
    """L2/3 -> decision synapses W (C, Q, F) over the Q on-object locations.

    Decision unit (c, d) belongs to hypothesis (c, d): while that hypothesis
    spikes, the unit integrates the code through the synapses of the object
    location l = p - d it currently occupies (no input off the object).

    Args:
        hypotheses (NeuronGroup): The frame-hypothesis population gating the units.
    """

    def __init__(self, *args, hypotheses, **kwargs):
        super().__init__(*args, hypotheses=hypotheses, **kwargs)

    def initialize(self, synapse):
        super().initialize(synapse)
        self.hypotheses = self.parameter("hypotheses", None, required=True)

    def calculate_input(self, synapse):
        f, b = self.frames, synapse.network.batch_size
        synapse.x = x = synapse.pre_spike.view(b, -1).to(self.def_dtype)
        synapse.slot = f.slot[self.locations(synapse)]  # (b, D), -1 off the object
        synapse.alive = self.hypotheses.spikes.view(b, f.C, f.D)
        drive = torch.einsum("cqf,bf->bcq", synapse.weights, x)
        drive = drive.gather(2, synapse.slot.clamp(min=0)[:, None, :].expand(-1, f.C, -1))
        return (drive * ((synapse.slot >= 0)[:, None, :] & synapse.alive)).view(-1)


class Conv2dWTASTDP(BaseLearning):
    """Competitive spike-timing-dependent plasticity of convolutional kernels.

    At each position only the feature whose membrane potential, less a
    homeostatic penalty on its share of past wins, is largest may learn, and
    only if it spiked (lateral inhibition across features). The winner's kernel
    moves towards the presynaptic activity p of its receptive field (spikes per
    sensory step) with soft bounds,
        dw = lr * (a_plus * p * (1 - w) - a_minus * (1 - p) * w),
    averaged over its winning positions. Raw weights stay in [0, 1]; the
    dendrite sees them normalised to unit sum per kernel (synaptic scaling),
    so ``weights`` should be initialised with the raw kernels.

    Note: PreTrace(tau_s=inf) must accumulate the presynaptic spikes of the
    sensory window, and PostSpikeCatcher must be added.

    Args:
        steps (int): Length of the sensory window.
        lr (float): Learning rate.
        a_plus (float): Potentiation coefficient.
        a_minus (float): Depression coefficient.
        homeostasis (float): Penalty per unit share of past wins.
    """

    def __init__(self, steps, *args, lr=0.05, a_plus=1.0, a_minus=0.1, homeostasis=0.05, **kwargs):
        super().__init__(*args, steps=steps, lr=lr, a_plus=a_plus, a_minus=a_minus,
                         homeostasis=homeostasis, **kwargs)

    def initialize(self, synapse):
        self.steps = self.parameter("steps", None, required=True)
        self.lr = self.parameter("lr", 0.05)
        self.a_plus = self.parameter("a_plus", 1.0)
        self.a_minus = self.parameter("a_minus", 0.1)
        self.homeostasis = self.parameter("homeostasis", 0.05)
        if not hasattr(synapse, "wins"):
            synapse.raw_weights = synapse.weights.clone()
            synapse.wins = synapse.weights.new_zeros(synapse.weights.shape[0])
            self.scale(synapse)

    @staticmethod
    def scale(synapse):
        raw = synapse.raw_weights
        synapse.weights = raw / raw.sum((1, 2, 3), keepdim=True).clamp(min=1e-6)

    def forward(self, synapse):
        n_features, b = synapse.weights.shape[0], synapse.network.batch_size
        pre = synapse.pre_trace.view(synapse.src_shape) / self.steps
        usage = synapse.wins / synapse.wins.sum().clamp(min=1.0)
        score = F.conv2d(pre, synapse.weights).view(b, n_features, -1) - self.homeostasis * usage[None, :, None]
        winners = F.one_hot(score.argmax(1), n_features).permute(0, 2, 1).to(pre.dtype)
        winners = winners * synapse.post_spike.view(b, n_features, -1)
        counts = winners.sum((0, 2))
        patches = F.unfold(pre, synapse.weights.shape[-2:])
        p = torch.einsum("bfp,bkp->fk", winners, patches) / counts[:, None].clamp(min=1.0)
        w = synapse.raw_weights.view(n_features, -1)
        dw = self.lr * (self.a_plus * p * (1 - w) - self.a_minus * (1 - p) * w)
        active = counts > 0
        w[active] = (w[active] + dw[active]).clamp(0.0, 1.0)
        self.scale(synapse)
        synapse.wins += counts


class Outstar(BaseLearning):
    """Outstar learning with an experience-dependent rate (Grossberg, 1976).

    Whenever presynaptic unit i spikes, its outgoing weights move towards the
    postsynaptic pattern x, w_ij <- w_ij + (x_j - w_ij) / n_i, where n_i counts
    the spikes of i so far. w_ij converges to P(post j spikes | pre i spikes).

    Note: PreSpikeCatcher and PostSpikeCatcher should be added.
    """

    def initialize(self, synapse):
        if not hasattr(synapse, "n"):
            synapse.n = synapse.weights.new_zeros(synapse.weights.shape[0])

    def forward(self, synapse):
        b = synapse.network.batch_size
        pre = synapse.pre_spike.view(b, -1).to(synapse.weights.dtype)
        post = synapse.post_spike.view(b, -1).to(synapse.weights.dtype)
        k = pre.sum(0)
        synapse.n = synapse.n + k
        synapse.weights += (pre.T @ post - k[:, None] * synapse.weights) / synapse.n.clamp(min=1)[:, None]


class GatedRSTDP(BaseLearning):
    """Three-factor plasticity of the frame-gated decision synapses.

    Eligibility of one trial: a presynaptic tag e[q, f] = 1 once feature f
    spiked while the active hypothesis occupied object location q, and a
    postsynaptic tag of +1 on the unit that spiked while receiving apical
    teaching input (the attended target) and -1 on the most active unit without
    it (the strongest rival, singled out by lateral inhibition). The global
    neuromodulator D (network.dopamine_concentration, one value per trial)
    gates the change
        dW[c, q, f] = lr * D * tag[c] * e[q, f] / max(1, ||e||).
    Increases and decreases accumulate in separate excitatory and inhibitory
    weights (Dale's principle), each bounded to [0, w_max].

    Note: needs FrameGatedDendriticInput (gates), PostTrace(tau_s=inf) (spike
    counts) and SimpleDendriteStructure on the decision units (apical input).

    Args:
        lr (float): Learning rate.
        w_max (float): Bound of the excitatory and inhibitory weights.
    """

    def __init__(self, *args, lr=0.05, w_max=5.0, **kwargs):
        super().__init__(*args, lr=lr, w_max=w_max, **kwargs)

    def initialize(self, synapse):
        self.lr = self.parameter("lr", 0.05)
        self.w_max = self.parameter("w_max", 5.0)
        if not hasattr(synapse, "w_exc"):
            synapse.w_exc = synapse.weights.clone()
            synapse.w_inh = torch.zeros_like(synapse.weights)
        _, Q, n_features = synapse.weights.shape
        synapse.eligibility = synapse.weights.new_zeros(synapse.network.batch_size, Q, n_features)

    def forward(self, synapse):
        net = synapse.network
        b = net.batch_size
        if getattr(synapse, "alive", None) is not None:
            # Presynaptic tags at the object location of the active anchor.
            anchor = synapse.alive.any(1).float().argmax(1)
            q = synapse.slot.gather(1, anchor[:, None]).squeeze(1)
            on = q >= 0
            synapse.eligibility[on, q[on]] = torch.maximum(synapse.eligibility[on, q[on]], synapse.x[on])
            synapse.anchor = anchor
            synapse.alive = None
        D = net.dopamine_concentration
        if not torch.is_tensor(D) or not D.any():
            return
        C = synapse.weights.shape[0]
        idx = synapse.anchor[:, None, None].expand(-1, C, 1)
        counts = synapse.post_trace.view(b, C, -1).gather(2, idx).squeeze(2)
        taught = synapse.dst.I_apical.view(b, C, -1).gather(2, idx).squeeze(2) > 0
        rival = counts.masked_fill(taught, -torch.inf).argmax(1)
        tag = (taught & (counts > 0)).to(counts.dtype)
        tag[torch.arange(b), rival] -= (counts.gather(1, rival[:, None]).squeeze(1) > 0).to(counts.dtype)
        e = synapse.eligibility / synapse.eligibility.flatten(1).norm(dim=1).clamp(min=1)[:, None, None]
        dw = self.lr * torch.einsum("b,bc,bqf->cqf", D, tag, e)
        synapse.w_exc = (synapse.w_exc + dw.clamp(min=0)).clamp(0, self.w_max)
        synapse.w_inh = (synapse.w_inh + (-dw).clamp(min=0)).clamp(0, self.w_max)
        synapse.weights = synapse.w_exc - synapse.w_inh
