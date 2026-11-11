"""Synaptic transmission and local STDP/R-STDP learning rules."""

import math

import torch
import torch.nn.functional as F
from conex.behaviors.synapses import BaseDendriticInput, BaseLearning
from pymonntorch import Behavior


class DriveDendriticInput(BaseDendriticInput):
    """Transmit a held feed-forward current, e.g. pooled L4 spikes of
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

    def __init__(
        self, steps, *args, lr=0.05, a_plus=1.0, a_minus=0.1, homeostasis=0.05, **kwargs
    ):
        super().__init__(
            *args,
            steps=steps,
            lr=lr,
            a_plus=a_plus,
            a_minus=a_minus,
            homeostasis=homeostasis,
            **kwargs,
        )

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
        score = (
            F.conv2d(pre, synapse.weights).view(b, n_features, -1)
            - self.homeostasis * usage[None, :, None]
        )
        winners = F.one_hot(score.argmax(1), n_features).permute(0, 2, 1).to(pre.dtype)
        winners = winners * synapse.post_spike.view(b, n_features, -1)
        counts = winners.sum((0, 2))
        patches = F.unfold(pre, synapse.weights.shape[-2:])
        p = torch.einsum("bfp,bkp->fk", winners, patches) / counts[:, None].clamp(
            min=1.0
        )
        w = synapse.raw_weights.view(n_features, -1)
        dw = self.lr * (self.a_plus * p * (1 - w) - self.a_minus * (1 - p) * w)
        active = counts > 0
        w[active] = (w[active] + dw[active]).clamp(0.0, 1.0)
        self.scale(synapse)
        synapse.wins += counts


class ImmediateIdentityInput(BaseDendriticInput):
    def calculate_input(self, synapse):
        return synapse.src.spikes.to(self.def_dtype)


class PairSTDP(Behavior):
    def initialize(self, synapse):
        self.scale = self.parameter("scale", 1.0)
        if not hasattr(synapse, "raw"):
            synapse.raw = torch.zeros_like(synapse.weights)
        b = synapse.network.batch_size
        synapse.local_pre = synapse.weights.new_zeros(b, synapse.weights.shape[0])
        synapse.local_post = synapse.weights.new_zeros(b, synapse.weights.shape[1])

    def forward(self, synapse):
        if not synapse.network.binding:
            return
        b = synapse.network.batch_size
        pre = synapse.src.spikes.reshape(b, -1).float()
        post = synapse.dst.spikes.reshape(b, -1).float()
        synapse.local_pre *= math.exp(-0.5)
        synapse.local_post *= math.exp(-0.5)
        synapse.raw.add_(
            0.001 * (synapse.local_pre.T @ post - 0.1 * pre.T @ synapse.local_post)
        ).clamp_(min=0)
        synapse.local_pre += pre
        synapse.local_post += post
        synapse.weights = (
            self.scale * synapse.raw / synapse.raw.sum(1, keepdim=True).clamp(min=1e-12)
        ).clamp(0, 1)


class BroadcastSynapse(Behavior):
    def initialize(self, synapse):
        synapse.indices = self.parameter("indices", required=True)

    def forward(self, synapse):
        if synapse.network.phase == "sense":
            synapse.I = synapse.src.spikes[:, synapse.indices].float()


class DenseCurrent(Behavior):
    def initialize(self, synapse):
        self.sign = self.parameter("sign", 1.0)

    def forward(self, synapse):
        if synapse.network.phase == "sense":
            synapse.I = self.sign * (synapse.src.spikes.float() @ synapse.weights)


class CenteredRSTDP(Behavior):
    """Three factors only: presynaptic activity, postsynaptic activity, dopamine.

    The expected output is the postsynaptic neuron's own noise-free input current.
    No location, class index, label, or unrelated population is read here.
    """

    def forward(self, synapse):
        net = synapse.network
        if not net.learning:
            return
        if net.phase == "feedback":
            synapse.eligibility *= 0.999
            synapse.weights.add_(
                net.lr
                * torch.einsum("bic,b->ic", synapse.eligibility, net.dopamine)
                / len(net.dopamine)
            ).clamp_(0, 1)
            synapse.eligibility.zero_()
            return
        if net.phase != "sense":
            return
        pre = synapse.src.spikes.float()
        post = synapse.dst.spike_window
        expected = synapse.dst.current.clamp(0, 1)
        residual = post - expected[None]
        steps = net.window
        lam = 0.999
        time = torch.arange(1, steps + 1, dtype=torch.float64, device=net.device)
        decay = (lam ** (steps - time)).float()[:, None, None]
        amplitude = (lam * (1 - lam**time) / (1 - lam)).float()[:, None, None]
        back = ((steps - time + 1) * lam ** (steps - time + 1)).float()[:, None, None]
        prior_post = synapse.post_trace - synapse.expected_trace
        synapse.eligibility *= lam**steps
        synapse.eligibility += (
            synapse.pre_trace[:, :, None] * (lam**steps * residual.sum(0))[:, None, :]
        )
        local_term = (decay * amplitude * residual).sum(0)
        local_term -= net.a_minus * (
            steps * lam**steps * prior_post + (back * residual).sum(0)
        )
        synapse.eligibility += pre[:, :, None] * local_term[:, None, :]
        synapse.pre_trace *= lam**steps
        synapse.pre_trace += pre * float(lam * (1 - lam**steps) / (1 - lam))
        synapse.post_trace *= lam**steps
        synapse.post_trace += (decay * lam * post).sum(0)
        synapse.expected_trace *= lam**steps
        synapse.expected_trace += expected * float(lam * (1 - lam**steps) / (1 - lam))


class PooledBurstInput(DriveDendriticInput):
    def calculate_input(self, synapse):
        if synapse.network.drive is None:
            return synapse.I.new_zeros(synapse.dst.size)
        return (
            (25 * synapse.src.held_current[:, synapse.network.slot]).round().flatten()
        )


class HeldSpikeTransmission(Behavior):
    def initialize(self, synapse):
        self.locations = self.parameter("locations", None)
        self.classes = self.parameter("classes", None)
        self.frame_size = self.parameter("frame_size", None)

    def forward(self, synapse):
        if synapse.network.phase != "sense":
            return
        spikes = synapse.src.held_spikes[:, synapse.network.slot]
        if self.locations is not None:
            # Fixed excitatory convergence of homologous cells in the frames.
            spikes = spikes.reshape(len(spikes), self.classes, self.frame_size)[
                :, :, self.locations
            ].any(1)
        synapse.dst.external = spikes
