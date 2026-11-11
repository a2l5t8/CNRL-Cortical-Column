"""Fresh retinal spike simulation, L4 burst coding and fixed spatial pooling.

Pooled activity is held only within the current trial for the subsequent
context and decision phases. There is no offline feature replay cache.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from conex import (
    AveragePool2D,
    Conv2dDendriticInput,
    Fire,
    NeuronAxon,
    PostSpikeCatcher,
    PreSpikeCatcher,
    PreTrace,
    SensorySetter,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    SynapseInit,
    WeightInitializer,
    prioritize_behaviors,
)
from pymonntorch import NeuronDimension, NeuronGroup, SynapseGroup

from .behavior.neurons import IF
from .behavior.specs import PoolTrace
from .behavior.synapse import Conv2dWTASTDP
from .simulation import activate, gate, set_batch, simulate


@dataclass
class L4Config:
    patch_size: int = 35
    pool: int = 5  # L2/3 pools L4 over a pool x pool grid
    steps: int = 48  # sensory window; L4 then releases for as many steps
    l4_threshold: float = 1.0
    stdp_lr: float = 0.05
    stdp_a_plus: float = 1.0
    stdp_a_minus: float = 0.1
    stdp_homeostasis: float = 0.05


def fixation_geometry(grid: int):
    """Return crop width and row-major fixation centers on an 80-pixel image."""
    if not 1 <= grid <= 80:
        raise ValueError("Grid side must be between 1 and 80")
    side = min(80, round(105 / grid))
    axis = [round(80 * (i + 1) / (grid + 1)) for i in range(grid)]
    axis = [min(max(center, side // 2), 80 - side + side // 2) for center in axis]
    return side, [(row, col) for row in axis for col in axis]


def phase_code(rates: torch.Tensor, steps: int) -> torch.Tensor:
    """Regular spike trains of non-leaky IF retinal cells with constant drive
    ``rates`` in [0, 1]: a cell fires whenever its phase crosses 1."""
    phase, spikes = torch.zeros_like(rates), []
    for _ in range(steps):
        phase = phase + rates
        s = phase >= 1
        phase = phase - s.to(phase.dtype)
        spikes.append(s)
    return torch.stack(spikes)


def extract_patches(
    images: torch.Tensor, centers: Sequence[Tuple[int, int]], size: int
) -> torch.Tensor:
    """(b, n_fixations, size, size) foveal patches centred on ``centers``."""
    images = images.reshape(len(images), *images.shape[-2:])
    half = size // 2
    return torch.stack(
        [
            images[:, r - half : r - half + size, c - half : c - half + size]
            for r, c in centers
        ],
        1,
    )


class L4:
    def __init__(self, net, kernels: torch.Tensor, cfg: L4Config | None = None):
        self.cfg = cfg or L4Config()
        self.net = net
        self.device = device = net.device
        prior = set([*self.net.NeuronGroups, *self.net.SynapseGroups])
        self.net.batch_size = 1
        n_features = kernels.shape[0]

        self._build_retina()
        self._build_l4(n_features, kernels.shape[-1])
        self._build_pool(n_features)
        self.pool.add_behavior(350, PoolTrace(), initialize=False)
        self._connect_l4(kernels.float().to(device))
        self._connect_pool()
        self.objects = [
            obj
            for obj in [*self.net.NeuronGroups, *self.net.SynapseGroups]
            if obj not in prior
        ]

    def _build_retina(self) -> None:
        """Retinal cells emitting the phase-coded spike trains set from outside."""
        size = self.cfg.patch_size
        self.retina = NeuronGroup(
            net=self.net,
            size=NeuronDimension(depth=1, height=size, width=size),
            tag="Retina",
            behavior=prioritize_behaviors([SensorySetter(), NeuronAxon()]),
        )

    def _build_l4(self, n_features: int, kernel_size: int) -> None:
        """Convolutional IF maps; ``release`` (their Fire) is off during the sensory window."""
        size = self.cfg.patch_size - kernel_size + 1
        self.release = Fire()
        self.neurons = NeuronGroup(
            net=self.net,
            size=NeuronDimension(depth=n_features, height=size, width=size),
            tag="L4",
            behavior=prioritize_behaviors(
                [
                    SimpleDendriteStructure(),
                    SimpleDendriteComputation(),
                    IF(threshold=self.cfg.l4_threshold),
                    self.release,
                    NeuronAxon(),
                ]
            ),
        )

    def _build_pool(self, n_features: int) -> None:
        """Non-firing IF integrators whose potential is the pooled L4 drive."""
        pool = self.cfg.pool
        self.pool = NeuronGroup(
            net=self.net,
            size=NeuronDimension(depth=n_features, height=pool, width=pool),
            tag="PoolIntegrators",
            behavior=prioritize_behaviors(
                [SimpleDendriteStructure(), SimpleDendriteComputation(), IF()]
            ),
        )

    def _connect_l4(self, kernels: torch.Tensor) -> None:
        """Retina -> L4 convolution with competitive STDP (``stdp``, off unless learning)."""
        cfg = self.cfg
        self.stdp = Conv2dWTASTDP(
            steps=cfg.steps,
            lr=cfg.stdp_lr,
            a_plus=cfg.stdp_a_plus,
            a_minus=cfg.stdp_a_minus,
            homeostasis=cfg.stdp_homeostasis,
        )
        self.conv = SynapseGroup(
            net=self.net,
            src=self.retina,
            dst=self.neurons,
            tag="Proximal",
            behavior=prioritize_behaviors(
                [
                    SynapseInit(),
                    WeightInitializer(weights=kernels),
                    Conv2dDendriticInput(),
                    PreSpikeCatcher(),
                    PostSpikeCatcher(),
                    PreTrace(tau_s=float("inf")),
                    self.stdp,
                ]
            ),
        )

    def _connect_pool(self) -> None:
        """L4 -> L2/3 average pooling."""
        SynapseGroup(
            net=self.net,
            src=self.neurons,
            dst=self.pool,
            tag="Proximal",
            behavior=prioritize_behaviors(
                [SynapseInit(), AveragePool2D(), PreSpikeCatcher()]
            ),
        )

    @property
    def kernels(self) -> torch.Tensor:
        return self.conv.raw_weights.detach().cpu().clone()

    @torch.no_grad()
    def drive(
        self,
        images: torch.Tensor,
        centers: Sequence[Tuple[int, int]],
        learn: bool = False,
    ) -> torch.Tensor:
        """Pooled L4 drive of L2/3, (b, n_fixations, L2/3 size), one fixation at a time."""
        activate(self.net, self.objects)
        steps = self.cfg.steps
        patches = extract_patches(
            images.to(self.device, torch.float32), centers, self.cfg.patch_size
        ).clamp(0, 1)
        out = []
        for i in range(patches.shape[1]):
            self.net.reset_pool_trace = i == 0
            set_batch(self.net, len(patches), self.objects)
            self.net.slot = i
            spikes = phase_code(patches[:, i].flatten(1), steps)
            gate(self.neurons, self.release, False)
            gate(self.conv, self.stdp, False)
            for t in range(steps):  # sensory window: L4 integrates silently
                self.net.x = spikes[t].view(-1)
                simulate(self.net)
            self.net.x = None
            gate(self.neurons, self.release, True)
            gate(self.conv, self.stdp, learn)  # learning sees the first release step
            simulate(self.net)
            gate(self.conv, self.stdp, False)
            simulate(self.net, steps - 1)
            gate(self.neurons, self.release, False)
            self.net.capture_pool = True
            simulate(self.net)  # L2/3 receives the last released spikes
            self.net.capture_pool = False
            out.append(self.pool.v.view(len(patches), -1).clone())
        return torch.stack(out, 1)
