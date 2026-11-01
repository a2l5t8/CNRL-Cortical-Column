"""Seeding and batched simulation of CoNeX networks."""
import random

import numpy as np
import torch
from conex import SynapseInit, WeightInitializer
from conex.behaviors.neurons import KWTA
from conex.behaviors.synapses import BaseDendriticInput
from pymonntorch import NeuronDimension

_STRUCTURAL = (SynapseInit, WeightInitializer, NeuronDimension)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_batch(net, b: int) -> None:
    """Hold ``b`` independent trials in every population and reset their state.

    Population tensors are flattened trial-major (b * size), so CoNeX's
    element-wise, convolutional and pooling behaviors run unchanged. Every
    behavior is re-initialised except the structural ones (synapse
    initialisers, NeuronDimension); learning behaviors keep their weights.
    """
    net.batch_size = b
    for ng in net.NeuronGroups:
        ng.unit_size = getattr(ng, "unit_size", ng.size)
        ng.size = b * ng.unit_size
    for sg in net.SynapseGroups:
        sg.unit_shapes = getattr(sg, "unit_shapes", (sg.src_shape, sg.dst_shape))
        sg.src_shape, sg.dst_shape = ((b, *s) for s in sg.unit_shapes)
        sg.src_delay = torch.zeros(sg.src.size, dtype=torch.long, device=net.device)
        sg.dst_delay = torch.zeros(sg.dst.size, dtype=torch.long, device=net.device)
    for obj in [net, *net.NeuronGroups, *net.SynapseGroups]:
        for key in sorted(obj.behavior):
            behavior = obj.behavior[key]
            if not isinstance(behavior, _STRUCTURAL):
                behavior.initialize(obj)
            if isinstance(behavior, KWTA):
                behavior.shape = (b, -1)


def simulate(net, steps: int = 1) -> None:
    """Advance ``steps`` iterations; ``net.monitor`` (if set) is called after each."""
    for _ in range(steps):
        net.simulate_iteration()
        if getattr(net, "monitor", None) is not None:
            net.monitor()


def gate(obj, behavior, enabled: bool) -> None:
    """Switch a behavior on or off; a silenced dendrite stops delivering current."""
    behavior.behavior_enabled = enabled
    if not enabled and isinstance(behavior, BaseDendriticInput):
        obj.I = torch.zeros_like(obj.I)
