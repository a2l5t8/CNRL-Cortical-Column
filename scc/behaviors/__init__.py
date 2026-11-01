"""Behaviors the cortical column adds to CoNeX.

Each behavior is registered in CoNeX's priority table, so populations are
assembled with ``conex.prioritize_behaviors`` like any CoNeX behavior.
"""
from conex.nn.priority import ALL_PRIORITIES, NETWORK_PRIORITIES, NEURON_PRIORITIES, SYNAPSE_PRIORITIES

from scc.behaviors.network import *  # noqa: F401,F403
from scc.behaviors.neuron import *  # noqa: F401,F403
from scc.behaviors.synapse import *  # noqa: F401,F403
from scc.behaviors import network, neuron, synapse

_PRIORITIES = (
    (NETWORK_PRIORITIES, {"MarginPayoff": 100}),
    (NEURON_PRIORITIES, {"IF": 260, "MarginWTA": 300}),
    (SYNAPSE_PRIORITIES, {
        "ReplayDendriticInput": 180, "DenseDendriticInput": 180, "PathIntegrationDendriticInput": 180,
        "FrameDendriticInput": 180, "FrameGatedDendriticInput": 180,
        "Conv2dWTASTDP": 500, "Outstar": 500, "GatedRSTDP": 500,
    }),
)
for table, entries in _PRIORITIES:
    table.update(entries)
    ALL_PRIORITIES.update(entries)

__all__ = [*network.__all__, *neuron.__all__, *synapse.__all__]
