"""Custom CoNeX/PyMoNNtorch behaviors used by the cortical column."""

from scc.behaviors.neuron import (
    SensoryPhaseEncoder,
    MotorCommandSpikes,
    CurrentDrivenL56LIF,
    BurstingL4Neurons,
    CompetitiveL23Neurons,
    DecisionLIF,
)
from scc.behaviors.synapse import (
    BatchedConv2dSTDP,
    MotorToL56Current,
    SpikingConvolutionInput,
    L4ToL23PoolingInput,
    L56ApicalFeedbackInput,
    L23ToL56ReciprocalBindingInput,
    L56GatedDecisionInput,
    SpikeTimingEligibility,
    DelayedDopaminePlasticity,
)

__all__ = [
    "SensoryPhaseEncoder",
    "MotorCommandSpikes",
    "CurrentDrivenL56LIF",
    "BurstingL4Neurons",
    "CompetitiveL23Neurons",
    "DecisionLIF",
    "BatchedConv2dSTDP",
    "MotorToL56Current",
    "SpikingConvolutionInput",
    "L4ToL23PoolingInput",
    "L56ApicalFeedbackInput",
    "L23ToL56ReciprocalBindingInput",
    "L56GatedDecisionInput",
    "SpikeTimingEligibility",
    "DelayedDopaminePlasticity",
]
