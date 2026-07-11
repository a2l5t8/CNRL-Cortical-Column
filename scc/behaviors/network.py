"""Custom behaviors attached directly to the cortical network.

The current runtime does not require a custom network-level behavior. Network
state transitions are orchestrated by FullySpikingCorticalColumn, while every
scheduled custom Behavior is owned by a NeuronGroup or SynapseGroup.
"""

__all__: list[str] = []

