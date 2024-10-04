import pymonntorch as pynt
import torch
import math


from conex import (
    LIF,
    Neocortex,
    NeuronDimension,
    prioritize_behaviors,
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    Fire,
    KWTA,
    SpikeTrace,
    NeuronAxon,
    ActivityBaseHomeostasis,
    SynapseInit,
    LateralDendriticInput,
    WeightInitializer,
)


### lateral inhibition
# max_inhibition = 3
# side = pc_shape[1] * 2 + 1
# lateral_kernel = torch.tensor([max_inhibition]).expand(side, side).to(dtype=torch.float)
# r = 3
# n = 9
# center_point = (side // 2, side // 2)

class GPCellLateralInhibition(LateralDendriticInput):
    def __init__(
            self, 
            kernel_side: int,
            r: float,
            n: float,
            max_inhibition: float,
            current_coef = 1,
            inhibitory = None,
            *args, 
            **kwargs
        ):
        super().__init__(kernel_side=kernel_side, r=r, n=n, max_inhibition=max_inhibition, current_coef=current_coef, inhibitory=inhibitory,*args, **kwargs)

    def initialize(self, synapse):
        self.kernel_side = self.parameter("kernel_side", required=True)
        self.r = self.parameter("r", required=True)
        self.n = self.parameter("n", required=True)
        self.max_inhibition = self.parameter("max_inhibition", required=True)   
        self.center_point = (self.kernel_side // 2, self.kernel_side // 2)
        self.lateral_kernel = torch.tensor([self.max_inhibition]).expand(self.kernel_side, self.kernel_side).to(dtype=torch.float)
        for _x in range(self.kernel_side):
            for _y in range(self.kernel_side):
                self.lateral_kernel[_x, _y] = self.lateral_function(_x, _y)
                if self.lateral_kernel[_x, _y] > self.max_inhibition:
                    self.lateral_kernel[_x, _y] = self.max_inhibition
        max = self.lateral_kernel.max()
        if max:
            self.lateral_kernel *= self.max_inhibition / self.lateral_kernel.max()
        synapse.weights = self.lateral_kernel.reshape(1, 1, 1, self.kernel_side, self.kernel_side)
        super().initialize(synapse)
        

    def lateral_function(self, x: float, y: float) -> float:
        result = ((x - self.center_point[0]) ** 2 + (y - self.center_point[1]) ** 2 - self.r**2) ** (1 / self.n)
        if type(result) == float:
            return result
        return 0


# for _x in range(side):
#     for _y in range(side):
#         lateral_kernel[_x, _y] = lateral_function(_x, _y)
#         if lateral_kernel[_x, _y] > max_inhibition:
#             lateral_kernel[_x, _y] = max_inhibition

# max = lateral_kernel.max()
# if max:

#     lateral_kernel *= max_inhibition / lateral_kernel.max()
# lateral_kernel = lateral_kernel.reshape(1, 1, 1, side, side)


