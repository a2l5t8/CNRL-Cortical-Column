from pymonntorch import *

class LayerKWTA(Behavior):
    def __init__(
        self,
        k: int, 
        group: str = None,
        *args, 
        **kwargs
    ):
        super().__init__(k = k, group = group,*args, **kwargs)
    
    def initialize(self, layer):
        self.k = self.parameter("k", required=True)
        self.group = self.parameter("group", required=False)
        self.neuron_groups = []
        for ngp in layer.neurongroups:
            if self.group:
                if self.group in ngp.tags:
                    self.neuron_groups.append(ngp)
                continue
            self.neuron_groups.append(ngp)
        self.threshold = self.neuron_groups[0].threshold
        self.v_reset = self.neuron_groups[0].v_reset
        return super().initialize(layer)

    def forward(self, layer):
        tple_v = tuple(map(lambda ng: ng.v, self.neuron_groups))
        all_v = torch.cat(tple_v, dim=0)
        will_spike = all_v >= self.threshold
        if all_v[will_spike].size(0) <= self.k:
            return
        v_values, indices = torch.topk(all_v, k=self.k, dim=0, sorted=False)
        ignored = will_spike
        ignored.scatter_(0, indices, False)
        all_v[ignored] = self.v_reset
        all_v = all_v.reshape(len(self.neuron_groups), -1)
        for ind, ng in enumerate(self.neuron_groups):
            ng.v = all_v[ind]
        return super().forward(layer)

  