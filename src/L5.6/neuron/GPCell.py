class GPCell(LIF) : 
    
    """
    Works as both L6 & L5 layers toghether as Grid-Place Cell getting speed vector as input.

    Args : 
        (float) L : The movement scaler, used as a scaler for the amount of effect the speed vector has on GPCells movement.
        (tuple) V : Speed-Vector throughout the iterations.
        (float) I_amp : Constant injected amplitude current.
    """

    def __init__(
        self,
        R,
        threshold,
        tau,
        v_reset,
        v_rest,
        *args,
        init_v=None,
        init_s=None,
        **kwargs
    ):
        super().__init__(
                *args,
                R=R,
                tau=tau,
                threshold=threshold,
                v_reset=v_reset,
                v_rest=v_rest,
                init_v=init_v,
                init_s=init_s,
                **kwargs
            )

    def initialize(self, neurons) :

        super().initialize(neurons)

        self.L = self.parameter("L", 1)
        self.V = self.parameter("V", required = True)
        self.I_amp = self.parameter("I_amp", 5)

        neurons.spike_prev = neurons.vector("zeros") < 0
    
    def forward(self, neurons) : 

        newPosX = neurons.x[neurons.spike_prev] + self.V[0][neurons.network.iteration] * self.L
        newPosX = ( (newPosX + torch.max(neurons.x) + torch.max(neurons.x)*2) % (torch.max(neurons.x) * 2)) - torch.max(neurons.x)
        newPosX = torch.round(newPosX)


        newPosY = neurons.y[neurons.spike_prev] + self.V[1][neurons.network.iteration] * self.L
        newPosY = ((newPosY + torch.max(neurons.y) + torch.max(neurons.y)*2) % (torch.max(neurons.y) * 2))  - torch.max(neurons.y)
        newPosY = torch.round(newPosY)

        inX = torch.isin(neurons.x, newPosX)
        inY = torch.isin(neurons.y, newPosY)

        # print("iteration :", neurons.network.iteration)
        # print("V: ", (self.V[0][neurons.network.iteration], self.V[1][neurons.network.iteration]))
        # print("prev X: ", neurons.x[neurons.spike_prev])
        # print("new X: ", newPosX)
        # print("-----------------------------------")

        # neurons.I += 1
        neurons.I[torch.logical_and(inX, inY)] += self.I_amp

        neurons.v += (
            (self._Fu(neurons) + self._RIu(neurons)) * neurons.network.dt / neurons.tau
        )

    def Fire(self, neurons) : 

        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset

        neurons.spike_prev = neurons.spikes



class SampleAncher(Behavior) : 

    """
    Sample Input Neuron to activate the anchering of GPCells in first iteration.

    Args : 
        ----
    """

    def initialize(self, neurons) : 
        neurons.I = neurons.vector("zeros")
        neurons.spikes = neurons.vector("zeros")
        neurons.v = neurons.vector(-80)

    def forward(self, neurons) : 
        if(neurons.network.iteration == 1) : 
            neurons.spikes = neurons.vector(1)
        else :
            neurons.spikes = neurons.vector(0)