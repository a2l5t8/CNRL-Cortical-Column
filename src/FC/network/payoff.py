from pymonntorch import *
from conex import *
import torch


class ConfidenceLevelPayOff(Payoff) : 

    """
    General Description of ConfidenceLevelPayOff Behavior.
    """

    def __init__(self, *args, initial_payoff=0.0, **kwargs):
        super().__init__(*args, initial_payoff=initial_payoff, **kwargs)

    def initialize(self, network) : 

        """
        Args : 
            confidence_level (float) : the percentage of the the maximum population activity to make a decision, if none has reached the threshold, is does not change payoff.
            interval (int) : to be added
            max_iter (int) : to be added
            reward (float) : to be added
            punish (float) : to be added
        """
        
        super().initialize(network)
        self.confidence_level = self.parameter("confidence_level", 0.6)
        self.interval = self.parameter("interval", 5)
        self.max_iter = self.parameter("max_iter", 200)
        
        self.reward = self.parameter("reward", 1)
        self.punish = self.parameter("punish", -1)

<<<<<<< Updated upstream
=======
        self.low_confidence_interval = 0
        self.classes = self.parameter("classes", 2)

        self.offset = self.parameter("offset", 500)

        network.decision = -1
        
>>>>>>> Stashed changes
    def forward(self, network) : 
        ng_classes = network.find_objects("target")

<<<<<<< Updated upstream
        tot = 0
        acts = []
        for ng in ng_classes :
            act = torch.sum(ng["spikes", 0][:,0] > max(0, network.iteration - self.interval), 0)
            acts.append(act)
            tot += act
=======
        ngs = network.find_objects("fc_pop")
        shaped_spikes = torch.Tensor([])
        for i in range(self.classes) : 
            shaped_spikes = torch.cat((shaped_spikes, ngs[i].spikes))
        shaped_spikes = shaped_spikes.reshape((self.classes, -1))
        acts = torch.sum(shaped_spikes, 1, dtype=torch.float32)
        tot = torch.sum(acts, 0).item()

        if(tot == 0) : 
            network.payoff = 0
            return 

        acts /= tot
        
        if(acts.max() < self.confidence_level and self.low_confidence_interval < self.max_iter) : 
            self.low_confidence_interval += 1
            network.payoff = 0
            return
        
        self.low_confidence_interval = 0
        network.decision = acts.argmax()

        if(network.decision == network.targets) : 
            network.payoff = self.reward
        else :
            network.payoff = self.punish

    def forward2(self, network) : 
        if(network.iteration < self.offset) : 
            return

        ng = network.find_objects("target")[0]
        shaped_spikes = ng.spikes.view((ng.depth, ng.width))

        acts = torch.sum(shaped_spikes, 1, dtype=torch.float32)
        tot = torch.sum(acts, 0).item()

        if(tot == 0) : 
            network.payoff = 0
            return 
>>>>>>> Stashed changes

        acts = torch.Tensor(acts)
        acts /= tot

        if(max(acts) < self.confidence_level) : 
            network.payoff = 0
            return
        
        prediction = acts.argmax()
        if(prediction == network.target) : 
            network.payoff = self.reward
        else :
            network.payoff = self.punish
        

class TimeWindowPayOff(Payoff) : 

    def __init__(self, *args, initial_payoff=0.0, **kwargs):
        super().__init__(*args, initial_payoff=initial_payoff, **kwargs)

    def initialize(self, network) : 
        super().initialize(network)

        self.time_window = self.parameter("time_window", 100)

    def forward(self, network) : 
        pass
