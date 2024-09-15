from pymonntorch import *
from conex import *
import torch

import matplotlib.pyplot as plt

from FC import FC

fclayer = FC(K = 5, N = 50)
fclayer.initialize()
fclayer.simulate_iterations(100)

for i in range(fclayer.K) : 
    plt.plot(fclayer.net.NeuronGroups[i]['spikes.t', 0], fclayer.net.NeuronGroups[i]['spikes.i', 0] + (i * 50), '.')
plt.show()