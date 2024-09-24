from pymonntorch import *
from conex import *
import torch

net = Neocortex(dt = 1)

net.initialize()
net.simulate_iterations(100)
