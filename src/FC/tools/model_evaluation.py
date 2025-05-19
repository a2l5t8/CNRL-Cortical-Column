import torch

from conex import *
from pymonntorch import *

def accuracy_score(fc_ng, K, window_size, targets, dataset_size, offset = 0) -> float : 
    acc_score = 0
    for i in range(dataset_size) : 
        y = targets[i]
        y_hat = model_prediction(
            fc_ng = fc_ng, 
            K = K, 
            st_iter = i * window_size, 
            en_iter = (i + 1) * window_size, 
            offset = offset
        )
        acc_score += (y == y_hat)
    
    return acc_score/dataset_size


def accuracy_score_ngs(fc_ngs, K, window_size, targets, dataset_size, offset = 0) -> float : 
    acc_score = 0
    for i in range(dataset_size) : 
        y = targets[i]
        y_hat = model_prediction_ngs(
            fc_ngs = fc_ngs, 
            K = K, 
            st_iter = i * window_size, 
            en_iter = (i + 1) * window_size, 
            offset = offset
        )
        acc_score += (y == y_hat)
    
    return acc_score/dataset_size

def model_prediction_ngs(fc_ngs, K, st_iter, en_iter, offset) : 
    acts = []
    shaped_spikes = torch.Tensor([])
    for i in range(K) : 
        fc_spike = torch.logical_and(fc_ngs[i]["spikes", 0][:, 0] >= offset + st_iter, fc_ngs[i]["spikes", 0][:, 0] <= offset + en_iter)
        act = torch.sum(fc_spike, dim = 0)
        acts.append(act)
    
    return torch.argmax(torch.tensor(acts))


def model_prediction(fc_ng, K, st_iter, en_iter, offset) : 
    ng_spikes = []
    N = fc_ng.size
    for i in range(K) : 
        st_ng = (N//K) * i
        en_ng = (N//K) * (i + 1)

        iteration_constraint = torch.logical_and(fc_ng["spikes", 0][:, 0] >= offset + st_iter, fc_ng["spikes", 0][:, 0] < offset + en_iter)
        class_constraint = torch.logical_and(fc_ng["spikes", 0][:, 1] >= st_ng, fc_ng["spikes", 0][:, 1] < en_ng)
        spikes = torch.sum(torch.logical_and(iteration_constraint, class_constraint), dim = 0)
        ng_spikes.append(spikes)
    
    return torch.argmax(torch.tensor(ng_spikes))