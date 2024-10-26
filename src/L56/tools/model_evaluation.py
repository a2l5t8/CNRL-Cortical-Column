import torch

def cosine_similarity(
    refrence_frame,
    f_iteration,
    s_iteration,
) -> torch.Tensor:
    fsp = refrence_frame.vector()
    fsp[refrence_frame["spikes", 0][refrence_frame["spikes", 0][:,0] == f_iteration][:,1]] = 1

    ssp = refrence_frame.vector()
    ssp[refrence_frame["spikes", 0][refrence_frame["spikes", 0][:,0] == s_iteration][:,1]] = 1
    
    value = torch.cosine_similarity(fsp, ssp, dim=0)
    return value
    

