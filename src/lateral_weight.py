import torch
from pymonntorch import Behavior


def triple_one_idx(i, v):
    res = [1, 1, 1]
    res[i] = v
    return res


class LatheralWeight2Sparse(Behavior):
    # Put this after weight initializtion timestamp
    # and use SparseDendriticInput instead of usual LateralDendriticInput
    # for dense, flip the value of sparse into False and use SimpleDendriticInput
    def __init__(self, *args, r_sparse=True, **kwargs):
        super().__init__(*args, r_sparse=r_sparse, **kwargs)
        self.r_sparse = r_sparse

    def initialize(self, sg):
        weight = sg.weights[0, 0]  # weight will be 3D
        src_shape = sg.src_shape
        src_stride = (src_shape[1] * src_shape[2], src_shape[2], 1)
        src_numel = src_stride[0] * src_shape[0]

        w_ranges = [
            torch.arange(-(x // 2), 1 + x // 2).reshape(-1, 1) for x in weight.shape
        ]
        inp_ranges = [torch.arange(x).to(torch.float64) for x in src_shape]

        dim_idx = [torch.add(w_ranges[i], inp_ranges[i]) for i in range(3)]
        for i in range(3):
            dim_idx[i][dim_idx[i] < 0] = torch.nan
            dim_idx[i][dim_idx[i] >= src_shape[i]] = torch.nan
        dim_idx = [dim_idx[i] * src_stride[i] for i in range(3)]

        src_idx = sum(
            [
                dim_idx[i].reshape(
                    *triple_one_idx(i, weight.shape[i]),
                    *triple_one_idx(i, src_shape[i])
                )
                for i in range(3)
            ]
        )
        src_idx = src_idx.reshape(weight.numel(), src_numel)

        new_w = weight.view(-1, 1).expand(-1, src_numel)
        dst_idx = torch.arange(src_numel).view(1, -1).expand(weight.numel(), -1)

        sp_idx = torch.stack(
            [
                src_idx.reshape(
                    -1,
                ),
                dst_idx.reshape(
                    -1,
                ),
            ]
        )
        new_w = new_w.reshape(
            -1,
        )

        ok_idx = ~sp_idx[0].isnan()
        sp_idx = sp_idx[:, ok_idx]
        new_w = new_w[ok_idx]
        
        sp_w = torch.sparse_coo_tensor(sp_idx, new_w, (src_numel, src_numel))
        if self.r_sparse:
            sg.weights = sp_w.to_sparse_csc()
        else:
            sg.weights = sp_w.to_dense()

        sg.weights = sg.weights.to(self.device).to(sg.def_dtype)
