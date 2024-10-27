import torch
import matplotlib.pyplot as plt

def refrence_frame_raster(
    refrence_frame,
    lib,
    last_itr = None,
    step = None,
):
    left_x = refrence_frame.x[0]
    right_x = refrence_frame.x[-1]
    down_y = refrence_frame.y[0]
    up_y = refrence_frame.y[-1]
    plt.xlim(left_x, right_x)
    plt.ylim(down_y, up_y)
    
    
    itr_spikes = refrence_frame["spikes", 0]
    for itr in range(last_itr):
        plt.title(f"raster for itr: {itr}")
        raster_data = itr_spikes[itr_spikes[:, 0] == itr][:, 1]
        x_events = refrence_frame.x[raster_data] 
        y_events = refrence_frame.y[raster_data]
        plt.scatter(x_events, y_events)
        plt.savefig("{}/fig{}.png".format(lib, itr))
        plt.close()
        

def iter_spike_multi_real(
    locs,
    ng,
    itr,
    save=False,
    step=False,
    color="red",
    scale=50,
    lib="figs",
    label="grid",
    offset_x=0,
    offset_y=0,
    base_offset_x=0,
    base_offset_y=0,
):
    plt.xlim(-25, +25)
    plt.ylim(-25, +25)

    prev = max(0, itr * step - 30)

    ng_idx = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step][:, 1]
    ng_idx_prev = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step - 4][:, 1]
    ng_idx_prev_prev = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step - 8][:, 1]

    pos_X = torch.Tensor(locs[:, 0])
    pos_Y = torch.Tensor(locs[:, 1])

    plt.plot(
        (pos_X[prev : itr * step] + base_offset_x) * -1,
        (pos_Y[prev : itr * step] + base_offset_y) * -1,
        color="gray",
    )

    # plt.plot((ng.x[ng_idx_prev_prev] + offset_x) *(50 / (max(ng.x)*2 + 1)), (ng.y[ng_idx_prev_prev] + offset_y) * (50 / (max(ng.x)*2 + 1)),'o',color = color, markersize = (4 * scale/(ng.shape.width)) + 2, alpha = 0.2)
    # plt.plot((ng.x[ng_idx_prev] + offset_x) *(50 / (max(ng.x)*2 + 1)), (ng.y[ng_idx_prev] + offset_y) * (50 / (max(ng.x)*2 + 1)),'o',color = color, markersize = (4 * scale/(ng.shape.width)) + 2, alpha = 0.35)
    plt.plot(
        (ng.x[ng_idx] + offset_x) * (50 / (max(ng.x) * 2 + 1)),
        (ng.y[ng_idx] + offset_y) * (50 / (max(ng.x) * 2 + 1)),
        "o",
        color=color,
        markersize=(4 * scale / (ng.shape.width)) + 2,
        label=label,
        alpha=1,
    )
    plt.legend(loc="upper right")

    plt.plot(
        (locs[:, 0][itr * step] + base_offset_x) * -1,
        (locs[:, 1][itr * step] + base_offset_y) * -1,
        "^",
        color="red",
        markersize=10,
    )

    plt.title(f"iteration = {itr * step}", y=-0.2)
    plt.suptitle("Grid Module Spikes")

    if save:
        plt.savefig("{}/fig{}.png".format(lib, itr))


def iter_spike_multi(
    pos_x,
    pos_y,
    ng,
    itr,
    save=False,
    step=False,
    color="red",
    scale=50,
    lib="figs",
    label="grid",
    offset_x=0,
    offset_y=0,
    base_offset_x=0,
    base_offset_y=0,
):
    plt.xlim(-25, +25)
    plt.ylim(-25, +25)

    prev = max(0, itr * step - 30)

    ng_idx = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step][:, 1]
    ng_idx_prev = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step - 4][:, 1]
    ng_idx_prev_prev = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step - 8][:, 1]

    pos_X = torch.Tensor(pos_x)
    pos_Y = torch.Tensor(pos_y)

    plt.plot(
        (pos_X[prev : itr * step] + base_offset_x) * -1,
        (pos_Y[prev : itr * step] + base_offset_y) * -1,
        color="gray",
    )

    plt.plot(
        (ng.x[ng_idx_prev_prev] + offset_x) * (50 / (max(ng.x) * 2 + 1)),
        (ng.y[ng_idx_prev_prev] + offset_y) * (50 / (max(ng.x) * 2 + 1)),
        "o",
        color=color,
        markersize=(4 * scale / (ng.shape.width)) + 2,
        alpha=0.2,
    )
    plt.plot(
        (ng.x[ng_idx_prev] + offset_x) * (50 / (max(ng.x) * 2 + 1)),
        (ng.y[ng_idx_prev] + offset_y) * (50 / (max(ng.x) * 2 + 1)),
        "o",
        color=color,
        markersize=(4 * scale / (ng.shape.width)) + 2,
        alpha=0.5,
    )
    plt.plot(
        (ng.x[ng_idx] + offset_x) * (50 / (max(ng.x) * 2 + 1)),
        (ng.y[ng_idx] + offset_y) * (50 / (max(ng.x) * 2 + 1)),
        "o",
        color=color,
        markersize=(4 * scale / (ng.shape.width)) + 2,
        label=label,
    )
    plt.legend(loc="upper right")

    plt.plot(
        (pos_x[itr * step] + base_offset_x) * -1,
        (pos_y[itr * step] + base_offset_y) * -1,
        "^",
        color="red",
        markersize=10,
    )

    plt.title(f"iteration = {itr * step}", y=-0.2)
    plt.suptitle("Grid Module Spikes")

    if save:
        plt.savefig("{}/fig{}.png".format(lib, itr))