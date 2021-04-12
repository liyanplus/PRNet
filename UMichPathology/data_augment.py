import torch as th
import math


def get_partial_data(raw_data):
    x_min = th.min(raw_data[:, 0]).item()
    x_max = th.max(raw_data[:, 0]).item() + 1
    y_min = th.min(raw_data[:, 1]).item()
    y_max = th.max(raw_data[:, 1]).item() + 1

    for lo_x in range(x_min, int(x_min + (x_max - x_min) / 5) + 1, int((x_max - x_min) / 5)):
        hi_x = lo_x + (x_max - x_min) / 5 * 4
        yield get_partial_data_proc(raw_data, lo_x, hi_x, y_min, y_max)


def get_partial_data_proc(raw_data, lo_x, hi_x, lo_y, hi_y):
    partial_data = raw_data[th.all(
        th.stack((
            raw_data[:, 0] >= lo_x,
            raw_data[:, 0] <= hi_x,
            raw_data[:, 1] >= lo_y,
            raw_data[:, 1] <= hi_y
        ), dim=1), dim=1
    )]
    partial_data[:, 0] -= lo_x
    partial_data[:, 1] -= lo_y
    return partial_data


def get_rotate_data(raw_data, rotation_count=4):
    x_avg = th.mean(raw_data[:, 0].float())
    y_avg = th.mean(raw_data[:, 1].float())

    x = raw_data[:, 0] - x_avg
    y = raw_data[:, 1] - y_avg

    ss = th.sin(th.arange(0, 2 * math.pi - 2 * math.pi / rotation_count + 1e-4, 2 * math.pi / rotation_count))
    cs = th.cos(th.arange(0, 2 * math.pi - 2 * math.pi / rotation_count + 1e-4, 2 * math.pi / rotation_count))

    x_new = th.outer(x, cs) - th.outer(y, ss) + x_avg
    y_new = th.outer(x, ss) + th.outer(y, cs) + y_avg

    for r_idx in range(rotation_count):
        res = th.IntTensor(raw_data)
        res[:, 0] = x_new[:, r_idx] - th.min(x_new[:, r_idx])
        res[:, 1] = y_new[:, r_idx] - th.min(y_new[:, r_idx])
        yield res