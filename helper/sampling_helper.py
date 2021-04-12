import torch as th
import math


def sampling_proc(raw_data, pointpairs, feature_type_count, sampling_ratio):
    core_point_idxs = th.zeros(raw_data.shape[0]).bool()
    for feature_idx in range(feature_type_count):
        feature_point_idxs = raw_data[:, 2] == feature_idx
        feature_point_count = th.sum(feature_point_idxs).item()
        if feature_point_count == 0:
            continue
        sampling_number = get_sampling_number(feature_point_count, sampling_ratio)
        core_point_idxs[
            th.arange(raw_data.shape[0])[feature_point_idxs][th.randint(0, feature_point_count, (sampling_number,))]
        ] = True

    selected_pointpair_idxs = th.zeros(pointpairs.shape[0]).bool()
    for cp_idx in th.arange(raw_data.shape[0])[core_point_idxs]:
        selected_pointpair_idxs[pointpairs[:, 0] == cp_idx] = True

    return core_point_idxs, selected_pointpair_idxs


def get_sampling_number(feature_count, sampling_ratio):
    sampling_number = int(math.ceil(feature_count * sampling_ratio))
    if sampling_number < 50:
        sampling_number = 50
    elif sampling_number > 200:
        sampling_number = 200

    if sampling_number > feature_count:
        sampling_number = feature_count

    return sampling_number
