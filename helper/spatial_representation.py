import torch as th
import math
import scipy.spatial as sci_spatial


def get_2d_neighborhood_pairs_from_cache(cache_file_path):
    with open(cache_file_path) as cache_file:
        res = [[ori_idx, int(dest_idx)] for ori_idx, line in enumerate(cache_file) for dest_idx in line.split(',')]
    return th.LongTensor(res)


def get_2d_neighborhood_pairs(raw_data, neighborhood_distance, cache_file_path):
    tree_index = sci_spatial.KDTree(raw_data[:, :2])
    res = []
    with open(cache_file_path, 'w') as cache_file_path:
        for cur_idx, n_idxs in enumerate(tree_index.query_ball_tree(tree_index, neighborhood_distance)):
            cache_file_path.write('{0}\n'.format(','.join(str(idx) for idx in n_idxs) if len(n_idxs) else ''))
            res.extend([[cur_idx, n] for n in n_idxs])
    return th.LongTensor(res)


def get_pointpairs_representation(raw_data, pointpairs,
                                  min_grid_scale, max_grid_scale, grid_scale_count):
    xs = raw_data[pointpairs[:, 1], :2] - raw_data[pointpairs[:, 0], :2]
    a = th.tensor([[1, 0], [-0.5, -math.sqrt(3) / 2], [-0.5, math.sqrt(3) / 2]])
    scales = th.tensor([min_grid_scale * (max_grid_scale / min_grid_scale) ** (s / (grid_scale_count - 1))
                        for s in range(grid_scale_count)])
    scaled_proj = th.einsum('qr, p->qrp', th.matmul(xs.float(), a.T), 1 / scales)
    return th.stack((th.cos(scaled_proj), th.sin(scaled_proj)), dim=3).reshape((scaled_proj.shape[0], -1))
