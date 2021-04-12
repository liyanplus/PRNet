import numpy as np

def get_pr_n_cross_k(all_data, target_feature, other_feature, radii):
    radii.sort()
    if target_feature not in all_data or other_feature not in all_data:
        return {r: 0 for r in radii}

    pr_res = {}
    crossk_res = {}
    target_tree = all_data[target_feature]
    other_tree = all_data[other_feature]
    for radius in radii:
        neighbors = other_tree.query_ball_point(target_tree.data, radius)
        pr_res[radius] = np.sum([len(n) > 0 for n in neighbors]) / target_tree.data.shape[0]
        crossk_res[radius] = np.mean([len(n) for n in neighbors]) / other_tree.data.shape[0]
    return pr_res, crossk_res


def get_participation_ratio_binary_pattern(all_data, target_feature, other_feature, radii):  
    radii.sort()
    if target_feature not in all_data or other_feature not in all_data:
        return {r: 0 for r in radii}

    results = {}
    target_tree = all_data[target_feature]
    other_tree = all_data[other_feature]
    participating_indices = np.array([False] * target_tree.data.shape[0])

    for radius in radii:
        for target_idx, neighbors in zip(np.arange(target_tree.data.shape[0])[participating_indices == False],
                                         other_tree.query_ball_point(target_tree.data[participating_indices == False], radius)):
            if len(neighbors):
                participating_indices[target_idx] = True
        results[radius] = np.sum(participating_indices) / \
            target_tree.data.shape[0]
    return results


def get_cross_k_function(all_data, target_feature, other_feature, radii, area):
    if target_feature not in all_data or other_feature not in all_data:
        return 0

    results = {}
    radii.sort()
    target_tree = all_data[target_feature]
    other_tree = all_data[other_feature]

    for radius in radii:
        neighbors = other_tree.query_ball_point(target_tree.data, radius)
        results[radius] = np.mean([len(n) for n in neighbors]) / (other_tree.data.shape[0] / area)
    return results

    