import sys

sys.path.insert(0, r'/Volumes/GoogleDrive/My Drive/Colab Notebooks/')
# sys.path.insert(0, r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/col/spatial_pathalogy')
# sys.path.insert(0, r'C:/Users/MajidCSci/Downloads/Experiment')

import os
import numpy as np
import torch as th
import torch.nn as th_nn
import torch.optim as optim
from src.UMichPathology.pathology_classifier import PathologyClassifier
import src.helper.multivariate_correlation_proc as mcp
import src.UMichPathology.file_operation as fo
import src.UMichPathology.data_augment as da
import random
from datetime import datetime
import gc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.spatial import KDTree
import itertools
from multiprocessing.pool import ThreadPool

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def valid_proc(valid_file_paths, valid_ys, classifier,
               min_grid_scale, max_grid_scale, grid_scale_count,
               neighborhood_distance,
               feature_type_count, sampling_ratio):
    pred_y_probs = []
    true_ys = []
    # validation
    for valid_file_path, valid_y in zip(valid_file_paths, valid_ys):
        raw_data = fo.read_file(valid_file_path)

        hf = fo.read_human_feature(valid_file_path)
        hf = th.from_numpy(hf).to(device)

        for partial_idx, partial_data in enumerate(da.get_partial_data(raw_data)):
            pointpairs = fo.get_pointpairs(valid_file_path, partial_data, partial_idx, neighborhood_distance)
            for augmented_data in da.get_rotate_data(partial_data):
                neighborhood_tensor, core_point_idxs = \
                    fo.get_neighborhood_representation(valid_file_path, augmented_data, pointpairs,
                                                       min_grid_scale, max_grid_scale, grid_scale_count,
                                                       feature_type_count, sampling_ratio)

                augmented_data = augmented_data[:, 2].int().to(device)
                neighborhood_tensor = neighborhood_tensor.to(device)
                core_point_idxs = core_point_idxs.to(device)

                y, _ = classifier(augmented_data, neighborhood_tensor,
                                  core_point_idxs, hf)
                pred_y_probs.append(y.item())
                true_ys.append(valid_y)
            th.cuda.empty_cache()

    pred_y_probs = np.array(pred_y_probs)
    true_ys = np.array(true_ys)

    print(true_ys)
    print(pred_y_probs)

    print('MCNet: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        roc_auc_score(true_ys, pred_y_probs), precision_score(true_ys, pred_y_probs > 0.5),
        recall_score(true_ys, pred_y_probs > 0.5), f1_score(true_ys, pred_y_probs > 0.5),
        accuracy_score(true_ys, pred_y_probs > 0.5)
    ))


def train_proc(min_grid_scale=1,
               max_grid_scale=100,
               grid_scale_count=10,
               neighborhood_distance=200,
               feature_type_count=9,
               pr_representation_dim=32,
               pp_representation_dim=128,
               learning_rate=1e-4,
               apply_attention=True,
               epoch=100,
               relu_slope=1e-2,
               regularization_weight=100,
               diff_weight=1e-3,
               batch_size=32,
               sampling_ratio=1,
               model_path=r'/content/drive/MyDrive/Colab Notebooks/prnet.model'):
    class1_file_paths = list(
        fo.file_system_scrawl('/content/drive/MyDrive/Research/Current/UMichCancer/Data/Anon_Group1', '.txt'))

    class2_file_paths = list(
        fo.file_system_scrawl('/content/drive/MyDrive/Research/Current/UMichCancer/Data/Anon_Group2', '.txt'))

    train_file_paths1, valid_file_paths1, train_ys1, valid_ys1 = \
        fo.split_train_valid(class1_file_paths, [0] * len(class1_file_paths))

    train_file_paths2, valid_file_paths2, train_ys2, valid_ys2 = \
        fo.split_train_valid(class2_file_paths, [1] * len(class2_file_paths))

    valid_file_paths = np.hstack((valid_file_paths1, valid_file_paths2))
    valid_ys = np.hstack((valid_ys1, valid_ys2))

    classifier = PathologyClassifier(feature_type_count, grid_scale_count,
                                     pr_representation_dim, pp_representation_dim,
                                     apply_attention, leaky_relu_slope=relu_slope).to(device)

    if os.path.exists(model_path):
        classifier.load_state_dict(th.load(model_path))

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    criterion = th_nn.CrossEntropyLoss()

    batch_pred_ys = th.zeros(batch_size, dtype=th.float, device=device)
    batch_true_ys = th.zeros(batch_size, dtype=th.long, device=device)

    train_true_pos = 0
    train_false_pos = 0
    train_true_neg = 0
    train_false_neg = 0

    idx = 0
    pr_diff_sum = 0
    for epoch_idx in range(epoch):
        # train
        batch_idx = 0

        for train_pair1, train_pair2 in zip(
                random.sample(list(zip(train_file_paths1, train_ys1)), min(len(train_ys1), len(train_ys2))),
                random.sample(list(zip(train_file_paths2, train_ys2)), min(len(train_ys1), len(train_ys2)))
        ):
            train_file_path1, cur_y1 = train_pair1
            train_file_path2, cur_y2 = train_pair2

            raw_data1 = fo.read_file(train_file_path1)
            raw_data2 = fo.read_file(train_file_path2)

            for partial_idx, (partial_data1, partial_data2) in enumerate(zip(
                    da.get_partial_data(raw_data1),
                    da.get_partial_data(raw_data2)
            )):
                pointpairs1 = fo.get_pointpairs(train_file_path1, partial_data1, partial_idx, neighborhood_distance)
                pointpairs2 = fo.get_pointpairs(train_file_path2, partial_data2, partial_idx, neighborhood_distance)

                for rotate_idx, (augmented_data1, augmented_data2) in enumerate(zip(
                        da.get_rotate_data(partial_data1),
                        da.get_rotate_data(partial_data2)
                )):
                    if rotate_idx != epoch_idx % 4:
                        continue

                    neighborhood_tensor1, core_point_idxs1 = \
                        fo.get_neighborhood_representation(train_file_path1, augmented_data1, pointpairs1,
                                                           min_grid_scale, max_grid_scale, grid_scale_count,
                                                           feature_type_count, sampling_ratio)

                    augmented_data1 = augmented_data1[:, 2].int().to(device)
                    neighborhood_tensor1 = neighborhood_tensor1.to(device)
                    core_point_idxs1 = core_point_idxs1.to(device)

                    batch_pred_ys[idx], pred_pr1 = classifier(augmented_data1, neighborhood_tensor1,
                                                              core_point_idxs1)
                    batch_true_ys[idx] = cur_y1

                    if batch_pred_ys[idx] >= 0.5 and batch_true_ys[idx] == 1:
                        train_true_pos += 1
                    elif batch_pred_ys[idx] <= 0.5 and batch_true_ys[idx] == 0:
                        train_true_neg += 1
                    elif batch_pred_ys[idx] >= 0.5 and batch_true_ys[idx] == 0:
                        train_false_pos += 1
                    else:
                        train_false_neg += 1

                    neighborhood_tensor2, core_point_idxs2 = \
                        fo.get_neighborhood_representation(train_file_path2, augmented_data2, pointpairs2,
                                                           min_grid_scale, max_grid_scale, grid_scale_count,
                                                           feature_type_count, sampling_ratio)

                    augmented_data2 = augmented_data2[:, 2].int().to(device)
                    neighborhood_tensor2 = neighborhood_tensor2.to(device)
                    core_point_idxs2 = core_point_idxs2.to(device)

                    batch_pred_ys[idx + 1], pred_pr2 = classifier(augmented_data2, neighborhood_tensor2,
                                                                  core_point_idxs2)
                    batch_true_ys[idx + 1] = cur_y2

                    pr_diff_sum += 1 / (th.norm(
                        pred_pr1 - pred_pr2, 1
                    ) + 1e-5)

                    if batch_pred_ys[idx + 1] >= 0.5 and batch_true_ys[idx + 1] == 1:
                        train_true_pos += 1
                    elif batch_pred_ys[idx + 1] <= 0.5 and batch_true_ys[idx + 1] == 0:
                        train_true_neg += 1
                    elif batch_pred_ys[idx + 1] >= 0.5 and batch_true_ys[idx + 1] == 0:
                        train_false_pos += 1
                    else:
                        train_false_neg += 1

                    idx += 2

                    if idx % 2 == 0:
                        th.cuda.empty_cache()

                    if idx >= batch_size:
                        # print('{0} - Epoch {1} batch {2} start training backward.'.format(
                        #     datetime.now().strftime("%H:%M:%S"), epoch_idx, batch_idx))
                        optimizer.zero_grad()
                        paras = th.cat([x.view(-1) for x in classifier.parameters()])
                        regularization = th.norm(paras, 1) / (paras.shape[0] + 1)
                        ce = criterion(th.stack([1 - batch_pred_ys, batch_pred_ys], 1), batch_true_ys)
                        loss = ce + regularization_weight * regularization + diff_weight * pr_diff_sum

                        loss.backward()
                        th_nn.utils.clip_grad_value_(classifier.parameters(), 0.5)
                        optimizer.step()

                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = 1e-3 - (1e-3 - 1e-4) / 10 * min(epoch_idx, 10)

                        print(
                            '{4} - Epoch: {0:5d}, Batch: {1:5d}, Training loss: {2:.3f}, Cross entropy: {3:.3f}'.format(
                                epoch_idx, batch_idx, loss.item(), ce.item(), datetime.now().strftime("%H:%M:%S")
                            ))

                        batch_pred_ys = th.zeros(batch_size, dtype=th.float, device=device)
                        batch_true_ys = th.zeros(batch_size, dtype=th.long, device=device)

                        idx = 0
                        pr_diff_sum = 0

                        print('{5} - Epoch: {0:5d}; Batch: {6:5d}; Training: '
                              'True positive: {1:5d}; '
                              'True negative: {2:5d}; '
                              'False positive: {3:5d}; '
                              'False negative: {4:5d}.'.format(epoch_idx,
                                                               train_true_pos, train_true_neg,
                                                               train_false_pos, train_false_neg,
                                                               datetime.now().strftime("%H:%M:%S"),
                                                               batch_idx))

                        train_true_pos = 0
                        train_false_pos = 0
                        train_true_neg = 0
                        train_false_neg = 0
                        gc.collect()
                        th.cuda.empty_cache()

                        batch_idx += 1

        th.save(classifier.state_dict(), model_path)

        if (epoch_idx + 1) % 4 == 0:
            valid_proc(valid_file_paths, valid_ys, classifier,
                       min_grid_scale, max_grid_scale, grid_scale_count,
                       neighborhood_distance, feature_type_count, 1)



if __name__ == "__main__":
    train_proc()
