#!/usr/bin/env python
# coding: utf-8
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import shutil
from copy import deepcopy
import warnings
import os
from sklearn.model_selection import KFold
import json

warnings.filterwarnings("ignore")
import random

random.seed(1234)
np.random.seed(1234)
from sklearn.model_selection import train_test_split


def obtain_data(task, task_id):
    pwd = r'../file/process_file/' + task_id + '/'
    node_feature_label = pd.read_csv(pwd + 'node_feature_label.csv', index_col=0)

    train_test_id_idx = np.load(
        r'../file/process_file/' + task_id + '/task_' + task + '__testlabel0_knn_edge_train_test_index_all.npz',
        allow_pickle=True)
    train_index_all = train_test_id_idx['train_index_all']
    test_index_all = train_test_id_idx['test_index_all']

    num_node = node_feature_label.shape[0]
    node_feat = node_feature_label.iloc[:, 3:]
    label = node_feature_label['label'].astype('int')

    gene_ids = list(set(node_feature_label['gene_idx']))
    trait_ids = list(set(node_feature_label['trait_idx']))
    random.shuffle(gene_ids)
    random.shuffle(trait_ids)

    return node_feature_label, num_node, node_feat, label, gene_ids, trait_ids, train_index_all, test_index_all


def generate_graphsaint_data(task, train_index_all, test_index_all, node_feat, label, num_node, task_id):
    # read knn_graph
    pwd = r'../file/process_file/' + task_id + '/'
    knn_graph_file = 'task_' + task + '__testlabel0_knn' + str(5) + 'neighbors_edge' + '.npz'
    knn_neighbors_graph = sp.load_npz(pwd + knn_graph_file)

    # nonzero()用于得到数组array中非零元素的位置（数组索引）的函数。
    edge_src_dst = knn_neighbors_graph.nonzero()
    # print(edge_src_dst)

    # save dir
    ## filepath = pwd + 'gene_trait_data/task_' + task + '' + '__testlabel0_' + str(5) + 'knn_edge_fold'
    # if (os.path.exists(filepath)):
    #     # 存在，则删除文件
    #     shutil.rmtree(filepath)
    save_dir_p = r'./gene_trait_data/task_Tg__testlabel0_5knn_edge_fold/'
    save_dir = pwd
    # save_dir = pwd + 'gene_trait_data/task_' + task + '' + '__testlabel0_' + str(5) + 'knn_edge_fold' + '/'
    #
    # try:
    #     os.mkdir(save_dir)
    # except OSError as error:
    #     print(error, save_dir)

    # feats.npy，不需要自己标准化！因为在utils.py中的load_data中有标准化的步骤哦！
    feats = np.array(node_feat, dtype='float32')
    np.save(save_dir + 'feats.npy', feats)
    np.save(save_dir_p + 'feats.npy', feats)
    time.sleep(30)

    train_idx, test_idx = train_index_all[0].tolist(), test_index_all[0].tolist()
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=1234)

    # role.json
    role = dict()
    role['tr'] = train_idx
    role['va'] = val_idx
    role['te'] = test_idx
    with open(save_dir + 'role.json', 'w') as f:
        json.dump(role, f)
    with open(save_dir_p + 'role.json', 'w') as f:
        json.dump(role, f)
    # class_map.json
    y = np.array(label)
    class_map = dict()
    for i in range(num_node):
        class_map[str(i)] = y[i].tolist()
    with open(save_dir + 'class_map.json', 'w') as f:
        json.dump(class_map, f)
    with open(save_dir_p + 'class_map.json', 'w') as f:
        json.dump(class_map, f)
        # adj_*.npz
    train_idx_set = set(train_idx)
    test_idx_set = set(test_idx)
    val_idx_set = set(val_idx)

    row_full, col_full = edge_src_dst[0], edge_src_dst[1]

    row_train = []
    col_train = []
    row_val = []
    col_val = []
    for i in tqdm(range(row_full.shape[0])):
        if row_full[i] in train_idx_set and col_full[i] in train_idx_set:
            row_train.append(row_full[i])
            col_train.append(col_full[i])
        if row_full[i] in val_idx_set and col_full[i] in val_idx_set:
            row_val.append(row_full[i])
            col_val.append(col_full[i])

    row_train = np.array(row_train)
    col_train = np.array(col_train)
    row_val = np.array(row_val)
    col_val = np.array(col_val)
    dtype = np.bool

    # sp.coo_matrix根据行列坐标生成矩阵，.tocsr()对矩阵进行压缩
    adj_full = sp.coo_matrix(
        (
            np.ones(row_full.shape[0], dtype=dtype),
            (row_full, col_full),
        ),
        shape=(num_node, num_node)
    ).tocsr()

    adj_train = sp.coo_matrix(
        (
            np.ones(row_train.shape[0], dtype=dtype),
            (row_train, col_train),
        ),
        shape=(num_node, num_node)
    ).tocsr()

    adj_val = sp.coo_matrix(
        (
            np.ones(row_val.shape[0], dtype=dtype),
            (row_val, col_val),
        ),
        shape=(num_node, num_node)
    ).tocsr()

    print('adj_full  num edges:', adj_full.nnz)
    print('adj_val   num edges:', adj_val.nnz)
    print('adj_train num edges:', adj_train.nnz)



    sp.save_npz(save_dir_p + 'adj_full.npz', adj_full)
    sp.save_npz(save_dir_p + 'adj_train.npz', adj_train)
    sp.save_npz(save_dir_p + 'adj_val.npz', adj_val)

    sp.save_npz(save_dir + 'adj_full.npz', adj_full)
    sp.save_npz(save_dir + 'adj_train.npz', adj_train)
    sp.save_npz(save_dir + 'adj_val.npz', adj_val)  # adj_val not used in GraphSAINT source code

    return feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst


def run(task, task_id):
    node_feature_label, num_node, node_feat, label, gene_ids, trait_ids, train_index_all, test_index_all = obtain_data(
        task, task_id)
    feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst = generate_graphsaint_data(task,
                                                                                                  train_index_all,
                                                                                                  test_index_all,
                                                                                                  node_feat,
                                                                                                  label,
                                                                                                  num_node, task_id)
    return node_feature_label, num_node, node_feat, label, gene_ids, trait_ids, train_index_all, test_index_all, \
           feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst


def two_step_run(task_id, select):
    run(task=select, task_id=task_id)
