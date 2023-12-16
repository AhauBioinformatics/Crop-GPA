from globals import *
from graphsaint.pytorch_version.models import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *
import numpy as np
import pandas as pd
import copy
import torch
import time

import warnings

warnings.filterwarnings("ignore")
device = torch.device("cpu")


def evaluate_full_batch(model, minibatch, path, mode='Test'):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        仅用于验证和测试集。在计算F1分数时，我们将对相关的根节点进行掩码   ####根节点：是没有父节点的节点
        (e.g., those belonging to the val / test sets).
    """
    loss, preds, labels = model.eval_step(
        *minibatch.one_batch(mode=mode))  ###model.eval_step向前传播 得到输出层的结果————————minibatch.one_batch
    if mode == 'val':
        printf('Val: loss = {:.4f}'.format(loss), style='red')
        node_target = [minibatch.node_val]

        node_target1 = np.array(node_target)
        # print('node_target1', node_target1)
        print('node_target1', node_target1.shape)

        accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc = [], [], [], [], [], [], [], []
        for n in node_target:
            print(labels[n, 1], preds[n, 1])
            ys, performances = metrics(to_numpy(labels[n, 1]), to_numpy(preds[n, 1]),
                                       model.sigmoid_loss)  #### metric.py代码
            ys_test = to_numpy(preds[n])
            accuracy.append(performances[0])
            precision.append(performances[1])
            recall.append(performances[2])
            f1.append(performances[3])
            roc_auc.append(performances[4])
            aupr.append(performances[5])
            pos_acc.append(performances[6])
            neg_acc.append(performances[7])

        pos_acc = pos_acc[0] if len(pos_acc) == 1 else pos_acc
        neg_acc = neg_acc[0] if len(neg_acc) == 1 else neg_acc
        roc_auc = roc_auc[0] if len(roc_auc) == 1 else roc_auc
        aupr = aupr[0] if len(aupr) == 1 else aupr
        f1 = f1[0] if len(f1) == 1 else f1
        recall = recall[0] if len(recall) == 1 else recall
        precision = precision[0] if len(precision) == 1 else precision
        accuracy = accuracy[0] if len(accuracy) == 1 else accuracy

        return loss, ys, (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc)

    else:
        ys_test = to_numpy(preds)
        print('****')
        print(ys_test)
        print(ys_test.shape)

        return ys_test


def prepare(train_data, train_params, arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train, feat_full, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]
    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())),
           style="yellow")
    minibatch_eval = Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)

    if args_global.gpu >= 0:
        model = model.to(device)
        # model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every, path):
    if not args_global.cpu_eval:
        minibatch_eval = minibatch
    epoch_ph_start = 0
    auc_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    path_saver = '{}/pytorch_models/gene_trait_saved_model_{}.pkl'.format(args_global.dir_log, timestamp)

    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip), style='underline')
        printf(phase)
        printf('*****')
        printf(phase['end'])
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()

        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e), style='bold')
            minibatch.shuffle()

            l_loss_tr, lr_accuracy_tr, lr_precision_tr, lr_recall_tr, lr_f1_tr, lr_roc_auc_tr, lr_aupr_tr, lr_pos_acc_tr, lr_neg_acc_tr = [], [], [], [], [], [], [], [], []

            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train, preds_train, labels_train = model.train_step(
                    *minibatch.one_batch(mode='train'))
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % args_global.eval_train_every:

                    ys_train, metrics_train = metrics(to_numpy(labels_train[:, 1]), to_numpy(preds_train[:, 1]),
                                                      model.sigmoid_loss, isprint=True)
                    l_loss_tr.append(loss_train)
                    lr_accuracy_tr.append(metrics_train[0])
                    lr_precision_tr.append(metrics_train[1])
                    lr_recall_tr.append(metrics_train[2])
                    lr_f1_tr.append(metrics_train[3])
                    lr_roc_auc_tr.append(metrics_train[4])
                    lr_aupr_tr.append(metrics_train[5])
                    lr_pos_acc_tr.append(metrics_train[6])
                    lr_neg_acc_tr.append(metrics_train[7])

            if (e + 1) % eval_val_every == 0:
                if args_global.cpu_eval:
                    torch.save(model.state_dict(), 'tmp.pkl')
                    model_eval.load_state_dict(torch.load('tmp.pkl', map_location=lambda storage, loc: storage))
                else:
                    model_eval = model

                printf('Train (Ep avg): loss = {:.4f} | Time = {:.4f}sec'.format(f_mean(l_loss_tr), time_train_ep),
                       style='yellow')
                printf(
                    'acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(
                        f_mean(lr_accuracy_tr), f_mean(lr_precision_tr), f_mean(lr_recall_tr), f_mean(lr_f1_tr),
                        f_mean(lr_roc_auc_tr), f_mean(lr_aupr_tr), f_mean(lr_pos_acc_tr), f_mean(lr_neg_acc_tr)),
                    style='yellow')

                loss_val, ys_val, metrics_val = evaluate_full_batch(model_eval, minibatch_eval, path, mode='val')
                auc_val = metrics_val[4]
                if auc_val > auc_best:
                    auc_best, ep_best = auc_val, e
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    printf('  Saving model ...', style='yellow')
                    torch.save(model.state_dict(), path_saver)
            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])  #####epoch 1000
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval = model
        printf('  Restoring model ...', style='yellow')

    printf('Best Epoch = ' + str(ep_best), style='red')
    ys_test = evaluate_full_batch(model_eval, minibatch_eval, path, mode='test')
    print(len(ys_test))
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')
    return ys_test


def train_saveys_main(task_id, email, name):
    path = r'../file/process_file/' + task_id
    ys_test_data = os.path.lexists(path + '/ys_test.csv')

    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP
    model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
    ys_test = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'], path)

    ys_test_df = pd.DataFrame(ys_test)
    ys_test_df.columns = ['prob_0', 'prob_1']
    print(ys_test_df)
    ys_test_df.to_csv(path + '/ys_test.csv')
    print("three step end")

    four_step(path, email, task_id, name)


import requests
from flask import jsonify

parentUrl = "https://api.crop.aielab.net"


def four_step(path, email, task_id, name):
    role = json.load(open(path + '/role.json'))
    y_prob = pd.read_csv(path + '/ys_test.csv')['prob_1']
    node_info = pd.read_csv(path + '/node_feature_label.csv')[['trait_idx', 'gene_idx', 'label']]

    a = pd.merge(node_info, y_prob, left_index=True, right_index=True)
    b = a.iloc[role.get('te')]

    gene = pd.read_excel(path + '/gene_name.xlsx')['gene']
    trait = pd.read_excel(path + '/trait_name.xlsx')['trait']
    c = pd.merge(pd.merge(b, gene, left_on='gene_idx', right_index=True), trait, left_on='trait_idx', right_index=True)

    print(c['prob_1'])
    y_pred = copy.deepcopy(c['prob_1'])

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    c['y_pred'] = y_pred
    print(c)

    c.drop(labels=['trait_idx', 'gene_idx'], axis=1).sort_values(
        by=["gene", "trait"]).reset_index(drop=True)
    order = ['gene', 'trait', 'prob_1', 'y_pred', 'label']
    result = c[order]

    result.to_excel(path + '/result.xlsx', index=False)

    read_xls(path + '/result.xlsx', task_id)

    print("four step end")
    subject = "Prediction Success Reminder"
    message = "Your taskId is " + task_id + ", If you want to see the detail, you can download the attached content or download task report in website: https://crop-gpa.aielab.net//crop/tools ."

    import to_email

    to_email.send_email_file(email, message, path + "/result.xlsx", subject, name, task_id)


import xlrd


def read_xls(filename, task_id):
    # 打开Excel文件
    data = xlrd.open_workbook(filename)
    # 读取第一个工作表
    table = data.sheets()[0]
    # 统计行数
    rows = table.nrows
    data = []  # 存放数据
    for v in range(1, rows):
        values = table.row_values(v)
        data_child = {
            "gene": str(values[0]),
            "trait": str(values[1]),  # 这里我只需要字符型数据，加了str(),根据实际自己取舍
            "prob": str(values[2]),
            "taskId": str(task_id),
            "pred": int(values[3]),
            "label": int(values[4]),
        }
        data.append(data_child)
    res = requests.post(url=parentUrl + "/crop/cropGpa/add_list_gpa", json=data,
                  headers={'Content-Type': 'application/json;charset=UTF-8'})
    print(res)
    print(data)
    return data
