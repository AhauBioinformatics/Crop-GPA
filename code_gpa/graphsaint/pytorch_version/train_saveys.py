from graphsaint.globals import *
from graphsaint.pytorch_version.models import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *
import numpy as np
import csv
import pandas as pd
import scipy.sparse

import torch
import time

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cpu")
def evaluate_full_batch(model, minibatch, mode='Test'):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        仅用于验证和测试集。在计算F1分数时，我们将对相关的根节点进行掩码   ####根节点：是没有父节点的节点
        (e.g., those belonging to the val / test sets).
    """
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=mode))  ###model.eval_step向前传播 得到输出层的结果————————minibatch.one_batch
    #print('preds',preds)
    #print('preds',preds.shape)
    ####输出

    #print('Labels \n', labels)

    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/old test data/old labels.txt", labels,fmt='%s', newline='\n')
    #print('Preds \n', len(preds))
    if mode == 'val':
        printf('Val: loss = {:.4f}'.format(loss), style = 'red')
        node_target = [minibatch.node_val]

        node_target1 = np.array(node_target)
        #print('node_target1', node_target1)
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
        printf('Test: loss = {:.4f}'.format(loss), style = 'red')
        node_target = [minibatch.node_test]

        node_target1 = np.array(node_target)
        print('node_target1',node_target1)
        print('node_target1',node_target1.shape)
        for n in node_target:
            ys_test = to_numpy(preds[n])

        return ys_test




def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """  ##  准备一些数据结构和初始化模型           minibatch处理程序之前实际的迭代训练
    ###初始化数据
    ############minibatch和minibatch_eval的前期的数据准备并进行采样都是一样的，但是在最后的训练迭代时是有所区别的#####
    adj_full, adj_train, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]   ###  m_classes 2
   #print('adj_ful',adj_full)
   #print('adj_train',adj_train)
   #print('adj_full_norm',adj_full_norm)
   #print( 'num_classes', num_classes)
    #print(' class_arr', class_arr)
    ####minibatch 处理程序之前实际的迭代训练
    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)    #####准备一些数据结构和初始化模型
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")##printf()函数是格式化输出函数
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)

    if args_global.gpu >= 0:
        model = model.to(device)
        #model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every):
    ###train_phases  是通过minibatch中的edge smaple 得到
    ###eval_val_every
    if not args_global.cpu_eval:
        minibatch_eval=minibatch
    epoch_ph_start = 0
    auc_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    path_saver = '{}/pytorch_models/gene_trait_saved_model_{}.pkl'.format(args_global.dir_log, timestamp)
    #print('dir_saver: ', dir_saver)
    #print('path_saver: ', path_saver)

    for ip, phase in enumerate(train_phases):  #用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        printf('START PHASE {:4d}'.format(ip),style='underline')
        printf(phase)
        printf('*****')
        printf(phase['end'])
        minibatch.set_sampler(phase)##用边采样模型
        num_batches = minibatch.num_training_batches()   ###2
        #print('num_batche',num_batches)
        for e in range(epoch_ph_start, int(phase['end'])):    ###epoch  设置为end=1000
            printf('Epoch {:4d}'.format(e),style='bold')  ###规定输出的格式
            minibatch.shuffle()   ####随机序列排序

            l_loss_tr, lr_accuracy_tr, lr_precision_tr, lr_recall_tr, lr_f1_tr, lr_roc_auc_tr, lr_aupr_tr, lr_pos_acc_tr, lr_neg_acc_tr = [], [], [], [], [], [], [], [], []

            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train,preds_train,labels_train = model.train_step(*minibatch.one_batch(mode='train'))####生成结果     *是指针，  指针可以任意转换类型，所以字符指针返回局部变量或临时变量的地址
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % args_global.eval_train_every:####  % 计算 a 除以 b 得出的余数。

                    ys_train, metrics_train = metrics(to_numpy(labels_train[:, 1]),to_numpy(preds_train[:, 1]),model.sigmoid_loss, isprint = True)
                    l_loss_tr.append(loss_train)
                    lr_accuracy_tr.append(metrics_train[0])
                    lr_precision_tr.append(metrics_train[1])
                    lr_recall_tr.append(metrics_train[2])
                    lr_f1_tr.append(metrics_train[3])
                    lr_roc_auc_tr.append(metrics_train[4])
                    lr_aupr_tr.append(metrics_train[5])
                    lr_pos_acc_tr.append(metrics_train[6])
                    lr_neg_acc_tr.append(metrics_train[7])

            if (e+1)%eval_val_every == 0:
                if args_global.cpu_eval:
                    torch.save(model.state_dict(),'tmp.pkl')
                    model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))
                else:
                    model_eval = model

                printf('Train (Ep avg): loss = {:.4f} | Time = {:.4f}sec'.format(f_mean(l_loss_tr), time_train_ep), style = 'yellow')
                printf('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(
f_mean(lr_accuracy_tr), f_mean(lr_precision_tr), f_mean(lr_recall_tr), f_mean(lr_f1_tr), f_mean(lr_roc_auc_tr), f_mean(lr_aupr_tr), f_mean(lr_pos_acc_tr), f_mean(lr_neg_acc_tr)),
style = 'yellow')

                loss_val, ys_val, metrics_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')
                auc_val = metrics_val[4]
                if auc_val > auc_best:
                    auc_best, ep_best = auc_val, e
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    printf('  Saving model ...', style='yellow')
                    torch.save(model.state_dict(), path_saver)
            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])#####epoch 1000
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval=model
        printf('  Restoring model ...', style='yellow')

    printf('Best Epoch = ' + str(ep_best), style = 'red')
    ys_test = evaluate_full_batch(model_eval, minibatch_eval, mode='test')####验证的ys_test运行  返回前面15的代码
    #print('ys_test',ys_test)
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')
    return ys_train, ys_test

if __name__ == '__main__':
    t1 = time.clock()
    # log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)   ##untill 代码中的log_dir函数调用  用于控制台的调用
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)###将带有参数的文件yml  导入的函数中
    t2 = time.clock()
    print('加载参数的时间: %s Seconds' % (t2 - t1))

    t3 = time.clock()
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP  ##在globls.py代码中 EVAL_VAL_EVERY_EP = 1
    model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)  ###调用前面的prepare的函数
    t4 = time.clock()
    print('数据预处理的时间: %s Seconds' % (t4 - t3))

    ys_train, ys_test = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'])
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/test data/ys_train.txt", ys_train,fmt='%s', newline='\n')
    ys_test_df = pd.DataFrame(ys_test)
    ys_test_df.columns = ['prob_0', 'prob_1']
    ys_test_df.to_csv('../../ys_test.csv')
    #print('ys',ys_train)
    #print('ys_test',ys_test[0].shape)
    #print('ys_test',ys_test[1].shape)
    #print('ys_test',ys_test[2].shape)
    #print('ys_test',ys_test[0])
    #print('ys_test',ys_test[1])
    #print('ys_test',ys_test[1])
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/MDA-GCNGS/ys_test[0]g.txt", ys_test[1] ,fmt='%s', newline='\n')
    #np.savetxt("C:/Users/Administrator/Desktop/图采样有向图代码/old test data/ys_train.txt", ys_train,fmt='%s', newline='\n')
    #print(ys_train,ys_test)
    #np.savez_compressed('C:/Users/Administrator/Desktop/GCNFTG CODE/miRNA_disease__case_study/task_Tp_balanced_10knn_lr0.001_fold00——epoch-1.npz', ys_train = ys_train, ys_test = ys_test)

    #python -m train_saveys --data_prefix gene_trait_data/task_Tp__testlabel0_5knn_edge_fold --train_config ./train_config/table2/ppi2_e_lr0.001.yml > ./miRNA_disease__case_study/20200115_Td_balanced_5knn_lr0.001_fold4.out --gpu 0
