from graphsaint.globals import *
import math
from graphsaint.utils import *
from graphsaint.graph_samplers import *
from graphsaint.norm_aggr import *
import torch
import scipy.sparse as sp
import scipy

import numpy as np
import time


device = torch.device("cpu")
def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch    转换一个scipy稀疏COO矩阵到troch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)  ###tensor的数据类型
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))  



class Minibatch:
    """
    Provides minibatches for the trainer or evaluator. This class is responsible for
    calling the proper graph sampler and estimating normalization coefficients.
    为训练和评估者提供小批量      这个类负责调用适当的图采样器并估计归一化系数。
    """
    def __init__(self, adj_full_norm, adj_train, role, train_params, cpu_eval=False):
        """
        Inputs:
            adj_full_norm       scipy CSR, adj matrix for the full graph (row-normalized)   完整图的矩阵（）
            adj_train           scipy CSR, adj matrix for the traing graph. Since we are
                                under transductive setting, for any edge in this adj,
                                both end points must be training nodes.   
                                训练图的Adj矩阵。由于我们处于转导设置下，对于这个adj中的任何边，两个端点都必须是训练节点。
                                
            role                dict, key 'tr' -> list of training node IDs;
                                      key 'va' -> list of validation node IDs;
                                      key 'te' -> list of test node IDs.
                                      
            train_params        dict, additional parameters related to training. e.g.,
                                how many subgraphs we want to get to estimate the norm
                                coefficients.
                                与培训相关的其他参数。例如，我们需要多少个子图来估计范数系数。
                                
            cpu_eval            bool, whether or not we want to run full-batch evaluation
                                on the CPU.
                                是否要在CPU上运行全批处理计算。

        Outputs:
            None
        """
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())
        self.adj_train = adj_train
        # -----------------------
        # sanity check (optional)
        # -----------------------
        #for role_set in [self.node_val, self.node_test]:
        #    for v in role_set:
        #        assert self.adj_train.indptr[v+1] == self.adj_train.indptr[v]
        #_adj_train_T = sp.csr_matrix.tocsc(self.adj_train)
        #assert np.abs(_adj_train_T.indices - self.adj_train.indices).sum() == 0
        #assert np.abs(_adj_train_T.indptr - self.adj_train.indptr).sum() == 0
        #_adj_full_T = sp.csr_matrix.tocsc(adj_full_norm)
        #assert np.abs(_adj_full_T.indices - adj_full_norm.indices).sum() == 0
        #assert np.abs(_adj_full_T.indptr - adj_full_norm.indptr).sum() == 0
        #printf("SANITY CHECK PASSED", style="yellow")
        if self.use_cuda:
            # now i put everything on GPU. Ideally, full graph adj/feat
            # should be optionally placed on CPU
            self.adj_full_norm = self.adj_full_norm.to(device)
            #self.adj_full_norm = self.adj_full_norm.cuda()

        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        # norm_loss_test is used in full batch evaluation (without sampling). Norm_loss_test用于全批评估(不取样)
        # so neighbor features are simply averaged.  所以邻居特征是简单平均的
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)  ##role的数据加在一起
        self.norm_loss_test[self.node_train] = 1. / _denom
        self.norm_loss_test[self.node_val] = 1. / _denom
        self.norm_loss_test[self.node_test] = 1. / _denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))  #torch.from_numpy创建一个张量 #astype 转换数组的数据类型
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.to(device)
            #self.norm_loss_test = self.norm_loss_test.cuda()
        self.norm_aggr_train = np.zeros(self.adj_train.size)

        self.sample_coverage = train_params['sample_coverage']###需要多个子图估计范数系数   sample_coverage= 50
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()

    def set_sampler(self, train_phases):
        """
        Pick the proper graph sampler. Run the warm-up phase to estimate
        loss / aggregation normalization coefficients.
        选择合适的图采样器   运行预热阶段进行估算损失/聚集标准化系数。
 
        Inputs:
            train_phases       dict, config / params for the graph sampler
            输入训练参数    为图形采样器配置/参数

        Outputs:
            None
        """
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'mrw':
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = mrw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                train_phases['size_frontier'],
                _deg_clip,
            )
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = rw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                int(train_phases['num_root']),
                int(train_phases['depth']),
            )
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2   ###4000*2
            self.graph_sampler = edge_sampling(
                self.adj_train,
                self.node_train,
                train_phases['size_subg_edge'],
            )
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                #train_phases['size_subgraph']
            )
        elif self.method_sample == 'full_batch':
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == "vanilla_node_python":
            self.size_subg_budget = train_phases["size_subgraph"]
            self.graph_sampler = NodeSamplingVanillaPython(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)##估计损失/聚集归一化因素

        # -------------------------------------------------------------
        # BELOW: estimation of loss / aggregation normalization factors   估计损失/聚集归一化因素
        # -------------------------------------------------------------
        # For some special sampler, no need to estimate norm factors, we can calculate
        # 对于一些特殊的采样器，不需要估计范数因子，即可直接计算节点/边缘的概率
        # the node / edge probabilities directly.
        # However, for integrity of the framework, we follow the same procedure 
        # for all samplers:  然而，为了框架的完整性，我们对所有采样器遵循相同的程序
        #   1. sample enough number of subgraphs    采样足够数量的子图
        #   2. update the counter for each node / edge in the training graph   更新训练图中每个节点/边的计数器
        #   3. estimate norm factor alpha and lambda   估计范数因子α和λ
        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample('train')     ###并行执行图采样
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)   ##子图数量
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1       ###聚集
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1           ###损失
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0    ###检测代码参数是否有问题
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s : i_e] = val
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.to(device)
            #self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self,phase):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        并行执行图形抽样。       graph_samplers.py的包装函数
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)  ##extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)    ####函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    def one_batch(self, mode='train'):
        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'val' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).
        为训练器生成一个minibatch  在'train'模式中，对应一个minibatch到训练图的一个子图
        在'val'或'test'模式下，一个bacth对应于完整的图
        原因是由于train时需要多个子图的迭代更新，更新权值，但是val或test是不需要再次训练数据，所以train的minibatch是对应多个子图
        而test这只需要一个全区的batch来进行预测即可
        
        
        
        Inputs:
            mode                str, can be 'train', 'val', 'test' or 'valtest'    输入   模型  分为 train  val  test  or   valtest
 
        Outputs:
            node_subgraph       np array, IDs of the subgraph / full graph nodes           np数组类型，子图/全图节点的id
            adj                 scipy CSR, adj matrix of the subgraph / full graph         子图/全图的adj矩阵
            norm_loss           np array, loss normalization coefficients. In 'val' or
                                'test' modes, we don't need to normalize, and so the values
                                in this array are all 1.
                                Np数组，
                                损失归一化系数。在'val'或'test'模式中，我们不需要标准化，因此这个数组中的值都是1。
        """
        if mode in ['val','test','valtest']:   ##如果模型是在[]中，这进行全图的标准化，不用子图进行标准化
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])
            adj = self.adj_full_norm
        else:
            assert mode == 'train'     ###------- assert为测试调试程序时而使用的
            if len(self.subgraphs_remaining_nodes) == 0: 
                self.par_graph_sample('train')
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()   #### pop() 函数用于移除列表中的一个元素（默认删除最后一个列表值），并且返回该元素的值。
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix(
                (
                    self.subgraphs_remaining_data.pop(),
                    self.subgraphs_remaining_indices.pop(),
                    self.subgraphs_remaining_indptr.pop()),
                    shape=(self.size_subgraph,self.size_subgraph,
                )
            )
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=args_global.num_cpu_core)
            # adj.data[:] = self.norm_aggr_train[adj_edge_index][:]      # this line is interchangable with the above line
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])
            adj = _coo_scipy2torch(adj.tocoo())
            if self.use_cuda:
                adj = adj.to(device)
               #adj = adj.cuda()
            self.batch_num += 1
        norm_loss = self.norm_loss_test if mode in ['val','test', 'valtest'] else self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph]
        return self.node_subgraph, adj, norm_loss   ##  图内的所有节点id   邻接矩阵  归一化的loss


    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))
    #### “向上取整”， 即小数部分直接舍去，并向正数部分进1      56 node_train = np.array(role['tr']) 8688
    ###size_subg_budget = train_phases['size_subg_edge'] * 2  子图预算=  4000*2   数据参数来自yml

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)  ###对(self.node_train)进行随机排序
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]
