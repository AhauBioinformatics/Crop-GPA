from graphsaint.globals import *
import numpy as np
import scipy.sparse
import time
import math
import pdb
from math import ceil
import graphsaint.cython_sampler as cy


class GraphSampler:
    """
    This is the sampler super-class. Any GraphSAINT sampler is supposed to perform
    the following meta-steps:
     1. [optional] Preprocessing: e.g., for edge sampler, we need to calculate the
            sampling probability for each edge in the training graph. This is to be
            performed only once per phase (or, once throughout the whole training,
            since in most cases, training only consists of a single phase. see
            ../train_config/README.md for definition of a phase).
            ==> Need to override the `preproc()` in sub-class
     2. Parallel sampling: launch a batch of graph samplers in parallel and sample
            subgraphs independently. For efficiency, the actual sampling operation
            happen in cython. And the classes here is mainly just a wrapper.
            ==> Need to set self.cy_sampler to the appropriate cython sampler
              in `__init__()` of the sampler sub-classz
     3. Post-processing: upon getting the sampled subgraphs, we need to prepare the
            appropriate information (e.g., subgraph adj with renamed indices) to
            enable the PyTorch trainer. Also, we need to do data conversion from C++
            to Python (or, mostly numpy). Post-processing is handled within the
            cython sampling file (./cython_sampler.pyx)

    Pseudo-code for the four proposed sampling algorithms (Node, Edge, RandomWalk,
    MultiDimRandomWalk) can be found in Appendix, Algo 2 of the GraphSAINT paper.

    Lastly, if you don't bother with writing samplers in cython, you can still code
    the sampler subclass in pure python. In this case, we have provided a function
    `_helper_extract_subgraph` for API consistency between python and cython. An
    example sampler in pure python is provided as `NodeSamplingVanillaPython` at the
    bottom of this file.
    """
    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):
        """
        Inputs:
            adj_train       scipy sparse CSR matrix of the training graph      稀疏CSR矩阵的训练图  
            node_train      1D np array storing the indices of the training nodes     存储训练节点索引的一维np数组
            size_subgraph   int, the (estimated) number of nodes in the subgraph      估计的子图中的节点数量
            args_preproc    dict, addition arguments needed for pre-processing        预处理所需的添加参数
            
        Outputs:
            None
        """
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)   ##对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
        # size in terms of number of vertices in subgraph     用子图中的顶点数表示大小
        self.size_subgraph = size_subgraph  ###子图中的节点数（估计）
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)    ###args_preproc ####添加预处理所需的参数

    def preproc(self, **kwargs):   ###预处理   **kwargs表示关键字参数，它是一个dict
        pass

    def par_sample(self, stage, **kwargs):   ####并行采样
        return self.cy_sampler.par_sample()     ###得到文件

    def _helper_extract_subgraph(self, node_ids):   ####辅助——选择——子图
        """
        ONLY used for serial Python sampler (NOT for the parallel cython sampler).
        Return adj of node-induced subgraph and other corresponding data struct.
        仅用于串行Python采样器(不用于并行cython采样器)。   返回节点诱导子图和其他相应数据结构的adj
        Inputs:
            node_ids        1D np array, each element is the ID in the original
                            training graph.   每个元素为原始训练图中的ID
        Outputs:
            indptr          np array, indptr of the subg adj CSR
            indices         np array, indices of the subg adj CSR
            data            np array, data of the subg adj CSR. Since we have aggregator
                            normalization, we can simply set all data values to be 1
                            因为我们有聚合器规范化，所以我们可以简单地将所有数据值设置为1
            subg_nodes      np array, i-th element stores the node ID of the original graph
                            for the i-th node in the subgraph. Used to index the full feats
                            and label matrices.
                            第i个元素存储子图中第i个节点的原始图的节点ID    用来索引完整的特征和标签矩阵。
            subg_edge_index np array, i-th element stores the edge ID of the original graph
                            for the i-th edge in the subgraph. Used to index the full array
                            of aggregation normalization.
                            第i个元素存储子图中第i条边的原始图的边ID。用于索引聚合规范化的完整数组
        """
        node_ids = np.unique(node_ids)   ##对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
        node_ids.sort()   ### 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。
        orig2subg = {n: i for i, n in enumerate(node_ids)}####   n为第一列    enumerate(node_ids)为第二列的数据
        n = node_ids.size   ###size 用来计算数组和矩阵中所有元素的个数
        indptr = np.zeros(node_ids.size + 1)   ###全零矩阵
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids
        for nid in node_ids:
            idx_s, idx_e = self.adj_train.indptr[nid], self.adj_train.indptr[nid + 1]
            neighs = self.adj_train.indices[idx_s : idx_e]    ##邻居   
            for i_n, n in enumerate(neighs):
                if n in orig2subg:
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64) ##cumsum（） 指定输出类型，或者按行累加或者按列累加
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size   ####断言函数就是针对某一行代码进行测试，得到输出结果，用来判断代码是否成功运行
        return indptr, indices, data, subg_nodes, subg_edge_index


# --------------------------------------------------------------------
# [BELOW] python wrapper for parallel samplers implemented with Cython 用于用Cython实现的并行采样器的python包装器
# --------------------------------------------------------------------

class rw_sampling(GraphSampler):##随机游走采样
    """
    The sampler performs unbiased random walk, by following the steps:
     1. Randomly pick `size_root` number of root nodes from all training nodes;
     2. Perform length `size_depth` random walk from the roots. The current node
            expands the next hop by selecting one of the neighbors uniformly
            at random;
     3. Generate node-induced subgraph from the nodes touched by the random walk.
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        """
        Inputs:
            adj_train       see super-class
            node_train      see super-class
            size_subgraph   see super-class
            size_root       int, number of root nodes (i.e., number of walkers)
            size_depth      int, number of hops to take by each walker

        Outputs:
            None
        """
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.RW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.size_root,
            self.size_depth
        )

    def preproc(self, **kwargs):
        pass


class edge_sampling(GraphSampler):##边采样
    def __init__(self,adj_train,node_train,num_edges_subgraph):
        """
        The sampler picks edges from the training graph independently, following
        a pre-computed edge probability distribution. i.e.,
        ******预先计算的边缘概率分布后，采样器从训练图中独立地选取边。
            p_{u,v} \\propto 1 / deg_u + 1 / deg_v
        Such prob. dist. is derived to minimize the variance of the minibatch
        estimator (see Thm 3.2 of the GraphSAINT paper).   
        ******根据最优边概率公式实现最小方差  从而选取边  
        """
        self.num_edges_subgraph = num_edges_subgraph
        # num subgraph nodes may not be num_edges_subgraph * 2 in many cases,
        # but it is not too important to have an accurate estimation of subgraph size. 
        # So it's probably just fine to use this number.
        # Num子图节点在很多情况下可能不是num_edges_subgraph * 2，
        # 但是精确估计子图的大小并不是很重要。所以用这个数就可以了
        
        self.size_subgraph = num_edges_subgraph * 2   #####子图节点     估计节点值
        self.deg_train = np.array(adj_train.sum(1)).flatten()   ###最优采样概率
        self.adj_train_norm = scipy.sparse.dia_matrix((1 / self.deg_train, 0), shape=adj_train.shape).dot(adj_train)###归一化
        super().__init__(adj_train, node_train, self.size_subgraph, {})    ###调用父类（超类）的一个方法 是用来解决多重继承的问题，直接使用类名调用 157
        self.cy_sampler = cy.Edge2(             ###使用cython 采样模块进行     
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.edge_prob_tri.row,
            self.edge_prob_tri.col,
            self.edge_prob_tri.data.cumsum(),
            self.num_edges_subgraph,
        )

    def preproc(self,**kwargs):####预处理
        """
        Compute the edge probability distribution p_{u,v}.
        计算边概率分布p
        """
        self.edge_prob = scipy.sparse.csr_matrix(
            (
                np.zeros(self.adj_train.size),
                self.adj_train.indices,
                self.adj_train.indptr
            ),
            shape=self.adj_train.shape,
        )
        self.edge_prob.data[:] = self.adj_train_norm.data[:]
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        self.edge_prob.data += _adj_trans.data      # P_e \propto a_{u,v} + a_{v,u}  公式
        self.edge_prob.data *= 2 * self.num_edges_subgraph / self.edge_prob.data.sum()
        # now edge_prob is a symmetric matrix, we only keep the upper triangle part, since adj is assumed to be undirected. 
        # 现在edge_probb是一个对称矩阵，我们只保留上面的三角形部分，因为adj被假定为无向的
        self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32)  # NOTE: in coo format


class mrw_sampling(GraphSampler):##多维的随机游走采样
    """
    A variant of the random walk sampler. The multi-dimensional random walk sampler
    is proposed in https://www.cs.purdue.edu/homes/ribeirob/pdf/ribeiro_imc2010.pdf

    Fast implementation of the sampler is proposed in https://arxiv.org/abs/1810.11899
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_frontier, max_deg=10000):
        """
        Inputs:
            adj_train       see super-class
            node_train      see super-class
            size_subgraph   see super-class
            size_frontier   int, size of the frontier during sampling process. The
                            size of the frontier is fixed during sampling.
            max_deg         int, the sampler picks iteratively pick a node from the
                            frontier by probability proportional to node degree. If
                            we specify the `max_deg`, we are essentially bounding the
                            probability of picking any frontier node. This may help
                            with improving sampling quality for skewed graphs.

        Outputs:
            None
        """
        self.p_dist = None
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.size_frontier = size_frontier
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'MRW'
        self.max_deg = int(max_deg)
        self.cy_sampler = cy.MRW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.p_dist,
            self.max_deg,
            self.size_frontier,
            self.size_subgraph
        )

    def preproc(self,**kwargs):
        _adj_hop = self.adj_train
        self.p_dist = np.array(
            [
                _adj_hop.data[_adj_hop.indptr[v] : _adj_hop.indptr[v + 1]].sum()
                for v in range(_adj_hop.shape[0])
            ],
            dtype=np.int32,
        )


class node_sampling(GraphSampler):###节点采样
    """
    Independently pick some nodes from the full training graph, based on
    pre-computed node probability distribution. The prob. dist. follows
    Sec 3.4 of the GraphSAINT paper. For detailed derivation, see FastGCN
    (https://arxiv.org/abs/1801.10247).
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        """
        Inputs:
            adj_train       see super-class
            node_train      see super-class
            size_subgraph   see super-class

        Outputs:
            None
        """
        self.p_dist = np.zeros(len(node_train))
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.Node(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.p_dist,
            self.size_subgraph,
        )

    def preproc(self, **kwargs):
        """
        Node probability distribution is derived in https://arxiv.org/abs/1801.10247
        """
        _p_dist = np.array(
            [
                self.adj_train.data[
                    self.adj_train.indptr[v] : self.adj_train.indptr[v + 1]
                ].sum()
                for v in self.node_train
            ],
            dtype=np.int64,
        )
        self.p_dist = _p_dist.cumsum()
        if self.p_dist[-1] > 2**31 - 1:
            print('warning: total deg exceeds 2**31')
            self.p_dist = self.p_dist.astype(np.float64)
            self.p_dist /= self.p_dist[-1] / (2**31 - 1)
        self.p_dist = self.p_dist.astype(np.int32)


class full_batch_sampling(GraphSampler):             #######test代码使用
    """
    Strictly speaking, this is not a sampler. It simply returns the full adj
    matrix of the training graph. This can serve as a baseline to compare
    full-batch vs. minibatch performance.
    严格来说，这不是采样器。 它只是返回训练图的完整的adj矩阵。
    这可以作为比较全批处理和小批处理性能的基线。  

    Therefore, the size_subgraph argument is not used here. 所以没有使用子图size的这个参数
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.FullBatch(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
        )


# --------------------------------------------
# [BELOW] Example sampler based on pure python   基于纯python的示例采样器
# --------------------------------------------

class NodeSamplingVanillaPython(GraphSampler):
    """
    This class is just to showcase how you can write the graph sampler in pure python.
    这个类只是为了展示如何用纯python编写图形采样器。

    The simplest and most basic sampler: just pick nodes uniformly at random and return the
    node-induced subgraph.  
    最简单和最基本的采样器:只需均匀随机地选取节点，并返回节点诱导子图。
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})

    def par_sample(self, stage, **kwargs):
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        pass
