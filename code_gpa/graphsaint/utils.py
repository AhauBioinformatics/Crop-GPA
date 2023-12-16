import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp
from graphsaint.globals import *


def load_data(prefix, normalize=True):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
                            CSR格式的稀疏矩阵，存储为scipy.sparse。csr_matrix形状为N乘N。矩阵中的非零对应于整个图中的所有边。
                            由一条边连接的两个节点是训练节点、验证节点还是测试节点都没有关系。对于非加权图，非零均为1。
                            
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
                            CSR格式的稀疏矩阵，存储为scipy.sparse。csr_matrix形状也是N * N。但是，矩阵中的非零只对应连接两个训练节点的边。
                            图采样器只从这个adj_train中选取节点/边，而不是adj_full。因此，在训练过程中既不显示属性信息，也不显示结构信息。
                            另外，注意，adj_train中只有xn行和cols包含非零。对于非加权图，非零均为1。
                            
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all  有三个键的字典。键'tr'对应所有的列表
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
                            注意，在原始数据中，节点可能具有字符串类型ID。
                            您需要为节点重新分配数字ID(0到N-1)，以便您可以在adj、特征和类标签的矩阵中建立索引。
                            
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
                            每个键是一个节点索引，每个值要么是一个长度为C的二进制列表(用于多类分类)，要么是一个整数(0到C-1，用于单类分类)。
                            
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.
                            第i行对应节点i的属性向量。

    Inputs:
        prefix              string, directory containing the above graph related files  字符串，包含上述图形相关文件的目录
        normalize           bool, whether or not to normalize the node features    Bool，是否对节点特性进行规范化

    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
                            scipy稀疏CSR(形状N x N， |E|非零)，的adj矩阵完整图，N为训练总数+ val +测试节点。
                            
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
                            scipy稀疏CSR(形状N x N， |E'|非零)，训练图的adj矩阵。当形状与adj_full相同时，adj_train中val/test节点对应的rows/cols为全零。
                                       
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
                            np数组(形状N x f)，节点特征矩阵，f为每个节点特征向量的长度
                            
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
                            其中key是节点ID, value是节点所属的类。
                            
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
                            tr'表示训练，'va'表示验证，'te'表示测试节点。该值是属于train/val/测试集的节点id列表。
    """
    ###type1
    
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))

    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    #print('class_map:',class_map)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def process_graph_data(adj_full, adj_train, feats, class_map, role):###预处理图的数据
    """
    setup vertex property map for output classes, train/val/test masks, and feats   为输出类、训练/val/测试掩码和专长设置顶点属性映射
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        #num_classes = max(class_map.values()) - min(class_map.values())
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        #print('offset',offset)
        #print('class_map.items()',class_map.items())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
            #class_arr[k]=-v
            #class_arr[k]=1
            #class_arr[v]=0
            #print(class_arr[k])
            #print(class_arr[v])
            #print(class_arr[k][v-offset])
            #print('k,v',k,v)
            #print('class_arr[k][v]',class_arr[k][v])

    #print('class_arr',class_arr)  
      
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(arch_gcn,dim_input):   #######arch 的具体内容，需要进一步理解
    """
    Parse the *.yml config file to retrieve the GNN structure.
    """
    num_layers = len(arch_gcn['arch'].split('-'))    ####4
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']]*num_layers
    act_layer = [arch_gcn['act']]*num_layers
    aggr_layer = [arch_gcn['aggr']]*num_layers
    dims_layer = [arch_gcn['dim']]*num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]    ###定义每个层计算时的层数量  并计算num_calsses类的数量
    print('num_layers',num_layers)
    return [dim_input]+dims_layer,order_layer,act_layer,bias_layer,aggr_layer


def parse_n_prepare(flags):     ###导入参数
    with open(flags.train_config) as f_train_config:
        train_config = yaml.load(f_train_config)
    arch_gcn = {           ###构造GCN需要的各个参数
        'dim': -1,
        'aggr': 'concat',
        'loss': 'softmax',
        'arch': '1',
        'act': 'I',
        'bias': 'norm'
    }
    arch_gcn.update(train_config['network'][0])
    train_params = {
        'lr': 0.01,
        'weight_decay': 0.,
        'norm_loss': True,
        'norm_aggr': True,
        'q_threshold': 50,
        'q_offset': 0
    }
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)  ###  process_graph_data函数通过输入数据得到   adj_full, adj_train, feats, class_arr, role
    print("Done loading training data..")
    return train_params,train_phases,train_data,arch_gcn    ####params phases 都是来自yml参数文件





def log_dir(f_train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def sess_dir(dims,train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return sess_dir


def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.
    根据rw归一化的方法归一化。注意，在原始GCN论文(kipf)中使用了sym规范，而在GraphSAGE和其他一些变体中使用了rw规范。
    这里我们不执行sym规范，因为它似乎无助于准确性的提高。

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    如果sort_indices为True，则对返回的adj的索引重新排序。
    注意，在'点'之后，节点的索引将按降序而不是升序排列
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))

