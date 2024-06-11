import sys
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


def load_node_mapping(datafile_path, index_col, offset=0):
    """
    将每个独特的节点映射到一个唯一的整数索引.

    从节点名到整数索引的映射
    """
    df = pd.read_csv(datafile_path, index_col=index_col, sep=",")
    data1 = df.index.unique()
    mapping = {index_id: i + offset for i, index_id in enumerate(data1)}
    return mapping


def load_edge_list(datafile_path, src_col, src_mapping, dst_col, dst_mapping):
    """
    给定节点映射, 返回按照节点整数索引的边列表.

    从节点名到整数索引的映射
    """
    df = pd.read_csv(datafile_path, sep=",")
    src_nodes = []
    dst_nodes = []
    # print(df[src_col])  # C0013182  C0023530  C0023903...
    # print(src_mapping)  # '{C0013182': 0, 'C0023530': 1, 'C0023903': 2, 'C0027794': 3...}
    # print(df.shape)  # (117738, 3)
    for i in range(df.shape[0]):
        # flag_list.append(df["flag"][i])
        src_nodes.append(src_mapping[df[src_col][i]])
        dst_nodes.append(dst_mapping[df[dst_col][i]])
    edge_index = torch.tensor([src_nodes, dst_nodes])
    return edge_index


def initialize_data(datafile_path, num_features=20):
    """
    给定tsv文件, 指明疾病-基因的相互作用, 索引节点, 并构建一 Data 对象.
    """
    # 获得疾病节点映射和基因节点映射.
    # 每个节点类型都有自己的整数id的集合.
    dz_col, gene_col = "Disease ID", "Gene ID"
    dz_mapping = load_node_mapping(datafile_path, dz_col, offset=0)
    gene_mapping = load_node_mapping(datafile_path, gene_col, offset=519)
    # 根据分配给节点的整数索引来获取边索引.
    edge_index = load_edge_list(
        datafile_path, dz_col, dz_mapping, gene_col, gene_mapping)

    x1 = torch.ones((len(dz_mapping) + len(gene_mapping), num_features))
    datas = []
    for i in range(10):
        chunks_edge = torch.chunk(edge_index, 10, dim=1)
        chunks_x1 = torch.chunk(x1, 10, dim=0)
        test_edge = chunks_edge[i]
        test_x1 = chunks_x1[i]
        train_edge = 0
        train_x1 = 0
        for j in range(10):
            if i == j:
                continue
            if type(train_edge) is int:
                train_edge = chunks_edge[j]
            else:
                train_edge = torch.cat((train_edge, chunks_edge[j]), dim=-1)
            if type(train_x1) is int:
                train_x1 = chunks_x1[j]
            else:
                train_x1 = torch.cat((train_x1, chunks_x1[j]), dim=0)
        data_train = Data(x=train_x1, edge_index=train_edge)
        data_test = Data(x=train_x1, edge_index=test_edge)
        datas.append([data_train, data_test])

    return datas, gene_mapping, dz_mapping


def get_mapping(data_path):
    df = pd.read_csv(data_path, index_col="Disease Name", sep=",")
    disease_mapping = [index_id for index_id in enumerate(df.index.unique())]
    df = pd.read_csv(data_path, index_col="Gene ID", sep=",")
    gene_mapping = [index_id[1] for index_id in enumerate(df.index.unique())]
    mapping = disease_mapping + gene_mapping
    return mapping


def flag_data(datafile_path, num_features=20):
    """
    给定tsv文件, 指明疾病-基因的相互作用, 索引节点, 并构建一 Data 对象.
    """
    # 获得疾病节点映射和基因节点映射.
    # 每个节点类型都有自己的整数id的集合.
    dz_col, gene_col, flag_col = "Disease ID", "Gene ID", "flag"
    df = pd.read_csv(datafile_path, sep=",")
    src_nodes = []
    dz_mapping = []
    gene_mapping = []
    dst_nodes = []
    flag_list = []
    weights_list = []
    old_part = []
    new_part = []

    for i in range(df.shape[0]):
        if df[dz_col][i] not in dz_mapping:
            dz_mapping.append(df[dz_col][i])
        if df[gene_col][i] not in gene_mapping:
            gene_mapping.append(df[gene_col][i])
        src_nodes.append(df[dz_col][i])
        dst_nodes.append(df[gene_col][i])
        flag_list.append(df[flag_col][i])

        if df[flag_col][i] == 0:
            old_part.append([df[dz_col][i], df[gene_col][i]])

        else:
            new_part.append([df[dz_col][i], df[gene_col][i]])


    new_part = torch.tensor(new_part).t()  # 需要转置
    old_part = torch.tensor(old_part).t()
    # ----------------------------------------------新加内容-------------------------------------------------------------
    x1 = torch.ones((len(dz_mapping) + len(gene_mapping), num_features))
    datas = []
    datas_weight=[]
    for i in range(10):
        chunks_edge = torch.chunk(old_part, 10, dim=1)
        test_edge = chunks_edge[i]
        train_edge = 0
        for j in range(10):
            if i == j:
                continue
            if type(train_edge) is int:
                train_edge = chunks_edge[j]
            else:
                train_edge = torch.cat((train_edge, chunks_edge[j]), dim=-1)
        # If the list is not empty, do the following.
        if len(new_part) > 0:

            train_edge = torch.cat((train_edge, new_part), dim=-1)

        data_train = Data(x=x1, edge_index=train_edge)
        data_test = Data(x=x1, edge_index=test_edge)
        datas.append([data_train, data_test])


    return datas, gene_mapping, dz_mapping
