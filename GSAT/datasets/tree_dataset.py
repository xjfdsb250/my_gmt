import os
import torch
import numpy as np
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix, degree
import ast
import scipy.sparse as sp


class TreeDataset(InMemoryDataset):
    """
    一个用于处理基于树结构的合成数据集的通用类。
    它会从三个独立的 .npy 文件 (_A, _X, _L) 和一个 explanations.txt 文件加载数据。
    如果节点特征文件 (_X.npy) 为空或为1D，它会自动创建或修正特征以确保它们是2D的。
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(TreeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}_A.npy', f'{self.name}_X.npy', f'{self.name}_L.npy', f'{self.name}_explanations.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        adj_path = osp.join(self.raw_dir, f'{self.name}_A.npy')
        feat_path = osp.join(self.raw_dir, f'{self.name}_X.npy')
        label_path = osp.join(self.raw_dir, f'{self.name}_L.npy')
        expl_path = osp.join(self.raw_dir, f'{self.name}_explanations.txt')

        adj_matrices = np.load(adj_path, allow_pickle=True)
        node_features = np.load(feat_path, allow_pickle=True)
        graph_labels = np.load(label_path, allow_pickle=True)

        with open(expl_path, 'r') as f:
            explanations_str = f.read()
            explanation_nodes = ast.literal_eval(explanations_str)

        data_list = []
        num_graphs_without_features = 0
        for i in range(len(graph_labels)):
            sparse_adj = sp.csr_matrix(adj_matrices[i])
            edge_index, _ = from_scipy_sparse_matrix(sparse_adj)

            current_features = node_features[i]
            if current_features is None or current_features.shape[0] == 0:
                if i == 0: num_graphs_without_features += 1
                num_nodes = adj_matrices[i].shape[0]
                if num_nodes == 0:
                    x = torch.empty((0, 1), dtype=torch.float)
                else:
                    degs = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
                    max_degree = degs.max().item()
                    x = torch.nn.functional.one_hot(degs, num_classes=max(1, max_degree + 1)).to(torch.float)
            else:
                # --- 这是核心修改：处理1D特征 ---
                x = torch.tensor(current_features, dtype=torch.float)
                if x.dim() == 1:
                    # 如果张量是1D的 (shape [N])，将其转换为2D (shape [N, 1])
                    x = x.unsqueeze(-1)
                # --- 修改结束 ---

            y = torch.tensor([graph_labels[i]], dtype=torch.long)

            expl_nodes_for_graph = set(explanation_nodes.get(i, []))
            edge_label = torch.zeros(edge_index.size(1), dtype=torch.long)
            for j in range(edge_index.size(1)):
                u, v = edge_index[0, j].item(), edge_index[1, j].item()
                if u in expl_nodes_for_graph and v in expl_nodes_for_graph:
                    edge_label[j] = 1

            data = Data(x=x, edge_index=edge_index, y=y, edge_label=edge_label)
            data_list.append(data)

        if num_graphs_without_features > 0:
            print(
                f"[INFO] Dataset '{self.name}': No/Empty node features found. Automatically generated degree-based features.")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
