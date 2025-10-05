import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
import os.path as osp
import numpy as np
import ast  # 用于安全地将字符串转换为 Python 对象


class MyGraphClassificationDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(MyGraphClassificationDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [
            f'{self.name}_A.npy',
            f'{self.name}_X.npy',
            f'{self.name}_L.npy',
            f'{self.name}_explanations.txt'
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # --- 修改点 2: 读取全部四个文件 ---
        adj_path = osp.join(self.raw_dir, self.raw_file_names[0])
        feat_path = osp.join(self.raw_dir, self.raw_file_names[1])
        label_path = osp.join(self.raw_dir, self.raw_file_names[2])
        explanation_path = osp.join(self.raw_dir, self.raw_file_names[3])

        adjs = np.load(adj_path)
        feats = np.load(feat_path)
        labels = np.load(label_path)

        # --- 新增: 读取并解析解释文件 ---
        with open(explanation_path, 'r') as f:
            # ast.literal_eval 可以安全地将字符串格式的字典转为真实的字典
            explanations = ast.literal_eval(f.read())

        data_list = []
        num_graphs = adjs.shape[0]
        print(f"正在处理数据集 '{self.name}'，共找到 {num_graphs} 个图...")

        for i in range(num_graphs):
            adj_matrix = adjs[i]
            node_features = feats[i]
            graph_label = labels[i]

            actual_nodes_mask = np.any(node_features, axis=1)
            num_actual_nodes = int(np.sum(actual_nodes_mask))

            adj_matrix = adj_matrix[actual_nodes_mask][:, actual_nodes_mask]
            node_features = node_features[actual_nodes_mask]

            edge_index = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))[0]
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor([graph_label], dtype=torch.long)

            # --- 新增: 创建 node_label 和 edge_label ---
            node_label = torch.zeros(num_actual_nodes, dtype=torch.float)

            # 从解释字典中获取当前图的重要节点索引
            # 注意：图的索引可能与文件名中的索引不完全对应，这里我们假设是按顺序的
            if i in explanations:
                important_nodes = explanations[i]
                # 将重要节点的标签设置为 1
                node_label[important_nodes] = 1.0

            # 根据重要的节点，生成重要的边标签
            # 一条边是重要的，当且仅当它的两个端点都是重要的
            edge_label_mask = (node_label[edge_index[0]] == 1) & (node_label[edge_index[1]] == 1)
            edge_label = edge_label_mask.float()

            # 将 node_label 和 edge_label 添加到 Data 对象中
            data = Data(x=x, edge_index=edge_index, y=y,
                        node_label=node_label, edge_label=edge_label)

            data_list.append(data)

        print(f"处理完成！成功创建 {len(data_list)} 个 Data 对象。")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])