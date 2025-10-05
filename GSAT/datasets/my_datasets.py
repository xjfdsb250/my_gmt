import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import numpy as np
import ast


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
        adj_path = osp.join(self.raw_dir, self.raw_file_names[0])
        feat_path = osp.join(self.raw_dir, self.raw_file_names[1])
        label_path = osp.join(self.raw_dir, self.raw_file_names[2])
        explanation_path = osp.join(self.raw_dir, self.raw_file_names[3])

        # 使用 allow_pickle=True 加载 .npy 对象数组
        adjs = np.load(adj_path, allow_pickle=True)
        feats = np.load(feat_path, allow_pickle=True)
        labels = np.load(label_path)

        with open(explanation_path, 'r') as f:
            explanations = ast.literal_eval(f.read())

        data_list = []
        num_graphs = len(labels)
        print(f"正在处理数据集 '{self.name}'，共找到 {num_graphs} 个图...")

        for i in range(num_graphs):
            # --- !! 最终核心修改点 !! ---

            # 1. 直接获取节点特征，并确定节点数
            node_features = feats[i]
            num_actual_nodes = node_features.shape[0]

            # 2. 获取一维的边列表
            edge_list_flat = adjs[i]

            # 3. 检查边列表是否为空或长度为奇数
            if edge_list_flat is None or len(edge_list_flat) % 2 != 0:
                print(f"警告: 图 {i} 的边列表格式不正确 (长度为 {len(edge_list_flat)})，已跳过。")
                continue

            # 4. 将一维边列表重塑为 [2, num_edges] 的 edge_index 格式
            if len(edge_list_flat) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list_flat, dtype=torch.long).reshape(-1, 2).t().contiguous()

            # --- 修改结束 ---

            # 如果特征是1D的，将其转换为 [N, 1] 的2D形状
            if node_features.ndim == 1:
                node_features = node_features.reshape(-1, 1)

            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor([labels[i]], dtype=torch.long)

            node_label = torch.zeros(num_actual_nodes, dtype=torch.float)

            if i in explanations:
                important_nodes = explanations[i]
                valid_important_nodes = [idx for idx in important_nodes if idx < num_actual_nodes]
                if valid_important_nodes:
                    node_label[valid_important_nodes] = 1.0

            edge_label_mask = (node_label[edge_index[0]] == 1) & (node_label[edge_index[1]] == 1)
            edge_label = edge_label_mask.float()

            data = Data(x=x, edge_index=edge_index, y=y,
                        node_label=node_label, edge_label=edge_label)

            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError("处理完成后没有生成任何有效的图数据。请检查你的原始 .npy 文件是否为空或格式完全错误。")

        print(f"处理完成！成功创建 {len(data_list)} 个 Data 对象。")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
