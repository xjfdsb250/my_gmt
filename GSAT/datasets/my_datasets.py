import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import numpy as np
import ast
from torch_geometric.utils import dense_to_sparse


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

        adjs = np.load(adj_path, allow_pickle=True)
        feats = np.load(feat_path, allow_pickle=True)
        labels = np.load(label_path)

        try:
            with open(explanation_path, 'r') as f:
                explanations = ast.literal_eval(f.read())
        except FileNotFoundError:
            print(f"警告: 未找到解釋文件 {explanation_path}。將為所有節點生成空的 'node_label'。")
            explanations = {}

        data_list = []
        num_graphs = len(labels)
        print(f"正在處理數據集 '{self.name}'，共找到 {num_graphs} 個圖...")

        for i in range(num_graphs):
            node_features = feats[i]
            adj_row = adjs[i]  # 獲取代表單個圖的數據行

            # --- !! 最終核心修改邏輯 !! ---

            # 1. 根據特徵文件確定實際節點數
            num_actual_nodes = node_features.shape[0]

            # 2. 計算扁平化鄰接矩陣所需的長度
            expected_len = num_actual_nodes * num_actual_nodes

            # 3. 檢查提供的鄰接信息數組長度是否足夠
            if len(adj_row) < expected_len:
                print(
                    f"警告: 圖 {i} 的鄰接信息長度 ({len(adj_row)}) 不足以構成一個 {num_actual_nodes}x{num_actual_nodes} 的矩陣 (需要 {expected_len})，已跳過。")
                continue

            # 4. 截取所需部分並重塑為二維鄰接矩陣
            try:
                # 截取前 expected_len 個元素並重塑
                adj_matrix = adj_row[:expected_len].reshape(num_actual_nodes, num_actual_nodes)
            except ValueError as e:
                print(f"警告: 圖 {i} 在重塑鄰接矩陣時出錯: {e}，已跳過。")
                continue

            # 5. 從稠密鄰接矩陣轉換為稀疏的 edge_index 格式
            edge_index = dense_to_sparse(torch.from_numpy(adj_matrix).float())[0]

            # --- 修改結束 ---

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

            if edge_index.numel() > 0:
                if edge_index.max() >= num_actual_nodes:
                    print(
                        f"警告: 圖 {i} 的 edge_index 包含越界索引 ({edge_index.max()})，節點數為 {num_actual_nodes}，已跳過。")
                    continue
                edge_label_mask = (node_label[edge_index[0]] == 1) & (node_label[edge_index[1]] == 1)
                edge_label = edge_label_mask.float()
            else:
                edge_label = torch.empty(0, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y, node_label=node_label, edge_label=edge_label)
            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError("處理完成後沒有生成任何有效的圖數據。請檢查你的原始 .npy 文件是否為空或格式完全錯誤。")

        print(f"處理完成！成功創建 {len(data_list)} 個 Data 對象。")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])