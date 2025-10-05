import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import numpy as np
import ast
from torch_geometric.utils import dense_to_sparse

# --- 新增偵錯開關 ---
# 如果問題解決，您可以將此設置為 False 以減少不必要的輸出
VERBOSE_DEBUG = True


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
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        adj_path = osp.join(self.raw_dir, f'{self.name}_A.npy')
        feat_path = osp.join(self.raw_dir, f'{self.name}_X.npy')
        label_path = osp.join(self.raw_dir, f'{self.name}_L.npy')

        adjs = np.load(adj_path, allow_pickle=True)
        feats = np.load(feat_path, allow_pickle=True)
        labels = np.load(label_path)

        # --- 增強的解釋文件加載和偵錯邏輯 ---
        specific_explanation_path = osp.join(self.raw_dir, f'{self.name}_explanations.txt')
        generic_explanation_path = osp.join(self.raw_dir, 'explanations.txt')

        explanation_path_to_use = None
        if osp.exists(specific_explanation_path):
            explanation_path_to_use = specific_explanation_path
        elif osp.exists(generic_explanation_path):
            explanation_path_to_use = generic_explanation_path

        explanations = {}
        if explanation_path_to_use:
            try:
                with open(explanation_path_to_use, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        raise ValueError("解釋文件為空。")
                    parsed_explanations = ast.literal_eval(content)
                    if not isinstance(parsed_explanations, dict):
                        raise TypeError(f"解釋文件內容不是一個字典，而是 {type(parsed_explanations)}。")

                    explanations = {int(k): v for k, v in parsed_explanations.items()}

                print(f"\n[INFO] 成功加載並解析解釋文件: '{explanation_path_to_use}'")
                print(f"[INFO] 共找到 {len(explanations)} 條解釋。")
                if VERBOSE_DEBUG and explanations:
                    # 打印前5個鍵值對以供診斷
                    print("[DEBUG] 解釋字典的前5個鍵值對:")
                    for i, (k, v) in enumerate(explanations.items()):
                        print(f"  - 鍵: {k} (類型: {type(k)}), 值: {v}")
                        if i == 4: break
                    print("-" * 20)

            except Exception as e:
                print(f"\n[嚴重錯誤] 加載或解析 '{explanation_path_to_use}' 時出錯: {e}")
                print("[嚴重錯誤] 請檢查文件是否存在、內容是否為標準的 Python 字典格式，且鍵是否能轉換為數字。")
                print("[嚴重錯誤] 由於解釋文件加載失敗，'explanation_accuracy' 將為 0。\n")
        else:
            print(f"\n[警告] 在 raw 目錄中未找到 '{specific_explanation_path}' 或 '{generic_explanation_path}'。")
            print("[警告] 'explanation_accuracy' 將為 0。\n")
        # --- 修改結束 ---

        data_list = []
        total_positive_edge_labels = 0
        num_graphs = len(labels)
        print(f"正在處理數據集 '{self.name}'，共找到 {num_graphs} 個圖...")

        for i in range(num_graphs):
            node_features = feats[i]
            adj_row = adjs[i]

            num_actual_nodes = node_features.shape[0]
            expected_len = num_actual_nodes * num_actual_nodes

            if len(adj_row) < expected_len:
                continue

            try:
                adj_matrix = adj_row[:expected_len].reshape(num_actual_nodes, num_actual_nodes)
            except ValueError:
                continue

            edge_index = dense_to_sparse(torch.from_numpy(adj_matrix).float())[0]

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
                    continue

                edge_label_mask = (node_label[edge_index[0]] == 1) | (node_label[edge_index[1]] == 1)
                edge_label = edge_label_mask.float()
            else:
                edge_label = torch.empty(0, dtype=torch.float)

            total_positive_edge_labels += edge_label.sum().item()

            data = Data(x=x, edge_index=edge_index, y=y, node_label=node_label, edge_label=edge_label)
            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError("處理完成後沒有生成任何有效的圖數據。請檢查你的原始 .npy 文件是否為空或格式完全錯誤。")

        if total_positive_edge_labels == 0:
            print("\n" + "=" * 80)
            print("!! 最終警告 !!")
            print(f"在數據集 '{self.name}' 的所有 {len(data_list)} 個圖中，未生成任何有效的解釋標籤 (edge_label > 0)。")
            print("這意味著 'explanations.txt' 未被找到、為空，或其中定義的 'important_nodes' 未能與任何邊匹配。")
            print("因此，'explanation_accuracy' 指標將始終為 0。")
            print("=" * 80 + "\n")

        print(f"處理完成！成功創建 {len(data_list)} 個 Data 對象。")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])