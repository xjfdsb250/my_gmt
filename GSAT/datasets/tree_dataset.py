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

        try:
            adj_matrices = np.load(adj_path, allow_pickle=True)
            node_features = np.load(feat_path, allow_pickle=True)
            graph_labels = np.load(label_path, allow_pickle=True)
        except Exception as e:
            print(f"严重错误：加载 .npy 文件失败: {e}")
            print("请确保 'raw' 目录中的 .npy 文件存在且未损坏。")
            data_list = []
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            return

        try:
            with open(expl_path, 'r') as f:
                explanations_str = f.read()
                explanation_nodes = ast.literal_eval(explanations_str)
        except Exception as e:
            print(f"警告：加载 explanations.txt 文件失败: {e}。 'edge_label' 将全为 0。")
            explanation_nodes = {}

        data_list = []
        num_graphs_without_features = 0
        num_graphs_A_X_mismatch = 0

        for i in range(len(graph_labels)):
            current_features = node_features[i]
            adj_row = adj_matrices[i]  # 可能是 1D 或 2D

            # --- 修复逻辑开始 ---

            # 步骤 1：确定 num_nodes 和节点特征 x
            # 优先级：X > A
            num_nodes = 0
            x = None

            if current_features is not None and hasattr(current_features, 'shape') and current_features.shape[0] > 0:
                # 情况 A：X 存在。num_nodes 由 X 决定。
                x = torch.tensor(current_features, dtype=torch.float)
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                num_nodes = x.shape[0]

            else:
                # 情况 B：X 不存在。num_nodes 必须由 A 推断。
                if i == 0: num_graphs_without_features += 1

                if adj_row.ndim == 1:
                    # A 是 1D。它必须是完美平方。
                    adj_N_float = np.sqrt(adj_row.shape[0])
                    adj_N = int(adj_N_float)
                    if adj_N_float != adj_N:
                        if adj_row.shape[0] == 0:
                            adj_N = 0
                        else:
                            print(f"[警告] 跳过图 {i}: X 不存在 且 A 是 1D 非平方 (shape {adj_row.shape})。数据已损坏。")
                            continue  # (这就是图 870 发生的情况)
                    num_nodes = adj_N

                elif adj_row.ndim == 2:
                    # A 是 2D。
                    num_nodes = adj_row.shape[0]
                else:
                    print(f"[警告] 跳过图 {i}: X 不存在 且 A 维度无效 (shape {adj_row.shape})。")
                    continue

            # 步骤 2：基于 num_nodes 处理邻接矩阵 A
            adj_matrix = None
            if num_nodes == 0:
                adj_matrix = np.empty((0, 0))

            elif adj_row.ndim == 1:
                # A 是 1D 扁平数组 (类似 my_datasets.py 的逻辑)
                expected_len = num_nodes * num_nodes
                if adj_row.shape[0] < expected_len:
                    if i == 0: num_graphs_A_X_mismatch += 1
                    print(
                        f"[警告] 跳过图 {i}: 1D 邻接矩阵 (len {adj_row.shape[0]}) 对于 N={num_nodes} (N*N={expected_len}) 来说太短了。")
                    continue

                try:
                    adj_matrix = adj_row[:expected_len].reshape(num_nodes, num_nodes)
                except ValueError as e:
                    print(f"[警告] 跳过图 {i}: Reshape 失败 (N={num_nodes}, shape={adj_row.shape}). Error: {e}")
                    continue

            elif adj_row.ndim == 2:
                # A 已经是 2D 数组
                if adj_row.shape[0] < num_nodes:
                    if i == 0: num_graphs_A_X_mismatch += 1
                    print(
                        f"[警告] 跳过图 {i}: 2D 邻接矩阵 (shape {adj_row.shape}) 对于 N={num_nodes} (来自 X) 来说太小了。")
                    continue

                if adj_row.shape[0] > num_nodes:
                    if i == 0: num_graphs_A_X_mismatch += 1

                adj_matrix = adj_row[:num_nodes, :num_nodes]  # 截取子矩阵

            # 此时, 'adj_matrix' 保证是 2D [num_nodes, num_nodes]

            # 步骤 3：如果 X 不存在，现在生成它
            if x is None:
                if num_nodes == 0:
                    x = torch.empty((0, 1), dtype=torch.float)
                else:
                    # （先创建 edge_index 来计算度数）
                    temp_sparse_adj = sp.csr_matrix(adj_matrix)
                    temp_edge_index, _ = from_scipy_sparse_matrix(temp_sparse_adj)

                    degs = degree(temp_edge_index[0], num_nodes=num_nodes, dtype=torch.long)
                    max_degree = degs.max().item() if num_nodes > 0 and temp_edge_index.numel() > 0 else 0
                    x = torch.nn.functional.one_hot(degs, num_classes=max(1, max_degree + 1)).to(torch.float)

            # --- 修复逻辑结束 ---

            # 步骤 4：创建 Edge Index
            if num_nodes == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                sparse_adj = sp.csr_matrix(adj_matrix)
                edge_index, _ = from_scipy_sparse_matrix(sparse_adj)

            y = torch.tensor([graph_labels[i]], dtype=torch.long)

            # 步骤 5：创建 edge_label
            expl_nodes_for_graph = set(explanation_nodes.get(i, []))
            edge_label = torch.zeros(edge_index.size(1), dtype=torch.long)
            for j in range(edge_index.size(1)):
                u, v = edge_index[0, j].item(), edge_index[1, j].item()
                if u in expl_nodes_for_graph or v in expl_nodes_for_graph:
                    edge_label[j] = 1

            data = Data(x=x, edge_index=edge_index, y=y, edge_label=edge_label)

            data_list.append(data)

        if num_graphs_without_features > 0:
            print(
                f"[INFO] Dataset '{self.name}': {num_graphs_without_features} 个图没有节点特征。已自动生成度数特征。")
        if num_graphs_A_X_mismatch > 0:
            print(
                f"[INFO] Dataset '{self.name}': 在多个图中检测到 A 和 X 维度不匹配。已自动截断或跳过。")

        if len(data_list) == 0:
            print("\n" + "=" * 80)
            print("!! 严重警告 !!")
            print(f"在处理完所有 {len(graph_labels)} 个图后，'data_list' 仍然为空。")
            print("这很可能意味着 *所有* 的原始图数据都已损坏 (例如 A/X 不匹配, 或 X 不存在且 A 非平方)。")
            print("请检查您的 'raw' 目录中的 .npy 文件。")
            print("=" * 80 + "\n")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
