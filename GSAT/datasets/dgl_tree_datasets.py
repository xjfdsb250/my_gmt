import torch
import os.path as osp
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_dgl

import dgl
import dgl.data


class DGLTreeDataset(InMemoryDataset):
    """
    一个 PyTorch Geometric (PyG) 包装器，用于加载 DGL 的 TreeCycle 和 TreeGrid 数据集。
    它会在内部下载 DGL 数据，并将其转换为 PyG 的 Data 对象格式。
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        if self.name not in ['tree_cycles', 'tree_grids']:
            raise ValueError(f"DGLTreeDataset 只支持 'tree_cycles' 或 'tree_grids'，但收到了 {name}")

        super(DGLTreeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        # 我们将 DGL 的下载也视为 'raw'
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # DGL 会自动处理下载，我们不需要原始文件
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # DGL 会在 'process' 阶段自动下载
        pass

    def process(self):
        data_list = []

        if self.name == 'tree_cycles':
            print("从 DGL 加载 TreeCycleDataset...")
            # DGL 数据集会自动下载到 ~/.dgl/
            dgl_dataset = dgl.data.TreeCycleDataset()

            print(f"Processing {len(dgl_dataset)} graphs from DGL TreeCycleDataset...")
            for i in range(len(dgl_dataset)):
                dgl_graph = dgl_dataset[i]
                if dgl_graph.num_nodes() > 0:
                    label = dgl_graph.ndata['label'][0]
                else:
                    default_label_val = 0
                    try:
                        # 检查 DGL 数据集是否存储了标签 dtype
                        if hasattr(dgl_dataset, 'labels') and dgl_dataset.labels is not None:
                            label_dtype = dgl_dataset.labels.dtype
                            label = torch.tensor(default_label_val, dtype=label_dtype)
                        elif hasattr(dgl_dataset, 'label') and dgl_dataset.label is not None:
                            label_dtype = dgl_dataset.label.dtype
                            label = torch.tensor(default_label_val, dtype=label_dtype)
                        else:
                            label = torch.tensor(default_label_val, dtype=torch.long)
                    except AttributeError:
                        label = torch.tensor(default_label_val, dtype=torch.long)

                data = from_dgl(dgl_graph)

                data.y = torch.tensor([label.item()], dtype=torch.long)

                if 'label' in dgl_graph.edata:
                    data.edge_label = dgl_graph.edata['label'].long()
                else:
                    data.edge_label = torch.zeros(data.num_edges, dtype=torch.long)

                if not hasattr(data, 'x') and 'feat' in dgl_graph.ndata:
                    data.x = dgl_graph.ndata['feat'].float()
                elif not hasattr(data, 'x'):
                    print(f"[警告] 图 {i} 缺少 'feat' 节点特征。使用全零占位符。")
                    data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

                data_list.append(data)

        elif self.name == 'tree_grids':
            print("从 DGL 加载 TreeGridDataset...")
            dgl_dataset = dgl.data.TreeGridDataset()

            print(f"Processing {len(dgl_dataset)} graphs from DGL TreeGridDataset...")
            for i in range(len(dgl_dataset)):
                dgl_graph = dgl_dataset[i]
                if dgl_graph.num_nodes() > 0:
                    label = dgl_graph.ndata['label'][0]
                else:
                    default_label_val = 0
                    try:
                        if hasattr(dgl_dataset, 'labels') and dgl_dataset.labels is not None:
                            label_dtype = dgl_dataset.labels.dtype
                            label = torch.tensor(default_label_val, dtype=label_dtype)
                        elif hasattr(dgl_dataset, 'label') and dgl_dataset.label is not None:  # 检查单数形式
                            label_dtype = dgl_dataset.label.dtype
                            label = torch.tensor(default_label_val, dtype=label_dtype)
                        else:
                            label = torch.tensor(default_label_val, dtype=torch.long)
                    except AttributeError:
                        label = torch.tensor(default_label_val, dtype=torch.long)

                data = from_dgl(dgl_graph)

                data.y = torch.tensor([label.item()], dtype=torch.long)

                if not hasattr(data, 'x') and 'feat' in dgl_graph.ndata:
                    data.x = dgl_graph.ndata['feat'].float()
                elif not hasattr(data, 'x'):
                    print(f"[警告] 图 {i} 缺少 'feat' 节点特征。使用全零占位符。")
                    data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

                if 'label' in dgl_graph.ndata:
                    node_expl_label = dgl_graph.ndata['label']
                    if data.edge_index.numel() > 0:
                        u, v = data.edge_index
                        # 确保索引在范围内（以防万一）
                        if u.max() < len(node_expl_label) and v.max() < len(node_expl_label):
                            edge_label = (node_expl_label[u] == 1) | (node_expl_label[v] == 1)
                            data.edge_label = edge_label.long()
                        else:
                            print(f"[警告] 图 {i} 的 edge_index 超出了 node_expl_label 的范围。边标签设为全零。")
                            data.edge_label = torch.zeros(data.num_edges, dtype=torch.long)
                    else:
                        data.edge_label = torch.empty(0, dtype=torch.long)
                else:
                    data.edge_label = torch.zeros(data.num_edges, dtype=torch.long)

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
