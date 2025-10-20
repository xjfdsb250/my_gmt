import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from datasets import SynGraphDataset, Mutag, SPMotif, MNIST75sp, graph_sst2
from datasets.my_datasets import MyGraphClassificationDataset
from datasets.tree_dataset import TreeDataset


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False):
    multi_label = False
    assert dataset_name in ['ba_2motifs', 'mutag', 'Graph-SST2', 'mnist',
                            'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9', 'spmotif_0.33',
                            'spmotif_0.50', 'spmotif_0.70', 'spmotif_0.90', 'spmotif_0.330',
                            'ogbg_molhiv', 'ogbg_moltox21', 'ogbg_molbace',
                            'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider',
                            'ba_community', 'ba_grids', 'ba_house_shapes',
                            'tree_cycles', 'tree_grids', 'ba_bottle_shaped']

    if dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs')
        split_idx = get_random_split_idx(dataset, splits)

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir / 'mutag')
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)

    # elif 'ogbg' in dataset_name:
    #     dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
    #     split_idx = dataset.get_idx_split()
    #     print('[INFO] Using default splits!')
    #     loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
    #     train_set = dataset[split_idx['train']]
    #
    # elif dataset_name == 'Graph-SST2':
    #     dataset = graph_sst2.get_dataset(dataset_dir=data_dir, dataset_name='Graph-SST2', task=None)
    #     dataloader, (train_set, valid_set, test_set) = graph_sst2.get_dataloader(dataset, batch_size=batch_size,
    #                                                                              degree_bias=True, seed=random_state)
    #     print('[INFO] Using default splits!')
    #     loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
    #     test_set = dataset  # used for visualization
    # elif 'spmotif' in dataset_name:
    #     b = dataset_name.split('_')[-1]
    #     train_set = SPMotif(root=data_dir / dataset_name, b=b, mode='train')
    #     valid_set = SPMotif(root=data_dir / dataset_name, b=b, mode='val')
    #     test_set = SPMotif(root=data_dir / dataset_name, b=b, mode='test')
    #     print('[INFO] Using default splits!')
    #     loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set,
    #                                                                              'test': test_set})
    #
    # elif dataset_name == 'mnist':
    #     n_train_data, n_val_data = 20000, 5000
    #     train_val = MNIST75sp(data_dir / 'mnist', mode='train')
    #     perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(random_state))
    #     train_val = train_val[perm_idx]
    #
    #     train_set, valid_set = train_val[:n_train_data], train_val[-n_val_data:]
    #     test_set = MNIST75sp(data_dir / 'mnist', mode='test')
    #     loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set,
    #                                                                              'test': test_set})
    #     print('[INFO] Using default splits!')
    elif dataset_name in ['ba_community', 'ba_grids', 'ba_house_shapes', 'ba_bottle_shaped']:
        dataset = MyGraphClassificationDataset(root=data_dir, name=dataset_name)
        split_idx = get_random_split_idx(dataset, splits)
    elif dataset_name in ['tree_cycles', 'tree_grids']:
        if splits is None:
            splits = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
            print(f"[INFO] No splits defined for {dataset_name}. Using default train/valid/test: 80/10/10.")
        dataset = TreeDataset(root=data_dir, name=dataset_name)
        split_idx = get_random_split_idx(dataset, splits)
    else:
        raise NotImplementedError

    loaders, test_set = get_loaders_and_test_set(batch_size, dataset, split_idx)
    x_dim = dataset.data.x.shape[1]
    edge_attr_dim = 0 if dataset.data.edge_attr is None else dataset.data.edge_attr.shape[1]

    if multi_label:
        num_class = dataset.num_tasks
    elif hasattr(dataset, 'num_classes'):
        num_class = dataset.num_classes
    # 确保 y 存在才调用 .max()
    elif hasattr(dataset.data, 'y') and dataset.data.y is not None:
        num_class = dataset.data.y.max().item() + 1
    else:
        num_class = 2

    # To follow PNA, we use the degree statistics of the training split
    deg = torch.zeros(12, dtype=torch.long)
    train_dataset = dataset[split_idx['train']]
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    aux_info = {'deg': deg, 'multi_label': multi_label}
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, mutag_x=False):
    # Fixed random seed
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(12345))

    if not mutag_x:
        # 确保 splits['test'] 存在
        test_split_ratio = splits.get('test', 0.1)  # 如果没有test键，默认为0.1
        train_split_ratio = splits['train']

        n_train = int(train_split_ratio * len(idx))
        n_valid = int(splits['valid'] * len(idx))

        # 确保即使比例加起来不完全等于1，测试集也能拿到数据
        n_test = len(idx) - n_train - n_valid
        if n_test < 0:  # 防止验证集比例过大导致测试集为负数
            n_valid = len(idx) - n_train
            n_test = 0

        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]

    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]

    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']

    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    return loaders, test_set
