import numpy as np
import os

# --- 用户需要修改以下两行 ---
# 1. 设置您正在尝试加载的数据集名称 (例如 'ba_community')
dataset_name = 'tree_grids'

# 2. 设置您的数据根目录。
#    这通常是 GSAT/configs/global_config.yml 文件中 'data_dir' 的值
#    请确保路径是正确的绝对路径或相对路径。
data_dir = './data'  # 例如: '/home/xjf/project/GMT-main/data'
# -----------------------------

# 构建文件路径
adj_path = os.path.join(data_dir, dataset_name, 'raw', f'{dataset_name}_A.npy')
feat_path = os.path.join(data_dir, dataset_name, 'raw', f'{dataset_name}_X.npy')

print(f"--- 正在调试数据集: {dataset_name} ---")
print(f"尝试从以下路径加载邻接数据: {adj_path}")
print(f"尝试从以下路径加载特征数据: {feat_path}\n")

try:
    # 加载数据
    adjs = np.load(adj_path, allow_pickle=True)
    feats = np.load(feat_path, allow_pickle=True)

    print(f"成功加载文件！")
    print(f"邻接数据 'adjs' 的类型: {type(adjs)}")
    if isinstance(adjs, np.ndarray):
        print(f"邻接数据 'adjs' 的形状 (shape): {adjs.shape}")
    print(f"邻接数据 'adjs' 的总长度 (图的数量): {len(adjs)}\n")

    # 检查前5个图的详细信息
    num_graphs_to_check = min(5, len(adjs))
    print(f"--- 正在检查前 {num_graphs_to_check} 个图的结构 ---\n")

    for i in range(num_graphs_to_check):
        print(f"--- 图索引: {i} ---")
        adj_info = adjs[i]
        feat_info = feats[i]
        num_nodes = feat_info.shape[0]

        print(f"  节点数 (来自特征文件): {num_nodes}")
        print(f"  邻接信息 (adj_info) 的类型: {type(adj_info)}")

        if isinstance(adj_info, np.ndarray):
            print(f"  邻接信息 (adj_info) 的维度 (ndim): {adj_info.ndim}")
            print(f"  邻接信息 (adj_info) 的形状 (shape): {adj_info.shape}")
            print(f"  邻接信息 (adj_info) 的长度 (len): {len(adj_info)}")
            if adj_info.ndim == 1:
                is_square_flattened = (num_nodes * num_nodes == len(adj_info))
                print(f"  检查是否为扁平化邻接矩阵 (len == num_nodes*num_nodes): {is_square_flattened}")
                print(f"  前10个元素: {adj_info[:10]}")

        print("-" * 20 + "\n")

except Exception as e:
    print(f"\n!!!!!! 在加载或检查文件时发生错误 !!!!!!")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    print("请确保上面的 'dataset_name' 和 'data_dir' 路径设置正确。")