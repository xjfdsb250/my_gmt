import numpy as np

file_path = './data/tree_cycles/raw/tree_cycles_X.npy'

try:
    data = np.load(file_path, allow_pickle=True)

    # 现在 'data' 变量包含了从 .npy 文件中加载的 NumPy 数组（或对象数组）
    print(f"成功加载文件: {file_path}")
    print("-" * 30)

    # 打印数据的基本信息
    print(f"数据类型: {type(data)}")

    # 检查 data 是否是 NumPy 数组
    if isinstance(data, np.ndarray):
        print(f"数组形状 (Shape): {data.shape}")
        print(f"数组数据类型 (dtype): {data.dtype}")
        print("-" * 30)

        # 打印数组内容
        # 注意：如果数组非常大，打印整个数组可能会导致输出过多
        # 这里我们只打印一部分内容作为示例

        # 如果是一维数组
        if data.ndim == 1:
            print("数组内容 (前 10 项):")
            print(data[:10])
        # 如果是二维数组
        elif data.ndim == 2:
            print("数组内容 (前 5 行，前 10 列):")
            print(data[:5, :10])
        # 如果是更高维数组
        elif data.ndim > 2:
            print(f"数组内容 (第一个元素，形状 {data[0].shape}):")
            print(data[0])
        # 如果是 0 维数组（标量）
        else:
            print("数组内容 (标量):")
            print(data)
    elif data.dtype == 'object':
        print("检测到对象数组 (Object Array)。")
        print(f"数组形状 (Shape): {data.shape}")
        print("-" * 30)
        print("对象数组内容 (前 3 个对象):")
        # 遍历对象数组的前几个元素
        for i, item in enumerate(data.flat):  # 使用 .flat 迭代所有元素
            if i >= 3:
                break
            print(f"  对象 {i}:")
            print(f"    类型: {type(item)}")
            # 如果对象本身是 NumPy 数组，打印其形状
            if isinstance(item, np.ndarray):
                print(f"    形状: {item.shape}")
                print(f"    内容 (前5项/行): {item[:5]}")
            # 如果是其他类型，直接打印
            else:
                print(f"    内容: {item}")
    else:
        print("加载的数据不是 NumPy 数组。")
        print("数据内容:", data)


except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。")
except Exception as e:
    print(f"读取或处理文件时发生错误: {e}")
