import pickle
import numpy as np

file_path = './data/ba_community/raw/ba_community.pkl'


def inspect_pkl(path):
    """
    加载并检查 .pkl 文件的内容和结构。
    """
    print(f"--- 开始检查文件: {path} ---")

    try:
        with open(path, 'rb') as f:
            # 加载数据
            data = pickle.load(f)

            # 1. 检查整体数据类型
            print(f"\n[1] 数据的整体类型: {type(data)}")

            # 2. 如果是列表或字典，检查其长度
            if isinstance(data, (list, dict, tuple)):
                print(f"[2] 数据的长度 (包含的元素/图的数量): {len(data)}")

                if len(data) > 0:
                    # 3. 检查第一个元素的结构
                    first_element = data[0]
                    print(f"\n[3] 第一个元素的类型: {type(first_element)}")

                    # 4. 如果第一个元素是字典，打印它的键和每个值的类型/形状
                    if isinstance(first_element, dict):
                        print("[4] 第一个元素的结构 (键和值的类型/形状):")
                        for key, value in first_element.items():
                            if isinstance(value, np.ndarray):
                                print(f"  - 键: '{key}', 值类型: numpy.ndarray, 形状: {value.shape}")
                            elif hasattr(value, 'shape'):  # 适用于 PyTorch Tensor 等
                                print(f"  - 键: '{key}', 值类型: {type(value)}, 形状: {value.shape}")
                            else:
                                print(f"  - 键: '{key}', 值类型: {type(value)}")

                    # 如果第一个元素是自定义对象，打印其属性
                    elif hasattr(first_element, '__dict__'):
                        print("[4] 第一个元素的属性:")
                        for attr, value in first_element.__dict__.items():
                            if isinstance(value, np.ndarray):
                                print(f"  - 属性: '{attr}', 值类型: numpy.ndarray, 形状: {value.shape}")
                            else:
                                print(f"  - 属性: '{attr}', 值类型: {type(value)}")

            else:  # 如果不是列表或字典
                if isinstance(data, np.ndarray):
                    print(f"[2] 数据的形状: {data.shape}")
                elif hasattr(data, 'shape'):
                    print(f"[2] 数据的形状: {data.shape}")
    except FileNotFoundError:
        print(f"错误: 文件 '{path}' 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"加载或检查文件时发生错误: {e}")

    print("\n--- 检查完毕 ---")


if __name__ == '__main__':
    inspect_pkl(file_path)
