import ast
import os

# --- 請根據您的實際情況修改以下變量 ---

# 1. 您的數據集名稱
dataset_name = 'ba_community'

# 2. 數據集的根目錄
#    這通常是 GSAT/configs/global_config.yml 文件中 'data_dir' 的值
data_dir = './data'  # 例如: '/home/xjf/project/GMT-main/GSAT/data'

# 3. 鍵的偏移量。根據您的日誌，鍵從300開始，所以我們需要減去300
offset = 300

# -----------------------------------------

# 構建文件路徑
raw_dir = os.path.join(data_dir, dataset_name, 'raw')
input_file_path = os.path.join(raw_dir, f'{dataset_name}_explanations.txt')
output_file_path = os.path.join(raw_dir, f'{dataset_name}_explanations_corrected.txt')
backup_file_path = os.path.join(raw_dir, f'{dataset_name}_explanations_original_backup.txt')


def correct_keys():
    """
    讀取解釋文件，將所有鍵減去指定的偏移量，並保存到新文件中。
    """
    if not os.path.exists(input_file_path):
        print(f"錯誤: 找不到輸入文件: {input_file_path}")
        return

    print(f"正在讀取文件: {input_file_path}")

    try:
        with open(input_file_path, 'r') as f:
            content = f.read()
            if not content.strip():
                print("錯誤: 文件為空。")
                return

            original_explanations = ast.literal_eval(content)
            if not isinstance(original_explanations, dict):
                print(f"錯誤: 文件內容不是一個有效的字典格式。")
                return

        # 創建一個新的字典，其中鍵是校正過的
        corrected_explanations = {int(k) - offset: v for k, v in original_explanations.items()}

        print(f"成功處理 {len(corrected_explanations)} 條解釋。")

        # 將校正後的字典寫入新文件
        with open(output_file_path, 'w') as f:
            # 使用 repr 確保輸出格式能被 ast.literal_eval 正確解析
            f.write(repr(corrected_explanations))

        print(f"已將校正後的解釋保存到: {output_file_path}")

        # 備份原始文件
        os.rename(input_file_path, backup_file_path)
        print(f"原始文件已備份到: {backup_file_path}")

        # 將校正後的文件重命名為原始文件名
        os.rename(output_file_path, input_file_path)
        print(f"校正後的文件已重命名為: {input_file_path}")
        print("\n校正完成！現在您可以重新運行訓練了。")

    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        print("請檢查文件內容和路徑是否正確。")


if __name__ == '__main__':
    correct_keys()
