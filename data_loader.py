import os
from datasets import load_from_disk, Dataset

# SYSTEM_PROMPT 保持不变
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

def load_dataset(dataset_path):
    """
    使用 datasets 库从磁盘加载 gsm8k 数据集（Arrow 格式）。
    """
    print(f"\n[调试] 加载数据集")
    print(f"  - 数据集路径: {dataset_path}")

    # 检查数据集目录是否存在
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
    
    # 使用 datasets.load_from_disk 加载数据
    try:
        print("  - 目录存在，尝试使用 `load_from_disk` 加载...")
        raw_dataset = load_from_disk(dataset_path)
        print(f"  - datasets 库加载成功，共 {len(raw_dataset)} 条原始数据。")
    except Exception as e:
        print(f"  [错误] 使用 datasets 库加载失败: {e}")
        raise

    data_list = []
    print(f"  - 开始转换数据格式...")

    # 遍历数据集的每一行
    for row in raw_dataset:
        # 从 Arrow 数据中获取 'question' 和 'answer'
        # 根据 dataset_info.json，列名是 'question' 和 'answer'
        user_content = row.get("question", "")
        solution_content = row.get("answer", "")
        
        # 如果 'question' 或 'answer' 为空，可以跳过
        if not user_content or not solution_content:
            continue

        # 构建 Chat 格式的消息列表
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # 构建符合 trl 要求的最终格式
        item = {
            "prompt": messages,      # 用于模型输入的 prompt
            "solution": solution_content # 用于计算奖励的正确答案
        }
        data_list.append(item)

    # 调试：打印转换后的前3条数据
    for i, item in enumerate(data_list[:3]):
        print(f"\n  [转换后样本 {i+1}]")
        # 注意 item['prompt'] 是一个列表，我们只打印 user 的内容
        user_msg = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'user'), "")
        print(f"    用户指令 (截取): {user_msg[:150]}...")
        print(f"    参考答案 (截取): {item['solution'][:150]}...")

    print(f"\n[调试] 数据集加载和转换完成")
    print(f"  - 成功处理: {len(data_list)} 条数据")
    
    if len(data_list) == 0:
        raise ValueError("数据处理后为空，请检查原始数据或处理逻辑")
    hf_dataset = Dataset.from_list(data_list)
    return hf_dataset

# --- 使用示例 ---
if __name__ == '__main__':
    # 将路径指向你的 gsm8k/train 目录
    path_to_dataset = '/home/tsinghua/slw/datasets/gsm8k/train' 
    
    try:
        processed_data = load_gsm8k_dataset(path_to_dataset)
        print(f"\n函数返回成功，共获得 {len(processed_data)} 条格式化数据。")
        # 你可以在这里继续使用 processed_data 进行后续的训练
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"\n程序出错: {e}")

