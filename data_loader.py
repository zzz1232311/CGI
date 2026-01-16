import os
from datasets import load_from_disk, Dataset

# SYSTEM_PROMPT 保持不变
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

def load_dataset(dataset_path):
    
    print(f"\n[debug] 加载数据集")
    print(f"  - 数据集路径: {dataset_path}")

    # 检查数据集目录是否存在
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
    
    # 使用 datasets.load_from_disk 加载数据
    try:
        raw_dataset = load_from_disk(dataset_path)

    except Exception as e:
        raise

    data_list = []
    print(f"开始转换数据格式")

    # 遍历数据集的每一行
    for row in raw_dataset:
        
        user_content = row.get("question", "")
        solution_content = row.get("answer", "")
        
        
        if not user_content or not solution_content:
            continue

        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        
        item = {
            "prompt": messages,      # 用于模型输入的 prompt
            "solution": solution_content # 用于计算奖励的正确答案
        }
        data_list.append(item)

    
    for i, item in enumerate(data_list[:3]):
        print(f"\n  [转换后样本 {i+1}]")
        
        user_msg = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'user'), "")
        print(f"    用户指令 : {user_msg[:150]}...")
        print(f"    参考答案 : {item['solution'][:150]}...")

    print(f"\n[debug] 数据集加载和转换完成")
    print(f"  - 成功处理: {len(data_list)} 条数据")
    
    if len(data_list) == 0:
        raise ValueError("数据处理后为空")
    hf_dataset = Dataset.from_list(data_list)
    return hf_dataset


if __name__ == '__main__':
   
    path_to_dataset = '/home/tsinghua/slw/datasets/gsm8k/train' 
    
    try:
        processed_data = load_gsm8k_dataset(path_to_dataset)
        print(f"\n返回成功，获得 {len(processed_data)} 条数据。")
        
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"\n出错: {e}")

