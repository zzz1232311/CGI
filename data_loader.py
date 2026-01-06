import json


SYSTEM_PROMPT = """You are a helpful AI assistant. You must first think about the reasoning process in the mind and then provide the answer. The reasoning process should be enclosed within <think> and </think> tags."""

def load_dataset(dataset_path):
    
    print(f"从 {dataset_path} 加载数据集...")
    
    data_list = []
    
    # 数据集是 jsonl 格式，每行 {"instruction": "...", "output": "..."}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                
                # 构建用户输入
                user_content = row.get("instruction", "")
                if "input" in row and row["input"]:
                    user_content += "\nInput: " + row["input"]
                
                # 构建 Chat 格式的消息列表
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                
                # 构建返回项
                # 注意：key 必须是 "prompt"，trl 才能识别
                item = {
                    "prompt": messages, 
                    # 需要计算正确性奖励
                    "label": row.get("output", "") 
                }
                data_list.append(item)
                print(item)
                print("-"*50)
            except json.JSONDecodeError:
                continue

    print(f"成功加载 {len(data_list)} 条数据 。")
    return data_list
