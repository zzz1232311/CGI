import re

def xml_count_reward(prompts, completions, **kwargs) -> list[float]:
   
    rewards = []
    for completion in completions:
        # 处理 completions 可能是 list[dict] (chat格式) 或 str 的情况
        content = completion[0]["content"] if isinstance(completion, list) else completion
        
        score = 0.0
        # 奖励包含思考标签
        if "<think>" in content:
            score += 0.5
        if "</think>" in content:
            score += 0.5
            
        # 简单的正则检查标签顺序
        pattern = r"<think>.*?</think>"
        if re.search(pattern, content, re.DOTALL):
            score += 1.0
            
        rewards.append(score)
    return rewards



# def accuracy_reward(prompts, completions, label, **kwargs):
#     ...
