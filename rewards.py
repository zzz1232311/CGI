#直接复用trl库里的奖励函数代码
import re


try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def think_format_reward(prompts, completions, **kwargs) -> list[float]:
    """
    奖励函数：检查推理过程是否被 <think> 和 </think> 包裹。
    """
    pattern = r"^<think>.*?</think>.*$"
    
    print(f"\n[调试] think_format_reward 被调用")
    print(f"  - completions 数量: {len(completions)}")
    print(f"  - completions[0] 类型: {type(completions[0])}")
    
    # 处理两种格式：对话格式（嵌套列表）和文本格式（字符串列表）
    if isinstance(completions[0], list):
        completion_contents = [completion[0]["content"] for completion in completions]
        print(f"  - 检测到对话格式")
    else:
        completion_contents = completions
        print(f"  - 检测到文本格式")
    
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    
    match_count = sum(rewards)
    print(f"  - 匹配 <think> 格式的数量: {match_count}/{len(rewards)}")
    
    return rewards


def reasoning_accuracy_reward(prompts, completions, solution, **kwargs) -> list[float]:
    """
    奖励函数：解析数学/逻辑答案并与标准答案(solution)对比。
    """
    if not HAS_MATH_VERIFY:
        raise ImportError("你需要安装 `pip install math_verify` 才能使用此奖励函数。或者使用简单的字符串匹配替代。")

    print(f"\n[调试] reasoning_accuracy_reward 被调用")
    print(f"  - completions 数量: {len(completions)}")
    print(f"  - solution 数量: {len(solution)}")
    
    # 处理两种格式
    if isinstance(completions[0], list):
        contents = [completion[0]["content"] for completion in completions]
        print(f"  - 检测到对话格式")
    else:
        contents = completions
        print(f"  - 检测到文本格式")
    
    rewards = []
    reasoning_delimiters = ["</think>"]

    for idx, (content, sol) in enumerate(zip(contents, solution)):
        # 1. 提取思考标签后的内容
        is_reasoning_complete = False
        for delim in reasoning_delimiters:
            if delim in content:
                content = content.split(delim)[-1]
                is_reasoning_complete = True
                break
        
        # 如果没有结束标签，惩罚（给0分）
        if not is_reasoning_complete:
            rewards.append(0.0)
            continue

        # 2. 解析标准答案
        gold_parsed = parse(sol)
        if len(gold_parsed) == 0:
            print(f"    [警告] 样本 {idx}: 标准答案解析失败")
            rewards.append(0.0)  # 改为 0.0 而不是 None
            continue

        # 3. 解析模型生成的答案
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(units=True),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        
        # 4. 验证两者是否相等
        reward = float(verify(gold_parsed, answer_parsed))
        rewards.append(reward)
    
    correct_count = sum(rewards)
    print(f"  - 正确答案数量: {correct_count}/{len(rewards)}")
    print(f"  - 准确率: {correct_count/len(rewards)*100:.2f}%")

    return rewards
