import torch
import os
import json
from tqdm import tqdm
from safetensors.torch import save_file


def compute_and_save_svd_lora(
    named_grads,
    output_dir,
    rank,
    alpha,
    target_modules_set,
    direction="ArBr",
    scale_mode="stable",
    stable_gamma=16,
    target_ab_norm=13.0,
    named_w0_norms=None,
    w0_ratio=0.01,
):
    """
    量级控制（二选一，优先 W0 自适应）：
    - named_w0_norms + w0_ratio: 补偿量 ‖(alpha/r)*B@A‖ 不超过该层 ‖W0‖ 的 w0_ratio（如 1%），
      即每层 target_ab_norm = w0_norm * w0_ratio / (alpha/rank)，有原则且 bf16 安全。
    - target_ab_norm: 全局目标 ‖B@A‖_F，当未传 named_w0_norms 时使用。
    """
    lora_state_dict = {}
    use_w0_adaptive = named_w0_norms is not None and len(named_w0_norms) > 0
    scaling = alpha / rank
    print(f"\n[debug] SVD分解开始")
    print(f"  - Rank: {rank}")
    print(f"  - Direction: {direction}")
    print(f"  - 量级: {'W0 自适应 (w0_ratio=' + str(w0_ratio) + ')' if use_w0_adaptive else 'target_ab_norm=' + str(target_ab_norm)}")
    print(f"  - 梯度数量: {len(named_grads)}")
    print(f"  - 目标模块: {target_modules_set}")

    skipped = 0
    processed = 0

    for name,grad in tqdm(named_grads.items(),desc = "lora矩阵初始化"):
        #过滤非目标层或非二维矩阵
        is_target = any(t in name for t in target_modules_set)
        if not is_target or grad.ndim != 2:
            if is_target: 
                skipped += 1
                print(f"  [跳过] {name}: 非二维矩阵 (shape={grad.shape})")
            continue
        
        
        grad = grad.float()
        if torch.cuda.is_available():
            grad = grad.cuda()

        #SVD分解
        try:
            U,S,V = torch.svd_lowrank(grad,q = 4 * rank,niter = 4)
        except Exception as e:
            print(f"  [警告] SVD分解失败 {name}: {e}")
            continue
        
        processed += 1
        V = V.T

        # 第一步：取方向（SVD 给的正交向量）
        if direction == "ArBr":
            B_mat = U[:, 0:2 * rank:2]
            A_mat = V[1:2 * rank:2, :]
        elif direction == "A2rBr":
            B_mat = U[:, :rank]
            A_mat = V[rank:2 * rank, :]
        elif direction == "ArB2r":
            B_mat = U[:, rank:2 * rank]
            A_mat = V[0:rank, :]
        else:
            B_mat = U[:, :rank]
            A_mat = V[:rank, :]

        # 第二步：计算当前 AB 乘积的 Frobenius 范数
        AB = B_mat @ A_mat
        ab_norm = float(AB.norm(p='fro'))

        # 第三步：目标范数。优先用 W0 推导：‖(alpha/r)*B@A‖ = w0_ratio * ‖W0‖ => ‖B@A‖ = w0_norm * w0_ratio / (alpha/rank)
        if use_w0_adaptive and name in named_w0_norms:
            w0_norm = float(named_w0_norms[name])
            layer_target = w0_norm * w0_ratio / scaling
        else:
            layer_target = target_ab_norm

        # 第四步：按目标范数缩放，方向不变。‖(c*B)@(c*A)‖ = c²‖B@A‖ => c = sqrt(layer_target / ab_norm)
        if ab_norm > 0:
            c = (layer_target / ab_norm) ** 0.5
            A_final = A_mat * c
            B_final = B_mat * c
        else:
            A_final = A_mat
            B_final = B_mat

        # 命名转换
        base_name = name.rsplit('.',1)[0]
        prefix = "base_model." + base_name

        lora_state_dict[f"{prefix}.lora_A.weight"] = A_final.cpu().contiguous()
        lora_state_dict[f"{prefix}.lora_B.weight"] = B_final.cpu().contiguous()

    #保存LoRA权重
    print(f"\n[debug] SVD分解完成")
    print(f"  - 处理的层数: {processed}")
    print(f"  - 跳过的层数: {skipped}")
    print(f"  - 生成的 LoRA 参数数量: {len(lora_state_dict)}")
    
    os.makedirs(output_dir,exist_ok=True)
    save_file(lora_state_dict,os.path.join(output_dir,"adapter_model.safetensors"))
    print(f"  - 保存路径: {os.path.join(output_dir, 'adapter_model.safetensors')}")


    config_dict = {
        "peft_type": "CGI",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": list(target_modules_set),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "init_params": {
            "direction": direction,
            "scale_mode": scale_mode,
            "stable_gamma": stable_gamma,
            "target_ab_norm": target_ab_norm,
            "w0_ratio": w0_ratio,
            "w0_adaptive": use_w0_adaptive,
        },
    }
    with open(os.path.join(output_dir,"adapter_config.json"),'w') as f:
        json.dump(config_dict,f,indent=2)
    print(f"\nCGI-LoRA权重已保存至：{output_dir}\n")
    
    
