import torch
import os
import json
from tqdm import tqdm
from safetensors.torch import save_file


def compute_and_save_svd_lora(named_grads,output_dir,rank,alpha,target_modules_set,direction = "ArBr",scale_mode = "stable",stable_gamma = 16):
    lora_state_dict = {}
    print(f"\n[debug] SVD分解开始")
    print(f"  - Rank: {rank}")
    print(f"  - Direction: {direction}")
    print(f"  - Scale mode: {scale_mode}")
    print(f"  - Stable gamma: {stable_gamma}")
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

        #向量分配
        if direction == "ArBr":
            B_mat = U[:,0:2 * rank:2] #偶数列
            A_mat = V[1:2 * rank:2,:] #奇数行 
        elif direction == "A2rBr":
            B_mat = U[:,:rank]
            A_mat = V[rank:2 * rank,:]
        elif direction == "ArB2r":
            B_mat = U[:,rank:2 * rank]
            A_mat = V[0:rank,:]
        else:
            B_mat = U[:,:rank]
            A_mat = V[:rank,:]
        

        #缩放
        m,n = grad.shape

        if scale_mode == "stable":
            B_final = B_mat * (m**0.25) / (stable_gamma**0.5)
            A_final = A_mat * (n**0.25) / (stable_gamma**0.5)
        elif scale_mode == "gd":
            B_final = B_mat
            A_final = A_mat
        else:
            B_final = B_mat
            A_final = A_mat
        

        #命名转换
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
        "r":rank,
        "lora_alpha":alpha,
        "target_modules":list(target_modules_set),
        "bias":"none",
        "task_type":"CAUSAL_LM",
        "init_params":{"direction":direction,"scale_mode":scale_mode,"stable_gamma":stable_gamma}
    }
    with open(os.path.join(output_dir,"adapter_config.json"),'w') as f:
        json.dump(config_dict,f,indent=2)
    print(f"\nCGI-LoRA权重已保存至：{output_dir}\n")
    
    
