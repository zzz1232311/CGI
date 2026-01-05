import torch
import os
import json
from tqdm import tqdm
from safetensors.torch import save_file


def compute_and_save_svd_lora(named_grads,output_dir,rank,alpha,target_modules_set,direction = "ArBr",scale_mode = "stable",stable_gamma = 16):
    lora_state_dict = {}
    print(f"\n[SVD分解]：Rank = {rank},Dir = {direction},Scale = {scale_mode}")

    skipped = 0

    for name,grad in tqdm(named_grads.item(),desc = "lora矩阵初始化"):
        #过滤非目标层或非二维矩阵
        is_target = any(t in name for t in target_modules_set)
        if not is_target or grad.ndim != 2:
            if is_target: skipped += 1
            continue
        grad = grad.float().cuda()

        #SVD分解
        try:
            U,S,V = torch.svd_lowrank(grad,q = 4 * rank,niter = 4)
        except:
            continue
        V = V.T

        #向量分配
        if direction == "ArBr":
            B_mat = U[:,0:2 * rank:2] #偶数列
            A_mat = V[1:2 * rank:2,:] #奇数行 
