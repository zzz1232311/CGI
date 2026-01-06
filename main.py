import argparse
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments
from trl import GRPOConfig

from grpo_init import GRPOTrainer
from data_loader import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",type=str,required=True)
    parser.add_argument("--dataset_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--num_samples",type=int,default=128)
    parser.add_argument("--batch_size",type=int,default=2)
    parser.add_argument("--max_length",type=int,default=512)
    parser.add_argument("--num_generations",type=int,default=8)
    parser.add_argument("--use_vllm",action="store_true")
    parser.add_argument("--lora_rank",type=int,default=8)
    parser.add_argument("--lora_alpha",type=int,default=16)
    parser.add_argument("--direction",type=str,default="ArBr",choices=["ArBr","A2rBr","ArB2r","A_B"])
    parser.add_argument("--scale_mode",type=str,default="stable",choices=["stable","gd","none"])

    return parser.parse_args()

def main():
    args = parse_args()

    print("\n" + "="*50)
    print(f"*** CGI初始化LoRA矩阵 ***")
    print(f"Model:       {args.model_path}")
    print(f"Output:      {args.output_dir}")
    print(f"Samples:     {args.num_samples} prompts")
    print(f"Batch Size:  {args.batch_size} prompts per step")
    print(f"Generations: {args.num_generations} responses per prompt (Real Batch = {args.batch_size * args.num_generations})")
    print(f"Strategy:    {args.direction} / {args.scale_mode}")
    print("="*50 + "\n")

    #加载模型
    print("加载模型")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
        
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    #加载数据集
    dataset = load_dataset(args.dataset_path)

    #初始化config
    training_args = GRPOConfig(
        output_dir = os.path.join(args.output_dir,"tmp"),
        per_device_train_batch_size = args.batch_size,
        num_generations = args.num_generations,
        max_completion_length = args.max_length,
        logging_steps = 10,
        report_to = "none",
        remove_unused_columns = False,
        use_vllm = args.use_vllm,
        bf16 = True,
    )
    
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        args = training_args,
        train_dataset = dataset,
        reward_funcs = [],
    )

    #梯度初始化
    target_modules = {"q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","fc1","fc2"}

  
    trainer.extract_gradients(
        train_dataloader=trainer.get_train_dataloader(),
        output_dir=args.output_dir,
        num_init_samples=args.num_samples, 
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        init_direction=args.direction,
        init_scale=args.scale_mode
    )

    print("\n" + "="*50)
    print(f"初始化完成！权重已保存至: {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()