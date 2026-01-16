from tqdm import tqdm
from typing import Any, Union
from collections import defaultdict


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


from transformers import Trainer,GenerationConfig
try:
    from transformers.utils import is_compiled_module
except ImportError:
    def is_compiled_module(module):
        """检查模块是否被 torch.compile 编译过"""
        return hasattr(module, "_orig_mod")





from accelerate.utils import (
    gather_object,
    broadcast_object_list,
)



try:
    from trl.trainer.utils import maybe_apply_chat_template, is_conversational
except ImportError:
    # 手动实现这些辅助函数
    
    def is_conversational(example):
        """
        检查样本是否为对话格式 (即 'prompt' 字段是消息列表而不是字符串)。
        """
        prompt = example.get("prompt")
        # 检查是否为 list，且内部元素看起来像消息字典 (包含 'role')
        if isinstance(prompt, list) and len(prompt) > 0:
            if isinstance(prompt[0], dict) and "role" in prompt[0]:
                return True
        return False

    def maybe_apply_chat_template(example, tokenizer):
        """
        如果样本是对话格式，则应用 tokenizer 的 chat_template 将其转换为字符串。
        如果已经是字符串，则原样返回。
        """
        if is_conversational(example):
            prompt = example["prompt"]
            # 应用模板转换为纯文本
            prompt_text = tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # 返回新字典，避免修改原数据
            return {"prompt": prompt_text}
        
        # 非对话格式，直接返回
        return example


from utils import compute_and_save_svd_lora
def pad(seqs, padding_value):
    return pad_sequence(seqs, batch_first=True, padding_value=padding_value)


from contextlib import contextmanager


@contextmanager
def unwrap_model_for_generation(model, accelerator, gather_deepspeed3_params=True):
    
    unwrapped_model = accelerator.unwrap_model(model)
    
    if hasattr(unwrapped_model, "_orig_mod"):
        unwrapped_model = unwrapped_model._orig_mod


    is_gradient_checkpointing = getattr(unwrapped_model, "is_gradient_checkpointing", False)
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_disable()


    
    try:
        if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
            if not gather_deepspeed3_params:
                yield unwrapped_model
            else:
                import deepspeed
                
                with deepspeed.zero.GatheredParameters(model.parameters()):
                    yield unwrapped_model
        else:
          
            yield unwrapped_model
            
    finally:
        
        if is_gradient_checkpointing:
            unwrapped_model.gradient_checkpointing_enable()




class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, nn.Module],
        reward_funcs: list,
        args: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: Any = None,
        callbacks: list = None,
        optimizers: tuple = (None, None),
    ):
       
        self.reward_funcs = reward_funcs
        self.num_generations = args.num_generations
        self.max_prompt_length = getattr(args, 'max_prompt_length', 1024)
        self.max_completion_length = args.max_completion_length
        
        print(f"[debug] GRPOTrainer 初始化:")
        print(f"  - num_generations: {self.num_generations}")
        print(f"  - max_prompt_length: {self.max_prompt_length}")
        print(f"  - max_completion_length: {self.max_completion_length}")
        print(f"  - reward_funcs 数量: {len(self.reward_funcs)}")
        
        
        self._metrics = defaultdict(list)

        
        super().__init__(
            model=model,
            args=args,
            data_collator=None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Ref Model 直接指向 Model
       
        self.ref_model = self.model

        
        if getattr(args, "generation_config", None) is not None:
             self.generation_config = args.generation_config
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature if hasattr(args, "temperature") else 1.0,
                pad_token_id=processing_class.pad_token_id,
                eos_token_id=processing_class.eos_token_id,
                num_return_sequences=1
            )

        #vLLM 初始化
        if hasattr(args, "use_vllm") and args.use_vllm:
            from vllm import LLM, SamplingParams
            model_path = model.config._name_or_path
            
            if self.accelerator.is_main_process:
               
                self.llm = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    dtype="auto",
                    gpu_memory_utilization=getattr(args, "vllm_gpu_memory_utilization", 0.05), 
                )
            else:
                self.llm = None
            
            self.sampling_params = SamplingParams(
                temperature=args.temperature if hasattr(args, "temperature") else 1.0,
                max_tokens=256,
                n=self.num_generations
            )
            self._last_loaded_step = -1
    def _get_per_token_logps(self,model,input_ids,attention_mask,logits_to_keep):

        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  

        
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
            input_ids = input_ids[:, -logits_to_keep:]
        
       
        return torch.gather(logits.log_softmax(-1), dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
        


    def _prepare_inputs(self,inputs: dict[str,Union[torch.Tensor,Any]]) -> dict[str,Union[torch.Tensor,Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example,self.processing_class)["prompt"] for example in inputs]
        
        print(f"\n[调试] _prepare_inputs 开始处理 {len(inputs)} 个样本")
        print(f"  - 设备: {device}")
        print(f"  - Prompts 数量: {len(prompts)}")
        
        prompt_inputs = self.processing_class(prompts_text,return_tensors = 'pt',padding = True,padding_side = 'left',add_special_tokens = False)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids,prompt_mask = prompt_inputs["input_ids"],prompt_inputs["attention_mask"]
        print(f"  - Prompt IDs 形状: {prompt_ids.shape}")

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:,-self.max_prompt_length : ]
            prompt_mask = prompt_mask[:,-self.max_prompt_length : ]
        
        prompt_ids = prompt_ids.repeat_interleave(self.num_generations,dim = 0)
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations,dim = 0)

        if self.args.use_vllm:
            # if self.state.global_step != self._last_loaded_step:
            #     with unwrap_model_for_generation(
            #         self.model,self.accelerator,gather_deepspeed3_params = getattr(self.args, 'ds3_gather_for_generation', False)
            #     ) as unwrapped_model:
            #         if is_compiled_module(unwrapped_model):
            #             state_dict = unwrapped_model._orig_mod.state_dict()
            #         else:
            #             state_dict = unwrapped_model.state_dict()
            #     if self.accelerator.is_main_process:
            #         llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            #         llm_model.load_weights(state_dict.items())
            #     self._last_loaded_step = self.state.global_step
            # all_prompts_text = gather_object(prompts_text)
            # if self.accelerator.is_main_process:
            #     outputs = self.llm.generate(all_prompts_text,sampling_params = self.sampling_params,use_tqdm = False)
            #     completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            # else:
            #     completion_ids = [None] * len(all_prompts_text) * self.num_generations
            # completion_ids = broadcast_object_list(completion_ids,from_process = 0)
            # process_slice = slice(
            #     self.accelerator.process_index * len(prompts) * self.num_generations,
            #     (self.accelerator.process_index + 1 )* len(prompts) * self.num_generations, 
            # )
            # completion_ids = completion_ids[process_slice]

            # completion_ids = [torch.tensor(ids,device = device)for ids in completion_ids]
            # completion_ids = pad(completion_ids,padding_value=self.processing_class.pad_token_id)
            # prompt_ids = torch.repeat_interleave(prompt_ids,self.num_generations,dim = 0)
            # prompt_mask = torch.repeat_interleave(prompt_mask,self.num_generations,dim = 0)
            # prompt_completions_ids = torch.cat([prompt_ids,completion_ids],dim = 1)
            raise RuntimeError("LoRA 初始化阶段禁止使用 vLLM")
        else:
            with torch.no_grad():
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    prompt_completions_ids = unwrapped_model.generate(
                        prompt_ids,
                        attention_mask=prompt_mask,
                        generation_config=self.generation_config
                    )

            torch.cuda.empty_cache()
            prompt_length = prompt_ids.size(1)
           
            completion_ids = prompt_completions_ids[:,prompt_length : ]
            prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)

        # with torch.inference_mode():
        #     if self.ref_model is not None:
        #         ref_per_token_logps = self._get_per_token_logps(self.ref_model,prompt_completions_ids,attention_mask,logits_to_keep)
        #     else:
        #         with self.accelerator.unwrap_model(self.model).disable_adapter():
        #             ref_per_token_logps = self._get_per_token_logps(self.model,prompt_completions_ids,attention_mask,logits_to_keep)
        completions = self.processing_class.batch_decode(completion_ids,skip_special_tokens = True)
        
        print(f"  - 生成的 completions 数量: {len(completions)}")
        print(f"  - 是否对话格式: {is_conversational(inputs[0])}")
        
        if is_conversational(inputs[0]):
            completions = [[{"role":"assistant","content":completion}]for completion in completions]
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        print(f"\n[调试] 开始计算奖励")
        print(f"  - Reward functions 数量: {len(self.reward_funcs)}")
        rewards_per_func = torch.zeros(len(prompts),len(self.reward_funcs),device = device)
        for i, reward_func in enumerate(self.reward_funcs):
        # 收集除 prompt / completion 之外的其他字段
            reward_kwargs = {
                key: []
                for key in inputs[0].keys()
                if key not in ["prompt", "completion"]
            }

            # 将 batch 中的字段按 num_generations 展开到 B*G
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend(
                        [example[key]] * self.num_generations
                    )

            # 调用 reward 函数
            output_rewards = reward_func(
                prompts=prompts,
                completions=completions,
                **reward_kwargs,
            )

            # 写入 rewards_per_func
            rewards_per_func[:, i] = torch.tensor(
                output_rewards,
                dtype=torch.float32,
                device=device,
            )
            print(f"  - Reward function {i} ({reward_func.__name__ if hasattr(reward_func, '__name__') else 'unknown'}): 平均={sum(output_rewards)/len(output_rewards):.4f}")
        
        rewards = rewards_per_func.sum(dim = 1)
        print(f"  - 总奖励: 平均={rewards.mean().item():.4f}, 最小={rewards.min().item():.4f}, 最大={rewards.max().item():.4f}")
                
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        print(f"\n[debug] 优势计算完成")
        print(f"  - 优势: 平均={advantages.mean().item():.4f}, 标准差={advantages.std().item():.4f}")
        
        # 打印每个 prompt 及其所有生成的响应、奖励和优势
        print(f"\n{'='*80}")
        print(f"详细生成结果 (每个 Prompt 的所有响应)")
        print(f"{'='*80}")
        
        num_prompts = len(inputs)
        for prompt_idx in range(num_prompts):
            print(f"\n[Prompt {prompt_idx+1}/{num_prompts}]")
            # 获取原始 prompt
            if is_conversational(inputs[0]):
                prompt_content = inputs[prompt_idx]["prompt"]
                if isinstance(prompt_content, list):
                    user_msg = [m for m in prompt_content if m.get("role") == "user"]
                    if user_msg:
                        print(f"问题: {user_msg[-1].get('content', '')[:200]}...")
            else:
                print(f"问题: {str(inputs[prompt_idx].get('prompt', ''))[:200]}...")
            
            # 打印该 prompt 的所有生成结果
            for gen_idx in range(self.num_generations):
                global_idx = prompt_idx * self.num_generations + gen_idx
                completion_text = completions[global_idx]
                if isinstance(completion_text, list) and len(completion_text) > 0:
                    completion_text = completion_text[0].get("content", "")
                
                reward_val = rewards[global_idx].item()
                advantage_val = advantages[global_idx].item()
                
                print(f"\n  生成 {gen_idx+1}:")
                print(f"    响应: {str(completion_text)[:300]}...")
                print(f"    奖励: {reward_val:.4f}")
                print(f"    优势: {advantage_val:.4f}")
        
        print(f"\n{'='*80}\n")
        
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module): 
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        print(
        "SHAPES | prompt_ids:", prompt_ids.shape,
        "completion_ids:", completion_ids.shape,
        "attention_mask:", attention_mask.shape
        )


        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            # "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    def compute_loss(self,model,inputs,return_outputs = False,num_items_in_batch = None):
        #获取当前策略下token的对数概率
        prompt_ids,prompt_mask = inputs["prompt_ids"],inputs["prompt_mask"]
        completion_ids,completion_mask = inputs["completion_ids"],inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids,completion_ids],dim = 1)
        attention_mask = torch.cat([prompt_mask,completion_mask],dim = 1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model,input_ids,attention_mask,logits_to_keep)

        #计算策略模型与参考模型的KL散度
        # ref_per_token_logps = inputs["ref_per_token_logps"]
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        #x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = - (per_token_logps * advantages.unsqueeze(1))
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss
    

    def extract_gradients(self,train_dataloader,output_dir,num_init_samples = 128,lora_r = 16,lora_alpha = 32,target_modules = {'q_proj',"v_proj"},init_direction = "ArBr",init_scale = "stable"):
        if not self.accelerator.is_main_process:
            return
        print("\nCGI：提取梯度并初始化LoRA参数")

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        named_grads = {}
        hooks = []

        #注册hook收集梯度
        def make_hook(name):
            def hook(grad):
                if any(t in name for t in target_modules):
                    if name not in named_grads:
                        named_grads[name] = grad.detach().cpu()
                    else:
                        named_grads[name] += grad.detach().cpu()
            return hook
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                hooks.append(param.register_hook(make_hook(name)))
        
        #前向/反传
        iterator = iter(train_dataloader)
        samples_processed = 0
        steps_taken = 0
        pbar = tqdm(total = num_init_samples,desc = "样本处理进度",unit = 'sample')
        
        print(f"\n[debug] 开始梯度提取")
        print(f"  - 目标样本数: {num_init_samples}")
        print(f"  - 目标模块: {target_modules}")

        while samples_processed < num_init_samples:
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                inputs = next(iterator)
            
            
            print(f"\n[debug] 批次大小计算")
            print(f"  - inputs 类型: {type(inputs)}")
            print(f"  - isinstance(inputs, list): {isinstance(inputs, list)}")
            if isinstance(inputs, list):
                print(f"  - len(inputs): {len(inputs)}")
            
            bsz = len(inputs) if isinstance(inputs, list) else 1
            print(f"  - 计算得到的批次大小 bsz: {bsz}")
            print(f"  - 当前已处理样本数: {samples_processed}")
            print(f"  - 本批次后总样本数: {samples_processed + bsz}")
            
            self.model.zero_grad()
            with torch.set_grad_enabled(True):

                rollout = self._prepare_inputs(inputs)

                prompt_ids = rollout["prompt_ids"]          
                completion_ids = rollout["completion_ids"]  
                advantages = rollout["advantages"]           
                prompt_mask = rollout["prompt_mask"]        
                completion_mask = rollout["completion_mask"] 

                batch_size = prompt_ids.size(0)  

               
                for idx in range(batch_size):
                    input_ids = torch.cat([
                        prompt_ids[idx], 
                        completion_ids[idx]
                    ], dim=0).unsqueeze(0)  

                    attention_mask = torch.cat([
                        prompt_mask[idx],
                        completion_mask[idx]
                    ], dim=0).unsqueeze(0) 

                    logps = self._get_per_token_logps(
                        self.model,
                        input_ids,
                        attention_mask,
                        logits_to_keep=completion_ids.size(-1),
                        )

                    current_mask = completion_mask[idx].unsqueeze(0).float()  
                    masked_logps = logps * current_mask
                    valid_tokens = current_mask.sum()
                    
                    if valid_tokens > 0:
                        mean_logps = masked_logps.sum() / valid_tokens
                    else:
                        mean_logps = logps.mean()
                    
                    loss = -(mean_logps * advantages[idx])
                    self.accelerator.backward(loss)
            del rollout
            torch.cuda.empty_cache()

            
            samples_processed += bsz
            steps_taken += 1
            pbar.update(bsz)
        pbar.close()
        for hook in hooks:
            hook.remove()
        
        #求平均并保存
        print(f"\nCGI：处理样本 {samples_processed}，梯度更新步数 {steps_taken}。开始SVD分解并保存LoRA参数。")
        for k in named_grads:
            named_grads[k] /= steps_taken
        
        compute_and_save_svd_lora(
            named_grads = named_grads,
            output_dir = output_dir,
            rank = lora_r,
            alpha = lora_alpha,
            target_modules_set = target_modules,
            direction = init_direction,
            scale_mode = init_scale,
        )

        del named_grads
        torch.cuda.empty_cache()






        









