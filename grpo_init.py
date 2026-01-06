from tqdm import tqdm
from typing import Any, Union



import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


from transformers import apply_chat_template
from transformers.trainer_utils import unwrap_model_for_generation
from transformers.utils import is_compiled_module


from accelerate.utils import (
    gather_object,
    broadcast_object_list,
)


from trl.trainer.utils import (
    maybe_apply_chat_template,
    is_conversational,
)

from utils import compute_and_save_svd_lora
def pad(seqs, padding_value):
    return pad_sequence(seqs, batch_first=True, padding_value=padding_value)




class GRPOTrainer(Trainer):
    def __init__():

        return


    def _get_per_token_logps(self,model,input_ids,attention_mask,logits_to_keep):

        return


    def _prepare_inputs(self,inputs: dict[str,Union[torch.tensor,any]]) -> dict[str,Union[torch.tensor,any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example,self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(prompts_text,return_tensors = 'pt',padding = True,padding_side = 'left',add_special_tokens = False)
        prompt_inputs = super._prepare_inputs(prompt_inputs)

        prompt_ids,prompt_mask = prompt_inputs["input_ids"],prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:,-self.max_prompt_length : ]
            prompt_mask = prompt_mask[:,-self.max_prompt_length : ]

        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                    self.model,self.accelerator,gather_deepspeed3_params = self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.loda_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text,sampling_params = self.sampling_params,use_tqdm = False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations
            completion_ids = broadcast_object_list(completion_ids,from_process = 0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1 )* len(prompts) * self.num_generations, 
            )
            completion_ids = completion_ids[process_slice]

            completion_ids = [torch.tensor(ids,device = device)for ids in completion_ids]
            completion_ids = pad(completion_ids,padding_value=self.processing_class.pad_token_id)
            prompt_ids = torch.repeat_interleave(prompt_ids,self.num_generations,dim = 0)
            prompt_mask = torch.repeat_interleave(prompt_mask,self.num_generations,dim = 0)
            prompt_completions_ids = torch.cat([prompt_ids,completion_ids],dim = 1)
        else:
            with unwrap_model_for_generation(self.model,self.accelerator) as unwrapped_model:
                prompt_completions_ids = unwrapped_model.generate(
                    prompt_ids,attention_mask = prompt_mask,generation_config = self.generation_config
                )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completions_ids[:, : prompt_length]
            completion_ids = prompt_completions_ids[:,prompt_length : ]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations,dim = 0)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model,prompt_completions_ids,attention_mask,logits_to_keep)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(self.model,prompt_completions_ids,attention_mask,logits_to_keep)
        completions = self.processing_class.batch_decode(completion_ids,skip_special_tokens = True)
        if is_conversational(inputs[0]):
            completions = [[{"role":"assistant","content":completion}]for completion in completions]
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

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

            # 调用 reward 函数（返回 list[float], 长度 B*G）
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
        rewards = rewards_per_func.sum(dim = 1)
                
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module): 
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
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
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        #x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss
    

    def extract_gradients(self,train_dataloader,output_dir,num_init_samples = 128,lora_r = 16,lora_alpha = 32,target_modules = {'q_prog',"v_proj"},init_direction = "ArBr",init_scale = "stable"):
        if not self.acclerator.is_main_process:
            return
        print("\nCGI：提取梯度并初始化LoRA参数")

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        named_grads = {}
        hooks = {}

        #注册hook收集梯度
        def make_hook(name):
            def hook(grad):
                if any(t in name for t in target_modules):
                    if name not in named_grads:
                        named_grads[name] = grad.detach().cpu()
                    else:
                        named_grads[name] += grad.detach().cpu()
                return hook
        for name,param in self.model.named_paramters():
            if param.requires_grad:
                hooks.append(param.register_hook(make_hook(name)))
        
        #前向/反传
        iterator = iter(train_dataloader)
        samples_processed = 0
        steps_taken = 0
        pbar = tqdm(total = num_init_samples,desc = "样本处理进度",unit = 'sample')

        while samples_processed < num_init_samples:
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                inputs = next(iterator)
            
            bsz = 1
            if isinstance(inputs,dict):
                if "prompt_ids" in inputs:
                    bsz = inputs["prompt_ids"].size(0)
                elif "input_ids" in inputs:
                    bsz = inputs["input_ids"].size(0)
            self.model.zero_grad()
            with torch.set_grad_enabled(True):
                processed_inputs = self._prepare_inputs(inputs)
                loss = self.compute_loss(self.model,processed_inputs)
                self.accelerator.backward(loss)
            
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






        









