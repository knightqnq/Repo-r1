#!/usr/bin/env python3
import os
import json
import torch
import random
import statistics
import gc
import warnings
import wandb
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

# ================= 1. 全局配置 (V3: 针对 313 数据集优化) =================
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

BASE_PATH = "/root/autodl-tmp/repor1/checkpoints/sft_gold_mpl_small_v3"
OUTPUT_PATH = "/root/autodl-tmp/repor1/checkpoints/dcda_sota_final_v3"

config = {
    # --- 训练参数优化 ---
    "group_size": 4,       
    "accum_steps": 4,      #  提升到4: 稳定梯度，模拟 Batch Size = 16
    "max_new_tokens": 256, #  提升长度: 允许更完整的思考 (128->256)
    "max_steps": 160,      #  覆盖率: 160*4 = 640 样本 ≈ 2 Epochs
    "lr": 1e-6,            #  降低LR: 跑得久一点，学得细一点
    "lora_r": 32,
    
    # --- DCDA 参数 ---
    "eps_low": 0.1,        
    "eps_high": 0.3,       
    "min_reward_gap": 0.005, #  进一步降低阈值，捕捉细微进步
    "max_resample": 3,     
}

wandb.init(project="search-r1-dcda", name="dcda_v3_full_dataset", config=config)

print(" DCDA V3 - Full Dataset Finetuning")
print(f"   Target: ~2 Epochs | Accum: {config['accum_steps']} | Temp Max: 1.4")

# ================= 2. 模型加载 =================
print("⏳ Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16, 
    device_map="auto", local_files_only=True
)
model.config.use_cache = False 
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=config["lora_r"], lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
optimizer = AdamW(model.parameters(), lr=config["lr"])

# ================= 3. 辅助函数 (Reward 升级) =================
def dap_reward(response):
    """Heuristic Reward Function V3 (更精细的指导)"""
    score = 0.5
    
    # 1. 基础思考奖励
    if "<think>" in response: score += 0.15
    if "</think>" in response: score += 0.05 #  鼓励闭合标签
    
    # 2. 检索工具奖励 (加权，引导突破 0.7)
    # 关键词扩充
    search_keywords = ['search', 'query', 'grep', 'find', 'ls', 'cat', '查询', '检索']
    if any(kw in response.lower() for kw in search_keywords): 
        score += 0.25 
    
    # 3. 思考深度奖励 (长度)
    # 鼓励写长一点，不要太敷衍
    think_len = 0
    if "<think>" in response and "</think>" in response:
        try:
            think_content = response.split("<think>")[1].split("</think>")[0]
            think_len = len(think_content)
        except: pass
    
    if think_len > 50: score += 0.05
    if think_len > 150: score += 0.05 # 越长分越高(有上限)
    
    # 4. 步骤奖励
    steps = min(response.count('第'), 5) * 0.04
    score += steps
    
    # 噪声与数值稳定
    noise = random.uniform(-0.005, 0.005)
    
    return max(min(score + noise, 1.0), -1.0)

def get_batch_log_probs(model, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

# ================= 4. 数据加载 (全量) =================
prompts = []
data_path = "/root/autodl-tmp/repor1/data/teacher_cn_multistep_longthink.jsonl"
try:
    with open(data_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompts.append(data['conversations'][0]['content'])
            except: continue
    print(f" Loaded ALL {len(prompts)} prompts from dataset")
except:
    print(" Dataset load failed, using dummy.")
    prompts = ["如何使用Python?", "介绍Transformer"]

# ================= 5. DCDA  训练循环 =================
global_step = 0
while global_step < config["max_steps"]:
    step_loss = 0
    step_rewards = []
    skipped_groups = 0
    valid_accumulation = 0
    
    #  梯度累积循环
    for _ in range(config["accum_steps"]):
        prompt_text = random.choice(prompts) # 随机采样
        
        inputs = tokenizer(
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n", 
            return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")
        prompt_len = inputs.input_ids.shape[1]
        
        group_data = [] 
        
        # --- Phase 1: 采样 (高探索) ---
        model.eval()
        model.config.use_cache = True
        
        batch_samples = []
        batch_rewards = []
        
        #  激进的温度设置: 强迫探索
        temperatures = [0.7, 1.0, 1.2, 1.4] 
        
        try:
            for temp in temperatures:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=config["max_new_tokens"],
                        temperature=temp, 
                        top_p=0.95, 
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    full_ids = outputs[0]
                    resp_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
                    r = dap_reward(resp_text)
                    batch_samples.append(full_ids)
                    batch_rewards.append(r)
        except:
            continue # 防止生成报错
        
        model.config.use_cache = False
        model.train()

        # --- Phase 2: Dynamic Sampling ---
        reward_gap = max(batch_rewards) - min(batch_rewards)
        
        # 前10步 Warmup，或者 Gap 足够大
        if global_step > 10 and reward_gap < config["min_reward_gap"]:
            del inputs, batch_samples
            skipped_groups += 1
            torch.cuda.empty_cache()
            continue 
            
        group_data = list(zip(batch_samples, batch_rewards))
        step_rewards.extend(batch_rewards)
        valid_accumulation += 1

        # --- Phase 3 & 4: Advantage & Update ---
        rewards = [r for _, r in group_data]
        mean_r = statistics.mean(rewards)
        std_r = statistics.stdev(rewards) + 1e-6
        advantages = [(r - mean_r) / std_r for r in rewards]

        batch_policy_loss = 0
        # 这里不 zero_grad，因为我们要累积梯度
        
        for (seq_ids, _), adv in zip(group_data, advantages):
            seq_ids = seq_ids.unsqueeze(0).to("cuda")
            new_log_probs = get_batch_log_probs(model, seq_ids, torch.ones_like(seq_ids))
            old_log_probs = new_log_probs.detach()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            loss_mask = torch.zeros_like(new_log_probs)
            loss_mask[:, prompt_len-1:] = 1.0 
            adv_t = torch.tensor(adv, device="cuda")
            
            # Decoupled Clip
            epsilon = torch.where(
                adv_t > 0, 
                torch.tensor(config["eps_high"], device="cuda"), 
                torch.tensor(config["eps_low"], device="cuda")
            )
            clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
            surr1 = ratio * adv_t
            surr2 = clipped_ratio * adv_t
            
            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            
            #  Loss 除以 accum_steps
            loss = (policy_loss / config["group_size"]) / config["accum_steps"]
            loss.backward()
            batch_policy_loss += policy_loss.item()
        
        step_loss += batch_policy_loss / len(group_data)
        
        del inputs, group_data, new_log_probs
        gc.collect(); torch.cuda.empty_cache()

    # --- Optimizer Step (Accumulation 结束) ---
    if valid_accumulation > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad() # Step 完再清空
        
        global_step += 1
        
        # 计算平均 Loss 和 Reward
        avg_loss = step_loss / valid_accumulation
        avg_reward = statistics.mean(step_rewards) if step_rewards else 0
        
        print(f"Step {global_step:3d} | Loss={avg_loss:.6f} | Reward={avg_reward:.3f} | Skip={skipped_groups}/{config['accum_steps']}")
        
        log_dict = {
            "train/loss": avg_loss, 
            "train/reward": avg_reward, 
            "train/step": global_step
        }
        
        if global_step % 10 == 0 and step_rewards:
            try:
                best_resp = tokenizer.decode(batch_samples[0][prompt_len:], skip_special_tokens=True)
                clean_resp = best_resp[:100].replace('\n', ' ')
                print(f" Sample: {clean_resp}...")
            except: pass
            wandb.log(log_dict)
    else:
        # 如果这一轮 Accumulation 全被 Skip 了
        print(f"Step {global_step:3d} | All Skipped in Batch (Looking for better prompts...)")
        # 防止死循环，强制换一批
        pass 

# ================= 6. 保存 =================
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
wandb.finish()
print(f"\n DCDA V3 训练结束! 模型已保存至: {OUTPUT_PATH}")
