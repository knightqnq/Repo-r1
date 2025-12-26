# /root/autodl-tmp/repor1/scripts/sft_train_dora_flashattn.py

import os
import sys
import yaml
from pathlib import Path

os.environ["TORCH_SDPA_FLASH_ATTENTION"] = "true"
os.environ["FLASH_ATTENTION_FORCE_ENABLE"] = "1"

VERL_PATH = "/scratch-shared/tc1proj043/verl"
if VERL_PATH not in sys.path:
    sys.path.insert(0, VERL_PATH)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

CONFIG_PATH = "/root/autodl-tmp/repor1/config/sft_config_dora_flashattn_stage3.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print("=" * 80)
print(" DoRA SFT 训练 (Flash Attention 2)")
print("=" * 80)
print(f"DoRA rank: {config['peft']['r']}")
print("=" * 80 + "\n")


class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    conv = item.get("conversations", [])
                    if len(conv) >= 2:
                        self.data.append(conv)
                except:
                    continue

        print(f" 加载 {len(self.data)} 条 conversations 样本\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conv = self.data[idx]
        user = conv[0]["content"]
        assistant = conv[1]["content"]

        text = (
            "<|im_start|>user\n"
            + user
            + "<|im_end|>\n<|im_start|>assistant\n"
            + assistant
            + "<|im_end|>"
        )

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attn_mask = encodings["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": input_ids.clone(),
        }


def train():
    print(" 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(" 加载模型 (Flash Attention 2)...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"],
        trust_remote_code=config["model"]["trust_remote_code"],
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    print(" 模型加载完成\n")

    print(" 配置 DoRA...")
    peft_conf = config["peft"]
    lora_config = LoraConfig(
        task_type=getattr(TaskType, peft_conf["task_type"]),
        r=peft_conf["r"],
        lora_alpha=peft_conf["lora_alpha"],
        lora_dropout=peft_conf["lora_dropout"],
        target_modules=peft_conf["target_modules"],
        bias=peft_conf["bias"],
        use_dora=peft_conf["use_dora"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(" DoRA 配置完成\n")

    train_file = config["data"]["train_files"]
    dataset = SFTDataset(
        train_file,
        tokenizer,
        config["data"]["max_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    from transformers import get_cosine_schedule_with_warmup

    total_steps = (
        len(dataloader)
        * config["training"]["num_train_epochs"]
        // config["training"]["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Flash Attention 训练开始")
    print("=" * 80)

    model.train()
    global_step = 0
    best_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config["training"]["num_train_epochs"]):
        print(f"\n Epoch {epoch + 1}/{config['training']['num_train_epochs']}")
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            with torch.amp.autocast("cuda"):
                batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / config["training"]["gradient_accumulation_steps"]

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * config["training"]["gradient_accumulation_steps"]

            if (step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config["training"]["logging_steps"] == 0:
                    avg_loss = epoch_loss / (step + 1)
                    pbar.set_postfix(
                        {"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
                    )
                    if avg_loss < best_loss:
                        best_loss = avg_loss

                if global_step % config["training"]["save_steps"] == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n 训练完成！最佳 loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
