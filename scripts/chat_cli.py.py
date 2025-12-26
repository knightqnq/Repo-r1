#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 配置路径 =================
BASE_MODEL_PATH = "/root/autodl-tmp/repor1/checkpoints/sft_gold_mpl_small_v3"
LORA_PATH = "/root/autodl-tmp/repor1/checkpoints/dcda_sota_final_v3"

# ================= 终端颜色 =================
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def main():
    print(f"{C.HEADER}{C.BOLD}CJ-R1 终端对话模式启动中...{C.RESET}")
    print(f"{C.HEADER}加载 Tokenizer...{C.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print(f"{C.HEADER}加载基础模型: {BASE_MODEL_PATH}{C.RESET}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"{C.HEADER}加载 CJ-R1 LoRA 适配器: {LORA_PATH}{C.RESET}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{C.GREEN}模型加载完成，使用设备: {device}{C.RESET}\n")
    print(f"{C.BOLD}输入问题开始对话，输入 'exit' 或 'quit' 退出。{C.RESET}\n")

    while True:
        try:
            print(f"{C.YELLOW}{'-' * 60}{C.RESET}")
            query = input(f"{C.YELLOW}User > {C.RESET}").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("退出 CJ-R1 对话。")
                break

            # 构造 ChatML 风格 prompt
            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            print(f"{C.BLUE}模型思考中，请稍候...{C.RESET}")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            resp = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # 解析 <think> 标签
            thought_text = ""
            answer_text = ""

            if "<think>" in resp and "</think>" in resp:
                before, after = resp.split("</think>", 1)
                thought_text = before.replace("<think>", "").strip()
                answer_text = after.strip()
            elif "<think>" in resp and "</think>" not in resp:
                # 未闭合，全部视为思考过程
                thought_text = resp.replace("<think>", "").strip()
                answer_text = "[提示] 思考过程过长或被截断，未生成明确结论。"
            else:
                # 没有思考标签，全部视为回答
                thought_text = "[提示] 未检测到显式思考过程。"
                answer_text = resp.strip()

            # 打印结果
            print(f"{C.BLUE}\n【思考过程】{C.RESET}")
            print(f"{C.BLUE}{thought_text}{C.RESET}\n")
            print(f"{C.GREEN}【解决方案】{C.RESET}")
            print(f"{C.GREEN}{answer_text}{C.RESET}\n")

        except KeyboardInterrupt:
            print("\n检测到中断，退出 CJ-R1 对话。")
            break
        except Exception as e:
            print(f"{C.RED}发生错误: {e}{C.RESET}")


if __name__ == "__main__":
    main()
