import gradio as gr
import torch
import re
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

# ================= 路径配置 =================
BASE_MODEL_PATH = "/root/autodl-tmp/repor1/checkpoints/sft_gold_mpl_small_v3"
LORA_PATH = "/root/autodl-tmp/repor1/checkpoints/dcda_sota_final_v3"

# ================= 加载模型 =================
print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print(f"正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"正在加载 CJ-R1 Adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("模型加载完成")

# ================= 核心逻辑 =================
def generate_cj_r1(query, temperature, top_p, max_tokens):
    if not query:
        yield "请输入内容...", "请输入内容..."
        return

    # 构造 Prompt
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_text = ""
    thought_text = ""
    answer_text = ""
    
    # 初始状态
    yield "思考中...", "等待生成..."

    for new_text in streamer:
        full_text += new_text
        
        if "</think>" in full_text:
            parts = full_text.split("</think>")
            # 1. 思考部分
            thought_text = parts[0].replace("<think>", "").strip()
            # 2. 回答部分
            if len(parts) > 1:
                answer_text = parts[1].strip()
            else:
                answer_text = "生成回答中..."
        else:
            # 还没闭合，全部视为思考
            thought_text = full_text.replace("<think>", "").strip()
            answer_text = "等待思考结束..."

        yield thought_text, answer_text

    # --- 兜底逻辑 ---
    if "</think>" not in full_text:
        # 如果到最后都没闭合，尝试智能截断
        if len(full_text) > 200:
            split_idx = int(len(full_text) * 0.8)
            thought_text = full_text[:split_idx].replace("<think>", "") + "\n\n[未闭合，已自动截断]"
            answer_text = full_text[split_idx:]
        else:
            answer_text = "[未生成有效结论，请尝试增大 Max Tokens]"
            
    yield thought_text, answer_text

# ================= 界面样式 =================
custom_css = """
.gradio-container { background-color: #1a1a1a !important; }

/* 思考过程：黑底蓝字 */
#thought_box {
    background-color: #000000 !important;
    border: 1px solid #444 !important;
    min-height: 400px;
}
#thought_box .prose {
    color: #4da6ff !important; /* 柔和蓝 */
    font-family: 'Consolas', monospace !important;
    font-size: 14px !important;
}

/* 解决方案：黑底白字 */
#answer_box {
    background-color: #000000 !important;
    border: 1px solid #444 !important;
    min-height: 400px;
}
#answer_box .prose {
    color: #ffffff !important; /* 纯白 */
    font-family: sans-serif !important;
    font-size: 16px !important;
    line-height: 1.6;
}

/* 标题样式 */
h1 { color: #ffffff !important; text-align: center; }
label { color: #cccccc !important; }
"""

# ================= 搭建界面 =================
with gr.Blocks(css=custom_css, title="CJ-R1 Demo") as demo:
    gr.Markdown("# CJ-R1")
    
    with gr.Row():
        with gr.Column(scale=4):
            input_box = gr.Textbox(
                label="输入问题", 
                lines=2, 
                placeholder="在此输入 Bug 描述或技术问题..."
                # 这里去掉了 value 参数，现在是空的
            )
        with gr.Column(scale=1):
            btn = gr.Button("生成", variant="primary", scale=2)
            clear_btn = gr.Button("清除")
    
    with gr.Accordion("参数设置", open=True):
        # 默认 2048 保证不被截断
        token_slider = gr.Slider(512, 4096, 2048, step=128, label="Max Tokens (生成长度)")
        temp_slider = gr.Slider(0.1, 1.5, 0.6, label="Temperature (随机性)")

    gr.Markdown("---")

    with gr.Row():
        # 左栏
        with gr.Column():
            gr.Markdown("### 思考过程")
            thought_output = gr.Markdown(elem_id="thought_box")
        
        # 右栏
        with gr.Column():
            gr.Markdown("### 解决方案")
            answer_output = gr.Markdown(elem_id="answer_box")

    # 事件绑定
    btn.click(generate_cj_r1, [input_box, temp_slider, gr.State(0.9), token_slider], [thought_output, answer_output])
    clear_btn.click(lambda: ("", "", ""), outputs=[input_box, thought_output, answer_output])

if __name__ == "__main__":
    # 为了方便手机访问，我默认开启了 share=True
    # 如果启动太慢，可以手动改回 False
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)
