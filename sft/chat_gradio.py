import argparse
import os

import torch
import gradio as gr
import tiktoken

from model import GPTConfig, GPT

# 指令-回复模板（与SFT训练数据格式保持一致）
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# 加载SFT微调后的模型权重
def load_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重文件: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # 推理阶段切换为评估模式
    return model.to(device)

# 对话机器人类（封装模型推理逻辑）
class ChatBot:
    def __init__(self, model, enc, device, max_new_tokens=256, temperature=0.8):
        self.model = model
        self.enc = enc
        self.device = device
        self.max_new_tokens = max_new_tokens  # 最大生成token数
        self.temperature = temperature        # 生成随机性控制

    def reply(self, instruction, history):
        # 拼接指令模板
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        # 编码提示文本为token
        prompt_ids = self.enc.encode(prompt)
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)[None, ...]
        
        # 模型生成回复（禁用梯度计算提升推理速度）
        with torch.no_grad():
            y = self.model.generate(
                idx,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=None,
            )
        
        # 解析生成结果（截断结束符前的内容）
        full = y[0].tolist()
        gen_ids = full[len(prompt_ids):]
        eos_id = self.enc._special_tokens["<|endoftext|>"]
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        response = self.enc.decode(gen_ids).strip()
        
        # 更新对话历史
        history = history + [(instruction, response)]
        return history, history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SFT模型权重文件路径 (*.pt)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--share", action="store_true", help="开启Gradio公网访问链接")
    args = ap.parse_args()

    # 自动选择运行设备（优先CUDA）
    device = 'cpu' 
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    # 初始化分词器和模型
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.ckpt, device)
    bot = ChatBot(model, enc, device, args.max_new_tokens, args.temperature)

    # 构建Gradio Web界面
    with gr.Blocks() as demo:
        gr.Markdown("# SFT Chat Demo")
        chatbot = gr.Chatbot()          # 对话历史展示组件
        instr = gr.Textbox(label="Instruction", placeholder="输入指令内容")  # 指令输入框
        send = gr.Button("Send")        # 发送按钮

        # 绑定提交事件（按钮点击/回车提交）
        def on_submit(message, history):
            return bot.reply(message, history or [])

        send.click(on_submit, inputs=[instr, chatbot], outputs=[chatbot, chatbot])
        instr.submit(on_submit, inputs=[instr, chatbot], outputs=[chatbot, chatbot])

    # 启动Gradio服务（share=True生成公网链接）
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()