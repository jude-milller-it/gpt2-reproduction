import argparse
import os

import torch

from model import GPTConfig, GPT
import tiktoken

# 指令-回复模板（匹配SFT训练数据格式）
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# 加载SFT训练后的模型权重
def load_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到checkpoint文件: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # 推理阶段设为评估模式
    return model.to(device)

# 生成回复文本
def generate(model: GPT, enc, prompt: str, device: str, max_new_tokens: int = 256, temperature: float = 0.8):
    # 编码提示文本
    prompt_ids = enc.encode(prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
    # 生成回复（禁用梯度计算加速推理）
    with torch.no_grad():
        y = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=None)
    # 解析生成结果（截断结束符前的内容）
    full = y[0].tolist()
    gen_ids = full[len(prompt_ids):]
    eos_id = enc._special_tokens["<|endoftext|>"]
    if eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos_id)]
    out = enc.decode(gen_ids)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SFT模型权重路径 (*.pt)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    # 自动选择运行设备
    device = 'cpu' 
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    # 初始化GPT-2分词器
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.ckpt, device)

    print(f"已加载模型 {args.ckpt} (设备: {device})。输入'exit'退出对话。")
    # 交互式对话循环
    while True:
        try:
            instr = input("Instruction> ").strip()
        except EOFError:
            break
        if instr.lower() in {"exit", "quit"}:
            break
        # 拼接指令模板
        prompt = PROMPT_TEMPLATE.format(instruction=instr)
        # 生成回复并打印
        resp = generate(model, enc, prompt, device, args.max_new_tokens, args.temperature)
        print("Response> ", resp.strip())

if __name__ == "__main__":
    main()