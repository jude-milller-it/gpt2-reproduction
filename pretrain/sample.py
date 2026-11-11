import os
import torch
import tiktoken
from model import GPTConfig, GPT
from contextlib import nullcontext

# 生成参数配置
init_from = 'resume'
out_dir = 'out'       # 本地权重目录
ckpt = 'ckpt_75000.pt' # 权重文件名
# 生成起始文本
start = "What is overfitting in machine learning, and how can we prevent it?"
num_return_sequences = 10  # 生成序列数量
max_new_tokens = 500       # 每个序列生成的最大token数
temperature = 0.8          # 随机性控制：<1更确定，>1更随机
top_k = 200                # Top-K采样候选集大小
seed = 1337
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
# 自动选择数据类型（优先bfloat16，兼容CUDA）
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
use_compile = False        # 是否启用torch.compile加速

# 初始化随机种子
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
# 开启TF32加速（提升CUDA矩阵运算速度）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 混合精度上下文（CPU不启用）
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# 加载模型
if init_from == 'resume':
    # 加载本地训练的权重文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, out_dir)
    ckpt_path = os.path.join(out_dir, ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # 恢复模型配置并初始化
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 清理torch.compile可能添加的前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 加载HuggingFace预训练GPT-2权重
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# 模型设为评估模式并移至指定设备
model.eval()
model.to(device)
if use_compile:
    model = torch.compile(model)

# 初始化GPT-2分词器
enc = tiktoken.get_encoding("gpt2")

# 编码起始文本
if start.startswith('FILE:'):
    # 从文件读取起始文本
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
tokens = enc.encode(start)
x = (torch.tensor(tokens, dtype=torch.long, device=device)[None, ...])

# 生成文本
with torch.no_grad():
    with ctx:
        for k in range(num_return_sequences):
            xgen = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(xgen[0].tolist()))
            print('----------------------------------')