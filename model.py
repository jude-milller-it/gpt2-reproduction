import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# 因果自注意力层（Causal Self-Attention）
# 实现GPT中的核心注意力机制，保证序列生成的因果性（只能关注前文）
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度能被注意力头数整除
        assert config.n_embd % config.n_head == 0
        # 一次性生成query、key、value的投影层（合并计算提升效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 注意力输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # 正则化层
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 保存配置参数
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # 检查是否支持PyTorch2.0+的FlashAttention（高效注意力实现）
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: 使用基础版注意力实现，Flash Attention需要PyTorch >= 2.0")
            # 注册因果掩码，确保每个位置只能关注自身及之前的位置
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # 获取输入维度：B=批次大小，T=序列长度，C=嵌入维度
        B, T, C = x.size()
        # 一次性计算query、key、value并拆分
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 调整维度：(B, T, nh, hs) -> (B, nh, T, hs)，nh=注意力头数，hs=单头维度
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 计算注意力权重并应用
        if self.flash:
            # 使用FlashAttention加速计算，自动处理因果掩码
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 手动实现注意力计算（兼容低版本PyTorch）
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 应用因果掩码，屏蔽未来位置的注意力
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 注意力加权求和
            y = att @ v
        # 拼接所有注意力头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影+残差dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

# 前馈网络层（MLP）
# 采用GPT-2的GELU激活+4倍维度扩张的经典结构
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')  # 近似tanh的GELU，与GPT-2一致
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer基础块
# 采用Pre-LN结构（LayerNorm在注意力/MLP之前），符合GPT-2实现
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 注意力层+残差连接
        x = x + self.attn(self.ln_1(x))
        # MLP层+残差连接
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT模型配置类
# 封装所有模型结构相关的超参数，使用dataclass简化配置管理
@dataclass
class GPTConfig:
    block_size: int = 1024  # 最大序列长度
    vocab_size: int = 50257 # 词汇表大小（GPT-2标准：50k BPE + 256字节 + 1结束符）
    n_layer: int = 12       # Transformer块数量
    n_head: int = 12        # 注意力头数
    n_embd: int = 768       # 嵌入维度
    dropout: float = 0.0    # Dropout概率
    bias: bool = True       # 是否使用偏置（GPT-2默认开启）

# GPT主模型类
# 实现完整的GPT-2架构，包含嵌入层、Transformer块、输出层及相关辅助功能
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 构建Transformer主体结构
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入层
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入层
            drop = nn.Dropout(config.dropout),                     # 嵌入层Dropout
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer块列表
            ln_f = nn.LayerNorm(config.n_embd),                    # 最终LayerNorm
        ))
        # 语言模型输出头（无偏置，与词嵌入共享权重）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：词嵌入与输出层共享权重（GPT-2核心优化）
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化模型参数
        self.apply(self._init_weights)

        # 打印参数量信息
        print("模型参数量: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        # 计算模型总参数量
        # non_embedding=True时，排除位置嵌入（仅计算核心Transformer参数量）
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # 自定义参数初始化策略（符合GPT-2初始化方式）
        if isinstance(module, nn.Linear):
            std = 0.02
            # 对输出投影层做尺度缩放，缓解深度网络梯度问题
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # 前向传播核心逻辑
        # idx: 输入序列，形状(B, T)；targets: 标签序列，用于计算损失
        B, T = idx.size()
        # 确保输入序列长度不超过模型最大支持长度
        assert T <= self.config.block_size, f"输入序列长度{T}超过模型最大长度{self.config.block_size}"

        # 生成位置索引并计算位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        # 计算词嵌入
        tok_emb = self.transformer.wte(idx)
        # 词嵌入+位置嵌入，加Dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # 依次通过所有Transformer块
        for block in self.transformer.h:
            x = block(x)

        # 最终LayerNorm + 输出头
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # 输出形状(B, T, vocab_size)

        # 计算交叉熵损失（如果提供标签）
        loss = None
        if targets is not None:
            # 将logits和targets展平，适配交叉熵计算
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # 从HuggingFace加载预训练的GPT-2权重
        # model_type支持: gpt2/gpt2-medium/gpt2-large/gpt2-xl
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("加载预训练GPT-2权重: %s" % model_type)

        # 根据模型类型确定基础配置
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
        }[model_type]
        # 固定GPT-2的基础配置
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        # 创建空模型并加载权重
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # 过滤掉不需要的bias缓冲区
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # 加载HuggingFace版本模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # 过滤HuggingFace模型中不需要的缓冲区
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # 需要转置的层（HuggingFace的Conv1D与原生Linear维度差异）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 权重映射赋值
        assert len(sd_keys_hf) == len(sd_keys), f"权重数量不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Conv1D转Linear需要转置权重
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device):
        # 配置优化器（AdamW），区分权重衰减策略
        # 仅对2D参数（权重矩阵）应用权重衰减，1D参数（偏置/LN）不衰减
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 分组参数：需要权重衰减 vs 不需要
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        # 打印参数分组信息
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"带权重衰减的参数张量数: {len(decay_params)}, 参数总数: {num_decay_params:,}")
        print(f"无权重衰减的参数张量数: {len(nodecay_params)}, 参数总数: {num_nodecay_params:,}")

        # 使用fused AdamW（CUDA环境下加速）
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        print(f"使用fused AdamW优化器: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)

        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # 文本生成函数
        # idx: 初始输入序列(B, T)；max_new_tokens: 生成的最大新token数
        # temperature: 控制生成随机性；top_k: 限制采样候选集大小
        for _ in range(max_new_tokens):
            # 截断过长序列，保证不超过模型最大长度
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播获取最后一个位置的logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 温度缩放

            # Top-K采样（可选）
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 计算概率分布并采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 将采样结果拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)

        return idx