import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken

# 指令-回复模板（构建SFT训练样本格式）
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}\n"

# Dolly-15k数据集加载与分词类
# 实时处理数据，生成适配模型输入的token序列
class Dolly15KDataset(Dataset):
    def __init__(self, split: str, block_size: int, seed: int = 1337):
        assert split in {"train", "val"}
        # Dolly-15k只有训练集，通过拆分生成验证集
        raw = load_dataset("databricks/databricks-dolly-15k")["train"]
        # 过滤长上下文依赖的类别（避免训练样本过长）
        drop_cats = {"summarization", "information_extraction", "closed_qa"}
        raw = raw.filter(lambda ex: ex.get("category") not in drop_cats)
        # 按95:5拆分训练/验证集
        split_data = raw.train_test_split(test_size=0.05, seed=seed)
        self.ds = split_data["train"] if split == "train" else split_data["test"]
        # 初始化GPT-2分词器
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]  # 结束符token
        self.block_size = block_size  # 序列最大长度

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # 拼接指令+回复文本
        text = PROMPT_TEMPLATE.format(
            instruction=item["instruction"],
            response=item["response"],
        )
        # 添加上下文分隔符，让模型学习结束生成
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(text))
        tokens.append(self.eot)
        # 截断到block_size+1（预留移位空间）
        tokens = tokens[: self.block_size + 1]
        return {"tokens": tokens}

# 批处理数据填充与移位（构建输入x和目标y）
def _pad_and_shift(batch, pad_id, block_size, device):
    # 取批次内最长样本长度（不超过block_size）
    max_len = min(block_size, max(len(x["tokens"]) for x in batch))
    B = len(batch)
    # 初始化填充后的输入和目标张量
    x = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    for i, sample in enumerate(batch):
        toks = sample["tokens"][:max_len]
        x[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
        # 目标序列是输入序列右移一位（因果语言模型）
        if len(toks) > 1:
            y[i, : len(toks) - 1] = x[i, 1:len(toks)]
    return x, y

# 构建训练/验证数据加载器
def build_dataloaders(batch_size, block_size, device, seed=1337):
    loaders = {}
    enc = tiktoken.get_encoding("gpt2")
    pad_id = enc._special_tokens["<|endoftext|>"]  # 用结束符作为填充id
    for split in ["train", "val"]:
        ds = Dolly15KDataset(split=split, block_size=block_size, seed=seed)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),  # 训练集打乱，验证集不打乱
            collate_fn=lambda b, p=pad_id: _pad_and_shift(b, p, block_size, device),
            drop_last=True,  # 丢弃最后不完整批次
        )
    return loaders