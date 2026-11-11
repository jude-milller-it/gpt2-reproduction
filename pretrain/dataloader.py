import os
import numpy as np
import torch
from numpy.random import default_rng


def load_tokens(filename):
    # 加载单个数据分片的token文件
    # 将numpy数组转换为torch长整型张量，适配模型输入格式
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


# 轻量级数据加载器
# 适配分布式训练，支持多进程分片读取、数据打乱和连续批次生成
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, master_process, split):
        # 初始化基础参数
        self.B = B  # 单进程批次大小
        self.T = T  # 序列长度
        self.process_rank = process_rank  # 当前进程编号（分布式训练）
        self.num_processes = num_processes  # 总进程数
        # 每个进程使用独立的随机数生成器，避免数据读取重复
        self.rng = default_rng(1337 + process_rank)
        # 校验数据集划分类型（训练/验证）
        assert split in {'train', 'val'}

        # 定位数据分片文件路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(base_dir, "edu_fineweb10B")
        shards = os.listdir(data_root)
        # 筛选对应划分的分片文件（train/val）
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        # 校验分片文件是否存在
        assert len(shards) > 0, f"未找到{split}划分对应的数据集分片"
        if master_process:
            print(f"找到{len(shards)}个{split}数据集分片")
        # 初始化数据读取状态
        self.reset()

    def reset(self):
        # 重置数据加载状态，打乱分片顺序（保证每个epoch数据顺序不同）
        self.shard_indices = list(range(len(self.shards)))
        self.rng.shuffle(self.shard_indices)
        self.current_shard_idx = 0
        # 加载第一个分片
        self._load_current_shard()

    def _load_current_shard(self):
        # 加载当前分片的token数据
        shard_path = self.shards[self.shard_indices[self.current_shard_idx]]
        self.tokens = load_tokens(shard_path)

        # 计算当前分片的最大起始偏移量，避免越界
        # stride为所有进程单次读取的总token数（B*T*进程数）
        stride = self.B * self.T * self.num_processes
        max_start = max(0, len(self.tokens) - (stride + 1))
        # 随机生成起始偏移量（保证每个进程读取不同的数据段）
        offset = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0

        # 对齐到当前进程的读取起始位置
        base = (offset // stride) * stride
        self.current_position = base + self.B * self.T * self.process_rank

    def next_batch(self):
        # 获取下一个训练批次（x为输入，y为目标，因果语言模型中y是x右移一位）
        B, T = self.B, self.T
        # 读取连续的B*T+1个token（+1是为了拆分输入和目标）
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # 输入序列（去掉最后一个token）
        y = (buf[1:]).view(B, T)  # 目标序列（去掉第一个token）

        # 推进当前读取位置（跨进程步长）
        self.current_position += B * T * self.num_processes

        # 检查是否超出当前分片范围，若超出则加载下一个分片
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard_idx += 1
            # 所有分片读取完毕后重新打乱分片顺序
            if self.current_shard_idx >= len(self.shards):
                self.rng.shuffle(self.shard_indices)
                self.current_shard_idx = 0
            # 加载新的分片
            self._load_current_shard()

        return x, y