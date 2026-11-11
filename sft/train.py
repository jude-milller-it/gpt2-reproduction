import os
import math
import time
import argparse
import yaml
import csv
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.amp import GradScaler

from model import GPTConfig, GPT
from sft.dataloader import build_dataloaders

# SFT默认配置（可通过yaml文件覆盖）
default_config = {
    "out_dir": "out",
    "log_dir": "log",
    "init_ckpt": None,          # 预训练权重路径（SFT必须指定）
    "total_batch_size": 131072, # 每步总token数（约128k）
    "eval_interval": 100,       # 验证间隔步数
    "save_interval": 500,       # 模型保存间隔步数
    "max_steps": 234,           # 总训练步数
    "batch_size": 16,           # 单设备微批次大小
    "block_size": 1024,         # 序列最大长度
    "learning_rate": 1e-5,      # SFT学习率（远低于预训练）
    "warmup_steps": 50,         # 学习率预热步数
    "lr_decay_steps": 234,      # 学习率衰减总步数
    "min_lr": 1e-6,             # 最小学习率
    "weight_decay": 0.0,        # SFT阶段禁用权重衰减
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,           # 梯度裁剪阈值
    "dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML配置文件路径（读取sft段配置）")
    return ap.parse_args()

# 加载配置（默认配置 + yaml覆盖）
def load_config(args):
    cfg = dict(default_config)
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            loaded_all = yaml.safe_load(f) or {}
            pretrain_cfg = loaded_all.get("pretrain", {})
            sft_cfg = loaded_all.get("sft", loaded_all)
        cfg.update(pretrain_cfg)
        cfg.update(sft_cfg)
    return cfg

# 初始化分布式训练环境
def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP训练需要CUDA环境"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ddp, ddp_rank, ddp_world_size, master_process, seed_offset, device

# 学习率调度函数（预热+余弦衰减）
def get_lr(it, cfg):
    warmup_steps = cfg["warmup_steps"]
    lr_decay_steps = cfg["lr_decay_steps"]
    learning_rate = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    if it < warmup_steps:
        return learning_rate * (it + 1) / warmup_steps
    if it > lr_decay_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    args = parse_args()
    cfg = load_config(args)
    ddp, ddp_rank, ddp_world_size, master_process, seed_offset, device = setup_ddp()

    # 初始化输出/日志目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, cfg["out_dir"])
    log_dir = os.path.join(base_dir, cfg["log_dir"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化随机种子和混合精度上下文
    torch.manual_seed(1337 + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

    # 加载预训练权重（SFT的基础模型）
    ckpt_name = cfg.get("init_ckpt")
    if not ckpt_name:
        raise FileNotFoundError("SFT训练必须指定init_ckpt（预训练权重路径）")
    else:
        # 定位预训练权重文件（从pretrain/out目录读取）
        root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        ckpt_path = os.path.join(root_dir, "pretrain", "out", ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到预训练权重: {ckpt_path}")
    
    # 加载权重并初始化模型
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)
    model.load_state_dict(checkpoint["model"])
    raw_model = model

    # 初始化混合精度缩放器（仅float16启用）
    if cfg["dtype"] == "float16":
        scaler = GradScaler(device=device, enabled=True)
    else:
        scaler = GradScaler(device=device, enabled=False)
    
    # 配置优化器（SFT禁用权重衰减）
    optimizer = raw_model.configure_optimizers(
        cfg["weight_decay"], cfg["learning_rate"], (cfg["beta1"], cfg["beta2"]), cfg["eps"], device
    )

    # 分布式包装（多卡训练）
    if ddp:
        model = DDP(model, device_ids=[ddp_rank])
        raw_model = model.module

    # 构建数据加载器（Dolly-15k）
    loaders = build_dataloaders(cfg["batch_size"], cfg["block_size"], device)
    train_loader, val_loader = loaders["train"], loaders["val"]
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # 初始化日志文件
    log_file = os.path.join(log_dir, "sft_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "stream", "loss", "ppl", "lr"])

    # 计算梯度累积步数（实现大批次训练）
    tokens_per_micro = cfg["batch_size"] * cfg["block_size"] * ddp_world_size
    assert cfg["total_batch_size"] % tokens_per_micro == 0, "total_batch_size需被单步token数整除"
    grad_accum_steps = cfg["total_batch_size"] // tokens_per_micro
    if master_process:
        print(f"单微步token数: {tokens_per_micro}, 梯度累积步数: {grad_accum_steps}")

    # 训练主循环
    best_val = 1e9
    for step in range(cfg["max_steps"]):
        t0 = time.time()
        last_step = (step == cfg["max_steps"] - 1)
        # 设置当前步学习率
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 验证损失评估
        if step % cfg["eval_interval"] == 0 or last_step:
            model.eval()
            with torch.no_grad():
                # 读取验证批次（迭代器耗尽则重置）
                try:
                    xb, yb = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    xb, yb = next(val_iter)
                with ctx:
                    val_logits, val_loss = model(xb, yb)
            # 分布式汇总验证损失
            if ddp:
                val_loss_tensor = val_loss.detach().clone()
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss_val = val_loss_tensor.item()
            else:
                val_loss_val = float(val_loss.item())
            # 计算困惑度（PPL）
            val_ppl_val = math.exp(val_loss_val) if val_loss_val < 50 else float("inf")
            
            # 记录验证日志并保存模型
            if master_process:
                with open(log_file, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([step, "val", val_loss_val, val_ppl_val, lr])
                print(f"验证损失: {val_loss_val:.4f}")
                print(f"验证困惑度:  {val_ppl_val:.2f}")
                # 保存最优模型权重
                if step > 0 and (step % cfg["save_interval"] == 0 or last_step):
                    if val_loss_val < best_val:
                        best_val = val_loss_val
                    ckpt = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "config": raw_model.config,
                        "step": step,
                        "best_val": best_val,
                    }
                    torch.save(ckpt, os.path.join(out_dir, f"sft_ckpt_{step:03d}.pt"))

        # 训练步骤
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(grad_accum_steps):
            # 读取训练批次（迭代器耗尽则重置）
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)
            with ctx:
                logits, loss = model(xb, yb)
            # 梯度累积时缩放损失（保证均值正确）
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            # 分布式梯度同步控制（仅最后微步同步）
            if ddp:
                raw_model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
            scaler.scale(loss).backward()
        
        # 分布式汇总训练损失
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # 梯度裁剪 + 优化器步进
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        
        # 计算训练指标
        dt = time.time() - t0
        tok_per_step = cfg["total_batch_size"]
        tok_per_sec = tok_per_step / dt

        # 记录训练日志
        if master_process:
            train_loss_val = float(loss_accum.item())
            train_ppl_val = math.exp(train_loss_val) if train_loss_val < 50 else float("inf")
            with open(log_file, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([step, "train", train_loss_val, train_ppl_val, lr])
            print(f"step {step:3d} | loss: {train_loss_val:8.6f} | ppl {train_ppl_val:.2f} | lr {lr:.4e} | norm: {norm:6.4f} | dt: {dt*1000:8.2f}ms | tok/sec: {tok_per_sec:8.2f}")

    # 销毁分布式进程组
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()