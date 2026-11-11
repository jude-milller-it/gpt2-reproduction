import os
import time
import math
import random
import csv
import argparse
import yaml
from contextlib import nullcontext

import tiktoken
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.amp import GradScaler

from model import GPTConfig, GPT
from pretrain.dataloader import DataLoaderLite
from pretrain.hellaswag import iterate_examples, render_example

# HellaSwag评测核心函数
# 计算各候选答案的损失，返回损失最小的候选索引（即模型预测结果）
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# 默认配置（可被yaml文件覆盖）
default_config = {
    "out_dir": "out",
    "log_dir": "log",
    "ckpt_path": None,
    "eval_interval": 250,
    "val_loss_steps": 20,
    "save_freq": 5000,
    "init_from": "scratch",
    "wandb_log": False,
    "wandb_project": "fineweb",
    "wandb_run_name": "gpt2",
    "total_batch_size": 524288,
    "batch_size": 32,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "bias": False,
    "learning_rate": 6e-4,
    "max_steps": 19073,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    "min_lr": None,
    "warmup_steps": 715,
    "lr_decay_steps": 19073,
    "backend": "nccl",
    "dtype": 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    "use_compile": False,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    return parser.parse_args()

# 加载配置（默认配置 + yaml覆盖）
def load_config(args):
    cfg = dict(default_config)
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            loaded_all = yaml.safe_load(f) or {}
            loaded_cfg = loaded_all.get("pretrain", loaded_all)
        cfg.update(loaded_cfg)
    if cfg["min_lr"] is None:
        cfg["min_lr"] = cfg["learning_rate"] * 0.1
    return cfg

# 初始化分布式训练环境
def setup_ddp(backend):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP训练需要CUDA环境"
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"使用设备: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset, device

# 写入训练日志（CSV格式）
def write_log_row(log_file, step, stream, val, lr):
    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step, stream, val, lr])

# 初始化日志文件（新建/追加）
def init_log_file(log_dir, init_from):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.csv")
    log_mode = "a" if init_from == "resume" else "w"
    with open(log_file, log_mode, newline='') as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["step", "stream", "value", "lr"])
    return log_file

def main():
    args = parse_args()
    cfg = load_config(args)

    # 初始化分布式训练环境
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset, device = setup_ddp(cfg["backend"])

    # 解析配置参数
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, cfg["out_dir"])
    log_dir = os.path.join(base_dir, cfg["log_dir"])
    ckpt_path = cfg["ckpt_path"]
    eval_interval = cfg["eval_interval"]
    val_loss_steps = cfg["val_loss_steps"]
    save_freq = cfg["save_freq"]
    init_from = cfg["init_from"]
    wandb_log = cfg["wandb_log"]
    wandb_project = cfg["wandb_project"]
    wandb_run_name = cfg["wandb_run_name"]
    total_batch_size = cfg["total_batch_size"]
    batch_size = cfg["batch_size"]
    block_size = cfg["block_size"]
    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    dropout = cfg["dropout"]
    bias = cfg["bias"]
    learning_rate = cfg["learning_rate"]
    max_steps = cfg["max_steps"]
    weight_decay = cfg["weight_decay"]
    beta1 = cfg["beta1"]
    beta2 = cfg["beta2"]
    eps = cfg["eps"]
    grad_clip = cfg["grad_clip"]
    min_lr = cfg["min_lr"]
    warmup_steps = cfg["warmup_steps"]
    lr_decay_steps = cfg["lr_decay_steps"]
    dtype = cfg["dtype"]
    use_compile = cfg["use_compile"]

    # 计算梯度累积步数（实现大批次训练）
    assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (batch_size * block_size * ddp_world_size)
    if master_process:
        print(f"目标总批次大小: {total_batch_size}")
        print(f"=> 梯度累积步数: {grad_accum_steps}")

    # 学习率调度函数（预热+余弦衰减）
    def get_lr(it):
        if it < warmup_steps:
            return learning_rate * (it+1) / warmup_steps
        if it > lr_decay_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # 配置信息整理（用于日志记录）
    config = dict(cfg)
    config["grad_accum_steps"] = grad_accum_steps

    # 初始化随机种子
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    # 初始化训练/验证数据加载器
    cur_step = 0
    best_val_loss = 1e9
    train_loader = DataLoaderLite(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="train")
    val_loader = DataLoaderLite(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="val")

    torch.set_float32_matmul_precision('high')

    # 模型初始化
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout)

    if init_from == 'scratch':
        print("从头初始化模型")
        if model_args['vocab_size'] is None:
            model_args['vocab_size'] = 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        # 从 checkpoint 恢复训练
        print(f"从{out_dir}恢复训练")
        ckpt_name = cfg.get("ckpt_path")
        if not ckpt_name:
            raise FileNotFoundError("resume模式需要指定ckpt_path")
        ckpt_path = os.path.join(out_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # 清理torch.compile可能添加的前缀
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        cur_step = checkpoint['cur_step'] + 1
        best_val_loss = checkpoint['best_val_loss']
        # 恢复随机数状态
        rng_state = checkpoint['rng_state']
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['cuda'])
    elif init_from.startswith('gpt2'):
        # 加载GPT-2预训练权重
        print(f"加载OpenAI GPT-2权重: {init_from}")
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

    model = model.to(device)

    # 模型编译/分布式包装
    if use_compile:
        print("编译模型... (约1分钟)")
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # 初始化混合精度缩放器
    scaler = GradScaler(device=device, enabled=(dtype == 'float16'))

    # 配置优化器
    optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), eps, device)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    # 初始化日志文件
    log_file = init_log_file(log_dir, init_from)

    # 初始化wandb日志（可选）
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # 训练主循环
    for step in range(cur_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # 设置当前步学习率
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 验证损失评估
        if step % eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with ctx:
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"验证损失: {val_loss_accum.item():.4f}")
                write_log_row(log_file, step, "val", round(val_loss_accum.item(), 6), lr)
                if wandb_log:
                    wandb.log({
                        "cur_step": step,
                        "val/loss": val_loss_accum.item(),
                        "lr": lr,
                    })
                # 保存checkpoint
                if step > 0 and (step % save_freq == 0 or last_step):
                    if val_loss_accum.item() < best_val_loss:
                        best_val_loss = val_loss_accum.item()
                    checkpoint_path = os.path.join(out_dir, f"ckpt_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'config': raw_model.config,
                        'cur_step': step,
                        'best_val_loss': best_val_loss,
                        'rng_state': {
                            'python': random.getstate(),
                            'numpy': np.random.get_state(),
                            'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        }
                    }
                    print(f"保存checkpoint到 {out_dir}")
                    torch.save(checkpoint, checkpoint_path)
        
        # HellaSwag评测
        if (step % eval_interval == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with ctx:
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # 分布式汇总评测结果
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag准确率: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                write_log_row(log_file, step, "hella", round(acc_norm, 6), lr)

        # 训练步骤
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with ctx:
                logits, loss = model(x, y)
            # 梯度累积时缩放损失（保证均值正确）
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            scaler.scale(loss).backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # 梯度裁剪
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0.0 else float('Inf'))
        # 优化器步进
        scaler.step(optimizer)
        scaler.update()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():>9.6f} | lr {lr:.4e} | norm: {norm:>7.4f} | dt: {dt*1000:>8.2f}ms | tok/sec: {tokens_per_sec:>8.2f}")
            write_log_row(log_file, step, "train", round(loss_accum.item(), 6), lr)
            if wandb_log:
                wandb.log({
                    "cur_step": step,
                    "train/loss": loss_accum.item(),
                    "lr": lr,
                })

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()