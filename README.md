# GPT-2 Reproduction and Instruction Tuning

A PyTorch reproduction of GPT-2 small that covers the full path from random initialization to interactive inference. The project has two stages:

- **Pretrain a language model and evaluate its continuations.** Train on FineWeb-Edu and track both next-token loss on held-out web text and the model’s ability to rank plausible endings on HellaSwag.
- **Adapt a pretrained checkpoint to instructions.** Fine-tune the same decoder on formatted Dolly-15k instruction–response pairs and use the resulting checkpoint in an interactive command-line interface.

## Model and training

The [decoder](model.py) has 12 Transformer blocks, 12 attention heads, and a 1,024-token context. It uses causal self-attention and shares weights between its token embeddings and output projection. The [pretraining loop](pretrain/train.py) reads tokenized FineWeb-Edu shards, accumulates gradients across microbatches, and supports distributed training.

For HellaSwag, the evaluator gives the model a shared prompt and four candidate endings. It ranks the endings by **mean negative log-likelihood over their own tokens**. The prompt provides context, and averaging puts endings of different lengths on a per-token scale.

The [instruction-tuning pipeline](sft/train.py) loads a pretrained checkpoint and continues next-token training on instruction–response sequences. The [data loader](sft/dataloader.py) excludes three categories that depend on additional context, then creates a fixed 95/5 train–validation split. The [chat CLI](sft/chat_cli.py) loads the tuned checkpoint and generates a response to an instruction entered at the terminal.

## Results

| Metric                                   | First logged evaluation | Final logged evaluation |
| ---------------------------------------- | ----------------------: | ----------------------: |
| FineWeb-Edu validation loss              |                   11.09 |                **2.90** |
| HellaSwag accuracy                       |                  25.39% |              **33.72%** |
| Filtered Dolly-15k validation perplexity |               45,100.05 |                **4.62** |

The FineWeb-Edu and HellaSwag results come from the same archived pretraining run. HellaSwag accuracy improved by **8.33 percentage points**. The Dolly-15k result measures next-token prediction on held-out instruction–response sequences from the filtered split.

## Run locally

From the repository root, review the batch size and training schedule in [config.yaml](config.yaml), then install dependencies, prepare the FineWeb-Edu shards, and start pretraining.

```bash
python -m pip install -r requirements.txt
python pretrain/fineweb.py
python -m pretrain.train --config config.yaml
```

Set `sft.init_ckpt` in `config.yaml` to the filename of a pretrained checkpoint in `pretrain/out/`. Fine-tune it, then pass the resulting checkpoint to the chat CLI.

```bash
python -m sft.train --config config.yaml
python -m sft.chat_cli --ckpt sft/out/YOUR_CHECKPOINT.pt
```
