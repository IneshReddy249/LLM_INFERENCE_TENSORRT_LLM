#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="" python3 -m tensorrt_llm.commands.build \
  --checkpoint_dir checkpoints/qwen2.5-7b \
  --output_dir engine/qwen2.5-7b-a100-fp16-lat \
  --max_batch_size 1 \
  --max_input_len 4096 \
  --max_seq_len 6144 \
  --context_fmha enable \
  --kv_cache_type paged \
  --gpt_attention_plugin float16
