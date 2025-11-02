#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="" python3 -m tensorrt_llm.commands.build \
  --checkpoint_dir checkpoints/qwen2.5-7b \
  --output_dir engine/qwen2.5-7b-a100-fp16-inflight \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_seq_len 6144 \
  --kv_cache_type paged \
  --context_fmha enable \
  --remove_input_padding enable
