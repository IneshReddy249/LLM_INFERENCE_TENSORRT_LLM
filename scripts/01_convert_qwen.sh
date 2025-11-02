#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="" python3 /workspace/trtllm-src/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /workspace/llm-trtllm/hf_models/qwen2.5-7b-instruct \
  --output_dir /workspace/llm-trtllm/checkpoints/qwen2.5-7b \
  --dtype float16 \
  --tp_size 1
