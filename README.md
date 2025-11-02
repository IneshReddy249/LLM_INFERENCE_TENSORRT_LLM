# ⚡ LLM Inference Optimization — Qwen2.5-7B + TensorRT-LLM (A100)

A complete, reproducible benchmark pipeline comparing **baseline Hugging Face inference** vs **TensorRT-LLM optimized inference** on NVIDIA A100 GPUs.  
This project demonstrates how to build TensorRT engines, measure inference efficiency, and analyze metrics such as **latency**, **TTFT**, **throughput (TPS)**, and **GPU utilization**.

---

## 🚀 Overview

**Goal:**  
Reduce inference latency and increase throughput for large-language models (LLMs) using **TensorRT-LLM** optimizations such as FP16 precision, paged KV-cache, inflight batching, and context FMHA.

**Model:**  
[`Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

**Hardware:**  
NVIDIA A100 80 GB GPU • CUDA 12.9 • TensorRT-LLM v0.20.0

---

## 🧱 Folder Structure

```text
/workspace/llm-trtllm/
├── hf_models/                     # Hugging Face raw model
│   └── qwen2.5-7b-instruct/
├── checkpoints/                   # TRT-LLM converted weights
│   └── qwen2.5-7b/
├── engine/                        # TensorRT serialized engines
│   ├── qwen2.5-7b-a100-fp16-lat/
│   └── qwen2.5-7b-a100-fp16-inflight/
├── results/                       # Benchmark JSON outputs
├── scripts/                       # Conversion / build / benchmark scripts
│   ├── 01_convert_qwen.sh
│   ├── 02_build_fp16_lat_a100.sh
│   ├── 03_build_fp16_inflight_a100.sh
│   └── benchmark.py
└── .gitignore


## ⚙️ Environment Setup
1️⃣ Start Container

docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/shadeform/projects/llm-trtllm:/workspace/llm-trtllm \
  nvcr.io/nvidia/tensorrt-llm/release:latest

2️⃣ Install Dependencies

cd /workspace/llm-trtllm
pip install --upgrade huggingface_hub transformers accelerate

📥 Download Model Weights

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="hf_models/qwen2.5-7b-instruct",
    local_dir_use_symlinks=False
)

🔄 Convert HF → TRT-LLM Checkpoints

bash scripts/01_convert_qwen.sh
✅ Outputs to /workspace/llm-trtllm/checkpoints/qwen2.5-7b/

⚡ Build Optimized Engines (A100)
Latency-optimized (batch = 1)
bash scripts/02_build_fp16_lat_a100.sh

Inflight-batching (batch = 8)
bash scripts/03_build_fp16_inflight_a100.sh

🧠 Benchmark Run
PYTHONPATH="" python3 scripts/benchmark.py

The benchmark measures:

-TTFT (Time to First Token)
-Total Latency
-Throughput (Tokens per Second)
-Output Validation

Results saved to:
results/latency.json

Example Output

{
  "prompt": "Explain paginated KV cache in transformers in simple
English.",
  "tokens": 250,
  "ttft_s": 0.046,
  "latency_s": 1.21,
  "tps": 168.7
}

📊 Key Metrics

| Metric         | Description                   | Tool                  |
| :------------- | :---------------------------- | :-------------------- |
| **TTFT**       | Time to first generated token | Benchmark script      |
| **Latency**    | Total generation time         | Benchmark script      |
| **TPS**        | Tokens per second             | Benchmark script      |
| **VRAM Usage** | GPU memory consumption        | `nvidia-smi`          |
| **GPU Util %** | Compute efficiency            | `nvidia-smi --loop=1` |


🧩 Optimization Summary

| Optimization      | Benefit                           | Verified |
| :---------------- | :-------------------------------- | :------: |
| FP16 precision    | Reduced compute & memory load     |     ✅    |
| Paged KV Cache    | Lower fragmentation, better reuse |     ✅    |
| Context FMHA      | Faster attention kernel           |     ✅    |
| Inflight Batching | Higher throughput                 |     ✅    |
| TensorRT Engine   | Up to 5–6× faster vs baseline     |     ✅    |




