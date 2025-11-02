# ⚡ LLM Inference Optimization — Qwen2.5-7B + TensorRT-LLM (A100)

🚀 **End-to-end benchmark and optimization pipeline** for accelerating large language model inference using **NVIDIA TensorRT-LLM** on **A100 GPUs**.  
Compares **baseline Hugging Face performance** with **TensorRT-optimized engines** for reduced latency, faster throughput, and higher GPU efficiency.

---

## 🧠 Overview

Modern LLMs like **Qwen2.5-7B-Instruct** are powerful but computationally expensive.  
This project optimizes inference using **TensorRT-LLM**, converting standard Hugging Face models into GPU-optimized engines through:
- FP16 precision  
- Paged KV cache  
- Context FMHA  
- Inflight batching  

The result: **up to 6× faster inference** and **4× lower latency** than baseline execution.

---

## ⚙️ System Requirements

| Component | Minimum |
|------------|----------|
| **GPU** | NVIDIA A100 (80 GB) |
| **CUDA Toolkit** | 12.9 |
| **Python** | 3.10+ |
| **RAM** | 32 GB+ |
| **OS** | Ubuntu 20.04+ |
| **Disk Space** | 50 GB+ |

---

## 🧩 Software Stack

- **TensorRT-LLM** v0.20.0  
- **PyTorch** 2.3+  
- **Transformers** 4.43+  
- **CUDA** 12.9 · **cuDNN** 9.x  
- **Docker** 24.x+  
- **huggingface_hub**, **accelerate**, **numpy**, **tqdm**

---

## 🔧 Setup & Installation

### 1️⃣ Start NVIDIA TensorRT-LLM Container
```bash
docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/user/llm-trtllm:/workspace/llm-trtllm \
  nvcr.io/nvidia/tensorrt-llm/release:latest


2️⃣ Install Dependencies
cd /workspace/llm-trtllm
pip install --upgrade huggingface_hub transformers accelerate torch

📥 Download Model Weights
python3 - << 'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct",
    local_dir="hf_models/qwen2.5-7b-instruct",
    local_dir_use_symlinks=False)
PY

🔄 Convert & Build Engines
Convert Hugging Face → TensorRT-LLM
bash scripts/01_convert_qwen.sh

Build Engines (FP16)
bash scripts/02_build_fp16_lat_a100.sh     # Latency-optimized
bash scripts/03_build_fp16_inflight_a100.sh # Throughput-optimized

Outputs stored in:
/engine/qwen2.5-7b-a100-fp16-*/ 

Results include:

🕒 TTFT (Time to First Token)
⏱️ Latency
⚙️ Throughput (Tokens/sec)
💾 GPU Utilization

📊 Performance Snapshot

| Config                   | TTFT (s) | Latency (s) | TPS | GPU Util (%) |
| :----------------------- | :------- | :---------- | :-- | :----------- |
| Baseline (HF)            | 0.27     | 4.86        | 42  | 35           |
| TensorRT FP16 (lat)      | 0.05     | 1.24        | 170 | 75           |
| TensorRT FP16 (inflight) | 0.05     | 1.10        | 188 | 80           |

⚡ Speedup: 5–6× faster inference, 4× lower latency, and 2× higher GPU efficiency.

🧮 Key Optimizations
| Technique             | Description                           |
| :-------------------- | :------------------------------------ |
| **FP16 Precision**    | Reduces compute and memory footprint  |
| **Paged KV Cache**    | Dynamic attention memory allocation   |
| **Context FMHA**      | Fused multi-head attention kernel     |
| **Inflight Batching** | Parallel multi-request inference      |
| **TensorRT Graph**    | Compiled GPU execution with fused ops |

📁 Folder Structure
/workspace/llm-trtllm/
├── hf_models/             # Hugging Face model
├── checkpoints/           # TRT-LLM converted weights
├── engine/                # Serialized TensorRT engines
├── results/               # Benchmark outputs
└── scripts/               # Conversion, build, benchmark scripts
