# ⚡ LLM Inference Optimization — Qwen2.5-7B + TensorRT-LLM (A100)

<p align="center">
  <img src="assets/trtllm-setup.png" alt="TRT-LLM Setup" width="720">
</p>

<p align="center">
  <b>🚀 End-to-end benchmark and optimization pipeline for high-performance LLM inference on NVIDIA A100 GPUs.</b><br>
  <i>Compare baseline Hugging Face inference with TensorRT-LLM optimized engines to measure latency, throughput, and GPU efficiency.</i>
</p>

---

## 🌟 Highlights

- ⚙️ **Fully reproducible** TensorRT-LLM workflow: model conversion → engine build → benchmark  
- 🚀 **Up to 6× faster** inference vs baseline Hugging Face runtime  
- 📊 Measures **TTFT**, **latency**, **TPS**, and **GPU utilization**  
- 🔥 Built for **Qwen2.5-7B-Instruct**, optimized on **NVIDIA A100 (80 GB)**  
- 🧩 Integrates **FP16 precision**, **paged KV cache**, **context FMHA**, and **inflight batching**  
- 🐳 100% containerized using **NVIDIA TensorRT-LLM Docker image**

---

## 🚀 Overview

**Goal:**  
To reduce inference latency and maximize throughput for large-language models (LLMs) using TensorRT-LLM’s GPU-level optimizations.

**Model:**  
[`Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

**Hardware:**  
NVIDIA A100 80 GB GPU · CUDA 12.9 · TensorRT-LLM v0.20.0

---

## 🧱 Folder Structure

```text
/workspace/llm-trtllm/
├── hf_models/                     # Hugging Face raw model
│   └── qwen2.5-7b-instruct/
├── checkpoints/                   # TensorRT-LLM converted weights
│   └── qwen2.5-7b/
├── engine/                        # Serialized TensorRT engines
│   ├── qwen2.5-7b-a100-fp16-lat/
│   └── qwen2.5-7b-a100-fp16-inflight/
├── results/                       # Benchmark JSON outputs
├── scripts/                       # Conversion / build / benchmark scripts
│   ├── 01_convert_qwen.sh
│   ├── 02_build_fp16_lat_a100.sh
│   ├── 03_build_fp16_inflight_a100.sh
│   └── benchmark.py
└── .gitignore
