import time, json
from pathlib import Path

import torch
from transformers import AutoTokenizer


HF_DIR = "/workspace/llm-trtllm/hf_models/qwen2.5-7b-instruct"

# engines you just built
LAT_ENGINE = "engine/qwen2.5-7b-a100-fp16-lat"
INFLIGHT_ENGINE = "engine/qwen2.5-7b-a100-fp16-inflight"

LAT_PROMPT = "Explain paginated KV cache in transformers in simple English."
INFLIGHT_PROMPTS = [
    "Explain paginated KV cache.",
    "What does remove_input_padding do in TensorRT-LLM?",
    "Give 3 ways to improve LLM throughput on A100.",
    "Difference between paged KV cache and normal KV cache?",
]

LAT_MAX_NEW = 128
INFLIGHT_MAX_NEW = 128


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _unwrap_output_ids(output_ids, seq_lens, i: int):
    """
    TRT-LLM 0.20 Python runtime sometimes gives:
      - output_ids: [B, T]
      - or output_ids: [B, 1, T]
    and seq_lens:
      - [B]
      - or [B, 1]
    We normalize all of that here.
    """
    # sequence length
    if seq_lens.ndim == 2:
        length = int(seq_lens[i, 0])
    else:
        length = int(seq_lens[i])

    # ids
    if output_ids.ndim == 3:
        # [B, 1, T]
        ids = output_ids[i, 0, :length].tolist()
    else:
        # [B, T]
        ids = output_ids[i, :length].tolist()

    return ids


def bench_latency(tok):
    from tensorrt_llm.runtime import ModelRunner

    engine_path = Path(LAT_ENGINE)
    if not engine_path.exists():
        raise SystemExit(f"[latency] engine not found: {engine_path}")

    runner = ModelRunner.from_dir(str(engine_path))

    # encode prompt
    prompt_ids = tok.encode(LAT_PROMPT)
    prompt_tokens = len(prompt_ids)
    inputs = [torch.tensor(prompt_ids, dtype=torch.int32)]

    # warmup
    runner.generate(
        inputs,
        max_new_tokens=4,
        end_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        return_dict=True,
        output_sequence_lengths=True,
    )

    # TTFT: 1 token
    t0 = time.time()
    runner.generate(
        inputs,
        max_new_tokens=1,
        end_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        return_dict=True,
        output_sequence_lengths=True,
    )
    ttft_s = time.time() - t0

    # full gen
    t1 = time.time()
    out = runner.generate(
        inputs,
        max_new_tokens=LAT_MAX_NEW,
        end_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        return_dict=True,
        output_sequence_lengths=True,
    )
    latency_s = time.time() - t1

    output_ids = out["output_ids"]
    seq_lens = out["sequence_lengths"]

    decoded_ids = _unwrap_output_ids(output_ids, seq_lens, 0)
    text = tok.decode(decoded_ids, skip_special_tokens=True)

    total_tokens = len(decoded_ids)
    generated_tokens = max(0, total_tokens - prompt_tokens)
    gen_time = max(1e-6, latency_s - ttft_s)
    tps = generated_tokens / gen_time

    return {
        "engine": str(engine_path),
        "prompt": LAT_PROMPT,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "generated_tokens": generated_tokens,
        "ttft_s": ttft_s,
        "latency_s": latency_s,
        "tps": tps,
        "text": text,
    }


def bench_inflight(tok):
    from tensorrt_llm.runtime import ModelRunner

    engine_path = Path(INFLIGHT_ENGINE)
    if not engine_path.exists():
        raise SystemExit(f"[inflight] engine not found: {engine_path}")

    runner = ModelRunner.from_dir(str(engine_path))

    inputs = []
    prompt_lens = []
    for p in INFLIGHT_PROMPTS:
        ids = tok.encode(p)
        inputs.append(torch.tensor(ids, dtype=torch.int32))
        prompt_lens.append(len(ids))

    # warmup
    runner.generate(
        inputs,
        max_new_tokens=4,
        end_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        return_dict=True,
        output_sequence_lengths=True,
    )

    # batch TTFT
    t0 = time.time()
    runner.generate(
        inputs,
        max_new_tokens=1,
        end_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        return_dict=True,
        output_sequence_lengths=True,
    )
    ttft_s = time.time() - t0

    # full batch
    t1 = time.time()
    outs = runner.generate(
        inputs,
        max_new_tokens=INFLIGHT_MAX_NEW,
        end_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        return_dict=True,
        output_sequence_lengths=True,
    )
    latency_s = time.time() - t1

    output_ids = outs["output_ids"]
    seq_lens = outs["sequence_lengths"]
    B = output_ids.shape[0]

    requests = []
    for i in range(B):
        decoded_ids = _unwrap_output_ids(output_ids, seq_lens, i)
        text = tok.decode(decoded_ids, skip_special_tokens=True)
        prompt_tokens = prompt_lens[i]
        total_tokens = len(decoded_ids)
        generated_tokens = max(0, total_tokens - prompt_tokens)
        gen_time = max(1e-6, latency_s - ttft_s)
        tps = generated_tokens / gen_time

        requests.append(
            {
                "prompt": INFLIGHT_PROMPTS[i],
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "generated_tokens": generated_tokens,
                "tps_est": tps,
                "text": text,
            }
        )

    return {
        "engine": str(engine_path),
        "batch_size": B,
        "ttft_s": ttft_s,
        "latency_s": latency_s,
        "requests": requests,
    }


def main():
    tok = load_tokenizer()
    lat_res = bench_latency(tok)
    inf_res = bench_inflight(tok)

    print("=== LATENCY ===")
    print(json.dumps(lat_res, indent=2))
    print("\n=== INFLIGHT ===")
    print(json.dumps(inf_res, indent=2))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "latency.json").write_text(json.dumps(lat_res, indent=2))
    (results_dir / "inflight.json").write_text(json.dumps(inf_res, indent=2))
    print("[+] wrote results/latency.json and results/inflight.json")


if __name__ == "__main__":
    main()
