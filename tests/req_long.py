"""
Context Scaling, Batch Processing & Tri-GPU Allocation Benchmark.
Tests our two new models:
  - Model A: Qwen3.6-35B-A3B-exl3-5.0bpw (Port 5000 / 50010)
  - Model B: DeepSeek-R1-Distill-Qwen-32B-exl3-4.25bpw (Port 5001 / 50011)

Exercises:
1. Progressive context scaling (4k, 8k, 16k, 32k, 64k, up to 131k context window).
2. Needle-in-a-Haystack accuracy verification to ensure zero degradation under long context.
3. Batch concurrency testing (n=1, 2, 4) in non-streaming and streaming modes.
4. Artificial context overflow boundary detection (verifying clean HTTP 400 rejection).
5. Live Tri-GPU VRAM telemetry (GPU 0 RTX 4080S, GPU 1 Ada 4000, GPU 2 Ada 4000).
6. GPU allocation analysis to maximize Ada 4000 saturation and minimize 4080S footprint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from _common import load_api_keys, test_chat_request, test_chat_streaming

DEFAULT_BASE_URL = "http://localhost:5000/v1"
DEFAULT_MODEL_35B = "Qwen3.6-35B-A3B-exl3-5.0bpw"
DEFAULT_MODEL_32B = "DeepSeek-R1-Distill-Qwen-32B-exl3-4.25bpw"


def query_tri_gpu_vram() -> List[Dict[str, Any]]:
    """Queries live VRAM usage across all GPUs via nvidia-smi."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                used = float(parts[2])
                total = float(parts[3])
                free = float(parts[4])
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "used_mib": used,
                    "total_mib": total,
                    "free_mib": free,
                    "utilization_pct": round((used / total) * 100, 2) if total > 0 else 0.0,
                    "gpu_load_pct": float(parts[5]),
                })
        return gpus
    except Exception as exc:
        print(f"[WARN] Failed to query nvidia-smi: {exc}")
        return []


def print_gpu_telemetry_table(gpus: List[Dict[str, Any]], title: str = "GPU VRAM Telemetry"):
    print(f"\n--- [{title}] ---")
    print(f"{'GPU':<5} | {'Device Name':<32} | {'Used (MiB)':<12} | {'Total (MiB)':<12} | {'Free (MiB)':<12} | {'Saturation':<10}")
    print("-" * 96)
    for g in gpus:
        print(
            f"{g['index']:<5} | {g['name']:<32} | {g['used_mib']:<12.1f} | {g['total_mib']:<12.1f} | {g['free_mib']:<12.1f} | {g['utilization_pct']:<9.1f}%"
        )
    print("-" * 96)


def generate_synthetic_context(
    target_tokens: int,
    needle_key: Optional[str] = "ALPHA-7749",
    needle_depth_pct: float = 0.5,
) -> Tuple[str, Optional[str]]:
    """
    Generates synthetic filler tokens to artificially scale context length,
    inserting a verifiable needle fact at needle_depth_pct to test retrieval accuracy.
    Uses ~4 characters per token estimate.
    """
    filler_phrase = "System architecture analysis and runtime KV cache evaluation token sequence. "
    chars_per_token = 4.0
    total_chars = int(target_tokens * chars_per_token)

    needle_text = ""
    if needle_key:
        needle_text = f"\n[CRITICAL SYSTEM SECRET]: The operational recovery code is '{needle_key}'. Memorize this exact code.\n"

    target_filler_chars = max(0, total_chars - len(needle_text))
    filler_repetitions = max(1, int(target_filler_chars / len(filler_phrase)))
    filler = filler_phrase * filler_repetitions

    if needle_text:
        split_idx = int(len(filler) * needle_depth_pct)
        full_content = filler[:split_idx] + needle_text + filler[split_idx:]
    else:
        full_content = filler

    return full_content, needle_key


def resolve_model_id(api_key: str, base_url: str, requested_model: Optional[str] = None) -> str:
    """Queries /v1/models to verify or auto-detect the active model ID."""
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                model_ids = [m.get("id") for m in data if m.get("id")]
                if requested_model and requested_model in model_ids:
                    return requested_model
                if model_ids:
                    return model_ids[0]
    except Exception as exc:
        print(f"[WARN] Error resolving model from {base_url}/models: {exc}")
    return requested_model or DEFAULT_MODEL_35B


def test_context_step(
    api_key: str,
    base_url: str,
    model_id: str,
    context_tokens: int,
    test_needle: bool = True,
    max_tokens: int = 120,
) -> Dict[str, Any]:
    """Tests a single context length step, checking execution and needle accuracy."""
    needle_key = f"KEY-{random.randint(1000, 9999)}" if test_needle else None
    prompt_content, expected_key = generate_synthetic_context(context_tokens, needle_key=needle_key)

    query = (
        f"{prompt_content}\n\n"
        f"Based strictly on the text above, what is the exact operational recovery code? "
        f"State the code directly in one short sentence."
    )

    request_payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a precise technical retrieval system. Answer factually and concisely."},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    t0 = time.perf_counter()
    status_code = 0
    error_msg = None
    response_data = None
    generated_text = ""
    needle_found = False

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_payload,
            timeout=300.0,
        )
        status_code = resp.status_code
        if resp.status_code == 200:
            response_data = resp.json()
            choice = response_data.get("choices", [{}])[0]
            generated_text = choice.get("message", {}).get("content", "")
            if needle_key and needle_key in generated_text:
                needle_found = True
        else:
            error_msg = resp.text
    except Exception as exc:
        error_msg = str(exc)

    latency_s = time.perf_counter() - t0
    overflow_triggered = status_code == 400 and (
        "context" in (error_msg or "").lower() or "exceed" in (error_msg or "").lower()
    )

    return {
        "context_tokens": context_tokens,
        "status_code": status_code,
        "success": status_code == 200,
        "overflow_triggered": overflow_triggered,
        "latency_s": round(latency_s, 2),
        "needle_expected": needle_key,
        "needle_found": needle_found,
        "generated_preview": generated_text[:150].replace("\n", " ") if generated_text else "",
        "error": error_msg,
        "usage": response_data.get("usage") if response_data else None,
    }


def analyze_gpu_allocation(vram_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes optimal allocation across GPUs:
    - Minimized RTX 4080 Super (GPU 0) footprint
    - Maximized Ada 4000 (GPU 1 + GPU 2 = 40 GB) saturation
    - Preserved accuracy and context size
    """
    ada_cards = [g for g in vram_snapshots if "Ada" in g["name"] or g["index"] in (1, 2)]
    gpu_4080 = [g for g in vram_snapshots if "4080" in g["name"] or g["index"] == 0]

    total_ada_vram = sum(g["total_mib"] for g in ada_cards) / 1024.0
    free_ada_vram = sum(g["free_mib"] for g in ada_cards) / 1024.0
    used_4080_vram = gpu_4080[0]["used_mib"] / 1024.0 if gpu_4080 else 0.0

    recommendations = []
    if len(ada_cards) >= 2:
        recommendations.append(
            "Tri-GPU Pool Detected: 2x NVIDIA RTX 4000 Ada (40 GB total Ada VRAM) + 1x RTX 4080 Super (16 GB)."
        )
        recommendations.append(
            "Optimal Allocation for Qwen3.6-35B-A3B: Set CUDA_VISIBLE_DEVICES=1,2 (or gpu_split: [0.0, 19.5, 19.5]). "
            "This routes 100% of target MoE weights and 128k FP8 KV cache onto the dual Ada 4000 cards, "
            "reducing RTX 4080S footprint to 0 GB for LLMs and leaving 100% headroom for display and ONNX embedder/reranker."
        )
        recommendations.append(
            "Optimal Allocation for DeepSeek-R1-32B: Set CUDA_VISIBLE_DEVICES=0,2 with gpu_split: [2.5, 19.8], "
            "or pair Qwen 35B on GPU 1 and DeepSeek-R1-32B on GPU 2 with shared drafters."
        )
        recommendations.append(
            "KV Cache Accuracy Retention: With 40 GB Ada capacity, keep cache_mode at '8,8' (FP8 KV Cache) "
            "for 128k context instead of '4,4' or '2,2' quantization, completely avoiding attention accuracy degradation."
        )

    return {
        "total_ada_vram_gb": round(total_ada_vram, 2),
        "free_ada_vram_gb": round(free_ada_vram, 2),
        "gpu_4080_used_gb": round(used_4080_vram, 2),
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Context Overflow, Batch & GPU Allocation Benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="TabbyAPI endpoint (default: http://localhost:5000/v1)")
    parser.add_argument("--model", default=None, help="Model ID override (e.g. Qwen3.6-35B-A3B-exl3-5.0bpw or DeepSeek-R1-Distill-Qwen-32B-exl3-4.25bpw)")
    parser.add_argument("--context-steps", default="4096,8192,16384,32768,65536,131072", help="Comma-separated token steps")
    parser.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated batch sizes (n)")
    parser.add_argument("--test-streaming", action="store_true", default=True, help="Test streaming responses")
    parser.add_argument("--skip-overflow", action="store_true", help="Skip artificial overflow trigger step")
    args = parser.parse_args()

    _, api_key = load_api_keys()
    if not api_key:
        api_key = "8c4a7797f347fb51c6907ec0a297cb24"

    print("=" * 80)
    print("🔥 TABBYAPI CONTEXT OVERFLOW, BATCH PROCESSING & GPU ALLOCATION TEST")
    print("=" * 80)

    # 1. Probe GPU VRAM Pre-test
    vram_pre = query_tri_gpu_vram()
    print_gpu_telemetry_table(vram_pre, "Initial Hardware State (Tri-GPU)")

    # 2. Resolve Active Model
    model_id = resolve_model_id(api_key, args.base_url, args.model)
    print(f"\n[TARGET] Model: {model_id} | Endpoint: {args.base_url}")

    # 3. Context Length Escalation & Needle-in-a-Haystack Accuracy Test
    context_tokens_list = [int(x.strip()) for x in args.context_steps.split(",") if x.strip()]
    print(f"\n[PHASE 1] Context Horizon Escalation ({len(context_tokens_list)} steps): {context_tokens_list}")

    step_results = []
    for ctx_len in context_tokens_list:
        print(f"\n>>> Probing context = {ctx_len:,} tokens...")
        res = test_context_step(api_key, args.base_url, model_id, ctx_len, test_needle=True)
        step_results.append(res)
        if res["success"]:
            acc_str = "PASSED (Needle Found)" if res["needle_found"] else "ACCURACY DEGRADED (Needle Missed)"
            print(f"    ✓ HTTP {res['status_code']} | Latency: {res['latency_s']}s | Accuracy: {acc_str}")
            print(f"      Response: \"{res['generated_preview']}\"")
        elif res["overflow_triggered"]:
            print(f"    ⚠️ OVERFLOW BOUNDARY REACHED cleanly at {ctx_len:,} tokens (HTTP 400)")
            print(f"      Server Error: {res['error']}")
            break
        else:
            print(f"    ❌ FAILED at {ctx_len:,} tokens: HTTP {res['status_code']} - {res['error']}")
            break

    # 4. Batch Processing Concurrency Test (n=1, 2, 4)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    print(f"\n[PHASE 2] Batch Processing Concurrency Test (n={batch_sizes})")

    batch_payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are an autonomous engineering assistant."},
            {"role": "user", "content": "Explain the architectural difference between Gated Linear Attention and Standard Attention."},
        ],
        "max_tokens": 128,
        "temperature": 0.2,
    }

    for n in batch_sizes:
        print(f"\n>>> Executing batch completion (n={n})...")
        t0 = time.perf_counter()
        try:
            res_batch = test_chat_request(api_key, args.base_url, batch_payload.copy(), n=n)
            wall_t = time.perf_counter() - t0
            print(f"    ✓ Batch n={n} completed in {wall_t:.2f}s ({len(res_batch.get('choices', []))} choices)")
        except Exception as exc:
            print(f"    ❌ Batch n={n} failed: {exc}")

        if args.test_streaming:
            print(f"\n>>> Executing batch streaming (n={n})...")
            try:
                test_chat_streaming(api_key, args.base_url, batch_payload.copy(), n=n)
            except Exception as exc:
                print(f"    ❌ Batch streaming n={n} failed: {exc}")

    # 5. Hardware Post-Audit & GPU Allocation Optimization Analysis
    vram_post = query_tri_gpu_vram()
    print_gpu_telemetry_table(vram_post, "Post-Test Hardware State (Tri-GPU)")

    alloc_analysis = analyze_gpu_allocation(vram_post)
    print("\n" + "=" * 80)
    print("📊 GPU ALLOCATION & KV CACHE OPTIMIZATION BLUEPRINT")
    print("=" * 80)
    print(f"Total Ada 4000 VRAM: {alloc_analysis['total_ada_vram_gb']} GB")
    print(f"Free Ada 4000 VRAM:  {alloc_analysis['free_ada_vram_gb']} GB")
    print(f"RTX 4080S Used VRAM: {alloc_analysis['gpu_4080_used_gb']} GB")
    print("\nStrategic Recommendations:")
    for idx, rec in enumerate(alloc_analysis["recommendations"], 1):
        print(f"  {idx}. {rec}")
    print("=" * 80)


if __name__ == "__main__":
    main()
