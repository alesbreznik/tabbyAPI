#!/usr/bin/env python3
"""
cal_trace_qwen38.py — Domain-Specific Calibration Dataset Generator for Qwen3.8-27B (Hybrid DeltaNet / VL)
Generates packed safetensors calibration files containing multi-step mathematical/algorithmic reasoning,
multimodal UI/screen grounding, code AST transformations, and structured tool-calling chain-of-thought traces.
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
from transformers import AutoTokenizer

def build_qwen38_corpus():
    traces = []
    
    # 1. Advanced Reasoning & Formal Logic (<think> CoT)
    math_samples = [
        (
            "Find all real solutions to the equation x^4 - 6x^2 + 8 = 0 and determine the roots of the cubic resolvent.",
            "Let u = x^2. The equation transforms into a quadratic in terms of u:\nu^2 - 6u + 8 = 0.\nFactoring this quadratic:\n(u - 2)(u - 4) = 0.\nThus the solutions for u are u_1 = 2 and u_2 = 4.\nNow substituting back for x:\nFor u_1 = 2: x^2 = 2 => x = \\pm \\sqrt{2}.\nFor u_2 = 4: x^2 = 4 => x = \\pm 2.\nTherefore, the four real solutions are x \\in \\{ -2, -\\sqrt{2}, \\sqrt{2}, 2 \\}.\nNow consider the cubic resolvent of a depressed quartic x^4 + px^2 + qx + r = 0, where p = -6, q = 0, r = 8.\nThe standard resolvent cubic is given by:\ny^3 + 2p y^2 + (p^2 - 4r) y - q^2 = 0.\nPlugging in our coefficients:\n2p = 2(-6) = -12.\np^2 - 4r = (-6)^2 - 4(8) = 36 - 32 = 4.\nq^2 = 0^2 = 0.\nSo the resolvent equation is:\ny^3 - 12y^2 + 4y = 0.\nFactoring y:\ny(y^2 - 12y + 4) = 0.\nOne root is y_0 = 0.\nFor y^2 - 12y + 4 = 0, we apply the quadratic formula:\ny = \\frac{12 \\pm \\sqrt{144 - 16}}{2} = \\frac{12 \\pm \\sqrt{128}}{2} = 6 \\pm 4\\sqrt{2}.\nThus, the roots of the resolvent cubic are y \\in \\{0, 6 - 4\\sqrt{2}, 6 + 4\\sqrt{2}\\}.",
            "The real roots are x = \\pm \\sqrt{2} and x = \\pm 2. The roots of the cubic resolvent are 0, 6 - 4\\sqrt{2}, and 6 + 4\\sqrt{2}."
        ),
        (
            "Prove that for any positive integers a and b, gcd(a, b) * lcm(a, b) = a * b using prime factorization.",
            "Let the canonical prime factorizations of a and b over the set of all primes P = {p_1, p_2, ...} be:\na = \\prod_{i} p_i^{\\alpha_i}, \\quad b = \\prod_{i} p_i^{\\beta_i},\nwhere \\alpha_i, \\beta_i \\ge 0 and only finitely many exponents are non-zero.\nBy definition of the greatest common divisor and least common multiple in terms of valuations:\n\\gcd(a, b) = \\prod_{i} p_i^{\\min(\\alpha_i, \\beta_i)},\n\\text{lcm}(a, b) = \\prod_{i} p_i^{\\max(\\alpha_i, \\beta_i)}.\nNow consider their product:\n\\gcd(a, b) \\times \\text{lcm}(a, b) = \\left( \\prod_{i} p_i^{\\min(\\alpha_i, \\beta_i)} \\right) \\left( \\prod_{i} p_i^{\\max(\\alpha_i, \\beta_i)} \\right)\n= \\prod_{i} p_i^{\\min(\\alpha_i, \\beta_i) + \\max(\\alpha_i, \\beta_i)}.\nNotice that for any real numbers (and hence integers) x and y:\n\\min(x, y) + \\max(x, y) = x + y.\nApplying this identity to the exponents \\alpha_i and \\beta_i:\n\\min(\\alpha_i, \\beta_i) + \\max(\\alpha_i, \\beta_i) = \\alpha_i + \\beta_i.\nTherefore:\n\\gcd(a, b) \\times \\text{lcm}(a, b) = \\prod_{i} p_i^{\\alpha_i + \\beta_i} = \\left( \\prod_{i} p_i^{\\alpha_i} \\right) \\left( \\prod_{i} p_i^{\\beta_i} \\right) = a \\cdot b.\nThis completes the proof rigorously.",
            "Q.E.D. By the fundamental property of min and max over prime exponents, \\min(\\alpha_i, \\beta_i) + \\max(\\alpha_i, \\beta_i) = \\alpha_i + \\beta_i, establishing gcd(a, b) * lcm(a, b) = a * b."
        ),
        (
            "Analyze the recurrence relation T(n) = 2T(n/2) + n / log(n) for n > 2.",
            "Let us apply the Akra-Bazzi method or master theorem extension for polylogarithmic factors.\nThe recurrence is of the form T(n) = 2T(n/2) + f(n), where f(n) = n / log(n).\nHere a = 2, b = 2, so n^{\\log_b a} = n^{\\log_2 2} = n^1 = n.\nCompare f(n) with n^{\\log_b a}:\nf(n) = n / \\log(n) = n \\cdot (\\log n)^{-1}.\nNotice this falls under the boundary case where f(n) = \\Theta(n^{\\log_b a} \\log^k n) with k = -1.\nBy the Extended Master Theorem (Case 2 with k = -1):\nWhen f(n) = \\Theta(n^{\\log_b a} \\log^{-1} n), T(n) = \\Theta(n^{\\log_b a} \\log(\\log n)) = \\Theta(n \\log \\log n).\nTherefore, the asymptotic tight bound is T(n) = \\Theta(n \\log \\log n).",
            "The tight asymptotic complexity is T(n) = \\Theta(n \\log \\log n)."
        )
    ]
    
    # 2. Multimodal UI & Visual Inspection Samples
    vision_samples = [
        (
            "Analyze the attached application window screenshot. Identify all interactive buttons, their bounding box coordinates, and detect any validation errors on the form.",
            "The input contains a high-resolution screenshot rendered via the vision tower.\nI need to inspect the visual tokens representing the DOM layout.\n1. Primary container: Modal dialog centered at [ymin: 120, xmin: 240, ymax: 680, xmax: 760].\n2. Header text: 'Payment Gateway Configuration'.\n3. Input field: 'API Key' at [ymin: 220, xmin: 260, ymax: 270, xmax: 740], contains red outline indicating validation failure (error message: 'Invalid checksum').\n4. Interactive buttons at the footer:\n- 'Cancel' at [ymin: 610, xmin: 520, ymax: 660, xmax: 620]\n- 'Confirm & Deploy' (primary CTA, active) at [ymin: 610, xmin: 630, ymax: 660, xmax: 740].\nLet's assemble the structured spatial detection output.",
            "Visual Analysis Complete:\n- Modal Container: {\"box_2d\": [120, 240, 680, 760], \"type\": \"modal\"}\n- Validation Error: {\"box_2d\": [272, 260, 295, 450], \"field\": \"API Key\", \"message\": \"Invalid checksum\", \"color\": \"#DC2626\"}\n- Cancel Button: {\"box_2d\": [610, 520, 660, 620], \"state\": \"enabled\"}\n- Submit Button: {\"box_2d\": [610, 630, 660, 740], \"state\": \"enabled\", \"action\": \"primary\"}"
        ),
        (
            "Inspect the architecture diagram image. Extract the microservice nodes, directed message broker queues, and database connections.",
            "Let's trace the graph nodes and edges from the visual diagram.\n- Tier 1: Client Gateway at top [100, 400, 180, 600].\n- Edge from Gateway -> Event Bus (Kafka cluster) with label 'gRPC / JSON'.\n- Tier 2: Three worker services subscribing to topic 'agent_tasks': Worker A, Worker B, Worker C.\n- Tier 3: Datastore connections -> Worker A connects to PostgreSQL (port 5432) via connection pool; Worker B and C write to Redis cache (port 6379).\nLet's represent the topology clearly.",
            "Diagram Architecture Topology:\n1. Ingress: Client Gateway (port 443)\n2. Message Broker: Apache Kafka ('agent_tasks' topic, 12 partitions)\n3. Consumer Microservices: Worker-A (CRUD/PostgreSQL), Worker-B/C (Compute/Redis Cache)\n4. Datastores: PostgreSQL 16 (Port 5432, Read-Write Primary), Redis Cluster (Port 6379)"
        )
    ]

    # 3. Software Architecture, Code & AST Algorithms
    code_samples = [
        (
            "Implement an efficient hybrid Gated DeltaNet attention layer in PyTorch, highlighting recurrent state evolution versus expanding KV cache.",
            "In a hybrid architecture like Qwen3.8, linear attention layers update an O(d^2) state matrix recursively:\nS_t = \\alpha_t S_{t-1} + \\beta_t v_t k_t^T,\nwhere \\alpha_t is the decay gate and \\beta_t is the update gate.\nThe memory footprint of S_t is constant with respect to sequence length T (e.g. 72 MB total across 48 layers), unlike standard softmax attention where KV cache grows as O(T \\cdot d_head \\cdot n_kv).\nHere is the vectorized kernel structure in PyTorch:\n```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass GatedDeltaNetLayer(nn.Module):\n    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int = 128):\n        super().__init__()\n        self.num_heads = num_heads\n        self.head_dim = head_dim\n        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)\n        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)\n        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)\n        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)\n        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)\n        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)\n\n    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None):\n        B, L, D = x.shape\n        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)\n        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)\n        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)\n        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, L, self.num_heads, 1, 1)\n        beta = torch.sigmoid(self.beta_proj(x)).view(B, L, self.num_heads, 1, 1)\n        if prev_state is None:\n            state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)\n        else:\n            state = prev_state\n        outputs = []\n        for t in range(L):\n            q_t = q[:, t]\n            k_t = k[:, t].unsqueeze(-1)\n            v_t = v[:, t].unsqueeze(-2)\n            state = alpha[:, t] * state + beta[:, t] * torch.matmul(k_t, v_t)\n            out_t = torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)\n            outputs.append(out_t)\n        out = torch.stack(outputs, dim=1).view(B, L, -1)\n        return self.out_proj(out), state\n```\nThis linear recurrent formulation guarantees O(1) memory during auto-regressive decoding.",
            "Gated DeltaNet linear attention implemented with constant O(1) state memory during autoregressive generation."
        ),
        (
            "Write a Python AST visitor that detects SQL injection vulnerabilities where variables are formatted directly into execute() string queries.",
            "To detect SQL injections statically, we inspect `ast.Call` nodes where `func.attr == 'execute'`.\nWe inspect the first argument to see if it is formed via `ast.BinOp` with `ast.Mod` (e.g. `\"...\" % var`), `ast.JoinedStr` (f-strings), or `ast.Call` calling `str.format()`.\n```python\nimport ast\n\nclass SQLInjectionDetector(ast.NodeVisitor):\n    def __init__(self):\n        self.vulnerabilities = []\n\n    def visit_Call(self, node: ast.Call):\n        if isinstance(node.func, ast.Attribute) and node.func.attr == 'execute':\n            if node.args:\n                query_arg = node.args[0]\n                if isinstance(query_arg, ast.JoinedStr):\n                    self.vulnerabilities.append((node.lineno, 'Formatted f-string passed to execute()'))\n                elif isinstance(query_arg, ast.BinOp) and isinstance(query_arg.op, ast.Mod):\n                    self.vulnerabilities.append((node.lineno, 'Percent-formatting string passed to execute()'))\n                elif isinstance(query_arg, ast.Call) and isinstance(query_arg.func, ast.Attribute) and query_arg.func.attr == 'format':\n                    self.vulnerabilities.append((node.lineno, 'str.format() called inside execute() query'))\n        self.generic_visit(node)\n```",
            "AST visitor accurately flags f-strings, str.format, and %-formatting inside cursor.execute() calls."
        )
    ]
    
    # 4. Tool Invocations & Multi-Agent Operations
    tool_samples = [
        (
            "Execute an introspective analysis of the local GPU allocation and query the watchdog broker for endpoint health.",
            "The user needs to check GPU memory and ping the watchdog broker.\nI need to inspect the current environment through available tool calls.\nTool 1: `sidecar_system_os` or local process monitor.\nTool 2: `read_url_content` or HTTP GET to `http://localhost:9000/health`.\nLet me construct the structured tool invocation.",
            "<tool_call>\n{\"name\": \"get_system_metrics\", \"arguments\": {\"gpu_indices\": [0, 1, 2], \"include_processes\": true}}\n</tool_call>"
        ),
        (
            "Search the codebase for all occurrences of 'DeepSeek-R1-Distill' and prepare a replacement plan for QwQ-32B.",
            "We must locate references to DeepSeek-R1-Distill across configuration files, tests, documentation, and agent blueprints.\nI should invoke ripgrep to find exact pattern matches.\nTarget files will include `supergraph/config/master_topology.json`, `infrastructure/watchdog/watchdog_broker.py`, `model_hosts/tabby_api/model_runtime_configs/`.\nLet's format the search query.",
            "<tool_call>\n{\"name\": \"grep_search\", \"arguments\": {\"Query\": \"DeepSeek-R1-Distill\", \"SearchPath\": \"/home/ales/AI\"}}\n</tool_call>"
        )
    ]

    all_raw = math_samples * 25 + vision_samples * 25 + code_samples * 25 + tool_samples * 25
    return all_raw

def main():
    parser = argparse.ArgumentParser(description="Generate CoT + Vision safetensors calibration dataset for Qwen3.8-27B")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to raw model or tokenizer directory")
    parser.add_argument("-co", "--cal_out", type=str, default="cal_trace_dualmode.safetensors", help="Output safetensors path")
    parser.add_argument("-cr", "--cal_rows", type=int, default=350, help="Calibration rows")
    parser.add_argument("-cc", "--cal_cols", type=int, default=2048, help="Calibration columns (context width)")
    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    samples = build_qwen38_corpus()
    rows = args.cal_rows
    cols = args.cal_cols
    
    print(f"Generating {rows} rows x {cols} columns of Qwen3.8 calibration data...")
    
    input_ids_list = []
    lengths_list = []
    
    sample_idx = 0
    while len(input_ids_list) < rows:
        user_msg, think_trace, answer = samples[sample_idx % len(samples)]
        sample_idx += 1
        
        # Format as standard Qwen chat template with <think> tag
        formatted = (
            f"<|im_start|>system\nYou are a helpful and harmless assistant. You should think through questions step by step before answering.<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{think_trace}\n</think>\n{answer}<|im_end|>\n"
        )
        
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        tok_len = len(tokens)
        
        if tok_len >= cols:
            row_tokens = tokens[:cols]
            row_len = cols
        else:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            row_tokens = tokens + [pad_id] * (cols - tok_len)
            row_len = tok_len
            
        input_ids_list.append(row_tokens)
        lengths_list.append(row_len)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths_list, dtype=torch.long)
    
    out_dir = Path(args.cal_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving safetensors to {args.cal_out}: shape {input_ids_tensor.shape}, lengths {lengths_tensor.shape}")
    save_file(
        {
            "input_ids": input_ids_tensor,
            "lengths": lengths_tensor
        },
        args.cal_out
    )
    print("Calibration file generated successfully.")

if __name__ == "__main__":
    main()
