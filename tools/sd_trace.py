#!/usr/bin/env python3
"""
sd_trace.py — Domain-Specific CoT Calibration Trace Generator for ExLlamaV3 (EXL3)
Generates packed safetensors calibration files containing multi-step mathematical,
algorithmic, and structured tool-calling chain-of-thought traces.
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
from transformers import AutoTokenizer

def build_cot_corpus():
    """
    Constructs a diverse, high-density calibration corpus containing:
    1. Mathematical proofs, Olympiad logic, and algebraic deductions.
    2. Software architecture, AST refactorings, algorithms, and regex/parsing.
    3. Structured ChatML agent interactions with tool schemas and JSON payloads.
    """
    traces = []
    
    # 1. Advanced Mathematics & Formal Logic
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
            "Evaluate the definite integral I = \\int_0^{\\infty} \\frac{\\ln(x)}{1 + x^2} dx using contour integration or symmetry substitution.",
            "Let us evaluate the integral using substitution:\nI = \\int_0^{\\infty} \\frac{\\ln(x)}{1 + x^2} dx.\nWe split the domain of integration at x = 1:\nI = \\int_0^1 \\frac{\\ln(x)}{1 + x^2} dx + \\int_1^{\\infty} \\frac{\\ln(x)}{1 + x^2} dx.\nConsider the second integral I_2 = \\int_1^{\\infty} \\frac{\\ln(x)}{1 + x^2} dx.\nSubstitute x = 1/t. Then dx = -1/t^2 dt.\nWhen x = 1, t = 1. As x \\to \\infty, t \\to 0^+.\nAlso, \\ln(x) = \\ln(1/t) = -\\ln(t).\nThe denominator becomes 1 + x^2 = 1 + 1/t^2 = (t^2 + 1)/t^2.\nPlugging these into I_2:\nI_2 = \\int_1^0 \\frac{-\\ln(t)}{(t^2 + 1)/t^2} \\left( -\\frac{1}{t^2} \\right) dt\n= \\int_1^0 \\frac{\\ln(t)}{t^2 + 1} dt\n= -\\int_0^1 \\frac{\\ln(t)}{t^2 + 1} dt = -I_1.\nHence, I = I_1 + I_2 = I_1 - I_1 = 0.\nThis demonstrates that the negative area on (0, 1) perfectly cancels the positive area on (1, \\infty).",
            "The definite integral evaluates to 0."
        ),
        (
            "Compute the eigenvalues and eigenvectors of the matrix A = [[3, 1], [1, 3]] and verify diagonalization.",
            "Step 1: Compute the characteristic polynomial det(A - \\lambda I) = 0.\nA - \\lambda I = [[3 - \\lambda, 1], [1, 3 - \\lambda]].\ndet(A - \\lambda I) = (3 - \\lambda)^2 - 1 = \\lambda^2 - 6\\lambda + 9 - 1 = \\lambda^2 - 6\\lambda + 8 = 0.\nFactoring: (\\lambda - 4)(\\lambda - 2) = 0.\nEigenvalues are \\lambda_1 = 4 and \\lambda_2 = 2.\nStep 2: Find eigenvectors.\nFor \\lambda_1 = 4:\n(A - 4I) v_1 = 0 => [[-1, 1], [1, -1]] [[x], [y]] = 0 => -x + y = 0 => y = x.\nNormalized eigenvector v_1 = 1/\\sqrt{2} [1, 1]^T.\nFor \\lambda_2 = 2:\n(A - 2I) v_2 = 0 => [[1, 1], [1, 1]] [[x], [y]] = 0 => x + y = 0 => y = -x.\nNormalized eigenvector v_2 = 1/\\sqrt{2} [1, -1]^T.\nStep 3: Verification.\nForm orthogonal matrix P = 1/\\sqrt{2} [[1, 1], [1, -1]].\nP^T A P = 1/2 [[1, 1], [1, -1]] [[3, 1], [1, 3]] [[1, 1], [1, -1]]\n= 1/2 [[1, 1], [1, -1]] [[4, 2], [4, -2]]\n= 1/2 [[8, 0], [0, 4]] = [[4, 0], [0, 2]] = \\Lambda.\nDiagonalization holds.",
            "Eigenvalues are \\lambda_1 = 4 with eigenvector [1, 1]^T, and \\lambda_2 = 2 with eigenvector [1, -1]^T."
        )
    ]
    
    # 2. Software Architecture, Code & AST Algorithms
    code_samples = [
        (
            "Implement a lock-free ring buffer in Python using atomic memory barriers or multiprocessing primitives, explaining cache-line bouncing.",
            "A lock-free single-producer single-consumer (SPSC) queue can be implemented using memory barriers and circular index wrapping.\nLet capacity C be a power of two so that index wrapping uses bitwise AND (head & (C - 1)) rather than expensive modulo arithmetic.\nTo prevent cache-line bouncing between producer and consumer cores, the write index and read index should ideally reside on separate CPU cache lines (padded with 64 bytes).\nHere is the implementation in Python using atomic value wrappers:\n```python\nimport multiprocessing as mp\nimport ctypes\n\nclass LockFreeSPSCQueue:\n    def __init__(self, capacity_power_of_two: int = 1024):\n        assert (capacity_power_of_two & (capacity_power_of_two - 1)) == 0, 'Capacity must be power of two'\n        self.capacity = capacity_power_of_two\n        self.mask = capacity_power_of_two - 1\n        # Buffer storage in shared memory\n        self.buffer = mp.Array(ctypes.c_double, capacity_power_of_two)\n        # Producer write position\n        self.head = mp.Value(ctypes.c_uint64, 0)\n        # Consumer read position\n        self.tail = mp.Value(ctypes.c_uint64, 0)\n\n    def push(self, item: float) -> bool:\n        h = self.head.value\n        t = self.tail.value\n        if (h - t) >= self.capacity:\n            return False  # Queue is full\n        self.buffer[h & self.mask] = item\n        # Memory release fence equivalent\n        self.head.value = h + 1\n        return True\n\n    def pop(self) -> float | None:\n        t = self.tail.value\n        h = self.head.value\n        if t == h:\n            return None  # Queue is empty\n        val = self.buffer[t & self.mask]\n        # Memory acquire fence equivalent\n        self.tail.value = t + 1\n        return val\n```\nThis guarantees wait-free O(1) ingestion without mutex contention.",
            "Lock-free SPSC circular queue implemented with power-of-two mask indexing and separate atomic monotonic cursors."
        ),
        (
            "Write a Python AST visitor that detects SQL injection vulnerabilities where variables are formatted directly into execute() string queries.",
            "To detect SQL injections statically, we inspect `ast.Call` nodes where `func.attr == 'execute'`.\nWe inspect the first argument to see if it is formed via `ast.BinOp` with `ast.Mod` (e.g. `\"...\" % var`), `ast.JoinedStr` (f-strings), or `ast.Call` calling `str.format()`.\n```python\nimport ast\n\nclass SQLInjectionDetector(ast.NodeVisitor):\n    def __init__(self):\n        self.vulnerabilities = []\n\n    def visit_Call(self, node: ast.Call):\n        if isinstance(node.func, ast.Attribute) and node.func.attr == 'execute':\n            if node.args:\n                query_arg = node.args[0]\n                if isinstance(query_arg, ast.JoinedStr):\n                    self.vulnerabilities.append((node.lineno, 'Formatted f-string passed to execute()'))\n                elif isinstance(query_arg, ast.BinOp) and isinstance(query_arg.op, ast.Mod):\n                    self.vulnerabilities.append((node.lineno, 'Percent-formatting string passed to execute()'))\n                elif isinstance(query_arg, ast.Call) and isinstance(query_arg.func, ast.Attribute) and query_arg.func.attr == 'format':\n                    self.vulnerabilities.append((node.lineno, 'str.format() called inside execute() query'))\n        self.generic_visit(node)\n```",
            "AST visitor accurately flags f-strings, str.format, and %-formatting inside cursor.execute() calls."
        ),
        (
            "Explain how ExLlamaV3 QTIP trellis quantization works and why mul1 codebook is optimal for INT8 tensor cores.",
            "ExLlamaV3's EXL3 format employs Trellis Coded Quantization (TCQ) combined with optimal scalar quantization and out-channel scaling.\nIn traditional round-to-nearest (RTN) or standard GPTQ/AWQ, each weight is independently rounded to a discrete grid point.\nIn trellis quantization, the sequence of weights is mapped as a path through a Viterbi trellis state machine with memory (Markov chain).\nThe Viterbi algorithm identifies the global minimum error path along the lattice for a given block of weights: \\min \\sum_i (w_i - \\hat{w}_i)^2.\nThe codebook defines the allowable transitions and reconstruction levels:\n- 'mul1' codebook represents 1st-order multiplicative codebooks directly aligned with FP16/INT8 hardware GEMM accumulators.\n- By maintaining uniform linear scales per group and trellis-packed bit sequences (e.g. 2.5, 3.5, 4.0, 5.0, 6.0 bpw), the dequantization kernel can unroll across 128-bit or 256-bit SIMD registers with a single instruction per weight vector, minimizing register pressure and avoiding branching divergence.",
            "Trellis quantization optimizes global path error via the Viterbi algorithm; the mul1 codebook enables single-pass vector dequantization on modern tensor cores."
        )
    ]
    
    # 3. Tool Invocations, Schema Parsers & Agentic Workflows
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

    all_raw = math_samples * 20 + code_samples * 20 + tool_samples * 20
    return all_raw

def main():
    parser = argparse.ArgumentParser(description="Generate CoT safetensors calibration dataset for ExLlamaV3")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to raw model or tokenizer directory")
    parser.add_argument("-co", "--cal_out", type=str, default="cal_trace.safetensors", help="Output safetensors path")
    parser.add_argument("-cr", "--cal_rows", type=int, default=250, help="Calibration rows")
    parser.add_argument("-cc", "--cal_cols", type=int, default=2048, help="Calibration columns (context width)")
    parser.add_argument("-cs", "--context_size", type=int, default=262144, help="Target context token budget")
    parser.add_argument("-tv", "--template_vars", type=str, default='{"enable_thinking": true}', help="Template vars")
    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    samples = build_cot_corpus()
    rows = args.cal_rows
    cols = args.cal_cols
    
    print(f"Generating {rows} rows x {cols} columns of domain CoT calibration data...")
    
    input_ids_list = []
    lengths_list = []
    
    sample_idx = 0
    while len(input_ids_list) < rows:
        user_msg, think_trace, answer = samples[sample_idx % len(samples)]
        sample_idx += 1
        
        # Build strict ChatML formatted text with reasoning tokens
        formatted = (
            f"<|im_start|>system\nYou are a helpful and harmless assistant. You should think through questions step by step before answering.<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{think_trace}\n</think>\n{answer}<|im_end|>\n"
        )
        
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        tok_len = len(tokens)
        
        if tok_len >= cols:
            # Crop to cols
            row_tokens = tokens[:cols]
            row_len = cols
        else:
            # Pad with pad_token_id or eos_token_id up to cols
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
