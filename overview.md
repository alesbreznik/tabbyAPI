# TabbyAPI Inference Backplane Overview & Compliance Report (`model_hosts/tabby_api`)

> [!NOTE]
> **Assigned Ports:** `5000` (Triage Engine - GPU 0) | `5001` (Reasoning Engine - GPU 1)
> **Engine Type:** EXL2 / EXL3 High-Performance ExLlamaV2 Engine
> **Target Hardware:** Dual NVIDIA GPUs (RTX 4080 SUPER 16GB + RTX 4000 Ada 20GB)

---

## 1. Executive Summary & Dual Architectural Role

`model_hosts/tabby_api` serves a critical dual function in the multi-agent system:

1. **Centralized Model Storage Repository (`model_hosts/tabby_api/models/`)**:
   Houses all centralized model weights, compiled ONNX computation graphs, and quantized model snapshots across the workspace (including Gemma 4 12B, Gemma 4 26B MoE, BGE-M3 embeddings, and BGE-Reranker v2).
2. **Dual-Instance EXL3 LLM Inference Engine**:
   Provides high-speed local text generation endpoints powered by ExLlamaV3:
   - **Triage Engine (12B)**: `gemma-4-12B-it-exl3` on GPU 0 (RTX 4080 SUPER) at `http://127.0.0.1:5000`
   - **Reasoning Engine (26B)**: `gemma-4-26B-A4B-it-exl3` on GPU 1 (RTX 4000 Ada) at `http://127.0.0.1:5001`

---

## 2. Architectural Standards Adherence Audit

| Compliance Criteria | Status | Notes & Audit Findings |
| :--- | :--- | :--- |
| **Centralized Model Storage** | ✅ **100% Compliant** | Houses central `models/` directory for all workspace models. |
| **Dynamic Relative Pathing** | ✅ **100% Compliant** | Dynamically resolves paths using `pathlib.Path`. Zero hardcoded Windows drive letters (`C:\`, `D:\`). |
| **Port Allocation Topology** | ✅ **100% Compliant** | Configured for Port 5000 (`config_4080S.yml`) and Port 5001 (`config_ada4000.yml`). |
| **Load-Once Initialization** | ✅ **100% Compliant** | Loads ExLlamaV3 weights into GPU VRAM once on startup. |
| **VRAM Telemetry Probe** | 🟡 **Partial** | Includes `/health` probe; recommends adding dynamic PyTorch/CUDA VRAM metrics. |

---

## 3. Configuration & Multi-GPU Topology

```text
model_hosts/tabby_api/
├── main.py                  # Primary FastAPI startup entrypoint
├── config_4080S.yml         # Port 5000 / GPU 0 configuration (Gemma 4 12B Triage)
├── config_ada4000.yml       # Port 5001 / GPU 1 configuration (Gemma 4 26B Reasoning)
├── models/                  # Centralized weights repository (bge-m3, gemma-4-12B, gemma-4-26B)
└── pyproject.toml           # Package build & dependency manifest
```

---

## 4. Identified Gaps & Refactoring Roadmap

1. **VRAM Telemetry Expansion**:
   - Update `/health` endpoint to report `vram_allocated_mb`, `vram_reserved_mb`, and `vram_headroom_mb` for GPU 0 and GPU 1.
2. **Unified Dual Launcher**:
   - Provide a standard launcher script (`start_server.bat`) that can launch both Port 5000 and Port 5001 concurrently.

---

## 5. Recommended Expansions

- **Logit Grammar Lock Tools**: Support JSON schema-constrained logit masking for 100% reliable structured tool calling.
- **Dynamic Context Headroom Adjuster**: Dynamically balance draft model KV cache allocation based on context length demands.
