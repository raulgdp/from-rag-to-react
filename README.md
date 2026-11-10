# From RAG to ReAct

[![Paper](https://img.shields.io/badge/Paper-IP%26M%202026-blue)](https://www.sciencedirect.com/journal/information-processing-and-management)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-CNA--EISC--RAFT--58K-yellow)](https://huggingface.co/datasets/raulgdp/CNA-EISC-RAFT-58K)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**A controlled four-way comparison of LLM architectures for low-resource institutional question answering**

> Gutiérrez de Piñerez Reyes, R. E. *"From RAG to ReAct: A Controlled Comparison of LLM Architectures for Low-Resource Institutional Question Answering"* — Information Processing & Management, 2026.

All four systems share identical retrieval infrastructure (BGE-M3 + BM25 + Qdrant, 79,879 chunks) and generator (DeepSeek-R1-0528), varying **only** in orchestration architecture — isolating orchestration as the sole independent variable.

---

## Systems

| System | Architecture | Description |
|--------|-------------|-------------|
| **SYS-A** | Sequential Semantic RAG | BGE-M3 + BM25 + RRF + BGE-Reranker-v2-m3 + DeepSeek-R1-0528 |
| **SYS-B** | Inference-time RAFT | SYS-A pipeline + CNA-EISC-RAFT-58K context enrichment (no fine-tuning) |
| **SYS-C** | Multi-Agent ReAct | Orchestrator (DeepSeek-R1) + Retriever + Validator (Gemma-3-12B), adaptive loop, Rmax=3 |
| **SYS-D** | LangGraph | Deterministic Python control flow, typed state transitions, guaranteed termination |

---

## Repository Structure

```
from-rag-to-react/
├── benchmark/
│   ├── benchmark_320q.jsonl      ← 320-question CNA-EISC evaluation benchmark
│   │                               (142 expert-curated + 178 corpus-derived)
│   └── README.md
├── systems/
│   ├── sysA_rag.py               ← SYS-A: Sequential Semantic RAG
│   ├── sysB_raft.py              ← SYS-B: Inference-time RAFT
│   ├── sysC_react.py             ← SYS-C: Multi-Agent ReAct
│   └── sysD_langgraph.py         ← SYS-D: LangGraph
├── evaluation/
│   ├── eval_sysA.py
│   ├── eval_sysB_sysD.py
│   ├── eval_sysC.py              ← Supports --resume for interrupted runs
│   └── ragas_judge.py            ← DeepSeek-R1-0528 as independent RAGAS judge
├── raft_dataset/
│   ├── build_raft_58k.py         ← Dataset construction pipeline
│   └── README.md
└── results/
    ├── main_results.csv
    ├── per_category.csv
    └── dre_per_round.csv
```

---

## Key Results

### RAGAS metrics — 320-question benchmark

| Metric | SYS-A (RAG) | SYS-B (RAFT) | SYS-C (ReAct) | SYS-D (LangGraph) |
|--------|:-----------:|:------------:|:-------------:|:-----------------:|
| Faithfulness | 0.832 ± .09 | 0.740 ± .16 | **0.873 ± .10** | 0.779 ± .16 |
| Answer Relevance | 0.851 ± .08 | 0.752 ± .18 | **0.878 ± .12** | 0.798 ± .18 |
| Context Precision | 0.744 ± .12 | 0.669 ± .16 | **0.842 ± .15** | 0.704 ± .15 |
| Context Recall | 0.711 ± .14 | 0.623 ± .21 | **0.757 ± .19** | 0.635 ± .20 |
| **RAGAS avg** | 0.785 | 0.696 | **0.838** | 0.729 |
| Latency (s) | **4.1 ± 0.8** | 38.0 ± 16.3 | 99.8 ± 72.4 | 29.4 ± 10.4 |
| Cost / query | **$0.0004** | $0.0007 | $0.0021 | $0.0012 |

**Main findings:**
- SYS-C (ReAct) achieves the highest quality on all four metrics (+0.053 RAGAS avg vs. SYS-A)
- **SYS-B (inference-time RAFT without fine-tuning) underperforms SYS-A by −0.089 at 9.3× the latency** — RAFT requires domain-adapted model training to be effective
- SYS-D (LangGraph) achieves 3.4× lower latency than SYS-C with 85% lower latency variance, suitable for strict SLAs
- Hybrid BM25+BGE-M3 contributes +14.3% context recall over dense-only search, uniformly across all architectures

### Decision Round Efficiency (DRE)

DRE measures marginal faithfulness gain per additional retrieval round: `DRE = (Fn − F1) / n`

| Round | SYS-C (ReAct) | SYS-D (LangGraph) |
|-------|:-------------:|:-----------------:|
| Round 1 | 0.841 | 0.754 |
| Round 2 | 0.863 | 0.779 |
| Round 3 | **0.873** | — |
| **DRE** | 0.437 | 0.390 |

78% of SYS-C's agentic quality gain is captured at round 1, motivating early-stopping schedulers.

### RAGAS average by category

| Category | n | SYS-A | SYS-B | SYS-C | SYS-D |
|----------|:-:|:-----:|:-----:|:-----:|:-----:|
| Statistical | 77 | 0.785 | 0.702 | **0.830** | 0.743 |
| Normative | 55 | 0.832 | 0.711 | **0.879** | 0.764 |
| Process | 54 | 0.851 | 0.714 | **0.854** | 0.737 |
| Comparison | 29 | **0.832** | 0.660 | 0.734 | 0.645 |
| Synthesis | 52 | 0.851 | 0.710 | **0.828** | 0.702 |
| General | 53 | 0.785 | 0.712 | **0.856** | 0.735 |

---

## Dataset: CNA-EISC-RAFT-58K

58,418 QA triplets (Q, D\*, D_k, A\*) built from 60 EISC accreditation documents following the RAFT formulation (Zhang et al., 2024).

```python
from datasets import load_dataset
ds = load_dataset("raulgdp/CNA-EISC-RAFT-58K")
# DatasetDict: train (46,734) / validation (5,841) / test (5,843)
# Evaluation benchmark (320q) is fully disjoint from all splits
```

| Statistic | Value |
|-----------|-------|
| Total QA triplets | 58,418 |
| Oracle-included examples (59.9%) | 35,002 |
| Distractor-only examples (40.1%) | 23,416 |
| Distractors per example (k) | 4 |
| Source documents | 60 EISC/CNA docs |
| Avg. CoT answer length | 55 words |
| Train / Val / Test split | 80 / 10 / 10 |

---

## Quick Start

### Prerequisites

```bash
pip install qdrant-client sentence-transformers langgraph openai tqdm ragas datasets
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-key"
export OPENROUTER_API_KEY="sk-or-..."
```

### Run a single system

```python
from systems.sysA_rag import SysA
from systems.sysC_react import SysC

sysA = SysA()
answer = sysA.answer("¿Cuántos factores tiene el modelo CNA del Decreto 1330?")
print(answer)
```

### Run the full evaluation

```bash
# SYS-A
python evaluation/eval_sysA.py \
  --benchmark benchmark/benchmark_320q.jsonl \
  --output results/sysA_results.json

# SYS-C (supports --resume for interrupted runs)
python evaluation/eval_sysC.py \
  --benchmark benchmark/benchmark_320q.jsonl \
  --delay 3 --output results/sysC_results.json

# SYS-B and SYS-D together
python evaluation/eval_sysB_sysD.py \
  --systems B D \
  --benchmark benchmark/benchmark_320q.jsonl \
  --delay 3
```

RAGAS metrics: `faithfulness` · `answer_relevance` · `context_precision` · `context_recall`  
Evaluator: `deepseek/deepseek-r1-0528` as independent judge

---

## Infrastructure

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Vector store | Qdrant Cloud (AWS us-east-1) |
| Embedding model | BAAI/bge-m3 (1024-dim) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Generator (all systems) | DeepSeek-R1-0528 via OpenRouter |
| Orchestrator fallbacks | Qwen3-32B → Claude Sonnet 4 |
| Validator (SYS-C) | Gemma-3-12B-IT → Llama-3.1-8B |
| Corpus size | 79,879 chunks (512 tokens, 64-token overlap) |
| Chunk sources | 47 EISC accreditation documents (2018–2025) |

---

## Related Repository

The fine-tuning study (ChatAcredita PRO, QLoRA on Qwen3-14B) that uses the same retrieval infrastructure is published separately:

> **[raulgdp/chatacredita-pro](https://github.com/raulgdp/chatacredita-pro)** — *ChatAcredita PRO: Fine-tuned LLMs with RAG-VSD Filtering for Colombian University Accreditation QA*

---

## Citation

```bibtex
@article{gutierrez2026fromrag,
  title   = {From {RAG} to {ReAct}: A Controlled Comparison of {LLM} Architectures
             for Low-Resource Institutional Question Answering},
  author  = {Gutiérrez de Piñerez Reyes, Raúl Ernesto},
  journal = {Information Processing \& Management},
  year    = {2026},
  note    = {EISC, Universidad del Valle, Cali, Colombia}
}
```

---

## Contact

For questions or issues, open a [GitHub Issue](https://github.com/raulgdp/from-rag-to-react/issues).
