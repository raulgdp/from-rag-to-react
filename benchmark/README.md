# CNA-EISC Benchmark — 320 Questions

Evaluation benchmark for the paper:

> Gutiérrez de Piñerez Reyes, R. E. *"From RAG to ReAct: A Controlled Comparison of LLM Architectures for Low-Resource Institutional Question Answering"* — Information Processing & Management, 2026.

---

## Overview

| Property | Value |
|----------|-------|
| Total questions | 320 |
| Language | Spanish |
| Domain | Colombian university accreditation (CNA / Decreto 1330 de 2019) |
| Institution | EISC, Universidad del Valle, Cali, Colombia |
| Format | JSON / JSONL |

The benchmark is **fully disjoint** from the CNA-EISC-RAFT-58K training dataset. No question or reference answer appears in any training, validation, or test split of that dataset.

---

## Composition

| Category | n | Description |
|----------|:-:|-------------|
| `statistical` | 77 | Factual counts, rates, and numerical data from EISC reports |
| `normative` | 55 | Regulatory definitions, criteria, and requirements (Decreto 1330, Resolución 21795) |
| `process` | 54 | Procedural steps, workflows, and institutional processes |
| `comparison` | 29 | Cross-document or cross-period comparisons |
| `synthesis` | 52 | Multi-document aggregation requiring information from several sources |
| `general` | 53 | General institutional knowledge |
| **Total** | **320** | |

### Construction

- **142 expert-curated questions** (IDs: N, S, P, C, Y, G prefixes): authored manually by domain experts with direct knowledge of the CNA framework and EISC accreditation documents. Each question was validated against the source corpus.
- **178 corpus-derived questions** (IDs: X prefix): generated automatically from corpus chunks and validated for factual grounding.

---

## File Format

Each record has four fields:

```json
{
  "id": "N01",
  "category": "normative",
  "question": "¿Cuáles son los diez factores del modelo de acreditación del CNA según el Decreto 1330 de 2019?",
  "reference": "Los diez factores CNA son: Misión y Proyecto Institucional, Estudiantes, ..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique question identifier. Prefix encodes category and construction method |
| `category` | string | One of: `normative`, `statistical`, `process`, `comparison`, `synthesis`, `general` |
| `question` | string | Question in Spanish |
| `reference` | string | Validated reference answer used for context recall evaluation (RAGAS) |

---

## Files

| File | Format | Description |
|------|--------|-------------|
| `benchmark_320q.json` | JSON array | All 320 questions as a JSON array |
| `benchmark_320q.jsonl` | JSONL | One question per line (recommended for streaming evaluation) |

---

## Usage

```python
import json

# Load from JSON
with open("benchmark_320q.json") as f:
    benchmark = json.load(f)

# Load from JSONL
benchmark = []
with open("benchmark_320q.jsonl") as f:
    for line in f:
        benchmark.append(json.loads(line))

print(f"Total questions: {len(benchmark)}")

# Filter by category
normative = [q for q in benchmark if q["category"] == "normative"]
print(f"Normative: {len(normative)}")
```

### Running evaluation with RAGAS

```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# For each question in the benchmark:
# 1. Run your QA system to get `answer` and `contexts`
# 2. Score with RAGAS using q["reference"] as ground truth

sample = {
    "question":  q["question"],
    "answer":    your_system.answer(q["question"]),
    "contexts":  your_system.retrieve(q["question"]),
    "ground_truth": q["reference"],
}
```

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

## License

Apache 2.0 — see [LICENSE](../LICENSE).
