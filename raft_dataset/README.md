# CNA-EISC-RAFT-58K — Dataset Construction

Script that builds the **CNA-EISC-RAFT-58K** dataset from raw PDF accreditation documents following the RAFT formulation (Zhang et al., 2024).

> Gutiérrez de Piñerez Reyes, R. E. *"From RAG to ReAct: A Controlled Comparison of LLM Architectures for Low-Resource Institutional Question Answering"* — manuscript under review, 2025.

---

## What it does

```
60 EISC/CNA PDFs
      │
      ▼
pymupdf4llm + Tesseract OCR          ← advanced extraction (text + tables)
      │
RecursiveCharacterTextSplitter       ← structural chunking (800 chars, 150 overlap)
      │
Llama-3.3-70B via OpenRouter         ← 5 Q/A pairs per chunk (differentiated table vs. text prompts)
      │
RAFT construction                    ← oracle + 4 distractors, 40% no-oracle ablation
      │
      ▼
dataset_raft_v4.json  →  58,418 examples
```

**Output format per example:**

```json
{
  "pregunta":        "¿Cuáles son los diez factores del modelo CNA?",
  "documentos":      ["<oracle chunk>", "<distractor 1>", "...", "<distractor 4>"],
  "oracle_incluido": true,
  "oracle_chunk":    "<the chunk that contains the answer>",
  "respuesta_corta": "Los diez factores CNA son: ...",
  "respuesta_cot":   "Para responder esta pregunta... ##begin_quote##...##end_quote## Por lo tanto...",
  "fuente":          "informe_autoevaluacion_2024.pdf",
  "es_tabla":        false
}
```

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total examples | 58,418 |
| Oracle-included (59.9%) | 35,002 |
| No-oracle ablation (40.1%) | 23,416 |
| Distractors per example (k) | 4 |
| Q/A pairs per chunk | 5 |
| Chunk size | 800 chars |
| Chunk overlap | 150 chars |
| Source PDFs | 60 EISC/CNA documents |
| Split | 80 / 10 / 10 (train/val/test) |

The dataset is available on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("raulgdp/CNA-EISC-RAFT-58K")
# DatasetDict: train (46,734) / validation (5,841) / test (5,843)
```

---

## Usage

### Prerequisites

```bash
pip install pymupdf pymupdf4llm langchain-text-splitters openai pillow
# Optional (for OCR of tables and scanned pages):
pip install opencv-python pytesseract
# Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki (Windows)
#                   sudo apt install tesseract-ocr tesseract-ocr-spa (Linux)
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Run

```bash
# Place your PDF files in a folder named 'pdfs/'
mkdir pdfs
# copy your accreditation PDFs there

python build_raft_58k.py
# Output: dataset_raft_v4.json  (~58K examples)
```

The script **auto-resumes** from a checkpoint (`dataset_raft_v4_checkpoint_qa.json`) if interrupted — safe to Ctrl+C and restart.

### Key parameters (top of script)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PARES_POR_CHUNK` | 5 | Q/A pairs generated per chunk |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between consecutive chunks |
| `N_DISTRACTORS` | 4 | Distractor documents per RAFT example |
| `P_NO_ORACLE` | 0.40 | Fraction of examples without the oracle document |
| `MODEL_TEXTO` | `meta-llama/llama-3.3-70b-instruct` | Model for narrative chunks |
| `MODEL_TABLA` | `meta-llama/llama-3.3-70b-instruct` | Model for table chunks |

---

## RAFT Formulation

Each training example follows Zhang et al. (2024):

- **Oracle document** `D*`: the chunk that contains the gold answer
- **Distractor documents** `D_k`: k=4 chunks from other sources (or distant positions)
- **40% no-oracle examples**: the oracle is withheld — the model must learn to say "I don't have enough information" instead of hallucinating
- **Chain-of-thought answer**: includes explicit reasoning + verbatim citation with `##begin_quote##...##end_quote##`

---

## Citation

```bibtex
@unpublished{gutierrez2025fromrag,
  title  = {From {RAG} to {ReAct}: A Controlled Comparison of {LLM} Architectures
            for Low-Resource Institutional Question Answering},
  author = {Gutiérrez de Piñerez Reyes, Raúl Ernesto},
  note   = {Manuscript submitted for publication. EISC, Universidad del Valle, Cali, Colombia},
  year   = {2025},
}
```

```bibtex
@article{zhang2024raft,
  title   = {{RAFT}: Adapting Language Model to Domain Specific {RAG}},
  author  = {Zhang, Tianhao and others},
  journal = {arXiv preprint arXiv:2403.10131},
  year    = {2024},
}
```

---

## License

Apache 2.0 — see [LICENSE](../LICENSE).
