# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Generador de Dataset RAFT desde PDF de Acreditacion — v4                   ║
# ║  CAMBIOS vs v3: Gemma 3 27B via OpenRouter (free) en vez de Ollama local    ║
# ║  CHUNK_SIZE=800, CHUNK_OVERLAP=150, P_NO_ORACLE=0.40, PARES_POR_CHUNK=5    ║
# ║  Uso: python generar_dataset_raft_v4.py                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import re
import io
import json
import time
import random
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import fitz                      # PyMuPDF
import pymupdf4llm
from PIL import Image
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import cv2
    import pytesseract
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN v3
# ─────────────────────────────────────────────────────────────────────────────
#GROQ_API_KEY      = ""
PDF_DIR           = "pdfs"
OUTPUT_JSON       = "dataset_raft_v4.json"          # ✅ nuevo nombre para no sobreescribir v3
PARES_POR_CHUNK   = 5        # ✅ 3 → 5: más ángulos por chunk
CHUNK_SIZE        = 800      # ✅ 500 → 800: contexto real de ~6-8 líneas
CHUNK_OVERLAP     = 150      # ✅ 100 → 150: más continuidad entre chunks
MIN_CHUNK_LEN     = 80       # ✅ 50 → 80: descartar chunks muy cortos

# ── Modelo via OpenRouter (free tier) ────────────────────────────────────────
MODEL_TABLA = "meta-llama/llama-3.3-70b-instruct"
MODEL_TEXTO = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_BASE   = "https://openrouter.ai/api/v1"
OPENROUTER_KEY    = os.getenv("OPENROUTER_API_KEY", "")  # ← pon tu key aquí

# ── Parámetros RAFT ───────────────────────────────────────────────────────────
N_DISTRACTORS     = 4
P_NO_ORACLE       = 0.40     # ✅ 0.20 → 0.40: 40% sin oracle, fiel al paper RAFT
SEED              = 42

random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN TESSERACT (igual que embeddings-avanzado.py)
# ─────────────────────────────────────────────────────────────────────────────
def is_tesseract_available():
    if not CV2_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

if CV2_AVAILABLE and not is_tesseract_available():
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

TESSERACT_AVAILABLE = is_tesseract_available()
print(f"{'✅' if TESSERACT_AVAILABLE else '⚠️'} Tesseract OCR: "
      f"{'Disponible' if TESSERACT_AVAILABLE else 'NO disponible (solo texto nativo)'}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXTRACCIÓN AVANZADA — pymupdf4llm + OCR de tablas/imágenes
#    (tomado de embeddings-avanzado.py — reemplaza el pdfplumber simple)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return ' '.join(text.split())


def extract_text_with_tesseract(page, page_num: int):
    """OCR de página completa — captura tablas e imágenes que el extractor de texto pierde."""
    if not TESSERACT_AVAILABLE:
        return None
    try:
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # aumentar contraste para mejor OCR

        custom_config = r'--oem 3 --psm 6 -l spa+eng'
        text = pytesseract.image_to_string(gray, config=custom_config)

        if text and text.strip():
            return f"[TABLA/IMAGEN PÁGINA {page_num + 1}]\n{normalize_text(text)}"
    except Exception as e:
        print(f"      ⚠️  OCR error página {page_num + 1}: {str(e)[:50]}")
    return None


def limpiar_encabezados(text: str) -> str:
    """Elimina encabezados/pies de página repetitivos típicos de docs de acreditación."""
    text = re.sub(r'Página \| \d+', '', text)
    text = re.sub(r'Documento para Solicitud de Renovación[^\n]*\n', '', text)
    text = re.sub(r'Programa Académico Ingeniería[^\n]*\n', '', text)
    text = re.sub(r'Versión\. Noviembre de \d{4}\n?', '', text)
    text = re.sub(r'Vicerrectoría\s*\nAcadémica\s*\n', '', text)
    text = re.sub(r'Dirección de Autoevaluación[^\n]*\n', '', text)
    return text


def process_pdf_advanced(pdf_path: Path):
    """
    Extracción avanzada por página:
    - pymupdf4llm.to_markdown() preserva estructura (encabezados, tablas en markdown)
    - Si la página tiene imágenes/tablas y Tesseract está disponible, agrega OCR
    Retorna: (texto_completo_markdown, num_tablas_detectadas)
    """
    print(f"   📄 {pdf_path.name}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"   ❌ Error abriendo PDF: {e}")
        return "", 0

    all_text = ""
    tables_detected = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extracción markdown (preserva tablas como | col | col |)
        try:
            page_text = pymupdf4llm.to_markdown(
                doc, pages=[page_num], show_progress=False, page_chunks=False
            )
        except Exception:
            page_text = page.get_text()

        # Detectar elementos visuales (tablas/imágenes) en la página
        has_visual = (
            "tabla" in page_text.lower()
            or "imagen" in page_text.lower()
            or "|" in page_text                 # markdown table syntax
            or len(page.get_images()) > 0
        )

        if has_visual and TESSERACT_AVAILABLE:
            ocr_text = extract_text_with_tesseract(page, page_num)
            if ocr_text:
                tables_detected += 1
                page_text += f"\n\n{ocr_text}"

        all_text += f"\n\n--- Página {page_num + 1} ---\n\n" + (page_text or "[Página sin texto]")

    doc.close()
    return limpiar_encabezados(all_text), tables_detected


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHUNKING ESTRUCTURAL — respeta encabezados, tablas y listas
#    (tomado de embeddings-avanzado.py — reemplaza la ventana deslizante simple)
# ─────────────────────────────────────────────────────────────────────────────
def structural_chunking(full_text: str, chunk_size: int = CHUNK_SIZE,
                        chunk_overlap: int = CHUNK_OVERLAP):
    """
    Chunking que respeta la estructura del documento:
    - Prioriza cortar en encabezados markdown (##, ###)
    - Nunca corta una tabla [TABLA ...] a la mitad si puede evitarlo
    - Fallback a separadores estándar si no hay estructura
    """
    if not full_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n## ", "\n\n### ", "\n\n#### ",
            "\n\n[TABLA ", "\n\n[IMAGEN ",
            "\n\n|", "\n\n", "\n", " ", "",
        ],
    )
    chunks = splitter.split_text(full_text)
    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]


def es_chunk_tabla(chunk: str) -> bool:
    """Detecta si un chunk es predominantemente una tabla (markdown o TABLA OCR)."""
    return (
        chunk.count("|") > 6
        or "[TABLA" in chunk
        or chunk.lower().count("tabla ") >= 2
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE DE EXTRACCIÓN — procesa todos los PDFs
# ─────────────────────────────────────────────────────────────────────────────
pdf_dir = Path(PDF_DIR)
if not pdf_dir.is_dir():
    raise SystemExit(f"❌ La carpeta '{PDF_DIR}' no existe. Crea la carpeta y coloca tus PDFs ahí.")

pdf_files = sorted(pdf_dir.glob("*.pdf"))
if not pdf_files:
    raise SystemExit(f"❌ No se encontraron archivos .pdf en '{PDF_DIR}'.")

print(f"📂 PDFs encontrados: {len(pdf_files)}")
for p in pdf_files:
    print(f"   - {p.name}")

all_chunks_with_source = []   # [(chunk_text, nombre_pdf, es_tabla), ...]
total_tablas = 0

print(f"\n📄 Extrayendo con pymupdf4llm + OCR (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
print("=" * 70)

for pdf_path in pdf_files:
    full_text, tablas_pdf = process_pdf_advanced(pdf_path)
    total_tablas += tablas_pdf

    if not full_text.strip():
        print(f"   ⚠️  {pdf_path.name}: texto vacío o no extraíble")
        continue

    file_chunks = structural_chunking(full_text)
    n_antes = len(all_chunks_with_source)

    for c in file_chunks:
        all_chunks_with_source.append((c, pdf_path.name, es_chunk_tabla(c)))

    n_tabla_chunks = sum(1 for c in file_chunks if es_chunk_tabla(c))
    print(f"   ✅ {pdf_path.name}: {len(file_chunks)} chunks "
          f"({n_tabla_chunks} con tablas) | {tablas_pdf} págs con OCR")

print("=" * 70)

# ── Deduplicar por prefijo ────────────────────────────────────────────────────
chunks_with_source, seen = [], set()
for c, src, is_table in all_chunks_with_source:
    key = c[:80]
    if key not in seen:
        seen.add(key)
        chunks_with_source.append((c, src, is_table))

chunks = [c for c, _, _ in chunks_with_source]
n_tabla_total = sum(1 for _, _, t in chunks_with_source if t)
print(f"\n✅ Total chunks únicos: {len(chunks)}  "
      f"(📊 {n_tabla_total} con tablas, 📝 {len(chunks) - n_tabla_total} texto narrativo)")
print(f"✅ Total páginas con OCR aplicado: {total_tablas}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERACIÓN DE PARES Q/A CON GROQ — prompt diferenciado tabla vs texto
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE)

SYSTEM_PROMPT_TEXTO = f"""Eres un generador experto de datasets RAFT para modelos de lenguaje.
Dado un contexto narrativo de un documento universitario, genera exactamente {PARES_POR_CHUNK} pares \
pregunta-respuesta en formato JSON.

Reglas estrictas:
- La respuesta DEBE estar contenida literalmente en el contexto proporcionado.
- Las preguntas deben ser VARIADAS y cubrir distintos aspectos del contexto:
  1. Pregunta factual directa (dato concreto)
  2. Pregunta inferencial (requiere razonar sobre el texto)
  3. Pregunta de detalle específico (cifra, fecha, nombre)
  4. Pregunta de relacion (conexion entre dos elementos del texto)
  5. Pregunta de definicion o proposito (que es, para que sirve)
- Responde UNICAMENTE con un array JSON valido, sin texto adicional, sin markdown, sin backticks.
- Formato exacto: [{{"pregunta": "...", "respuesta": "..."}}, ...]"""

SYSTEM_PROMPT_TABLA = f"""Eres un generador experto de datasets RAFT especializado en TABLAS de \
documentos universitarios (estadisticas, indicadores, cifras de acreditacion).
Dada una tabla (en formato markdown o texto extraido por OCR), genera exactamente {PARES_POR_CHUNK} \
pares pregunta-respuesta en formato JSON.

Reglas estrictas:
- Las preguntas deben referirse a VALORES ESPECIFICOS de la tabla (numeros, porcentajes, nombres, fechas).
- Genera preguntas variadas:
  1. Valor especifico de una celda
  2. Comparacion entre dos valores
  3. Total o suma de valores
  4. Valor maximo o minimo
  5. Tendencia o variacion entre periodos
- Si la tabla tiene columnas/filas claras, formula preguntas tipo "Cual es el valor de X en Y?"
- La respuesta DEBE estar contenida literalmente en la tabla proporcionada.
- Responde UNICAMENTE con un array JSON valido, sin texto adicional, sin markdown, sin backticks.
- Formato exacto: [{{"pregunta": "...", "respuesta": "..."}}, ...]"""


def generar_pares_qa(chunk: str, es_tabla: bool, forzar_8b: bool = False):
    """
    Genera pares Q/A usando Gemma 3 27B via OpenRouter (free tier).
    OpenRouter requiere headers HTTP-Referer y X-Title.
    """
    system_prompt = SYSTEM_PROMPT_TABLA if es_tabla else SYSTEM_PROMPT_TEXTO

    resp = client.chat.completions.create(
        model=MODEL_TABLA,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Contexto:\n{chunk}"},
        ],
        temperature=0.4,
        max_tokens=900,
        timeout=60,
        extra_headers={
            "HTTP-Referer": "https://eisc.univalle.edu.co",
            "X-Title": "RAFT Dataset Generator EISC",
        },
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


qa_pairs = []   # [{"contexto", "pregunta", "respuesta", "fuente", "es_tabla", "chunk_idx"}, ...]
errores = 0

# ── Checkpoint: reanudar si ya existe un progreso parcial ────────────────────
CHECKPOINT_JSON = OUTPUT_JSON.replace(".json", "_checkpoint_qa.json")
start_idx = 0
if os.path.exists(CHECKPOINT_JSON):
    with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    qa_pairs = checkpoint.get("qa_pairs", [])
    start_idx = checkpoint.get("last_idx", -1) + 1
    print(f"📂 Checkpoint encontrado: {len(qa_pairs)} pares ya generados, "
          f"retomando desde chunk {start_idx}/{len(chunks_with_source)}")


def guardar_checkpoint(idx: int):
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump({"last_idx": idx, "qa_pairs": qa_pairs}, f, ensure_ascii=False)


print(f"\nGenerando pares Q/A con Gemma 3 27B via OpenRouter (free)...\n")

MAX_RETRIES        = 5
BACKOFF_BASE       = 10    # OpenRouter free tiene rate limits — esperar más
CHECKPOINT_EVERY   = 25

for i in range(start_idx, len(chunks_with_source)):
    chunk, fuente, es_tabla = chunks_with_source[i]

    intento = 0
    while intento < MAX_RETRIES:
        try:
            pares = generar_pares_qa(chunk, es_tabla)
            count = 0
            for par in pares:
                if "pregunta" in par and "respuesta" in par:
                    qa_pairs.append({
                        "contexto":  chunk,
                        "pregunta":  par["pregunta"].strip(),
                        "respuesta": par["respuesta"].strip(),
                        "fuente":    fuente,
                        "es_tabla":  es_tabla,
                        "chunk_idx": i,
                    })
                    count += 1

            tag = "TABLA" if es_tabla else "TEXTO"
            print(f"  [{i+1:03d}/{len(chunks_with_source)}] OK {tag} {count} pares · {fuente}")
            time.sleep(1.5)   # OpenRouter free: respetar rate limit con pausa entre llamadas
            break

        except json.JSONDecodeError as e:
            errores += 1
            print(f"  [{i+1:03d}/{len(chunks_with_source)}] JSON invalido: {str(e)[:60]}")
            break

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()

            if is_rate_limit and intento < MAX_RETRIES - 1:
                espera = BACKOFF_BASE * (2 ** intento)
                print(f"  [{i+1:03d}/{len(chunks_with_source)}] Rate limit OpenRouter — esperando {espera}s (intento {intento+1}/{MAX_RETRIES})...")
                time.sleep(espera)
                intento += 1
                continue
            elif is_rate_limit:
                # Rate limit persistente — guardar y parar
                guardar_checkpoint(i - 1)
                raise SystemExit(
                    f"\nRate limit persistente en OpenRouter tras {MAX_RETRIES} reintentos.\n"
                    f"Progreso guardado: {len(qa_pairs)} pares en '{CHECKPOINT_JSON}'.\n"
                    f"Vuelve a correr el script para retomar desde el chunk {i}/{len(chunks_with_source)}."
                )
            else:
                errores += 1
                print(f"  [{i+1:03d}/{len(chunks_with_source)}] Error: {err_str[:80]}")
                time.sleep(2.0)
                break

    if (i + 1) % CHECKPOINT_EVERY == 0:
        guardar_checkpoint(i)
        print(f"  Checkpoint guardado ({i+1}/{len(chunks_with_source)} procesados, {len(qa_pairs)} pares)")

# Guardar checkpoint final y limpiar
guardar_checkpoint(len(chunks_with_source) - 1)
print(f"\nPares Q/A generados: {len(qa_pairs)}  (errores: {errores})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONSTRUCCIÓN DEL DATASET RAFT — oracle + distractores + ablation
#    Implementa el formato real de RAFT (Zhang et al., 2024):
#    - Cada ejemplo tiene k=1+N_DISTRACTORS documentos en el contexto
#    - Solo UNO es el "oracle" (contiene la respuesta real)
#    - El resto son distractores: chunks de OTROS temas/fuentes
#    - P_NO_ORACLE% de ejemplos NO incluyen el oracle (fuerza "no sé" honesto)
#    - La respuesta incluye razonamiento con cita ##begin_quote##...##end_quote##
# ─────────────────────────────────────────────────────────────────────────────
def seleccionar_distractores(idx_actual: int, fuente_actual: str,
                             todos_chunks, n: int):
    """
    Selecciona N chunks distractores, priorizando:
    1. Chunks de OTRAS fuentes (más realista — info irrelevante de otro PDF)
    2. Si no hay suficientes, chunks de la misma fuente pero lejos del actual
    """
    otros_indices = [
        j for j, (_, src, _) in enumerate(todos_chunks)
        if j != idx_actual and src != fuente_actual
    ]
    mismos_indices = [
        j for j, (_, src, _) in enumerate(todos_chunks)
        if j != idx_actual and src == fuente_actual
        and abs(j - idx_actual) > 3   # evitar chunks adyacentes (muy relacionados)
    ]

    pool = otros_indices if len(otros_indices) >= n else otros_indices + mismos_indices
    if len(pool) < n:
        pool = [j for j in range(len(todos_chunks)) if j != idx_actual]

    elegidos = random.sample(pool, min(n, len(pool)))
    return [todos_chunks[j][0] for j in elegidos]


def construir_respuesta_con_cita(respuesta: str, oracle_chunk: str) -> str:
    """
    Formatea la respuesta al estilo RAFT con razonamiento explícito y cita.
    Busca el fragmento de la respuesta dentro del oracle para citarlo literalmente.
    """
    oracle_sentences = re.split(r'(?<=[.!?])\s+', oracle_chunk)
    mejor_oracion = ""
    mejor_overlap = 0
    resp_words = set(normalize_text(respuesta).split())

    for sent in oracle_sentences:
        sent_words = set(normalize_text(sent).split())
        overlap = len(resp_words & sent_words)
        if overlap > mejor_overlap:
            mejor_overlap = overlap
            mejor_oracion = sent.strip()

    if not mejor_oracion:
        mejor_oracion = oracle_chunk[:200]

    return (
        f"Para responder esta pregunta, reviso el documento y encuentro la información relevante: "
        f"##begin_quote##{mejor_oracion}##end_quote##\n\n"
        f"Por lo tanto, la respuesta es: {respuesta}"
    )


def construir_respuesta_sin_oracle(pregunta: str) -> str:
    """Respuesta honesta cuando el oracle NO está en el contexto (ablation)."""
    return (
        "Para responder esta pregunta, reviso los documentos proporcionados, pero no encuentro "
        "información relevante que permita responder con certeza. No tengo suficiente contexto "
        "para dar una respuesta verificada sobre esto."
    )


print(f"\n🔨 Construyendo dataset RAFT final "
      f"(k={1+N_DISTRACTORS} docs/ejemplo, {N_DISTRACTORS} distractores, "
      f"{P_NO_ORACLE:.0%} sin oracle)...\n")

raft_dataset = []

for qa in qa_pairs:
    oracle_chunk = qa["contexto"]
    fuente = qa["fuente"]
    idx = qa["chunk_idx"]

    incluir_oracle = random.random() > P_NO_ORACLE
    distractores = seleccionar_distractores(idx, fuente, chunks_with_source, N_DISTRACTORS)

    if incluir_oracle:
        documentos = distractores + [oracle_chunk]
        random.shuffle(documentos)
        respuesta_final = construir_respuesta_con_cita(qa["respuesta"], oracle_chunk)
        oracle_incluido = True
    else:
        # Ablation: sin oracle, solo distractores (+1 extra para mantener k constante)
        documentos = seleccionar_distractores(idx, fuente, chunks_with_source, N_DISTRACTORS + 1)
        random.shuffle(documentos)
        respuesta_final = construir_respuesta_sin_oracle(qa["pregunta"])
        oracle_incluido = False

    raft_dataset.append({
        "pregunta":        qa["pregunta"],
        "documentos":      documentos,        # k documentos en el contexto (mezclados)
        "oracle_incluido": oracle_incluido,    # True/False — para evaluar distractor robustness
        "oracle_chunk":    oracle_chunk if oracle_incluido else None,
        "respuesta_corta": qa["respuesta"],    # respuesta gold sin formato (para EM/F1)
        "respuesta_cot":   respuesta_final,    # respuesta con razonamiento + cita (target real)
        "fuente":          fuente,
        "es_tabla":        qa["es_tabla"],
    })

random.shuffle(raft_dataset)


# ─────────────────────────────────────────────────────────────────────────────
# 6. GUARDAR DATASET
# ─────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(raft_dataset, f, ensure_ascii=False, indent=2)

# Limpiar checkpoint — ya no se necesita, el dataset final está completo
if os.path.exists(CHECKPOINT_JSON):
    os.remove(CHECKPOINT_JSON)
    print(f"🧹 Checkpoint '{CHECKPOINT_JSON}' eliminado (proceso completado)")

n_con_oracle = sum(1 for d in raft_dataset if d["oracle_incluido"])
n_sin_oracle = len(raft_dataset) - n_con_oracle
n_de_tablas  = sum(1 for d in raft_dataset if d["es_tabla"])

print(f"""
╔════════════════════════════════════════════════════╗
║  Dataset RAFT v3 generado                          ║
║  PDFs procesados        : {len(pdf_files):<24}║
║  Chunks unicos          : {len(chunks):<24}║
║  Chunks con tablas      : {n_tabla_total:<24}║
║  Paginas con OCR        : {total_tablas:<24}║
║  Total ejemplos RAFT    : {len(raft_dataset):<24}║
║  Con oracle document    : {n_con_oracle:<24}║
║  Sin oracle (40% abla.) : {n_sin_oracle:<24}║
║  Provenientes de tablas : {n_de_tablas:<24}║
║  Distractores/ejemplo   : {N_DISTRACTORS:<24}║
║  Pares por chunk        : {PARES_POR_CHUNK:<24}║
║  Chunk size             : {CHUNK_SIZE:<24}║
║  P_NO_ORACLE            : {P_NO_ORACLE:<24}║
║  Errores generacion     : {errores:<24}║
║  Archivo                : {OUTPUT_JSON:<24}║
╚════════════════════════════════════════════════════╝
""")

# Desglose por fuente
por_fuente = Counter(item["fuente"] for item in raft_dataset)
print("📊 Ejemplos RAFT por PDF:")
for fuente, n in por_fuente.items():
    print(f"   - {fuente}: {n} ejemplos")

# Muestra 2 ejemplos completos (uno con oracle, uno sin)
print("\n📋 Ejemplo CON oracle:")
ejemplo_con = next((d for d in raft_dataset if d["oracle_incluido"]), None)
if ejemplo_con:
    print(f"  Pregunta    : {ejemplo_con['pregunta']}")
    print(f"  # documentos: {len(ejemplo_con['documentos'])}")
    print(f"  Respuesta CoT: {ejemplo_con['respuesta_cot'][:250]}...")

print("\n📋 Ejemplo SIN oracle (ablation):")
ejemplo_sin = next((d for d in raft_dataset if not d["oracle_incluido"]), None)
if ejemplo_sin:
    print(f"  Pregunta    : {ejemplo_sin['pregunta']}")
    print(f"  # documentos: {len(ejemplo_sin['documentos'])}")
    print(f"  Respuesta   : {ejemplo_sin['respuesta_cot'][:250]}...")

print(f"\n🚀 Listo para fine-tuning RAFT. Próximo paso: convertir a ChatML para SFTTrainer.")
