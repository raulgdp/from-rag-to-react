# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita · Sys-D LangGraph — Frontend Streamlit                       ║
# ║  Interfaz WhatsApp-style para el grafo de estados CNA/EISC                 ║
# ║  Comando: streamlit run langgraph_chat.py                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import streamlit as st
import time, html
from pathlib import Path

# ── Importar el motor LangGraph ───────────────────────────────────────────────
try:
    from langgraph_acredita import answer_with_langgraph, GENERATOR_MODEL, GRADER_MODEL
    BACKEND_OK = True
except ImportError as e:
    BACKEND_OK = False
    BACKEND_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatAcredita · LangGraph",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — ESTILO WHATSAPP DARK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fondo general ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b141a !important;
    color: #e9edef;
}
[data-testid="stSidebar"] {
    background-color: #111b21 !important;
}
[data-testid="stSidebar"] * { color: #e9edef !important; }

/* ── Input de chat ─────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background-color: #1f2c33 !important;
    color: #e9edef !important;
    border: 1px solid #2a3942 !important;
    border-radius: 8px !important;
}

/* ── Burbujas usuario ──────────────────────────────────────────── */
[data-testid="stChatMessageContent"]:has(p) {
    background: none;
}
.wa-user {
    background-color: #005c4b;
    color: #e9edef;
    border-radius: 12px 2px 12px 12px;
    padding: 10px 14px;
    margin: 4px 0 4px 18%;
    max-width: 80%;
    float: right;
    clear: both;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
}

/* ── Burbujas bot ──────────────────────────────────────────────── */
.wa-bot {
    background-color: #1f2c33;
    color: #e9edef;
    border-radius: 2px 12px 12px 12px;
    padding: 10px 14px;
    margin: 4px 18% 4px 0;
    max-width: 80%;
    float: left;
    clear: both;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
}

/* ── Traza de nodos ────────────────────────────────────────────── */
.wa-trace {
    background-color: #12232b;
    border-left: 3px solid #00a884;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 6px 0 4px 0;
    font-size: 12px;
    color: #8696a0;
    clear: both;
}
.wa-step       { color: #00a884; }
.wa-step.done  { color: #53bdeb; }
.wa-step.warn  { color: #ffa500; }
.wa-step.error { color: #ff6b6b; }

/* ── Fuentes ───────────────────────────────────────────────────── */
.wa-sources {
    font-size: 11px;
    color: #8696a0;
    margin-top: 6px;
    clear: both;
}
.wa-src-pill {
    display: inline-block;
    background: #2a3942;
    border-radius: 10px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 10px;
}

/* ── Timestamp ─────────────────────────────────────────────────── */
.wa-time {
    font-size: 10px;
    color: #8696a0;
    text-align: right;
    margin-top: 4px;
}

/* ── Header ────────────────────────────────────────────────────── */
.wa-header {
    background: #202c33;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.wa-avatar {
    width: 40px; height: 40px;
    background: #00a884;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.wa-header-text h3 { color: #e9edef; font-size: 15px; margin: 0; }
.wa-header-text p  { color: #8696a0; font-size: 12px; margin: 0; }

/* ── Clearfix ──────────────────────────────────────────────────── */
.cf::after { content: ""; display: table; clear: both; }

/* ── Métricas ──────────────────────────────────────────────────── */
.metric-row {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin: 6px 0; clear: both;
}
.metric-pill {
    background: #2a3942;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 11px;
    color: #53bdeb;
}

/* ── Info box nodos ────────────────────────────────────────────── */
.node-badge {
    display: inline-block;
    background: #00a88422;
    border: 1px solid #00a88466;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 10px;
    color: #00a884;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 ChatAcredita PRO")
    st.markdown("**Sys-D · LangGraph**")
    st.divider()

    st.markdown("### 🔗 Estado del sistema")
    if BACKEND_OK:
        st.success("✅ Motor LangGraph listo")
        st.caption(f"Generador: `{GENERATOR_MODEL[:30]}`")
        st.caption(f"Evaluador: `{GRADER_MODEL[:30]}`")
    else:
        st.error(f"❌ Error: {BACKEND_ERR}")
        st.info("Verifica que `langgraph_acredita.py` esté en el mismo directorio.")

    st.divider()
    st.markdown("### 📊 Grafo de estados")
    st.markdown("""
<div style="font-size:12px; color:#8696a0; line-height:2">
<span class="node-badge">retrieve</span> →
<span class="node-badge">grade_docs</span><br>
&nbsp;&nbsp;✓ → <span class="node-badge">generate</span> →
<span class="node-badge">grade_answer</span><br>
&nbsp;&nbsp;✗ → <span class="node-badge">rewrite</span> → retrieve<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
supported → <b>END</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
unsupported → rewrite
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Configuración")
    max_iter = st.slider("Máx. iteraciones", 1, 5, 3)

    st.divider()
    st.markdown("### 📈 Sesión actual")
    st.metric("Consultas", st.session_state.total_queries)
    if st.session_state.total_queries > 0:
        avg_lat = st.session_state.total_latency / st.session_state.total_queries
        st.metric("Latencia media", f"{avg_lat:.1f}s")

    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_latency = 0.0
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ÁREA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="wa-header">
  <div class="wa-avatar">🎓</div>
  <div class="wa-header-text">
    <h3>ChatAcredita PRO · LangGraph</h3>
    <p>Sys-D · Grafo de estados determinista · CNA/EISC · Universidad del Valle</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Renderizar historial ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]

    if role == "user":
        st.markdown(f"""
<div class="cf">
  <div class="wa-user">{html.escape(msg['content'])}</div>
</div>""", unsafe_allow_html=True)

    elif role == "assistant":
        # Traza de nodos
        if msg.get("trace"):
            trace_html = "<br>".join(
                f'<span class="wa-step">{html.escape(str(t))}</span>'
                for t in msg["trace"]
            )
            st.markdown(f'<div class="wa-trace">{trace_html}</div>',
                        unsafe_allow_html=True)

        # Métricas rápidas
        metrics_html = ""
        if msg.get("latency_s"):
            metrics_html += f'<span class="metric-pill">⏱ {msg["latency_s"]:.1f}s</span>'
        if msg.get("iterations") is not None:
            metrics_html += f'<span class="metric-pill">🔄 {msg["iterations"]+1} iteración(es)</span>'
        if msg.get("doc_grade"):
            icon = "✅" if msg["doc_grade"] == "relevant" else "⚠️"
            metrics_html += f'<span class="metric-pill">{icon} docs: {msg["doc_grade"]}</span>'
        if msg.get("answer_grade"):
            icon = "✅" if msg["answer_grade"] == "supported" else "⚠️"
            metrics_html += f'<span class="metric-pill">{icon} answer: {msg["answer_grade"]}</span>'
        if metrics_html:
            st.markdown(f'<div class="metric-row">{metrics_html}</div>',
                        unsafe_allow_html=True)

        # Burbuja respuesta — escape primero, luego restaurar saltos de línea
        content_raw = msg.get("content", "") or "⚠️ Sin respuesta"
        answer_safe = html.escape(content_raw).replace("\n", "<br>")
        # Convertir **negrita** a <b> después del escape
        import re as _re
        answer_safe = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', answer_safe)
        ts = html.escape(msg.get("timestamp", ""))
        st.markdown(f"""
<div class="cf">
  <div class="wa-bot">
    {answer_safe}
    <div class="wa-time">{ts}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Fuentes
        if msg.get("sources"):
            pills = "".join(
                f'<span class="wa-src-pill">📄 {html.escape(s)}</span>'
                for s in list(dict.fromkeys(msg["sources"]))[:6]
            )
            st.markdown(f'<div class="wa-sources">{pills}</div>',
                        unsafe_allow_html=True)

# ── Preguntas de ejemplo ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**💬 Preguntas de ejemplo:**")
    examples = [
        "¿Cuáles son los diez factores del modelo de acreditación CNA?",
        "¿Qué establece el Decreto 1330 de 2019 para programas académicos?",
        "¿Cuáles son las etapas del proceso de autoevaluación de la EISC?",
        "¿Qué fortalezas identificó el CNA en el último informe de pares?",
        "¿Cómo se compara la productividad investigativa de la EISC?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex[:55] + "…" if len(ex) > 55 else ex,
                         key=f"ex_{i}", use_container_width=True):
                st.session_state._pending_query = ex
                st.rerun()

# ── Procesar pregunta de ejemplo pendiente ────────────────────────────────────
if hasattr(st.session_state, "_pending_query"):
    pending = st.session_state._pending_query
    del st.session_state._pending_query
    st.session_state.messages.append({"role": "user", "content": pending})

    with st.spinner("🔄 Ejecutando grafo LangGraph…"):
        if BACKEND_OK:
            import langgraph_acredita as lg
            lg.MAX_ITERATIONS = max_iter
            result = answer_with_langgraph(pending)
        else:
            result = {"answer": f"⚠️ Backend no disponible: {BACKEND_ERR}",
                      "trace": [], "sources": [], "latency_s": 0,
                      "iterations": 0, "doc_grade": "", "answer_grade": ""}

    st.session_state.total_queries += 1
    st.session_state.total_latency += result.get("latency_s", 0)

    ts = time.strftime("%H:%M")
    answer_text = (result.get("answer") or "").strip()
    if not answer_text:
        answer_text = "⚠️ El sistema no generó respuesta. Intenta reformular la pregunta."

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer_text,
        "trace":        result.get("trace", []),
        "sources":      result.get("sources", []),
        "latency_s":    result.get("latency_s", 0),
        "iterations":   result.get("iterations", 0),
        "doc_grade":    result.get("doc_grade", ""),
        "answer_grade": result.get("answer_grade", ""),
        "timestamp":    ts,
    })
    st.rerun()

# ── Input de chat ─────────────────────────────────────────────────────────────
if query := st.chat_input("Escribe tu pregunta sobre acreditación CNA/EISC…"):
    st.session_state.messages.append({"role": "user", "content": query})

    trace_placeholder = st.empty()
    trace_lines = []

    def update_trace(line: str):
        trace_lines.append(line)
        trace_html = "<br>".join(
            f'<span class="wa-step">{html.escape(l)}</span>'
            for l in trace_lines
        )
        trace_placeholder.markdown(
            f'<div class="wa-trace">{trace_html}</div>',
            unsafe_allow_html=True
        )

    update_trace("🚀 Iniciando grafo LangGraph…")

    with st.spinner(""):
        if BACKEND_OK:
            import langgraph_acredita as lg
            lg.MAX_ITERATIONS = max_iter

            # Ejecutar con streaming de traza simulado
            update_trace("🔍 Nodo: retrieve — buscando fragmentos relevantes…")
            result = answer_with_langgraph(query)

            # Mostrar traza real del grafo
            for t in result.get("trace", []):
                update_trace(t)
        else:
            result = {"answer": f"⚠️ Backend no disponible: {BACKEND_ERR}",
                      "trace": [], "sources": [], "latency_s": 0,
                      "iterations": 0, "doc_grade": "", "answer_grade": ""}

    trace_placeholder.empty()

    st.session_state.total_queries += 1
    st.session_state.total_latency += result.get("latency_s", 0)

    ts = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role":         "assistant",
        "content":      result.get("answer", "Sin respuesta"),
        "trace":        result.get("trace", []),
        "sources":      result.get("sources", []),
        "latency_s":    result.get("latency_s", 0),
        "iterations":   result.get("iterations", 0),
        "doc_grade":    result.get("doc_grade", ""),
        "answer_grade": result.get("answer_grade", ""),
        "timestamp":    ts,
    })
    st.rerun()
