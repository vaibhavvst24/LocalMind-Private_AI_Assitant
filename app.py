"""
Offline Local AI Assistant — Streamlit front end.

Run with:  streamlit run app.py
Requires:  Ollama running locally (https://ollama.com) with at least one
           chat model and the `nomic-embed-text` embedding model pulled.
"""

import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config, memory
from src.llm_client import OllamaClient, OllamaConnectionError
from src.rag_engine import DocumentStore

import streamlit as st

st.set_page_config(
    page_title="LocalMind — Private AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium UI theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200&icon_names=auto_awesome,chat,description,delete,download,folder_open,bolt,check_circle,cloud_off,settings,storage,analytics,science">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080A0D;
        --panel: #101318;
        --panel-2: #141820;
        --panel-3: #191E27;
        --border: rgba(255,255,255,.08);
        --border-strong: rgba(192,192,192,.9);
        --text: #F4F7FA;
        --muted: #8C96A5;
        --accent: #8B7CFF;
        --accent-2: #6EA8FE;
        --success: #45D483;
        --danger: #FF6B6B;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 48% -4%, rgba(139,124,255,.10), transparent 45%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8,10,13,.75);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1015 0%, #090B0E 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.2rem;
        padding-bottom: 7rem;
    }

    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 22px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }

    /* Brand */
    .brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0 0 2.4rem 0;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-icon {
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: linear-gradient(135deg, #8B7CFF, #6EA8FE);
        color: white;
        box-shadow: 0 8px 28px rgba(139,124,255,.22);
    }

    .brand-name {
        font-size: 18px;
        font-weight: 750;
        letter-spacing: -.4px;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 11px;
        margin-top: 1px;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 2.8rem 0 2rem;
    }

    .hero-icon {
        display: inline-flex;
        width: 62px;
        height: 62px;
        align-items: center;
        justify-content: center;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(139,124,255,.22), rgba(110,168,254,.13));
        border: 1px solid rgba(139,124,255,.22);
        color: #B9B0FF;
        margin-bottom: 1.15rem;
        box-shadow: 0 12px 50px rgba(139,124,255,.10);
    }

    .hero h1 {
        font-size: clamp(30px, 4vw, 46px);
        line-height: 1.1;
        letter-spacing: -1.8px;
        margin: 0;
        font-weight: 750;
        color: var(--text);
    }

    .hero p {
        max-width: 620px;
        margin: .9rem auto 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 1.1rem;
        padding: 6px 11px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,.025);
        color: #B9C1CC;
        font-size: 11px;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 12px rgba(69,212,131,.55);
    }

    /* Cards */
    .ui-card {
        background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.018));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
    }

    .feature-card {
        min-height: 112px;
        transition: border-color .2s ease, transform .2s ease;
        margin-bottom: 15px
    }

    .feature-card:hover {
        border-color: var(--border-strong);
        transform: translateY(-2px);
    }

    .feature-icon {
        color: #AFA7FF;
        margin-bottom: 11px;
    }

    .feature-title {
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .feature-text {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.55;
    }

    .section-label {
        color: #737D8C;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 1.1rem 0 .55rem;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px 16px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .8px;
        font-weight: 700;
    }

    .metric-value {
        color: var(--text);
        font-size: 24px;
        font-weight: 750;
        margin-top: 5px;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: 0;
        padding-top: .7rem;
        padding-bottom: .7rem;
    }

    [data-testid="stChatMessageContent"] {
        max-width: 820px;
        line-height: 1.75;
        font-size: 14px;
    }

    [data-testid="stChatInput"] {
        border-top: 0 !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] > div {
        background: #11151B !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 45px rgba(0,0,0,.28);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Manrope', sans-serif !important;
        font-weight: 650 !important;
        color: #778190 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text) !important;
    }

    div[data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #8B7CFF, #6EA8FE) !important;
    }

    /* Inputs */
    .stSelectbox > div > div,
    .stTextArea textarea,
    .stTextInput input {
        background: #11151B !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        font-weight: 650 !important;
        transition: all .2s ease !important;
    }

    .stButton > button:hover {
        border-color: rgba(139,124,255,.45) !important;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px 15px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border) !important;
    }

    /* Sidebar */
    .sidebar-brand {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 20px;
    }

    .sidebar-brand-icon {
        color: #AFA7FF;
        font-size: 25px;
    }

    .sidebar-brand-title {
        font-size: 15px;
        font-weight: 750;
    }

    .sidebar-brand-sub {
        color: var(--muted);
        font-size: 10px;
    }

    .connection {
        padding: 11px 12px;
        border-radius: 11px;
        border: 1px solid rgba(69,212,131,.16);
        background: rgba(69,212,131,.045);
        color: #B8EBCB;
        font-size: 11px;
        font-weight: 650;
    }

    .connection.offline {
        border-color: rgba(255,107,107,.16);
        background: rgba(255,107,107,.045);
        color: #FFB2B2;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main brand / hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">
            <span class="material-symbols-outlined" style="font-size:32px;">auto_awesome</span>
        </div>
        <h1>Offline Local AI Assistant</h1>
        <p>
            A private AI workspace powered by local models.
            Chat, search your documents, benchmark models, and evaluate quality —
            without sending your data to the cloud.
        </p>
        <div class="status-pill">
            <span class="status-dot"></span>
            100% Local · Private by Design
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_ollama_client() -> OllamaClient:
    return OllamaClient()


@st.cache_resource
def get_doc_store() -> DocumentStore:
    return DocumentStore()


client = get_ollama_client()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    memory.ensure_session(st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = memory.load_history(st.session_state.session_id)

# ---------------------------------------------------------------------------
# Sidebar — connection status, model selection, settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="material-symbols-outlined sidebar-brand-icon">auto_awesome</span>
            <div>
                <div class="sidebar-brand-title">LocalMind</div>
                <div class="sidebar-brand-sub">Private AI workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    connected = client.health_check()
    if connected:
        st.markdown('<div class="connection">● &nbsp; Ollama connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection offline">● &nbsp; Ollama not reachable</div>', unsafe_allow_html=True)
        st.caption(
            f"Expected at `{config.OLLAMA_HOST}`. Run `ollama serve` "
            "(or open the Ollama app) and refresh."
        )

    st.divider()
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_label = st.selectbox(
        "Chat model",
        options=list(config.MODEL_OPTIONS.keys()),
        index=list(config.MODEL_OPTIONS.keys()).index(config.DEFAULT_MODEL_LABEL),
    )
    model_info = config.MODEL_OPTIONS[model_label]
    model_tag = model_info["tag"]

    if connected:
        try:
            pulled = client.is_model_pulled(model_tag)
        except OllamaConnectionError:
            pulled = False
        if pulled:
            st.caption(f"✅ `{model_tag}` is pulled ({model_info['params']}, ~{model_info['approx_ram_gb']} GB RAM)")
        else:
            st.warning(f"Not pulled yet. Run: `ollama pull {model_tag}`")

    temperature = st.slider("Temperature", 0.0, 1.0, config.DEFAULT_TEMPERATURE, 0.05)
    max_tokens = st.slider("Max response tokens", 64, 1024, config.DEFAULT_MAX_TOKENS, 64)

    st.divider()
    st.markdown('<div class="section-label">Knowledge / RAG</div>', unsafe_allow_html=True)
    use_rag = st.toggle("Ground answers in my documents", value=False)
    doc_store = get_doc_store()
    n_chunks = 0
    try:
        n_chunks = doc_store.count()
    except Exception:
        pass
    st.caption(f"{n_chunks} chunks indexed across {len(doc_store.list_sources()) if n_chunks else 0} document(s)")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        memory.delete_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        memory.ensure_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    st.caption("Runs 100% locally. No API calls, no data leaves your machine.")

    st.markdown(
    """
    <div style="
        text-align:center;
        color:#808080;
        font-size:15px;
        padding:2.5rem 0 1rem;
        border-top:1px solid rgba(255,255,255,.05);
        margin-top:3rem;">
        Made by Vaibhav 
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_docs, tab_bench, tab_eval = st.tabs(
    ["💬 Chat", "📄 Documents", "📊 Benchmark", "✅ Evaluation"]
)

# ---------------------------------------------------------------------------
# CHAT TAB
# ---------------------------------------------------------------------------
with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="hero" style="padding-top:2.2rem;padding-bottom:1.1rem;">
                <div class="hero-icon" style="width:50px;height:50px;border-radius:16px;">
                    <span class="material-symbols-outlined">chat</span>
                </div>
                <h1 style="font-size:30px;">What can I help you with?</h1>
                <p>Ask a question, analyze an idea, or enable RAG to chat with your own documents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(
                '<div class="ui-card feature-card"><div class="feature-icon">📄</div>'
                '<div class="feature-title">Ask your documents</div>'
                '<div class="feature-text">Enable RAG and get responses of your given Query.</div></div>',
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                '<div class="ui-card feature-card"><div class="feature-icon">⚡</div>'
                '<div class="feature-title">Run locally</div>'
                '<div class="feature-text">Inference happens through Ollama on your own machine.</div></div>',
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                '<div class="ui-card feature-card"><div class="feature-icon">📊</div>'
                '<div class="feature-title">Measure performance</div>'
                '<div class="feature-text">Compare model speed and evaluate answer quality offline.</div></div>',
                unsafe_allow_html=True,
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Message the assistant...")

    if prompt:
        if not connected:
            st.error("Ollama isn't reachable — start it and try again.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            memory.save_message(st.session_state.session_id, "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build the message list sent to the model
            retrieved_hits = []
            system_prompt = config.SYSTEM_PROMPT_BASE
            user_content = prompt

            if use_rag:
                if n_chunks == 0:
                    st.warning(
                        "RAG is on but no documents are indexed yet. "
                        "Upload some in the Documents tab, or turn RAG off."
                    )
                else:
                    from src.rag_engine import build_context_block

                    retrieved_hits = doc_store.query(prompt, top_k=config.TOP_K)
                    context_block = build_context_block(retrieved_hits)
                    system_prompt = config.SYSTEM_PROMPT_RAG
                    user_content = (
                        f"Context:\n{context_block}\n\nQuestion: {prompt}"
                    )

            api_messages = [{"role": "system", "content": system_prompt}]
            # include prior turns (not the RAG-augmented version) for continuity
            for m in st.session_state.messages[:-1]:
                api_messages.append(m)
            api_messages.append({"role": "user", "content": user_content})

            with st.chat_message("assistant"):
                if retrieved_hits:
                    with st.expander("📎 Retrieved context", expanded=False):
                        for h in retrieved_hits:
                            st.markdown(f"**{h['source']}** (distance: {h['distance']:.3f})")
                            st.caption(h["text"][:400] + ("..." if len(h["text"]) > 400 else ""))

                placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in client.chat(
                        api_messages,
                        model=model_tag,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    ):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                    stats = getattr(client, "last_stats", {})
                    tps = stats.get("tokens_per_sec")
                    total_s = stats.get("total_seconds")
                    caption_bits = []
                    if total_s:
                        caption_bits.append(f"{total_s}s")
                    if tps:
                        caption_bits.append(f"{tps} tok/s")
                    if caption_bits:
                        st.markdown(
                            '<div style="color:#7F8998;font-size:10px;margin-top:5px;">'
                            '⚡ ' + " · ".join(caption_bits) + ' · Local inference'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                except OllamaConnectionError as e:
                    full_response = f"⚠️ {e}"
                    placeholder.error(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            memory.save_message(
                st.session_state.session_id, "assistant", full_response
            )

# ---------------------------------------------------------------------------
# DOCUMENTS TAB
# ---------------------------------------------------------------------------
with tab_docs:
    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)
    st.subheader("Build your local knowledge base")
    st.caption(
        "Files are chunked, embedded locally with "
        f"`{config.EMBED_MODEL_TAG}`, and stored in a local Chroma vector "
        "database. Nothing is uploaded anywhere."
    )

    uploaded = st.file_uploader(
        "Upload .txt, .md, or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )

    if uploaded and st.button("📥 Ingest documents"):
        if not connected:
            st.error("Ollama isn't reachable — start it first.")
        else:
            progress = st.progress(0.0, text="Starting...")
            for i, f in enumerate(uploaded):
                tmp_path = config.DOCS_DIR / f.name
                tmp_path.write_bytes(f.getbuffer())
                progress.progress((i) / len(uploaded), text=f"Embedding {f.name}...")
                try:
                    n = doc_store.add_document(tmp_path)
                    st.success(f"Indexed {f.name}: {n} chunks")
                except Exception as e:
                    st.error(f"Failed on {f.name}: {e}")
                progress.progress((i + 1) / len(uploaded))
            progress.empty()
            st.rerun()

    st.divider()
    st.subheader("Indexed documents")
    sources = doc_store.list_sources()
    if not sources:
        st.caption("No documents indexed yet.")
    else:
        for s in sources:
            col1, col2 = st.columns([5, 1])
            col1.write(f"📄 {s}")
            if col2.button("Remove", key=f"rm_{s}"):
                doc_store.delete_source(s)
                st.rerun()

        if st.button("🗑️ Clear entire index", type="secondary"):
            doc_store.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# BENCHMARK TAB
# ---------------------------------------------------------------------------
with tab_bench:
    st.markdown('<div class="section-label">Performance Lab</div>', unsafe_allow_html=True)
    st.subheader("Model performance & tradeoffs")
    st.caption(
        "This is the centerpiece for a resume writeup: show the tradeoff "
        "between model size, RAM footprint, and real tokens/sec on YOUR "
        "hardware."
    )

    df = pd.DataFrame(
        [
            {
                "Model": label,
                "Params": info["params"],
                "Default Quant": info["default_quant"],
                "Approx RAM (GB)": info["approx_ram_gb"],
                "Ollama tag": info["tag"],
            }
            for label, info in config.MODEL_OPTIONS.items()
        ]
    )
    total_models = len(df)
    avg_ram = df["Approx RAM (GB)"].mean() if total_models else 0
    bm1, bm2, bm3 = st.columns(3)
    with bm1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Available models</div>'
            f'<div class="metric-value">{total_models}</div></div>',
            unsafe_allow_html=True,
        )
    with bm2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Average RAM</div>'
            f'<div class="metric-value">{avg_ram:.1f} GB</div></div>',
            unsafe_allow_html=True,
        )
    with bm3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Benchmark mode</div>'
            f'<div class="metric-value" style="font-size:18px;">Local</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    bench_prompt = st.text_area(
        "Benchmark prompt",
        value="Explain what a vector database is in three sentences.",
    )
    selected_for_bench = st.multiselect(
        "Models to benchmark (must already be pulled)",
        options=list(config.MODEL_OPTIONS.keys()),
        default=[model_label],
    )

    if st.button("▶️ Run benchmark"):
        if not connected:
            st.error("Ollama isn't reachable — start it first.")
        else:
            results = []
            bar = st.progress(0.0)
            for i, label in enumerate(selected_for_bench):
                tag = config.MODEL_OPTIONS[label]["tag"]
                try:
                    r = client.chat_sync(
                        [{"role": "user", "content": bench_prompt}],
                        model=tag,
                    )
                    results.append(
                        {
                            "Model": label,
                            "Total time (s)": r["total_seconds"],
                            "Tokens generated": r["eval_count"],
                            "Tokens/sec": r["tokens_per_sec"],
                        }
                    )
                except OllamaConnectionError as e:
                    results.append(
                        {"Model": label, "Total time (s)": None, "Tokens generated": None, "Tokens/sec": f"error: {e}"}
                    )
                bar.progress((i + 1) / max(len(selected_for_bench), 1))

            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            numeric = result_df.dropna(subset=["Tokens/sec"]).copy()
            if not numeric.empty and pd.api.types.is_numeric_dtype(numeric["Tokens/sec"]):
                st.bar_chart(numeric.set_index("Model")["Tokens/sec"])

# ---------------------------------------------------------------------------
# EVALUATION TAB
# ---------------------------------------------------------------------------
with tab_eval:
    st.markdown('<div class="section-label">Evaluation Lab</div>', unsafe_allow_html=True)
    st.subheader("Answer quality evaluation")
    st.caption(
        "Runs a small hand-written Q&A set against the selected model and "
        "scores answers by keyword overlap with a reference answer — simple, "
        "fully offline, and good enough to show a directional accuracy number "
        "in a resume writeup or README."
    )

    import json

    if config.EVAL_QUESTIONS_PATH.exists():
        eval_set = json.loads(config.EVAL_QUESTIONS_PATH.read_text())
        st.caption(f"{len(eval_set)} questions loaded from `eval/eval_questions.json`")
    else:
        eval_set = []
        st.warning("No eval_questions.json found.")

    eval_model_label = st.selectbox(
        "Model to evaluate", options=list(config.MODEL_OPTIONS.keys()), key="eval_model"
    )
    use_rag_eval = st.checkbox("Use RAG context for evaluation", value=False)

    if st.button("▶️ Run evaluation") and eval_set:
        if not connected:
            st.error("Ollama isn't reachable — start it first.")
        else:
            from eval.run_eval import run_evaluation

            with st.spinner("Running evaluation..."):
                results_df, avg_score, avg_latency = run_evaluation(
                    client=client,
                    doc_store=doc_store if use_rag_eval else None,
                    model_tag=config.MODEL_OPTIONS[eval_model_label]["tag"],
                    eval_set=eval_set,
                )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Average score</div>'
                    f'<div class="metric-value">{avg_score:.0%}</div></div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Average latency</div>'
                    f'<div class="metric-value">{avg_latency:.2f}s</div></div>',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Questions evaluated</div>'
                    f'<div class="metric-value">{len(results_df)}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # persist results for the README / portfolio writeup
            out_path = config.EVAL_RESULTS_DIR / f"{eval_model_label.replace(' ', '_')}.csv"
            results_df.to_csv(out_path, index=False)
            st.caption(f"Saved to `{out_path.relative_to(config.BASE_DIR)}`")


# ---------------------------------------------------------------------------
# App footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="
        text-align:center;
        color:#596272;
        font-size:15px;
        padding:2.5rem 0 1rem;
        border-top:1px solid rgba(255,255,255,.05);
        margin-top:3rem;">
        ✦ LocalMind &nbsp;·&nbsp; Private AI &nbsp;·&nbsp; Powered by Ollama
    </div>
    """,
    unsafe_allow_html=True,
)
