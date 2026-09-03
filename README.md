# Offline Local AI Assistant

A retrieval-augmented AI assistant that runs **entirely on your own machine**
— no API keys, no cloud calls, no per-token cost. Powered by small
open-weight language models (1.5B–4B parameters) served locally through
[Ollama](https://ollama.com), with a Streamlit UI for chat, document
grounding (RAG), model benchmarking, and quality evaluation.

Built as a portfolio project to demonstrate resource-constrained LLM
engineering: quantization tradeoffs, retrieval pipelines, and offline
inference — not just "call an API."

---

## Live App

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red)](https://he2cdeudepvqgtpaiqjmkm.streamlit.app/)

---

## Features

- **💬 Chat** — streaming responses, conversation memory persisted to SQLite
  across restarts.
  
- **📄 Documents / RAG** — upload PDF/TXT/MD files; they're chunked, embedded
  locally, and stored in a persistent Chroma vector database. Answers can be
  grounded in your own documents with source citations.
  
- **📊 Benchmark** — compare tokens/sec, latency, and RAM footprint across
  five small models (1.5B–3.8B params) on your actual hardware.
  
- **✅ Evaluation** — run a hand-written Q&A set against any model and get a
  quality score (keyword-overlap against reference answers) plus latency,
  fully offline.

---

## Architecture

```
┌─────────────────┐
│   Streamlit UI   │  app.py
└────────┬─────────┘
         │
    ┌────┴────┬───────────────┬───────────────┐
    ▼         ▼               ▼               ▼
 llm_client  rag_engine      memory         eval/run_eval
 (Ollama     (chunk, embed,  (SQLite chat   (Q&A scoring
  REST API)   Chroma store)   history)       harness)
    │             │
    ▼             ▼
┌─────────────────────┐
│   Ollama (local)     │  serves chat model + nomic-embed-text
│   localhost:11434     │  handles GGUF quantization/inference
└─────────────────────┘
```

Both the chat model and the embedding model are served by Ollama, so the
entire pipeline — generation *and* retrieval — runs offline with a single
local dependency.

---

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) (macOS, Windows, Linux) and
make sure it's running:

```bash
ollama serve   # or just open the Ollama app — it usually runs in the background
```

### 2. Pull a chat model and the embedding model

Pick at least one chat model. Smaller = faster, less accurate; larger = slower,
more capable. `phi3.5:3.8b` is a good default.

```bash
ollama pull phi3.5:3.8b
ollama pull nomic-embed-text     # required for RAG — do this regardless
```

Optional, for the benchmark tab to have something to compare:

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen2.5:1.5b
```

### 3. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Two sample documents are included in
`data/sample_docs/` — upload them in the Documents tab to try RAG
immediately (they cover local LLMs and RAG architecture, so you can ask the
assistant to explain its own architecture).

---

## Running evaluation from the CLI

```bash
python -m eval.run_eval --model phi3.5:3.8b
python -m eval.run_eval --model phi3.5:3.8b --rag   # grounded in indexed docs
```

Results are saved as CSVs in `eval/results/` — useful for pasting a
benchmark table straight into a portfolio writeup.

---

## Benchmark methodology (fill in after you run it)

This is the part that actually differentiates the project. Run the
Benchmark and Evaluation tabs on your own hardware, then record results
here:

| Model | Params | Quant | RAM (GB) | Tokens/sec | Eval score |
|---|---|---|---|---|---|
| Qwen2.5 (1.5B) | 1.5B | Q4_K_M | ~1.0 | _fill in_ | _fill in_ |
| Gemma2 (2B) | 2B | Q4_0 | ~1.6 | _fill in_ | _fill in_ |
| Qwen2.5 (3B) | 3B | Q4_K_M | ~2.0 | _fill in_ | _fill in_ |
| Llama 3.2 (3B) | 3B | Q4_K_M | ~2.0 | _fill in_ | _fill in_ |
| Phi-3.5 mini (3.8B) | 3.8B | Q4_K_M | ~2.4 | _fill in_ | _fill in_ |

Hardware tested on: _fill in your machine, e.g. "MacBook Air M2, 16GB RAM"_

---

## Project structure

```
local-llm-assistant/
├── app.py                      # Streamlit UI (chat, docs, benchmark, eval tabs)
├── requirements.txt
├── src/
│   ├── config.py                # model options, paths, prompts
│   ├── llm_client.py            # Ollama REST wrapper (chat, streaming, embeddings)
│   ├── rag_engine.py            # chunking, Chroma vector store, retrieval
│   └── memory.py                # SQLite conversation history
├── eval/
│   ├── eval_questions.json      # hand-written Q&A test set
│   ├── run_eval.py              # scoring harness (keyword overlap + latency)
│   └── results/                 # CSV outputs land here
├── data/
│   └── sample_docs/             # demo documents for RAG
└── .streamlit/config.toml       # theme
```

---

## Design decisions worth mentioning in an interview

- **Ollama over raw llama.cpp bindings**: trades a small amount of
  low-level control for reliable quantization handling, model management,
  and a stable REST API — a defensible pragmatic choice, and one worth
  being able to argue the other side of (llama.cpp gives you more control
  over sampling internals and no server dependency).
  
- **Chroma over FAISS**: Chroma persists to disk out of the box and has a
  simpler metadata-filtering API, which mattered more here than FAISS's
  raw speed advantage at this data scale (thousands, not millions, of
  vectors).
  
- **Keyword-overlap eval over LLM-as-judge**: keeps the eval harness
  dependency-free and deterministic. The README calls out explicitly that
  this is a simplification — an LLM-judge using the same local model would
  be a natural next step and is worth mentioning as a limitation, not
  hiding it.
  
- **SQLite for chat memory**: no server process, ships as a single file,
  more than sufficient for a single-user local assistant.

---

## Possible extensions

- Add an LLM-as-judge scoring mode (using the same local model) for a
  more nuanced eval than keyword overlap.
- Add reranking of retrieved chunks before they're inserted into the prompt.
- Add LoRA fine-tuning on a narrow task (see the "fine-tuned SLM" project
  variant) and compare against the base model in the Benchmark tab.
- Swap Ollama for raw `llama.cpp` bindings to demonstrate lower-level
  control over sampling and quantization.

---

## 👨‍💻 About Me

**Vaibhav Singh Bains**  
*Aspiring Data Scientist | Machine Learning Enthusiast*

- 📧 [Email](mailto:vaibhavvst8@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/vaibhav-singh-bains/)
- 🐙 [GitHub](https://github.com/vaibhavvst24)

---
