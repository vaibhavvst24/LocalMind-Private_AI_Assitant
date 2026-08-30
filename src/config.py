"""
Central configuration for the offline local AI assistant.

All model serving happens through Ollama (http://localhost:11434), which
handles GGUF quantization, memory-mapping, and inference for you. This keeps
the app itself lightweight and lets you swap models with one line.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_store"
SQLITE_PATH = BASE_DIR / "data" / "memory.db"
EVAL_DIR = BASE_DIR / "eval"
EVAL_QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

for p in (DATA_DIR, DOCS_DIR, CHROMA_DIR, EVAL_RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama connection
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Chat model options
#
# These are small models chosen specifically because they run comfortably
# on consumer laptops (8-16GB RAM, no GPU required with 4-bit quantization).
# "tag" is the exact string Ollama uses. Pull with: ollama pull <tag>
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "Phi-3.5 mini (3.8B)": {
        "tag": "phi3.5:3.8b",
        "params": "3.8B",
        "default_quant": "Q4_K_M",
        "approx_ram_gb": 2.4,
        "notes": "Strong reasoning-to-size ratio; Microsoft's flagship SLM.",
    },
    "Qwen2.5 (3B)": {
        "tag": "qwen2.5:3b",
        "params": "3B",
        "default_quant": "Q4_K_M",
        "approx_ram_gb": 2.0,
        "notes": "Good multilingual + instruction following for its size.",
    },
    "Llama 3.2 (3B)": {
        "tag": "llama3.2:3b",
        "params": "3B",
        "default_quant": "Q4_K_M",
        "approx_ram_gb": 2.0,
        "notes": "Meta's small model, solid general-purpose baseline.",
    },
    "Gemma2 (2B)": {
        "tag": "gemma2:2b",
        "params": "2B",
        "default_quant": "Q4_0",
        "approx_ram_gb": 1.6,
        "notes": "Smallest/fastest option; best for low-RAM machines.",
    },
    "Qwen2.5 (1.5B)": {
        "tag": "qwen2.5:1.5b",
        "params": "1.5B",
        "default_quant": "Q4_K_M",
        "approx_ram_gb": 1.0,
        "notes": "Fastest, most quality-constrained. Good for benchmarking floor.",
    },
}

DEFAULT_MODEL_LABEL = "Phi-3.5 mini (3.8B)"

# ---------------------------------------------------------------------------
# Embedding model (also served by Ollama, so no HuggingFace download needed)
# Pull with: ollama pull nomic-embed-text
# ---------------------------------------------------------------------------
EMBED_MODEL_TAG = "nomic-embed-text"
EMBED_DIMENSIONS = 768

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 120       # characters of overlap between chunks
TOP_K = 4                 # chunks retrieved per query
COLLECTION_NAME = "local_assistant_docs"

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 512
SYSTEM_PROMPT_BASE = (
    "You are a helpful, concise offline assistant running entirely on the "
    "user's local machine. Answer directly. If you are not confident in an "
    "answer, say so rather than guessing."
)
SYSTEM_PROMPT_RAG = (
    "You are a helpful, concise offline assistant. Use ONLY the provided "
    "context to answer the question. If the context does not contain the "
    "answer, say you don't have enough information in the documents rather "
    "than guessing. Cite the source filename in brackets, e.g. [source.txt], "
    "when you use information from it."
)
