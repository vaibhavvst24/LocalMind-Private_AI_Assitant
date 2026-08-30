"""
Simple offline evaluation harness.

Scoring uses keyword/token overlap against a reference answer rather than an
LLM-as-judge, so the whole pipeline stays dependency-free and offline. This
is intentionally simple — the point for a resume project is to *show you
measure quality*, not to build a research-grade eval framework. Swap in an
LLM-judge (using the same local model) for a stronger version if you want to
go further.

Usable both from the Streamlit app and standalone:
    python -m eval.run_eval --model phi3.5:3.8b
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src import config
from src.llm_client import OllamaClient
from src.rag_engine import build_context_block


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in",
    "on", "for", "it", "this", "that", "with", "as", "by", "be", "or", "at",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def score_answer(candidate: str, reference: str) -> float:
    """Fraction of reference keywords present in the candidate answer."""
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0
    cand_tokens = _tokenize(candidate)
    overlap = ref_tokens & cand_tokens
    return len(overlap) / len(ref_tokens)


def run_evaluation(
    client: OllamaClient,
    model_tag: str,
    eval_set: list[dict],
    doc_store=None,
) -> tuple[pd.DataFrame, float, float]:
    rows = []
    for item in eval_set:
        question = item["question"]
        reference = item["reference_answer"]

        messages = [{"role": "system", "content": config.SYSTEM_PROMPT_BASE}]
        if doc_store is not None:
            hits = doc_store.query(question, top_k=config.TOP_K)
            context = build_context_block(hits)
            messages[0]["content"] = config.SYSTEM_PROMPT_RAG
            messages.append(
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            )
        else:
            messages.append({"role": "user", "content": question})

        result = client.chat_sync(messages, model=model_tag)
        score = score_answer(result["text"], reference)

        rows.append(
            {
                "question": question,
                "reference_answer": reference,
                "model_answer": result["text"][:300],
                "score": round(score, 2),
                "latency_s": result["total_seconds"],
                "tokens_per_sec": result["tokens_per_sec"],
            }
        )

    df = pd.DataFrame(rows)
    avg_score = df["score"].mean() if not df.empty else 0.0
    avg_latency = df["latency_s"].mean() if not df.empty else 0.0
    return df, avg_score, avg_latency


def main():
    parser = argparse.ArgumentParser(description="Run offline eval against a local model.")
    parser.add_argument("--model", default="phi3.5:3.8b", help="Ollama model tag")
    parser.add_argument("--rag", action="store_true", help="Ground answers in indexed documents")
    args = parser.parse_args()

    eval_set = json.loads(config.EVAL_QUESTIONS_PATH.read_text())
    client = OllamaClient()

    doc_store = None
    if args.rag:
        from src.rag_engine import DocumentStore
        doc_store = DocumentStore()

    df, avg_score, avg_latency = run_evaluation(client, args.model, eval_set, doc_store)

    print(df.to_string(index=False))
    print(f"\nAverage score: {avg_score:.0%}")
    print(f"Average latency: {avg_latency:.2f}s")

    out_path = config.EVAL_RESULTS_DIR / f"{args.model.replace(':', '_')}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
