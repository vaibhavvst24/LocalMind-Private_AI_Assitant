"""
Thin client around the local Ollama server.

Ollama exposes a REST API on localhost once installed and running
(`ollama serve`, or it auto-starts on most installs). This module wraps
that API so the rest of the app never talks HTTP directly.
"""

from __future__ import annotations

import time
import json
import requests
from typing import Generator, Optional

from . import config


class OllamaConnectionError(Exception):
    """Raised when the local Ollama server can't be reached."""


class OllamaClient:
    def __init__(self, host: str = config.OLLAMA_HOST):
        self.host = host.rstrip("/")

    # ------------------------------------------------------------------
    # Health / model management
    # ------------------------------------------------------------------
    def health_check(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=timeout)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_local_models(self) -> list[str]:
        """Return tags of models already pulled locally."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(str(e)) from e

    def is_model_pulled(self, tag: str) -> bool:
        local = self.list_local_models()
        # Ollama tags list may include ":latest" variants; match prefix too
        return any(tag == m or m.startswith(tag.split(":")[0] + ":") for m in local)

    # ------------------------------------------------------------------
    # Chat / generation
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
        stream: bool = True,
    ) -> Generator[str, None, dict]:
        """
        Yields text chunks as they arrive. After the generator is exhausted,
        `self.last_stats` holds timing/token info for the last call.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.time()
        first_token_time = None
        full_text = []

        try:
            with requests.post(
                f"{self.host}/api/chat", json=payload, stream=stream, timeout=300
            ) as r:
                r.raise_for_status()
                if not stream:
                    data = r.json()
                    text = data.get("message", {}).get("content", "")
                    full_text.append(text)
                    yield text
                else:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get("done"):
                            self.last_stats = self._parse_stats(chunk, start)
                            break
                        piece = chunk.get("message", {}).get("content", "")
                        if piece:
                            if first_token_time is None:
                                first_token_time = time.time()
                            full_text.append(piece)
                            yield piece
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host}. Is it running? ({e})"
            ) from e

        if not hasattr(self, "last_stats"):
            elapsed = time.time() - start
            self.last_stats = {
                "total_seconds": round(elapsed, 2),
                "time_to_first_token": round((first_token_time - start), 2)
                if first_token_time
                else None,
                "response_text": "".join(full_text),
            }

    def chat_sync(
        self,
        messages: list[dict],
        model: str,
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
    ) -> dict:
        """Non-streaming call that returns the full response + timing stats."""
        start = time.time()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host}. Is it running? ({e})"
            ) from e

        elapsed = time.time() - start
        eval_count = data.get("eval_count")
        eval_duration_ns = data.get("eval_duration")  # nanoseconds
        tokens_per_sec = None
        if eval_count and eval_duration_ns:
            tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 2)

        return {
            "text": data.get("message", {}).get("content", ""),
            "total_seconds": round(elapsed, 2),
            "eval_count": eval_count,
            "tokens_per_sec": tokens_per_sec,
        }

    @staticmethod
    def _parse_stats(final_chunk: dict, start_time: float) -> dict:
        elapsed = time.time() - start_time
        eval_count = final_chunk.get("eval_count")
        eval_duration_ns = final_chunk.get("eval_duration")
        tokens_per_sec = None
        if eval_count and eval_duration_ns:
            tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 2)
        return {
            "total_seconds": round(elapsed, 2),
            "eval_count": eval_count,
            "tokens_per_sec": tokens_per_sec,
        }

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, text: str, model: str = config.EMBED_MODEL_TAG) -> list[float]:
        try:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["embedding"]
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host} for embeddings. ({e})"
            ) from e

    def embed_batch(
        self, texts: list[str], model: str = config.EMBED_MODEL_TAG
    ) -> list[list[float]]:
        # Ollama's /api/embeddings is single-prompt; loop with light backoff.
        return [self.embed(t, model=model) for t in texts]
