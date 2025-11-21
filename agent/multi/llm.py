"""Local LLM factory tailored for the multi-agent stack."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional, Any, List, Dict


LOG = logging.getLogger("multi.llm")


try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None

try:
    from langchain_community.chat_models import ChatGemini
except Exception:  # pragma: no cover
    ChatGemini = None

try:
    from langchain_community.chat_models import ChatDeepSeek
except Exception:  # pragma: no cover
    ChatDeepSeek = None


class GroqChatWrapper:
    """Simplified Groq LLM wrapper compatible with LangChain interfaces."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = kwargs.get("api_url") or os.getenv("GROQ_API_URL")
        try:
            from groq import Groq  # type: ignore

            self._client = Groq(api_key=self.api_key)
        except Exception:  # pragma: no cover - HTTP fallback path
            self._client = None

    def invoke(self, messages):  # type: ignore[override]
        if self._client is None:
            return self._invoke_via_http(messages)
        role_map = {
            "human": "user",
            "user": "user",
            "system": "system",
            "ai": "assistant",
            "assistant": "assistant",
            "tool": "tool",
        }
        payload = []
        for msg in messages:
            role = getattr(msg, "type", "user")
            mapped_role = role_map.get(role, "user")
            payload.append({"role": mapped_role, "content": getattr(msg, "content", str(msg))})
        response = self._client.chat.completions.create(model=self.model, messages=payload)
        text = response.choices[0].message.content
        return type("GroqResponse", (), {"content": text})()

    def _invoke_via_http(self, messages):
        import requests

        prompt = "\n\n".join(getattr(msg, "content", str(msg)) for msg in messages)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if not self.api_url:
            raise RuntimeError("GROQ_API_URL not configured for Groq HTTP fallback.")
        payload = {"input": prompt, "model": self.model}
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        body = resp.json()
        text = body.get("output") or body.get("generated_text") or resp.text
        return type("GroqResponse", (), {"content": text})()


# --- Optional tokenization support (best-effort) ---
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False


def _estimate_tokens_from_messages(messages: List[Any], model_name: Optional[str]) -> Optional[int]:
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        if model_name:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages or []:
            content = getattr(m, "content", str(m))
            total += len(enc.encode(content))
        return total
    except Exception:
        return None


def _estimate_tokens_from_text(text: str, model_name: Optional[str]) -> Optional[int]:
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        if model_name:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return None


class MeteredLLM:
    """Thin wrapper that meters token usage per invoke.

    - Records best-effort tokens_in (messages) and tokens_out (response)
    - Tries to read provider metadata if available (e.g., OpenAI via LangChain)
    - Exposes reset_stats() and get_stats() for per-turn aggregation
    """

    def __init__(self, base_llm: Any, *, model: Optional[str], provider: Optional[str]) -> None:
        self._base = base_llm
        self.model = model
        self.provider = provider
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    # Preserve interface used by agents
    def invoke(self, messages: List[Any]):  # type: ignore[override]
        # Pre-count input tokens
        est_in = _estimate_tokens_from_messages(messages, self.model)
        resp = self._base.invoke(messages)

        # Extract content and potential metadata
        try:
            text = getattr(resp, "content", None)
        except Exception:
            text = None

        est_out = None
        if text is not None:
            est_out = _estimate_tokens_from_text(str(text), self.model)

        # Try provider-native token usage via LangChain response metadata
        used_in = None
        used_out = None
        try:
            meta: Dict[str, Any] = getattr(resp, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or meta.get("usage") or {}
            used_in = usage.get("prompt_tokens") or usage.get("input_tokens")
            used_out = usage.get("completion_tokens") or usage.get("output_tokens")
        except Exception:
            pass

        tin = used_in if isinstance(used_in, int) else (est_in or 0)
        tout = used_out if isinstance(used_out, int) else (est_out or 0)

        self._calls += 1
        self._tokens_in += int(tin)
        self._tokens_out += int(tout)
        return resp

    # Helpers for metrics collection
    def reset_stats(self) -> None:
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    def get_stats(self) -> Dict[str, Any]:
        total = self._tokens_in + self._tokens_out
        return {
            "calls": self._calls,
            "tokens_input": self._tokens_in,
            "tokens_output": self._tokens_out,
            "tokens_total": total,
            "model": self.model,
            "provider": self.provider,
        }


@lru_cache(maxsize=8)
def ensure_llm(*, model: Optional[str] = None, provider: Optional[str] = None, **kwargs):
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    model = model or os.getenv("MODEL", "gpt-4o-mini")
    LOG.info("Instantiating LLM provider=%s model=%s", provider, model)

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("ChatOpenAI backend not available. Install langchain-community.")
        return ChatOpenAI(model=model, openai_api_key=os.getenv("OPENAI_API_KEY"), **kwargs)

    if provider == "groq":
        return GroqChatWrapper(model=model, api_key=os.getenv("GROQ_API_KEY"), **kwargs)

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("ChatOllama backend not available. Install langchain-community/ollama.")
        host = kwargs.pop("ollama_host", None) or os.getenv("OLLAMA_HOST")
        if host:
            kwargs.setdefault("base_url", host)
        return ChatOllama(model=model, **kwargs)

    if provider == "gemini":
        if ChatGemini is None:
            raise RuntimeError("ChatGemini backend not available.")
        return ChatGemini(model=model, **kwargs)

    if provider == "deepseek":
        if ChatDeepSeek is None:
            raise RuntimeError("ChatDeepSeek backend not available.")
        return ChatDeepSeek(model=model, **kwargs)

    raise ValueError(f"Unsupported LLM provider '{provider}'. Configure LLM_PROVIDER.")


def wrap_with_meter(llm: Any, *, model: Optional[str], provider: Optional[str]) -> MeteredLLM:
    """Return an LLM wrapped with token metering."""
    return MeteredLLM(llm, model=model, provider=provider)
