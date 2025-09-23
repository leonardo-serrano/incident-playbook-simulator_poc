# orchestrator/llm.py
"""
Provider-agnostic LLM adapters for planner/summarizer/decision nodes.

- Adapters:
    * Google Gemini (google-genai)
    * OpenAI
    * Anthropic
- Common interface:
    * plan(payload: Dict|Any, system_prompt: str) -> List[Dict] | str
    * summarize(state: Dict, system_prompt: str) -> str
- Factory: build_llm_or_none(provider=..., ...)

Design goals:
- Keep provider-specific SDK differences isolated within each adapter.
- Avoid scattering conditional logic across the codebase.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import asyncio
import inspect
import logging

log = logging.getLogger("llm")

# ---------------------------------------------------------------------
# Helpers to robustly call Google GenAI across SDK signature variants
# ---------------------------------------------------------------------

def _as_text(resp: Any) -> str:
    """
    Best-effort to extract text from Google GenAI responses.
    Falls back to str(resp) if needed.
    """
    if resp is None:
        return ""
    # Most SDK versions expose .text
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    # Sometimes text lives in candidates[0].content.parts[...].text
    cand = getattr(resp, "candidates", None)
    if cand and isinstance(cand, (list, tuple)) and len(cand) > 0:
        try:
            parts = cand[0].content.parts  # type: ignore[attr-defined]
            if parts and len(parts) > 0:
                maybe = getattr(parts[0], "text", None)
                if isinstance(maybe, str):
                    return maybe
        except Exception:
            pass
    # Fallback
    return str(resp)


def _genai_generate_robust(
    *,
    client: Any = None,
    model_obj: Any = None,
    model_name: str,
    contents: Any,
    system_instruction: Optional[str] = None,
    generation_config: Optional[dict] = None,
) -> str:
    """
    Robust wrapper tolerant to multiple Google GenAI SDK variants and signatures.
    It tries, in order:
      A) model_obj.generate_content(..., generation_config=..., system_instruction=...)
      B) model_obj.generate_content(..., config=...,           system_instruction=...)
      C) client.models.generate_content(model=..., contents=..., config=...,           system_instruction=...)
      D) client.models.generate_content(model=..., contents=..., generation_config=..., system_instruction=...)
      E) (fallback) NO system_instruction: send combined text as contents
      F) (last resort) contents only (no config, no system)
    """

    # Normalize user contents as text
    def _to_text(x: Any) -> str:
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    user_text = _to_text(contents)
    combined_text = (
        f"{system_instruction.strip()}\n\n{user_text}" if system_instruction else user_text
    )

    # 1) Try model instance first (older/working path en muchos entornos)
    if model_obj is not None:
        # A) old signature + system
        try:
            return _as_text(
                model_obj.generate_content(
                    contents=user_text,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
            )
        except Exception as e:
            log.debug("model.generate_content(gen_config + system) failed: %s", e)

        # B) new signature (config=) + system
        try:
            return _as_text(
                model_obj.generate_content(
                    contents=user_text,
                    config=generation_config,
                    system_instruction=system_instruction,
                )
            )
        except Exception as e:
            log.debug("model.generate_content(config + system) failed: %s", e)

        # E) fallback: no system_instruction (combine system+user como texto)
        try:
            return _as_text(
                model_obj.generate_content(
                    contents=combined_text
                )
            )
        except Exception as e:
            log.debug("model.generate_content(combined only) failed: %s", e)

        # F) last resort: contents only (user_text)
        try:
            return _as_text(
                model_obj.generate_content(
                    contents=user_text
                )
            )
        except Exception as e:
            log.debug("model.generate_content(user only) failed: %s", e)

    # 2) Try client.models entrypoint (newer SDK style)
    if client is not None:
        # C) new signature (config=) + system
        try:
            return _as_text(
                client.models.generate_content(
                    model=model_name,
                    contents=user_text,
                    config=generation_config,
                    system_instruction=system_instruction,
                )
            )
        except Exception as e:
            log.debug("client.models.generate_content(config + system) failed: %s", e)

        # D) old signature (generation_config=) + system
        try:
            return _as_text(
                client.models.generate_content(
                    model=model_name,
                    contents=user_text,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
            )
        except Exception as e:
            log.debug("client.models.generate_content(gen_config + system) failed: %s", e)

        # E) fallback: no system_instruction (combine system+user)
        try:
            return _as_text(
                client.models.generate_content(
                    model=model_name,
                    contents=combined_text
                )
            )
        except Exception as e:
            log.debug("client.models.generate_content(combined only) failed: %s", e)

        # F) last resort: contents only (user_text)
        try:
            return _as_text(
                client.models.generate_content(
                    model=model_name,
                    contents=user_text
                )
            )
        except Exception as e:
            log.debug("client.models.generate_content(user only) failed: %s", e)

    raise RuntimeError("No valid GenAI entrypoint available")


# ---------------------------------------------------------------------
# Primary adapter class
# ---------------------------------------------------------------------

class GeminiLLM:
    """Thin wrapper over Google Gen AI SDK for our LLM calls."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-001"):
        # Lazy import keeps dependency optional for environments without LLM
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        # Create a model instance; use correct keyword (model_name=) to avoid signature issues
        try:
            self.model_obj = genai.GenerativeModel(model_name=model)
        except Exception as e:
            log.debug("GenerativeModel(model_name=...) failed, continuing without model_obj: %s", e)
            self.model_obj = None
        # Default low-variance config; can be extended from env if needed
        self.gen_config: Dict[str, Any] = {"temperature": 0, "max_output_tokens": 1024}

    def _generate(self, system: str, user: str) -> str:
        """
        Call Gemini using the robust helper, passing system_instruction separately.
        """
        return _genai_generate_robust(
            client=self.client,
            model_obj=self.model_obj,
            model_name=self.model_name,
            contents=user,                       # plain text is fine
            system_instruction=system,
            generation_config=self.gen_config,   # tries both config= and generation_config=
        ).strip()

    # ---------- Planner ---------- #
    def plan(self, payload: Dict[str, Any], system_prompt: str) -> Any:
        """
        Return a list of step dicts if the model emits JSON list; otherwise return raw text.
        """
        user = json.dumps(payload, ensure_ascii=False)
        text = self._generate(system_prompt, user)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return text  # caller may still parse later

    # --------- Summarizer -------- #
    def summarize(self, state: Dict[str, Any], system_prompt: str) -> str:
        """Return a concise summary string."""
        payload = {
            "alert": state.get("alert", {}),
            "plan": state.get("plan", []),
            "retrieved_docs_count": len(state.get("retrieved_docs", []) or []),
            "metrics_present": bool(state.get("metrics")),
            "logs_count": len(state.get("logs", []) or []),
            "actions_taken": state.get("actions_taken", []),
        }
        user = json.dumps(payload, ensure_ascii=False)
        return self._generate(system_prompt, user)

# ---------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------

class OpenAILLM:
    """Thin wrapper over OpenAI SDK with the same interface as GeminiLLM."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        # Lazy import keeps dependency optional
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(f"OpenAI SDK not available: {e}")
        self._OpenAI = OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model
        # Low variance defaults; tune via env if desired
        self.gen_config: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 1024}

    def _generate(self, system: str, user: str) -> str:
        # Use responses API for unified text output
        try:
            resp = self.client.responses.create(
                model=self.model_name,
                input=f"{system}\n\n{user}",
                temperature=self.gen_config.get("temperature", 0.0),
                max_output_tokens=self.gen_config.get("max_tokens", 1024),
            )
            # Robust text extraction
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt
            # Fallbacks
            return str(resp)
        except Exception as e:
            log.debug("OpenAI generate failed: %s", e)
            raise

    def plan(self, payload: Dict[str, Any], system_prompt: str) -> Any:
        user = json.dumps(payload, ensure_ascii=False)
        text = self._generate(system_prompt, user)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return text

    def summarize(self, state: Dict[str, Any], system_prompt: str) -> str:
        payload = {
            "alert": state.get("alert", {}),
            "plan": state.get("plan", []),
            "retrieved_docs_count": len(state.get("retrieved_docs", []) or []),
            "metrics_present": bool(state.get("metrics")),
            "logs_count": len(state.get("logs", []) or []),
            "actions_taken": state.get("actions_taken", []),
        }
        user = json.dumps(payload, ensure_ascii=False)
        return self._generate(system_prompt, user)


# ---------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------

class AnthropicLLM:
    """Thin wrapper over Anthropic SDK with the same interface as GeminiLLM."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Anthropic SDK not available: {e}")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
        self.gen_config: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 1024}

    def _generate(self, system: str, user: str) -> str:
        try:
            msg = self.client.messages.create(
                model=self.model_name,
                system=system,
                max_tokens=self.gen_config.get("max_tokens", 1024),
                temperature=self.gen_config.get("temperature", 0.0),
                messages=[{"role": "user", "content": user}],
            )
            # Extract text from content blocks
            try:
                blocks = getattr(msg, "content", None)
                if isinstance(blocks, list) and blocks:
                    for b in blocks:
                        t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                        if isinstance(t, str) and t.strip():
                            return t
            except Exception:
                pass
            return str(msg)
        except Exception as e:
            log.debug("Anthropic generate failed: %s", e)
            raise

    def plan(self, payload: Dict[str, Any], system_prompt: str) -> Any:
        user = json.dumps(payload, ensure_ascii=False)
        text = self._generate(system_prompt, user)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return text

    def summarize(self, state: Dict[str, Any], system_prompt: str) -> str:
        payload = {
            "alert": state.get("alert", {}),
            "plan": state.get("plan", []),
            "retrieved_docs_count": len(state.get("retrieved_docs", []) or []),
            "metrics_present": bool(state.get("metrics")),
            "logs_count": len(state.get("logs", []) or []),
            "actions_taken": state.get("actions_taken", []),
        }
        user = json.dumps(payload, ensure_ascii=False)
        return self._generate(system_prompt, user)


# ---------------------------------------------------------------------
# Generic text adapter (kept for compatibility with older code paths)
# ---------------------------------------------------------------------

class _TextLLMAdapter:
    """
    Adapter that provides .ainvoke/.invoke(text) on top of a variety of LLM shapes:
    - callable(prompt) -> str | Awaitable[str]
    - google genai client (generate_content / generate_content_async)
    - objects exposing text/invoke/ainvoke already
    """
    def __init__(self, inner: Any):
        self._inner = inner

    async def ainvoke(self, text: str) -> str:
        # Already async APIs
        if hasattr(self._inner, "ainvoke") and callable(getattr(self._inner, "ainvoke")):
            return await self._inner.ainvoke(text)
        if hasattr(self._inner, "atext") and callable(getattr(self._inner, "atext")):
            return await self._inner.atext(text)
        if hasattr(self._inner, "generate_content_async") and callable(getattr(self._inner, "generate_content_async")):
            resp = await self._inner.generate_content_async(text)
            return getattr(resp, "text", str(resp))

        # Callable coroutine function
        if inspect.iscoroutinefunction(self._inner):
            return await self._inner(text)

        # Fallback: run sync call in executor
        loop = asyncio.get_running_loop()
        if hasattr(self._inner, "invoke") and callable(getattr(self._inner, "invoke")):
            return await loop.run_in_executor(None, self._inner.invoke, text)
        if hasattr(self._inner, "text") and callable(getattr(self._inner, "text")):
            return await loop.run_in_executor(None, self._inner.text, text)
        if hasattr(self._inner, "generate_content") and callable(getattr(self._inner, "generate_content")):
            def _run():
                resp = self._inner.generate_content(text)
                return getattr(resp, "text", str(resp))
            return await loop.run_in_executor(None, _run)
        if callable(self._inner):
            return await loop.run_in_executor(None, self._inner, text)

        raise RuntimeError("LLM adapter: no compatible async/sync text interface found")

    def invoke(self, text: str) -> str:
        # Prefer sync if available
        if hasattr(self._inner, "invoke") and callable(getattr(self._inner, "invoke")):
            return self._inner.invoke(text)
        if hasattr(self._inner, "text") and callable(getattr(self._inner, "text")):
            return self._inner.text(text)
        if hasattr(self._inner, "generate_content") and callable(getattr(self._inner, "generate_content")):
            resp = self._inner.generate_content(text)
            return getattr(resp, "text", str(resp))
        if callable(self._inner) and not inspect.iscoroutinefunction(self._inner):
            return self._inner(text)
        # As last resort, run async interface in a blocking way (rarely used in our code path)
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(text))

def ensure_text_iface(llm_like: Any) -> Any:
    """
    Wrap whatever you have into a safe object exposing .ainvoke/.invoke(text).
    If it already has those, returns as-is.
    """
    if llm_like is None:
        return None
    if hasattr(llm_like, "ainvoke") or hasattr(llm_like, "invoke") or hasattr(llm_like, "text"):
        return llm_like
    if hasattr(llm_like, "generate_content") or hasattr(llm_like, "generate_content_async"):
        return _TextLLMAdapter(llm_like)
    if callable(llm_like):
        return _TextLLMAdapter(llm_like)
    return _TextLLMAdapter(llm_like)

# -------------------------- Factory helper -------------------------- #
def build_llm_or_none(
    provider: str = "google",
    *,
    google_key: Optional[str] = None,
    google_model: str = "gemini-2.0-flash-001",
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
    anthropic_key: Optional[str] = None,
    anthropic_model: str = "claude-3-5-sonnet-latest",
    **_: Any,
):
    """
    Return a provider-specific LLM adapter or None if misconfigured/disabled.

    Provider values: 'google' | 'openai' | 'anthropic'
    """
    try:
        p = (provider or "").strip().lower()
        if p == "openai":
            if not openai_key:
                return None
            return OpenAILLM(api_key=openai_key, model=openai_model)
        if p == "anthropic":
            if not anthropic_key:
                return None
            return AnthropicLLM(api_key=anthropic_key, model=anthropic_model)
        # default to google
        if not google_key:
            return None
        return GeminiLLM(api_key=google_key, model=google_model)
    except Exception as e:
        log.debug("build_llm_or_none failed: %s", e)
        return None
