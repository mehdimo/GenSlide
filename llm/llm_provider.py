"""
LLM provider factory for GenSlide.

Reads LLM_PROVIDER from the environment and returns the appropriate
LangChain-compatible LLM instance. All agents import from here — they
never instantiate their own LLM directly.

Supported providers (set in .env):
    LLM_PROVIDER=openai   → ChatOpenAI (gpt-4o), requires OPENAI_API_KEY
    LLM_PROVIDER=local    → gpt4all wrapper (Llama 3 by default, runs on CPU)

Environment variables:
    LLM_PROVIDER        "openai" | "local"   (default: "openai")
    OPENAI_API_KEY      required for openai provider
    LOCAL_MODEL_NAME    gpt4all model name   (default: "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    LOCAL_MODEL_PATH    directory to store downloaded models (default: ~/.cache/gpt4all)
"""

import logging
import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ProviderName = Literal["openai", "local"]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
DEFAULT_PROVIDER    = "openai"


def get_provider_name() -> ProviderName:
    """Return the active provider name from the environment."""
    raw = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if raw not in ("openai", "local"):
        logger.warning(
            "Unknown LLM_PROVIDER='%s'. Falling back to 'openai'.", raw
        )
        return "openai"
    return raw  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.3):
    """
    Return a cached LangChain-compatible LLM for the configured provider.

    Both providers expose the same .invoke(messages) interface so agents
    can call them identically. The local wrapper adapts the plain-text
    gpt4all output into a LangChain AIMessage.

    Args:
        temperature: sampling temperature (cached per value — agents
                     that need different temperatures call get_llm()
                     with their specific value).

    Returns:
        A LangChain BaseChatModel-compatible object.
    """
    provider = get_provider_name()

    if provider == "openai":
        return _build_openai_llm(temperature)
    else:
        return _build_local_llm(temperature)


def _build_openai_llm(temperature: float):
    """Build a ChatOpenAI instance for gpt-4o."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add it to your .env file."
        )

    logger.info("LLM provider: OpenAI gpt-4o (temperature=%.1f)", temperature)
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        max_tokens=1024,
        response_format={"type": "json_object"},
        api_key=api_key,
    )


def _build_local_llm(temperature: float):
    """Build the gpt4all wrapper for local inference."""
    from llm.gpt4all_wrapper import GPT4AllChatWrapper

    model_name = os.getenv("LOCAL_MODEL_NAME", DEFAULT_LOCAL_MODEL)
    model_path = os.getenv("LOCAL_MODEL_PATH", "")  # empty → gpt4all default cache

    logger.info(
        "LLM provider: local gpt4all  model=%s  temperature=%.1f",
        model_name, temperature,
    )
    return GPT4AllChatWrapper(
        model_name=model_name,
        model_path=model_path or None,
        temperature=temperature,
    )