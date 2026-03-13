"""
LangChain-compatible wrapper around the gpt4all local inference library.

Why a wrapper instead of using langchain_community.llms.GPT4All directly:
    - The community integration is a plain text LLM (BaseLLM), not a chat
      model. It accepts a raw string prompt and returns a raw string.
    - Our agents pass a list of LangChain messages (SystemMessage +
      HumanMessage) and expect an AIMessage back — the same interface
      as ChatOpenAI.
    - This wrapper bridges that gap: it formats the message list into a
      single plain-text prompt that a local Llama model understands
      (the Llama 3 instruct chat template), calls gpt4all, and returns
      a mock AIMessage so the agents never know which provider they're
      talking to.

Llama 3 instruct chat template used:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

For non-Llama models (Mistral, Falcon, etc.) the template degrades
gracefully to a simple "System: ... \nUser: ..." format.
"""

import logging
import os
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Llama 3 instruct special tokens
_LLAMA3_BOS        = "<|begin_of_text|>"
_LLAMA3_SYS_START  = "<|start_header_id|>system<|end_header_id|>\n\n"
_LLAMA3_EOT        = "<|eot_id|>"
_LLAMA3_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
_LLAMA3_ASST_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _is_llama3(model_name: str) -> bool:
    return "llama-3" in model_name.lower() or "llama3" in model_name.lower()


def _format_messages(messages: List[BaseMessage], model_name: str) -> str:
    """
    Convert a LangChain message list into a single prompt string.

    Uses Llama 3 instruct chat template for Llama 3 models,
    falls back to a simple System/User format for everything else.
    """
    system_text = ""
    user_text   = ""

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_text = msg.content
        elif isinstance(msg, HumanMessage):
            user_text = msg.content
        # AssistantMessage etc. are ignored for now (single-turn)

    if _is_llama3(model_name):
        prompt = (
            f"{_LLAMA3_BOS}"
            f"{_LLAMA3_SYS_START}{system_text}{_LLAMA3_EOT}\n"
            f"{_LLAMA3_USER_START}{user_text}{_LLAMA3_EOT}\n"
            f"{_LLAMA3_ASST_START}"
        )
    else:
        # Generic instruct format — works for Mistral, Falcon, Phi, etc.
        prompt = (
            f"### System:\n{system_text}\n\n"
            f"### User:\n{user_text}\n\n"
            f"### Assistant:\n"
        )

    return prompt


class GPT4AllChatWrapper:
    """
    Wraps langchain_community.llms.GPT4All to expose a ChatModel-like
    interface (accepts List[BaseMessage], returns AIMessage).

    The gpt4all library auto-downloads the model file on first use
    if it is not already present in the cache directory.

    Args:
        model_name:   gpt4all model filename (e.g. "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        model_path:   directory for model storage (None → gpt4all default: ~/.cache/gpt4all)
        temperature:  sampling temperature (0.0–1.0)
        max_tokens:   maximum tokens to generate per call
        n_threads:    CPU threads to use (None → gpt4all auto-detects)
    """

    def __init__(
        self,
        model_name:  str,
        model_path:  Optional[str] = None,
        temperature: float = 0.5,
        max_tokens:  int   = 1024,
        n_threads:   Optional[int] = None,
    ):
        self.model_name  = model_name
        self.model_path  = model_path
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.n_threads   = n_threads
        self._llm        = None   # lazy-loaded on first call
        self.allow_download = True # If model does not exist in ~/.cache/gpt4all/, download it.

    # ------------------------------------------------------------------
    # Lazy loader — model is large; only download/load when first needed
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        try:
            from langchain_community.llms import GPT4All
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-community' and 'gpt4all' packages are required "
                "for local inference.\n"
                "Install them with: pip install langchain-community gpt4all"
            ) from exc

        init_kwargs: dict[str, Any] = {
            "model":       self.model_name,
            "temp":        self.temperature,
            "max_tokens":  self.max_tokens,
            "verbose":     False,
            "allow_download": self.allow_download
        }

        if self.model_path:
            # gpt4all uses 'model' param which can be a full path
            full_path = os.path.join(self.model_path, self.model_name)
            if os.path.exists(full_path):
                init_kwargs["model"] = full_path
            else:
                logger.warning(
                    "LOCAL_MODEL_PATH set to '%s' but model file not found there. "
                    "gpt4all will look in its default cache (~/.cache/gpt4all) "
                    "and download if needed.",
                    self.model_path,
                )

        if self.n_threads is not None:
            init_kwargs["n_threads"] = self.n_threads

        logger.info(
            "Loading local model '%s'  (first call may download ~4 GB)…",
            self.model_name,
        )
        self._llm = GPT4All(**init_kwargs)
        logger.info("Local model loaded.")
        return self._llm

    # ------------------------------------------------------------------
    # Public interface — mirrors ChatOpenAI.invoke()
    # ------------------------------------------------------------------

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        Accept a list of LangChain messages, format them into a prompt,
        call gpt4all, and return the response wrapped in an AIMessage.
        """
        prompt  = _format_messages(messages, self.model_name)
        llm     = self._get_llm()

        logger.debug("GPT4AllChatWrapper prompt (truncated):\n%s…", prompt[:300])

        raw: str = llm.invoke(prompt)

        # Strip any echoed prompt prefix that some models include
        if _LLAMA3_ASST_START in raw:
            raw = raw.split(_LLAMA3_ASST_START, 1)[-1]
        if "### Assistant:" in raw:
            raw = raw.split("### Assistant:", 1)[-1]

        content = raw.strip()
        logger.debug("GPT4AllChatWrapper response: %s…", content[:200])

        return AIMessage(content=content)