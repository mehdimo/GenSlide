"""
Orchestrator node for the GenSlide LangGraph pipeline.

Uses a single unified prompt for all LLM providers (OpenAI and local).
The prompt is designed to be unambiguous enough that well-instructed
models follow it, while the parser is robust enough to recover from
models that don't produce perfect JSON.

Parser strategy (tried in order):
    1. Direct JSON parse of full response
    2. Extract first {...} or [...] block via regex
    3. Extract quoted strings
    4. Numbered / bulleted plain-text lines
"""

import json
import logging
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState
from llm.llm_provider import get_llm, get_provider_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a presentation strategist. Analyse the source text and return a \
slide outline as a JSON array of slide title strings.

Requirements:
- Between 5 and 10 slide titles.
- Each title is 3 to 8 words.
- Titles follow a logical narrative arc.
- Do NOT include "Introduction", "Agenda", "Thank You", or "Q&A".

Return ONLY a JSON array. Example:
["Title one", "Title two", "Title three"]
""".strip()

REVISION_SUFFIX = """

The user reviewed the previous deck and requested changes: "{feedback}"

Previous titles:
{prev_outline}

Return the updated JSON array of titles only.
""".strip()

# ---------------------------------------------------------------------------
# Parser — 4-strategy fallback chain
# ---------------------------------------------------------------------------

def _extract_outline(raw: str) -> list[str]:
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # 1. Full JSON parse — handles both [...] and {"outline": [...]}
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _validate(data)
        if isinstance(data, dict):
            for key in ("outline", "titles", "slides"):
                if isinstance(data.get(key), list):
                    return _validate(data[key])
            for v in data.values():
                if isinstance(v, list):
                    return _validate(v)
    except json.JSONDecodeError:
        pass

    # 2. Extract first JSON array anywhere in the text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate(data)
        except json.JSONDecodeError:
            pass

    # 3. Extract quoted strings
    titles = re.findall(r'"([^"]{5,80})"', text)
    if len(titles) >= 3:
        return _validate(titles)

    # 4. Numbered or bulleted plain-text lines
    lines = [
        re.sub(r"^[\d\.\-\*\•]+\s*", "", ln).strip()
        for ln in text.splitlines()
        if ln.strip() and len(ln.strip()) > 4
    ]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 3:
        return _validate(lines)

    raise ValueError(
        f"Could not extract a slide outline from model output:\n{raw[:400]}"
    )


def _validate(titles: list) -> list[str]:
    clean = [str(t).strip() for t in titles if str(t).strip()]
    if len(clean) < 2:
        raise ValueError(f"Outline too short — only {len(clean)} title(s).")
    return clean


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState) -> AgentState:
    """
    LangGraph node: plan the slide outline using the configured LLM.

    Reads:
        state["parsed_text"]  — clean source content
        state["feedback"]     — revision notes (may be None)
        state["outline"]      — previous outline (used on revisions)
        state["iteration"]    — current iteration count

    Writes:
        state["outline"]  — ordered List[str] of slide titles
        state["error"]    — error message on failure (else None)
    """
    if state.get("error"):
        logger.warning("orchestrator_node skipped — upstream error: %s", state["error"])
        return state

    parsed_text  = state.get("parsed_text", "").strip()
    if not parsed_text:
        state["error"] = "parsed_text is empty — cannot plan outline."
        return state

    iteration    = state.get("iteration", 1)
    feedback     = state.get("feedback")
    prev_outline = state.get("outline")

    # Build human turn
    human_content = f"Source text:\n\n{parsed_text}"
    if feedback and iteration > 1 and prev_outline:
        human_content += "\n\n" + REVISION_SUFFIX.format(
            feedback=feedback.strip(),
            prev_outline=json.dumps(prev_outline, indent=2),
        )

    try:
        llm = get_llm(temperature=0.2)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        logger.info(
            "orchestrator_node: calling %s (iteration=%d, revision=%s)",
            get_provider_name(), iteration, bool(feedback),
        )

        response = llm.invoke(messages)
        outline  = _extract_outline(response.content)

        logger.info("orchestrator_node: planned %d slides → %s", len(outline), outline)

        state["outline"] = outline
        state["error"]   = None

    except Exception as exc:
        logger.exception("orchestrator_node failed: %s", exc)
        state["error"] = f"Orchestrator failed: {exc}"

    return state