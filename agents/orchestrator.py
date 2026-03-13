"""
Orchestrator node for the GenSlide LangGraph pipeline.

Responsibilities:
    - Receive parsed_text from the input parser node
    - Use GPT-4o to analyse the content and plan a logical slide outline
    - Incorporate user feedback on revision iterations
    - Write the ordered list of slide titles into state["outline"]
    - Write a human-readable error into state["error"] on failure

The orchestrator is the "architect" of the deck — it decides HOW MANY slides
are needed, what each slide should be called, and what logical order they
should appear in. It does NOT write slide content (that is the content agent's job).

Prompt strategy:
    - System prompt gives GPT-4o the role of a senior presentation strategist
    - It is instructed to return ONLY a JSON array of strings (slide titles)
    - On revision iterations it receives the previous outline + user feedback
      so it can make targeted adjustments rather than starting from scratch
    - Temperature is low (0.2) for consistent, structured output
"""

import json
import logging
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,       # low — we want consistent structure
    max_tokens=1024,
    response_format={"type": "json_object"},   # enforce JSON mode
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior presentation strategist with expertise in structuring \
complex information into clear, compelling slide decks.

Your task is to analyse source text and produce a slide outline — \
an ordered list of slide TITLES only (no content yet).

Rules:
- Produce between 5 and 12 slides. Never fewer than 5, never more than 12.
- The first slide title should be the deck's main topic or thesis.
- Follow a logical narrative arc: context → problem/opportunity → \
body points → conclusion or call-to-action.
- Each title must be concise: 3–8 words maximum.
- Titles must be distinct — no two slides should cover the same angle.
- Do NOT include a generic "Introduction" or "Agenda" slide unless \
the content genuinely calls for one.
- Do NOT include a "Thank You" or "Q&A" slide — those are added automatically.
- Return ONLY valid JSON in this exact shape:
  {"outline": ["Title one", "Title two", ...]}
""".strip()

REVISION_SUFFIX = """

--- REVISION REQUEST ---
This is iteration {iteration}. The user reviewed the previous deck and \
provided the following feedback:

\"{feedback}\"

Previous outline for reference:
{prev_outline}

Adjust the outline to address the feedback. You may reorder, rename, \
add, or remove slides as needed. Return the same JSON format.
""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_outline(raw: str) -> list[str]:
    """
    Parse the JSON response from GPT-4o.
    Handles both:
        {"outline": [...]}   ← preferred (json_object mode)
        ["title", ...]       ← fallback if model ignores the wrapper
    Also strips any accidental markdown code fences.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GPT-4o returned non-JSON output: {raw!r}") from exc

    # Unwrap {"outline": [...]} or accept a bare list
    if isinstance(data, dict):
        if "outline" in data:
            data = data["outline"]
        else:
            # Take the first list value in the dict, if any
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                raise ValueError(f"JSON object has no list value: {data}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of slide titles, got: {type(data)}")

    titles = [str(t).strip() for t in data if str(t).strip()]

    if len(titles) < 2:
        raise ValueError(
            f"Outline too short — only {len(titles)} title(s) returned. "
            "Model may have misunderstood the prompt."
        )

    return titles


def _build_human_message(
    parsed_text: str,
    feedback: Optional[str],
    prev_outline: Optional[list[str]],
    iteration: int,
) -> str:
    """Compose the human turn content, appending revision context if needed."""
    base = f"Source text:\n\n{parsed_text}"

    if feedback and iteration > 1 and prev_outline:
        suffix = REVISION_SUFFIX.format(
            iteration=iteration,
            feedback=feedback.strip(),
            prev_outline=json.dumps(prev_outline, indent=2),
        )
        return base + "\n\n" + suffix

    return base


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState) -> AgentState:
    """
    LangGraph node: plan the slide outline using GPT-4o.

    Reads:
        state["parsed_text"]  — clean source content from input_parser
        state["feedback"]     — revision notes from human_approval (may be None)
        state["outline"]      — previous outline (used on revision iterations)
        state["iteration"]    — current iteration count

    Writes:
        state["outline"]  — ordered List[str] of slide titles
        state["error"]    — error message on failure (else None)
    """
    # Guard: if a previous node errored, skip and propagate
    if state.get("error"):
        logger.warning("orchestrator_node skipped — upstream error: %s", state["error"])
        return state

    parsed_text = state.get("parsed_text", "").strip()
    if not parsed_text:
        state["error"] = "parsed_text is empty — cannot plan outline."
        return state

    iteration   = state.get("iteration", 1)
    feedback    = state.get("feedback")
    prev_outline = state.get("outline")  # may be None on first run

    try:
        human_content = _build_human_message(
            parsed_text, feedback, prev_outline, iteration
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        logger.info(
            "orchestrator_node: calling GPT-4o (iteration=%d, feedback=%s)",
            iteration,
            bool(feedback),
        )

        response = llm.invoke(messages)
        raw = response.content

        outline = _extract_outline(raw)

        logger.info(
            "orchestrator_node: planned %d slides → %s",
            len(outline),
            outline,
        )

        state["outline"] = outline
        state["error"]   = None

    except Exception as exc:
        logger.exception("orchestrator_node failed: %s", exc)
        state["error"] = f"Orchestrator failed: {exc}"

    return state