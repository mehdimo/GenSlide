"""
Content agent node for the GenSlide LangGraph pipeline.

Responsibilities:
    - Receive state["outline"] (List[str] of slide titles) from the orchestrator
    - For each title, call GPT-4o to write bullet points and speaker notes
    - Respect any revision feedback when rewriting slides
    - Write List[SlideContent] into state["slides"]
    - Write a human-readable error into state["error"] on failure

Design decisions:
    Parallelism  : Slides are generated concurrently using ThreadPoolExecutor
                   to minimise wall-clock time (a 10-slide deck takes ~1s
                   instead of ~10s when calls are sequential).
    Temperature  : 0.5 — slightly higher than the orchestrator to allow
                   varied, natural-sounding bullet phrasing.
    Retry        : Each slide call is retried once on transient failures
                   before the whole node errors out.
    Context      : Full parsed_text is sent with every slide call so GPT-4o
                   can draw accurate, grounded facts — not hallucinated filler.
    Feedback     : On revision iterations, the previous slide content is
                   included so the model makes targeted edits rather than
                   rewriting everything from scratch.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState, SlideContent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    max_tokens=512,
    response_format={"type": "json_object"},
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert slide content writer. Your job is to write the body \
content for a single presentation slide.

Given:
  - A slide title
  - The full source text the presentation is based on
  - (Optionally) previous content for this slide and revision instructions

Produce a JSON object with exactly these three keys:

  "title"         : string — the slide title (echo it back unchanged)
  "bullets"       : array of 3 to 5 strings — concise, informative bullet \
points drawn from the source text. Each bullet should be one sentence \
(max 15 words). No bullet should start with a dash or bullet character.
  "speaker_notes" : string — 2 to 4 sentences expanding on the bullets, \
written as natural spoken prose for the presenter. Do NOT repeat bullet \
text verbatim.

Return ONLY the JSON object. No markdown fences, no extra keys, no preamble.
""".strip()

REVISION_SYSTEM_PROMPT = """
You are an expert slide content writer performing a targeted revision.

The user has reviewed the deck and provided feedback. Your job is to \
revise the content for a single slide based on that feedback.

Only change what the feedback asks for. Preserve anything the feedback \
does not mention. Return the same JSON shape as the original:

  "title"         : string
  "bullets"       : array of 3 to 5 strings
  "speaker_notes" : string

Return ONLY the JSON object. No markdown fences, no preamble.
""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_slide_content(raw: str, expected_title: str) -> SlideContent:
    """
    Parse the JSON response for a single slide.
    Validates required keys and coerces types.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"GPT-4o returned non-JSON for slide '{expected_title}': {raw!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data)} for '{expected_title}'")

    title   = str(data.get("title", expected_title)).strip() or expected_title
    bullets = data.get("bullets", [])
    notes   = str(data.get("speaker_notes", "")).strip()

    # Coerce bullets to a clean list of strings
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    bullets = [str(b).strip() for b in bullets if str(b).strip()]

    if not bullets:
        raise ValueError(f"No bullets returned for slide '{expected_title}'")

    return SlideContent(title=title, bullets=bullets, speaker_notes=notes)


def _call_with_retry(messages: list, slide_title: str, max_retries: int = 1) -> str:
    """
    Invoke the LLM with one retry on failure, with a short backoff.
    Returns the raw response content string.
    """
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt   # 1s, 2s, ...
                logger.warning(
                    "Slide '%s' — attempt %d failed (%s). Retrying in %ds...",
                    slide_title, attempt + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                raise


def _generate_slide(
    title: str,
    parsed_text: str,
    feedback: Optional[str],
    prev_slide: Optional[SlideContent],
    iteration: int,
) -> SlideContent:
    """
    Generate content for a single slide. Used as the unit of work
    for parallel execution.
    """
    is_revision = bool(feedback and iteration > 1 and prev_slide)

    if is_revision:
        system = SYSTEM_PROMPT  # reuse base system; revision context goes in human turn
        human_content = (
            f"Slide title: {title}\n\n"
            f"Source text:\n{parsed_text}\n\n"
            f"--- Previous slide content ---\n"
            f"Bullets:\n"
            + "\n".join(f"- {b}" for b in prev_slide["bullets"])
            + f"\n\nSpeaker notes:\n{prev_slide['speaker_notes']}\n\n"
            f"--- Revision feedback ---\n{feedback.strip()}\n\n"
            f"Revise the slide above to address the feedback. "
            f"Keep what the feedback does not mention."
        )
    else:
        system = SYSTEM_PROMPT
        human_content = (
            f"Slide title: {title}\n\n"
            f"Source text:\n{parsed_text}"
        )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human_content),
    ]

    raw = _call_with_retry(messages, title)
    return _extract_slide_content(raw, title)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def content_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: write slide content for every title in state["outline"].

    Reads:
        state["outline"]      — List[str] of slide titles from orchestrator
        state["parsed_text"]  — full source text (context for each slide)
        state["feedback"]     — revision notes from human_approval (may be None)
        state["slides"]       — previous slides (used on revision iterations)
        state["iteration"]    — current iteration count

    Writes:
        state["slides"]   — List[SlideContent] in outline order
        state["error"]    — error message on failure (else None)
    """
    # Guard: propagate upstream errors
    if state.get("error"):
        logger.warning(
            "content_agent_node skipped — upstream error: %s", state["error"]
        )
        return state

    outline     = state.get("outline") or []
    parsed_text = state.get("parsed_text", "").strip()
    feedback    = state.get("feedback")
    prev_slides = state.get("slides") or []
    iteration   = state.get("iteration", 1)

    if not outline:
        state["error"] = "Outline is empty — orchestrator may have failed."
        return state

    if not parsed_text:
        state["error"] = "parsed_text is empty — cannot generate slide content."
        return state

    # Build a title → prev_slide lookup for revision context
    prev_by_title: dict[str, SlideContent] = {
        s["title"]: s for s in prev_slides
    }

    logger.info(
        "content_agent_node: generating %d slides (iteration=%d, parallel=True)",
        len(outline),
        iteration,
    )

    slides: list[Optional[SlideContent]] = [None] * len(outline)
    errors: list[str] = []

    # Generate all slides concurrently — each is an independent GPT-4o call
    with ThreadPoolExecutor(max_workers=min(len(outline), 6)) as executor:
        future_to_index = {
            executor.submit(
                _generate_slide,
                title,
                parsed_text,
                feedback,
                prev_by_title.get(title),
                iteration,
            ): idx
            for idx, title in enumerate(outline)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            title = outline[idx]
            try:
                slides[idx] = future.result()
                logger.info("  ✓ Slide %d/%d: '%s'", idx + 1, len(outline), title)
            except Exception as exc:
                err_msg = f"Slide '{title}' failed: {exc}"
                logger.error("  ✗ %s", err_msg)
                errors.append(err_msg)
                # Insert a placeholder so indexing stays consistent
                slides[idx] = SlideContent(
                    title=title,
                    bullets=["Content could not be generated for this slide."],
                    speaker_notes="",
                )

    # Report partial failures as a warning, not a hard stop
    if errors:
        state["error"] = (
            f"{len(errors)} slide(s) failed to generate: "
            + "; ".join(errors)
        )
        logger.warning("content_agent_node partial failure: %s", state["error"])
    else:
        state["error"] = None

    state["slides"] = [s for s in slides if s is not None]

    logger.info(
        "content_agent_node: completed %d/%d slides",
        len(state["slides"]),
        len(outline),
    )

    return state