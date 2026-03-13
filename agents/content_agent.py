"""
Content agent node for the GenSlide LangGraph pipeline.

Uses a single unified prompt for all LLM providers (OpenAI and local).
The parser uses a 3-strategy fallback chain to extract usable slide
content from any model output, regardless of formatting quality.

Parallelism:
    OpenAI  → ThreadPoolExecutor(max_workers=6) — independent cloud calls
    Local   → Sequential (max_workers=1) — CPU inference; parallel calls
              would thrash memory and be slower than sequential
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState, SlideContent
from llm.llm_provider import get_llm, get_provider_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a slide content writer. Write the content for a single \
PowerPoint slide.

Given a slide title and source text, return a JSON object with \
exactly these three keys:

  "title"         : string — the slide title, copied exactly
  "bullets"       : array of 3 to 5 strings — concise bullet points
                    drawn from the source text; each bullet max 15 words;
                    no leading dashes or bullet characters
  "speaker_notes" : string — 2 to 4 sentences for the presenter,
                    expanding on the bullets in natural spoken prose;
                    do not repeat bullet text verbatim

Return ONLY a JSON object. Example:
{
  "title": "The slide title",
  "bullets": ["First point", "Second point", "Third point"],
  "speaker_notes": "Spoken notes for the presenter."
}
""".strip()

REVISION_SUFFIX = """

The user requested changes: "{feedback}"

Previous content for this slide:
{prev_content}

Revise only what the feedback asks for. Keep everything else the same.
Return the updated JSON object only.
""".strip()

# ---------------------------------------------------------------------------
# Parser — 3-strategy fallback chain
# ---------------------------------------------------------------------------

def _parse_response(raw: str, expected_title: str) -> SlideContent:
    """
    Extract SlideContent from the model response.

    Tries in order:
        1. Full JSON parse — {"title":..., "bullets":[...], "speaker_notes":...}
        2. Regex extraction of a JSON object block
        3. Heuristic line-by-line extraction as a last resort
    """
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # 1. Full JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _coerce(data, expected_title)
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _coerce(data, expected_title)
        except json.JSONDecodeError:
            pass

    # 3. Heuristic: scrape any non-empty lines as bullets
    logger.warning(
        "Slide '%s': JSON parse failed, falling back to line extraction.",
        expected_title,
    )
    lines = [
        re.sub(r"^[-\*\•\d\.]+\s*", "", ln).strip()
        for ln in text.splitlines()
        if ln.strip() and len(ln.strip()) > 4
    ]
    bullets = [ln for ln in lines if ln][:5]
    if not bullets:
        bullets = ["Content could not be parsed for this slide."]

    return SlideContent(
        title=expected_title,
        bullets=bullets,
        speaker_notes="",
    )


def _coerce(data: dict, expected_title: str) -> SlideContent:
    """Coerce a parsed dict into a SlideContent, filling missing fields safely."""
    title   = str(data.get("title",   expected_title)).strip() or expected_title
    bullets = data.get("bullets", [])
    notes   = str(data.get("speaker_notes", data.get("notes", ""))).strip()

    if not isinstance(bullets, list):
        # Model may have returned a single string or a dict
        bullets = [str(bullets)]
    bullets = [str(b).strip() for b in bullets if str(b).strip()]

    if not bullets:
        bullets = ["Content could not be extracted for this slide."]

    return SlideContent(title=title, bullets=bullets, speaker_notes=notes)


# ---------------------------------------------------------------------------
# Single-slide generation with retry
# ---------------------------------------------------------------------------

def _generate_slide(
    title: str,
    parsed_text: str,
    feedback: Optional[str],
    prev_slide: Optional[SlideContent],
    iteration: int,
    max_retries: int = 1,
) -> SlideContent:
    is_revision = bool(feedback and iteration > 1 and prev_slide)

    if is_revision:
        prev_content = (
            f"Bullets:\n"
            + "\n".join(f"- {b}" for b in prev_slide["bullets"])
            + f"\nSpeaker notes: {prev_slide['speaker_notes']}"
        )
        human_content = (
            f"Slide title: {title}\n\n"
            f"Source text:\n{parsed_text}\n\n"
            + REVISION_SUFFIX.format(
                feedback=feedback.strip(),
                prev_content=prev_content,
            )
        )
    else:
        human_content = f"Slide title: {title}\n\nSource text:\n{parsed_text}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
    llm = get_llm(temperature=0.5)

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            return _parse_response(response.content, title)
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Slide '%s' attempt %d failed (%s). Retrying in %ds…",
                    title, attempt + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def content_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: write slide content for every title in state["outline"].

    Reads:
        state["outline"]      — List[str] of slide titles
        state["parsed_text"]  — full source text
        state["feedback"]     — revision notes (may be None)
        state["slides"]       — previous slides (used on revisions)
        state["iteration"]    — current iteration count

    Writes:
        state["slides"]  — List[SlideContent] in outline order
        state["error"]   — error message on failure (else None)
    """
    if state.get("error"):
        logger.warning("content_agent_node skipped — upstream error: %s", state["error"])
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

    prev_by_title = {s["title"]: s for s in prev_slides}

    # Local models run on a single CPU thread — parallel calls are slower
    max_workers = 1 if get_provider_name() == "local" else min(len(outline), 6)

    logger.info(
        "content_agent_node: generating %d slides via %s (workers=%d, iteration=%d)",
        len(outline), get_provider_name(), max_workers, iteration,
    )

    slides: list[Optional[SlideContent]] = [None] * len(outline)
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            idx   = future_to_index[future]
            title = outline[idx]
            try:
                slides[idx] = future.result()
                logger.info("  ✓ Slide %d/%d: '%s'", idx + 1, len(outline), title)
            except Exception as exc:
                err_msg = f"Slide '{title}' failed: {exc}"
                logger.error("  ✗ %s", err_msg)
                errors.append(err_msg)
                slides[idx] = SlideContent(
                    title=title,
                    bullets=["Content could not be generated for this slide."],
                    speaker_notes="",
                )

    if errors:
        state["error"] = f"{len(errors)} slide(s) failed: " + "; ".join(errors)
        logger.warning("content_agent_node partial failure: %s", state["error"])
    else:
        state["error"] = None

    state["slides"] = [s for s in slides if s is not None]
    logger.info(
        "content_agent_node: completed %d/%d slides",
        len(state["slides"]), len(outline),
    )
    return state