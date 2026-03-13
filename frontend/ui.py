"""
GenSlide agentic UI — Streamlit frontend for the LangGraph pipeline.

Run from the project root:
    streamlit run frontend/ui.py

Flow:
    1. User provides text or uploads PDF/DOCX
    2. "Generate" triggers the graph — runs until interrupt_before=["human_approval"]
    3. UI shows the deck outline + slide previews + download button
    4. User either approves (graph → END) or adds feedback and revises (graph loops)

State management:
    All LangGraph state lives in st.session_state["thread"] (the checkpoint key)
    and st.session_state["graph_state"] (the last snapshot for UI rendering).
    Streamlit session state holds UI-only flags like "phase" and "generating".
"""

import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Path setup — run from project root OR from frontend/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.graph import graph
from graph.state import AgentState

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GenSlide",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — "Midnight Executive" design language matching the deck theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root tokens ─────────────────────────────────────────── */
:root {
    --navy:      #1E2761;
    --ice:       #CADCFC;
    --white:     #FFFFFF;
    --off-white: #F4F6FF;
    --muted:     #8892B0;
    --border:    rgba(202,220,252,0.18);
    --card-bg:   rgba(30,39,97,0.06);
    --danger:    #E55353;
    --success:   #2EC4B6;
    --font:      'DM Sans', sans-serif;
    --mono:      'DM Mono', monospace;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, [class*="css"] { font-family: var(--font) !important; }

.stApp { background: var(--off-white); }

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1100px; }

/* ── Top wordmark bar ────────────────────────────────────── */
.gs-topbar {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.gs-logo {
    width: 36px; height: 36px; background: var(--navy);
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; color: var(--ice);
    font-weight: 600; flex-shrink: 0;
}
.gs-wordmark { font-size: 1.35rem; font-weight: 600; color: var(--navy); letter-spacing: -0.02em; }
.gs-tagline  { font-size: 0.78rem; color: var(--muted); margin-left: auto; letter-spacing: 0.05em; }

/* ── Section headings ────────────────────────────────────── */
.gs-section-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 0.6rem;
}

/* ── Input card ──────────────────────────────────────────── */
.gs-input-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.8rem 2rem; margin-bottom: 1.5rem;
}

/* ── Streamlit widget overrides ──────────────────────────── */
.stTextArea textarea {
    font-family: var(--font) !important; font-size: 0.93rem !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    background: var(--off-white) !important; color: var(--navy) !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(30,39,97,0.1) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important; background: var(--off-white) !important;
}

/* Radio pills */
.stRadio [role="radiogroup"] { display: flex; gap: 8px; flex-wrap: wrap; }
.stRadio [role="radiogroup"] label {
    background: var(--off-white); border: 1px solid var(--border);
    border-radius: 20px; padding: 4px 14px; font-size: 0.82rem;
    color: var(--navy); cursor: pointer; transition: all 0.15s;
}
.stRadio [role="radiogroup"] label:has(input:checked) {
    background: var(--navy); color: var(--ice); border-color: var(--navy);
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: var(--navy) !important; color: var(--white) !important;
    border: none !important; border-radius: 10px !important;
    font-family: var(--font) !important; font-size: 0.9rem !important;
    font-weight: 500 !important; padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.01em !important; transition: opacity 0.15s !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.85 !important; }

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: transparent !important; color: var(--navy) !important;
    border: 1.5px solid var(--navy) !important; border-radius: 10px !important;
    font-family: var(--font) !important; font-size: 0.9rem !important;
    font-weight: 500 !important; padding: 0.55rem 1.4rem !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--navy) !important; color: var(--white) !important;
}

/* ── Status pills ────────────────────────────────────────── */
.gs-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.04em;
}
.gs-pill-running  { background: rgba(30,39,97,0.1);  color: var(--navy); }
.gs-pill-ready    { background: rgba(46,196,182,0.15); color: #1a7a72; }
.gs-pill-error    { background: rgba(229,83,83,0.12);  color: var(--danger); }
.gs-pill-approved { background: rgba(46,196,182,0.15); color: #1a7a72; }

/* ── Progress step bar ───────────────────────────────────── */
.gs-steps {
    display: flex; align-items: center; gap: 0;
    margin: 1.5rem 0 2rem; font-size: 0.78rem;
}
.gs-step {
    display: flex; align-items: center; gap: 6px;
    color: var(--muted); white-space: nowrap;
}
.gs-step.active { color: var(--navy); font-weight: 500; }
.gs-step.done   { color: var(--success); }
.gs-step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--border); flex-shrink: 0;
}
.gs-step.active .gs-step-dot { background: var(--navy); }
.gs-step.done   .gs-step-dot { background: var(--success); }
.gs-step-line {
    flex: 1; height: 1px; background: var(--border);
    margin: 0 10px; min-width: 32px;
}

/* ── Outline panel ───────────────────────────────────────── */
.gs-outline {
    background: var(--white); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.gs-outline-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid var(--border);
}
.gs-outline-item:last-child { border-bottom: none; }
.gs-outline-num {
    width: 22px; height: 22px; border-radius: 6px;
    background: var(--navy); color: var(--ice);
    font-size: 0.68rem; font-weight: 600; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
    margin-top: 1px;
}
.gs-outline-title { font-size: 0.88rem; color: var(--navy); font-weight: 500; }

/* ── Slide preview card ──────────────────────────────────── */
.gs-slide-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; margin-bottom: 1rem;
}
.gs-slide-header {
    background: var(--navy); color: var(--white);
    padding: 10px 16px; display: flex; align-items: center; gap: 10px;
}
.gs-slide-num {
    background: var(--ice); color: var(--navy);
    width: 22px; height: 22px; border-radius: 5px;
    font-size: 0.7rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.gs-slide-title-text { font-size: 0.9rem; font-weight: 500; color: var(--white); }
.gs-slide-body { padding: 12px 16px; }
.gs-slide-bullet {
    display: flex; align-items: flex-start; gap: 8px;
    font-size: 0.82rem; color: #3a3f5c; padding: 3px 0; line-height: 1.5;
}
.gs-slide-bullet::before {
    content: "▸"; color: var(--ice); font-size: 0.72rem;
    margin-top: 3px; flex-shrink: 0; filter: brightness(0.7);
}
.gs-notes {
    margin-top: 8px; padding: 8px 12px;
    background: var(--off-white); border-radius: 8px;
    border-left: 3px solid var(--ice);
    font-size: 0.78rem; color: var(--muted); line-height: 1.5;
    font-style: italic;
}

/* ── Approval panel ──────────────────────────────────────── */
.gs-approval-banner {
    background: var(--navy); border-radius: 14px;
    padding: 1.6rem 2rem; margin: 1.5rem 0;
    display: flex; align-items: center; gap: 16px;
}
.gs-approval-icon { font-size: 1.6rem; flex-shrink: 0; }
.gs-approval-text h3 { color: var(--white); font-size: 1rem; margin: 0 0 4px; }
.gs-approval-text p  { color: var(--ice);   font-size: 0.83rem; margin: 0; }

/* ── Iteration badge ─────────────────────────────────────── */
.gs-iter-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(30,39,97,0.08); border-radius: 20px;
    padding: 3px 10px; font-size: 0.75rem; color: var(--navy);
    font-weight: 500; margin-bottom: 1rem;
}

/* ── Download button ─────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    background: var(--success) !important; color: var(--white) !important;
    border: none !important; border-radius: 10px !important;
    font-family: var(--font) !important; font-size: 0.9rem !important;
    font-weight: 500 !important; padding: 0.55rem 1.6rem !important;
}

/* ── Error box ───────────────────────────────────────────── */
.gs-error {
    background: rgba(229,83,83,0.06); border: 1px solid rgba(229,83,83,0.3);
    border-radius: 10px; padding: 12px 16px; margin: 1rem 0;
    font-size: 0.83rem; color: var(--danger);
}

/* ── Divider ─────────────────────────────────────────────── */
.gs-divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
DEFAULTS = {
    "phase":        "input",       # "input" | "generating" | "review" | "done"
    "thread":       {"configurable": {"thread_id": "genslide-1"}},
    "graph_state":  None,          # last AgentState snapshot
    "error_msg":    None,
    "show_notes":   False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset():
    """Return to the input phase, clearing all run state."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # Fresh thread ID to avoid checkpoint collisions
    import uuid
    st.session_state["thread"] = {"configurable": {"thread_id": str(uuid.uuid4())}}


def _get_snapshot() -> Optional[AgentState]:
    """Pull the latest AgentState from the LangGraph checkpoint."""
    try:
        snap = graph.get_state(st.session_state["thread"])
        return snap.values if snap else None
    except Exception:
        return None


def _run_graph(initial_state: AgentState):
    """
    Stream the graph from START until the interrupt fires.
    Updates st.session_state["graph_state"] as nodes complete.
    """
    status_placeholder = st.empty()
    node_labels = {
        "parse_input":    "Parsing input...",
        "orchestrator":   "Planning slide outline...",
        "content_agent":  "Writing slide content...",
        "build_pptx":     "Building presentation...",
        "human_approval": "Awaiting your review...",
    }

    try:
        for event in graph.stream(
            initial_state,
            st.session_state["thread"],
            stream_mode="updates",
        ):
            for node_name in event:
                label = node_labels.get(node_name, node_name)
                status_placeholder.markdown(
                    f'<span class="gs-pill gs-pill-running">⬡ {label}</span>',
                    unsafe_allow_html=True,
                )

        status_placeholder.empty()
        st.session_state["graph_state"] = _get_snapshot()
        st.session_state["phase"] = "review"
        st.session_state["error_msg"] = None

    except Exception as exc:
        status_placeholder.empty()
        logger.exception("Graph stream failed: %s", exc)
        st.session_state["error_msg"] = str(exc)
        st.session_state["phase"] = "input"


def _resume_approve():
    """Resume the graph with approval → terminal END."""
    try:
        graph.invoke(
            Command(resume={"approved": True}),
            st.session_state["thread"],
        )
        st.session_state["graph_state"] = _get_snapshot()
        st.session_state["phase"] = "done"
        st.session_state["error_msg"] = None
    except Exception as exc:
        logger.exception("Resume (approve) failed: %s", exc)
        st.session_state["error_msg"] = str(exc)


def _resume_revise(feedback: str):
    """Resume the graph requesting a revision with the given feedback."""
    prev_state = st.session_state["graph_state"] or {}
    iteration = prev_state.get("iteration", 1) + 1
    try:
        # Inject the revision values, then stream again until next interrupt
        graph.invoke(
            Command(resume={
                "approved":  False,
                "feedback":  feedback.strip(),
                "iteration": iteration,
            }),
            st.session_state["thread"],
        )
        # Now re-stream the revision run
        st.session_state["phase"] = "generating"
        st.rerun()
    except Exception as exc:
        logger.exception("Resume (revise) failed: %s", exc)
        st.session_state["error_msg"] = str(exc)


def _save_upload(uploaded_file) -> str:
    """Save an uploaded file to a temp path and return the path string."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def _render_topbar():
    st.markdown("""
    <div class="gs-topbar">
        <div class="gs-logo">⬡</div>
        <div class="gs-wordmark">GenSlide</div>
        <div class="gs-tagline">AGENTIC PRESENTATION BUILDER</div>
    </div>
    """, unsafe_allow_html=True)


def _render_steps(phase: str):
    steps = [
        ("input",      "Input"),
        ("generating", "Generating"),
        ("review",     "Review"),
        ("done",       "Done"),
    ]
    order = [s[0] for s in steps]
    current_idx = order.index(phase) if phase in order else 0

    html = '<div class="gs-steps">'
    for i, (key, label) in enumerate(steps):
        idx = order.index(key)
        cls = "done" if idx < current_idx else ("active" if idx == current_idx else "")
        icon = "✓ " if cls == "done" else ""
        html += f'<div class="gs-step {cls}"><div class="gs-step-dot"></div>{icon}{label}</div>'
        if i < len(steps) - 1:
            html += '<div class="gs-step-line"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_outline(outline: list[str]):
    if not outline:
        return
    st.markdown('<div class="gs-section-title">Slide outline</div>', unsafe_allow_html=True)
    html = '<div class="gs-outline">'
    for i, title in enumerate(outline, 1):
        html += f"""
        <div class="gs-outline-item">
            <div class="gs-outline-num">{i}</div>
            <div class="gs-outline-title">{title}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_slide_previews(slides: list, show_notes: bool):
    if not slides:
        return
    st.markdown('<div class="gs-section-title">Slide previews</div>', unsafe_allow_html=True)
    for i, slide in enumerate(slides, 1):
        bullets_html = "".join(
            f'<div class="gs-slide-bullet">{b}</div>'
            for b in slide.get("bullets", [])
        )
        notes_html = ""
        if show_notes and slide.get("speaker_notes"):
            notes_html = f'<div class="gs-notes">{slide["speaker_notes"]}</div>'

        st.markdown(f"""
        <div class="gs-slide-card">
            <div class="gs-slide-header">
                <div class="gs-slide-num">{i}</div>
                <div class="gs-slide-title-text">{slide.get("title", "")}</div>
            </div>
            <div class="gs-slide-body">
                {bullets_html}
                {notes_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_error(msg: str):
    st.markdown(
        f'<div class="gs-error">⚠ {msg}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Phase: INPUT
# ---------------------------------------------------------------------------

def phase_input():
    st.markdown('<div class="gs-section-title">Content source</div>', unsafe_allow_html=True)

    input_type = st.radio(
        "Input type",
        options=["Plain text", "Upload PDF", "Upload DOCX"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_input  = ""
    file_path  = None
    input_key  = "text"

    st.markdown('<div class="gs-input-card">', unsafe_allow_html=True)

    if input_type == "Plain text":
        input_key = "text"
        raw_input = st.text_area(
            "Paste your content",
            height=220,
            placeholder="Paste the text you want to turn into a presentation…",
            label_visibility="collapsed",
        )

    elif input_type == "Upload PDF":
        input_key = "pdf"
        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded:
            file_path = _save_upload(uploaded)
            st.caption(f"Loaded: {uploaded.name}  ({uploaded.size:,} bytes)")

    elif input_type == "Upload DOCX":
        input_key = "docx"
        uploaded = st.file_uploader(
            "Upload a DOCX",
            type=["docx"],
            label_visibility="collapsed",
        )
        if uploaded:
            file_path = _save_upload(uploaded)
            st.caption(f"Loaded: {uploaded.name}  ({uploaded.size:,} bytes)")

    st.markdown("</div>", unsafe_allow_html=True)

    # Error display
    if st.session_state["error_msg"]:
        _render_error(st.session_state["error_msg"])

    # Generate button
    can_generate = bool(raw_input.strip()) or bool(file_path)
    if st.button("Generate presentation", type="primary", disabled=not can_generate):
        actual_input = file_path if file_path else raw_input.strip()
        initial_state: AgentState = {
            "raw_input":    actual_input,
            "input_type":   input_key,
            "parsed_text":  "",
            "outline":      [],
            "slides":       [],
            "pptx_path":    None,
            "feedback":     None,
            "approved":     False,
            "iteration":    1,
            "error":        None,
        }
        st.session_state["phase"] = "generating"
        st.session_state["_pending_state"] = initial_state
        st.rerun()


# ---------------------------------------------------------------------------
# Phase: GENERATING
# ---------------------------------------------------------------------------

def phase_generating():
    _render_steps("generating")

    st.markdown("### Generating your deck")
    st.markdown(
        '<p style="color:var(--muted);font-size:0.87rem;margin-bottom:1.5rem">'
        'The agents are working — this usually takes 15–30 seconds.</p>',
        unsafe_allow_html=True,
    )

    # Check if we're continuing a revision (no _pending_state)
    if "_pending_state" in st.session_state:
        initial_state = st.session_state.pop("_pending_state")
        _run_graph(initial_state)
    else:
        # Revision continuation — stream from current checkpoint
        state = st.session_state.get("graph_state") or {}
        status = st.empty()
        status.markdown(
            '<span class="gs-pill gs-pill-running">⬡ Resuming revision...</span>',
            unsafe_allow_html=True,
        )
        try:
            for event in graph.stream(
                None,
                st.session_state["thread"],
                stream_mode="updates",
            ):
                pass
            status.empty()
            st.session_state["graph_state"] = _get_snapshot()
            st.session_state["phase"] = "review"
            st.session_state["error_msg"] = None
        except Exception as exc:
            status.empty()
            logger.exception("Revision stream failed: %s", exc)
            st.session_state["error_msg"] = str(exc)
            st.session_state["phase"] = "review"

    st.rerun()


# ---------------------------------------------------------------------------
# Phase: REVIEW
# ---------------------------------------------------------------------------

def phase_review():
    _render_steps("review")

    state = st.session_state.get("graph_state") or {}
    iteration  = state.get("iteration", 1)
    outline    = state.get("outline") or []
    slides     = state.get("slides") or []
    pptx_path  = state.get("pptx_path")
    error      = state.get("error")

    # Iteration badge
    if iteration > 1:
        st.markdown(
            f'<div class="gs-iter-badge">↻ Revision {iteration}</div>',
            unsafe_allow_html=True,
        )

    # Partial error warning (non-fatal)
    if error:
        _render_error(f"Note: {error}")

    # ── Layout: left = outline + previews, right = actions ──
    left, right = st.columns([3, 1.4], gap="large")

    with left:
        _render_outline(outline)

        st.markdown('<hr class="gs-divider">', unsafe_allow_html=True)

        show_notes = st.toggle("Show speaker notes", value=st.session_state["show_notes"])
        st.session_state["show_notes"] = show_notes

        _render_slide_previews(slides, show_notes)

    with right:
        # Approval banner
        st.markdown("""
        <div class="gs-approval-banner">
            <div class="gs-approval-icon">⬡</div>
            <div class="gs-approval-text">
                <h3>Ready to review</h3>
                <p>Download the deck, then approve or request changes below.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download button
        if pptx_path and Path(pptx_path).exists():
            with open(pptx_path, "rb") as f:
                st.download_button(
                    label="Download .pptx",
                    data=f.read(),
                    file_name=Path(pptx_path).name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
        else:
            st.warning("PPTX file not found. Try regenerating.")

        st.markdown('<hr class="gs-divider">', unsafe_allow_html=True)

        # Approve
        st.markdown('<div class="gs-section-title">Approve</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.8rem;color:var(--muted);margin-bottom:0.7rem">'
            'Happy with the deck? Finalise it here.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Approve deck", type="primary", use_container_width=True):
            _resume_approve()
            st.rerun()

        st.markdown('<hr class="gs-divider">', unsafe_allow_html=True)

        # Revise
        st.markdown('<div class="gs-section-title">Request revision</div>', unsafe_allow_html=True)
        feedback = st.text_area(
            "Feedback",
            height=110,
            placeholder="e.g. Add a slide about pricing. Make the intro more concise.",
            label_visibility="collapsed",
        )
        if st.button(
            "Revise",
            type="secondary",
            disabled=not feedback.strip(),
            use_container_width=True,
        ):
            _resume_revise(feedback)
            st.rerun()

        st.markdown('<hr class="gs-divider">', unsafe_allow_html=True)

        # Start over
        if st.button("Start over", use_container_width=True):
            _reset()
            st.rerun()

        # Deck stats
        if slides:
            st.markdown(
                f'<p style="font-size:0.75rem;color:var(--muted);text-align:center;margin-top:1rem">'
                f'{len(slides)} slides · iteration {iteration}</p>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Phase: DONE
# ---------------------------------------------------------------------------

def phase_done():
    _render_steps("done")

    state     = st.session_state.get("graph_state") or {}
    pptx_path = state.get("pptx_path")
    slides    = state.get("slides") or []
    iteration = state.get("iteration", 1)

    st.markdown("""
    <div style="text-align:center;padding:3rem 0 2rem">
        <div style="font-size:2.8rem;margin-bottom:0.8rem">⬡</div>
        <h2 style="color:var(--navy);font-weight:600;margin-bottom:0.4rem">
            Presentation approved
        </h2>
        <p style="color:var(--muted);font-size:0.9rem">
            Your deck has been finalised and is ready to use.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if pptx_path and Path(pptx_path).exists():
            with open(pptx_path, "rb") as f:
                st.download_button(
                    label="Download final .pptx",
                    data=f.read(),
                    file_name=Path(pptx_path).name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
        else:
            st.warning("File not found. Check frontend/generated/")

        st.markdown(
            f'<p style="font-size:0.78rem;color:var(--muted);text-align:center;margin-top:0.6rem">'
            f'{len(slides)} slides · {iteration} iteration{"s" if iteration > 1 else ""}</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 1, 2])
    with col_b:
        if st.button("Build another deck", type="secondary", use_container_width=True):
            _reset()
            st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main():
    _render_topbar()

    phase = st.session_state["phase"]
    _render_steps(phase)

    if phase == "input":
        phase_input()
    elif phase == "generating":
        phase_generating()
    elif phase == "review":
        phase_review()
    elif phase == "done":
        phase_done()


if __name__ == "__main__":
    main()