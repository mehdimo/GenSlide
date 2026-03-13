"""
Defines the shared AgentState TypedDict that flows through every node
in the LangGraph pipeline. All nodes read from and write to this state.
"""

from typing import TypedDict, Optional, List, Literal


class SlideContent(TypedDict):
    """Structured content for a single slide."""
    title: str
    bullets: List[str]
    speaker_notes: str


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the GenSlide LangGraph.

    Lifecycle:
        raw_input      → set by the user via Streamlit
        parsed_text    → filled by input_parser node
        outline        → filled by orchestrator node
        slides         → filled by content_agent node
        pptx_path      → filled by pptx_builder node
        feedback       → set by human_approval node on revision
        approved       → set True by human_approval node on accept
        iteration      → incremented on each revision loop
        error          → set if any node raises an exception
    """

    # --- Input ---
    raw_input: str
    # "text" | "pdf" | "docx"
    input_type: Literal["text", "pdf", "docx"]

    # --- Parsed content ---
    parsed_text: str

    # --- Orchestrator output ---
    # List of slide titles forming the deck outline
    outline: List[str]

    # --- Content agent output ---
    # One SlideContent dict per slide title in outline
    slides: List[SlideContent]

    # --- Builder output ---
    # Filesystem path to the generated .pptx file
    pptx_path: Optional[str]

    # --- Human-in-the-loop ---
    # Free-text revision notes from the user (empty string = no notes)
    feedback: Optional[str]
    # Set to True when the user clicks "Approve" in the Streamlit UI
    approved: bool

    # --- Control ---
    # Counts how many times the pipeline has run (starts at 1)
    iteration: int
    # Holds a human-readable error message if a node fails
    error: Optional[str]