"""
Defines and compiles the GenSlide LangGraph StateGraph.

Graph topology:
    START
      └─► parse_input
            └─► orchestrator
                  └─► content_agent
                        └─► build_pptx
                              └─► human_approval  ← interrupt_before here
                                    ├─► END          (approved)
                                    └─► orchestrator  (revise)
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from tools.input_parser import parse_input_node
from tools.pptx_builder import build_pptx_node
from agents.orchestrator import orchestrator_node
from agents.content_agent import content_agent_node
from graph.human_approval import human_approval_node


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_approval(state: AgentState) -> str:
    """
    Conditional edge coming out of human_approval.

    Returns:
        "end"          — user approved the deck → terminate the graph
        "orchestrator" — user requested revisions → loop back to re-plan
    """
    if state.get("approved"):
        return "end"
    return "orchestrator"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Assembles and returns the compiled GenSlide StateGraph.

    The graph is compiled with:
        - MemorySaver checkpointer  → persists state across the
          interrupt so the Streamlit UI can resume after user input
        - interrupt_before=["human_approval"]  → pauses execution
          before the approval node so the UI can show the draft deck
    """
    builder = StateGraph(AgentState)

    # --- Register nodes ---
    builder.add_node("parse_input",    parse_input_node)
    builder.add_node("orchestrator",   orchestrator_node)
    builder.add_node("content_agent",  content_agent_node)
    builder.add_node("build_pptx",     build_pptx_node)
    builder.add_node("human_approval", human_approval_node)

    # --- Entry point ---
    builder.set_entry_point("parse_input")

    # --- Linear edges ---
    builder.add_edge("parse_input",   "orchestrator")
    builder.add_edge("orchestrator",  "content_agent")
    builder.add_edge("content_agent", "build_pptx")
    builder.add_edge("build_pptx",    "human_approval")

    # --- Conditional edge: approve → END, revise → re-plan ---
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "end":          END,
            "orchestrator": "orchestrator",
        },
    )

    # --- Compile with checkpointing and human-in-the-loop interrupt ---
    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )

    return graph


# ---------------------------------------------------------------------------
# Module-level singleton — import this in the Streamlit UI
# ---------------------------------------------------------------------------

graph = build_graph()