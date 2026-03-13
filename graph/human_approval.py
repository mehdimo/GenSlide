"""
The human_approval node is a pass-through — it does nothing by itself.
LangGraph pauses execution *before* this node (interrupt_before=["human_approval"])
and waits for the Streamlit UI to resume the graph via Command(resume={...}).

The actual approval/feedback values are injected by the UI into the state
at resume time, so this node simply returns state unchanged.
"""

from graph.state import AgentState


def human_approval_node(state: AgentState) -> AgentState:
    """
    Pass-through node. State is mutated by the Streamlit UI at resume time:

        Approve path:
            graph.invoke(Command(resume={"approved": True}), thread)

        Revise path:
            graph.invoke(Command(resume={
                "approved": False,
                "feedback": "<user notes>",
                "iteration": state["iteration"] + 1,
            }), thread)
    """
    return state