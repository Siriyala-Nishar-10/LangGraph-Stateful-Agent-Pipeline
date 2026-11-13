"""W4D4: human-in-the-loop approval enhancement (sub-task 1 of docs/enhancement_plan.md).

The original pipeline (week1_run.py) is left untouched. This module reuses its nodes and adds:
critique -> approval (pauses for a human) -> send_report (the irreversible action) or end.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from week1_run import MAX_LOOPS, State, critique, research, summarise


class ApprovalState(State):
    approved: bool
    sent: bool


def route_after_critique(state: ApprovalState) -> str:
    """Loop back to research until the summary is good enough, then ask for approval."""
    if state["score"] >= 7 or state["loops"] >= MAX_LOOPS:
        return "approval"
    return "research"


def approval(state: ApprovalState) -> dict:
    """Pause the run and wait for a human decision ("approve" or "reject").

    interrupt() saves the state and stops the graph. When the run is resumed, this node restarts
    from the top, so nothing with side effects may run before the interrupt() call.
    """
    decision = interrupt({
        "action": "send_report",
        "summary": state["summary"],
        "score": state["score"],
    })
    return {"approved": decision == "approve"}


def route_after_approval(state: ApprovalState) -> str:
    return "send_report" if state["approved"] else END


def send_report(state: ApprovalState) -> dict:
    """The irreversible action (a safe stub for now). It only runs after an approval."""
    # TODO (later): replace the stub with a real tool call, e.g. email the report to the user
    print("[stub] report sent")
    return {"sent": True}


def build_approval_graph(checkpointer=None):
    builder = StateGraph(ApprovalState)
    builder.add_node("research", research)
    builder.add_node("summarise", summarise)
    builder.add_node("critique", critique)
    builder.add_node("approval", approval)
    builder.add_node("send_report", send_report)

    builder.add_edge(START, "research")
    builder.add_edge("research", "summarise")
    builder.add_edge("summarise", "critique")
    builder.add_conditional_edges("critique", route_after_critique, ["research", "approval"])
    builder.add_conditional_edges("approval", route_after_approval, ["send_report", END])
    builder.add_edge("send_report", END)
    return builder.compile(checkpointer=checkpointer)


# A checkpointer is required for interrupt() and for resuming a run by thread_id.
# In memory: saved state is lost when the process restarts.
# TODO (later): use a persistent checkpointer (for example SQLite) for production.
approval_app = build_approval_graph(InMemorySaver())
