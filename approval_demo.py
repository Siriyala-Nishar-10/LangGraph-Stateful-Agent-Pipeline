"""W4D4 skeleton: start a run and show that it pauses at the approval step.

Run from the project root:  python approval_demo.py
"""
from approval_graph import approval_app

config = {"configurable": {"thread_id": "demo-1"}}
start = {"question": "What is LangGraph?", "notes": "", "summary": "", "score": 0, "loops": 0}

# Sub-task 1: the run stops at the approval node instead of sending the report
approval_app.invoke(start, config=config)
snapshot = approval_app.get_state(config)
print("Paused before node:", snapshot.next)  # expected: ('approval',)
print("State so far:", snapshot.values)
print("Report sent yet?", snapshot.values.get("sent", False))  # expected: False

# TODO (sub-task 2): ask the user to type approve or reject, then resume the run with
#   from langgraph.types import Command
#   approval_app.invoke(Command(resume=decision), config=config)
# TODO (sub-task 2): show the final state for both an approved and a rejected run
