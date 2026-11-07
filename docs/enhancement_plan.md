# Enhancement Plan: Human-in-the-Loop Approval Node

Project: LangGraph Stateful Agent Pipeline
Author: Siriyala Nishar
Status: **Awaiting mentor approval** (required before Week 4 starts)

## Goal

Add a human-in-the-loop checkpoint so the agent **pauses for approval before any irreversible tool call**
and only continues when a person approves. A rejection stops the run without the action happening.

Today the pipeline loops through research, summarise and critique until the score is high enough, then ends.
`data/sample_dataset.csv` already marks the final step (`send_report`) with `requires_approval = True`, but
nothing in the graph enforces it. This enhancement makes that flag real.

## Technique

LangGraph's interrupt and resume mechanism with a checkpointer:

- A new `approval` node calls `interrupt(...)` with the pending action (what will be sent, and the critique score).
  The graph stops and its state is saved.
- A **checkpointer** stores the state after each step, keyed by `thread_id`, so the run can pause for as long
  as needed and resume later.
- The caller resumes with `Command(resume="approve")` or `Command(resume="reject")`. A conditional edge routes
  approve to the `send_report` node (the irreversible action, a safe stub for now) and reject to the end.
- `thread_id` in the config identifies the run. This matches the `/invoke` endpoint planned for W4D5, which
  accepts `{"input": "...", "thread_id": "..."}`.

Design note: when a graph resumes, the interrupted node restarts from its beginning, so any side effect must
sit after the `interrupt()` call or in a later node. The irreversible action therefore lives in `send_report`,
never in `approval`.

The exact function names and imports will be checked against the documentation for the installed
langgraph version (1.2.16) before coding.

## Expected improvement

This is a control feature, not a model, so success is measured by behaviour rather than an accuracy score.

| Measure                                              | Now          | Target                               |
| ---------------------------------------------------- | ------------ | ------------------------------------ |
| Irreversible actions executed without approval       | Not enforced | 0 in all test scenarios              |
| Run pauses at the approval step and keeps its state  | Not possible | Yes, resumes on the same `thread_id` |
| Rejected run performs the action                     | n/a          | Never                                |
| Approved run completes with the final state returned | n/a          | Yes                                  |

Test scenarios: approve, reject, resume with the wrong or an unknown `thread_id`, and a run that never
reaches approval (low critique score loops back to research).

## Files to change

| File                                       | Change                                                                                        |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `week1_run.py` (or a new `agent/graph.py`) | Add `approval` and `send_report` nodes, the conditional edge, and compile with a checkpointer |
| `approval_demo.py` (new)                   | Command-line demo: start a run, show the pause, then approve or reject                        |
| `tests/test_approval.py` (new)             | Checks that nothing is sent without approval                                                  |
| `README.md`                                | Document the approval flow and add a diagram                                                  |
| `app.py` (Week 4, Day 5)                   | `/invoke` endpoint reuses the same graph and `thread_id`                                      |

## New libraries and data

- No new required libraries: `langgraph` already provides `interrupt`, `Command` and an in-memory checkpointer.
- `pytest` for the tests (optional; plain assert-based scripts also work).
- Data: reuse `data/sample_dataset.csv`, specifically its `requires_approval` column.
- Known limit: an in-memory checkpointer loses saved state when the process restarts. That is fine for the demo
  and for a running API server. A persistent checkpointer (for example SQLite) would be a later improvement.

## Sub-tasks and time estimate

| #   | Sub-task                                                                                                               | Estimate |
| --- | ---------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | Add the `approval` node with `interrupt()` and the `send_report` stub, wire the conditional edge, add the checkpointer | 1.5 h    |
| 2   | Resume handling with `Command(resume=...)` and `thread_id`, plus `approval_demo.py` showing pause, approve and reject  | 1.5 h    |
| 3   | Tests for the scenarios above, then README update and diagram                                                          | 1.5 h    |

Total: about 4.5 hours, spread over Week 4 (Day 4 starts sub-task 1).

## Risks

- Interrupt and resume API details differ between langgraph versions, so check the docs first.
- Forgetting the `thread_id` or the checkpointer means the graph cannot resume; the tests cover this.
- Putting a side effect before `interrupt()` would run it twice on resume; the design keeps it after.
