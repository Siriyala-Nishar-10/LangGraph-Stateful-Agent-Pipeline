"""W4D4 skeleton: tests for the approval flow (sub-task 3 of docs/enhancement_plan.md).

Run with:  pytest tests
"""
import pytest

TODO = "TODO sub-task 3"


@pytest.mark.skip(reason=TODO)
def test_run_pauses_before_sending():
    """A new run stops at the approval node and has not sent the report."""


@pytest.mark.skip(reason=TODO)
def test_approve_sends_report():
    """Resuming with "approve" sends the report and returns the final state."""


@pytest.mark.skip(reason=TODO)
def test_reject_never_sends_report():
    """Resuming with "reject" ends the run and the report is never sent."""


@pytest.mark.skip(reason=TODO)
def test_unknown_thread_id_does_not_send():
    """Resuming a thread that was never started must not send anything."""
