"""Typed run outcomes and nested-work completion (hermes-sip-live-voice#49, #50).

The lane used to return a plain string for every terminal state and mark
every normal return ``done``, so a cancelled or failed api_server run was
announced as "finished". Here every terminal shape the host can produce is
driven through the worker and the registry, and the assertion is the same
each time: only a genuine success may ever be spoken as finished.

Zero network: ``start_run``/``get_run``/session reads are faked at the module
seam, exactly as the rest of the lane suite does.
"""

from __future__ import annotations

import contextlib
import threading
import time

import pytest

import talk_apiserver
import talk_approvals
import talk_cli
import talk_host
import talk_runs
import talk_tools


def _wait_terminal(run_id: int, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = talk_runs.get_run(run_id)
        if run and run["status"] in talk_runs.TERMINAL_STATUSES:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()
    talk_apiserver.reset_for_tests()
    talk_approvals.reset_for_tests()
    talk_runs.attach_owner(
        talk_session_id="ts-test",
        generation_id="gen-test",
        hermes_session_id="sess-test",
        operator="test",
        profile=None,
    )
    monkeypatch.setattr(talk_host, "hermes_binary", lambda: None)
    monkeypatch.setenv("TALK_API_SERVER_POLL_S", "0.01")
    monkeypatch.setenv("TALK_AGENT_TIMEOUT_S", "2")
    monkeypatch.setattr(talk_approvals, "watch_run", lambda run_id, api_run_id: None)
    monkeypatch.setattr(talk_runs, "_history_enabled", lambda: False)
    yield
    talk_runs.reset_for_tests()
    talk_approvals.reset_for_tests()


def _fake_remote(monkeypatch, states: list[dict], run_id: str = "run_remote_1") -> None:
    it = iter(states)
    last: dict = {}

    def get_run(_rid):
        nonlocal last
        with contextlib.suppress(StopIteration):
            last = next(it)
        return last

    monkeypatch.setattr(talk_apiserver, "start_run", lambda *a, **k: run_id)
    monkeypatch.setattr(talk_apiserver, "get_run", get_run)


def _spoken(run: dict) -> str:
    """The announcement headline the model would be handed for this run."""

    commands = talk_cli.run_finished_commands(run)
    return commands[0].text


# -- run_to_outcome: one typed shape per host terminal state -----------------


def test_success_is_the_only_outcome_that_reads_as_finished(monkeypatch):
    _fake_remote(
        monkeypatch, [{"status": "running"}, {"status": "completed", "output": "72 and sunny"}]
    )
    out = talk_apiserver.run_to_outcome("weather")
    assert out.outcome == talk_apiserver.OUTCOME_SUCCESS
    assert out.succeeded
    assert out.output == "72 and sunny"
    assert out.remote_status == "completed"


@pytest.mark.parametrize(
    "remote, outcome",
    [
        ("failed", talk_apiserver.OUTCOME_FAILED),
        ("cancelled", talk_apiserver.OUTCOME_CANCELLED),
        ("interrupted", talk_apiserver.OUTCOME_INTERRUPTED),
    ],
)
def test_each_non_success_terminal_state_keeps_its_own_category(monkeypatch, remote, outcome):
    _fake_remote(
        monkeypatch,
        [
            {"status": "running"},
            {"status": remote, "error": "operator stopped it", "output": "half a report"},
        ],
    )
    out = talk_apiserver.run_to_outcome("go")
    assert out.outcome == outcome
    assert not out.succeeded
    assert out.remote_status == remote
    # Partial output survives, labelled as such by the category, never lost.
    assert out.output == "half a report"
    assert "operator stopped it" in out.speakable()
    assert "finished" not in out.speakable()


def test_an_unknown_future_status_is_never_success_and_never_polled_forever(monkeypatch):
    _fake_remote(
        monkeypatch, [{"status": "running"}, {"status": "quarantined", "error": "new host state"}]
    )
    started = time.monotonic()
    out = talk_apiserver.run_to_outcome("go")
    assert time.monotonic() - started < 1.0, "an unrecognized status must not spin to the deadline"
    assert out.outcome == talk_apiserver.OUTCOME_UNKNOWN
    assert out.remote_status == "quarantined"
    assert "don't recognize" in out.speakable()


def test_lost_polling_is_a_timeout_outcome_with_partial_output(monkeypatch):
    monkeypatch.setattr(talk_apiserver.talk_config, "agent_timeout_s", lambda: 0.05)
    _fake_remote(monkeypatch, [{"status": "running", "output": "so far: 3 of 9 files"}])
    out = talk_apiserver.run_to_outcome("go")
    assert out.outcome == talk_apiserver.OUTCOME_TIMEOUT
    assert out.output == "so far: 3 of 9 files"
    assert "stopped waiting" in out.speakable()


def test_the_string_wrapper_keeps_its_documented_shapes(monkeypatch):
    _fake_remote(monkeypatch, [{"status": "cancelled", "error": "stop requested"}])
    assert talk_apiserver.run_to_completion("go").startswith("the agent run cancelled")
    monkeypatch.setattr(talk_apiserver.talk_config, "agent_timeout_s", lambda: 0.05)
    _fake_remote(monkeypatch, [{"status": "running"}])
    with pytest.raises(talk_apiserver.TalkApiServerError):
        talk_apiserver.run_to_completion("go")


def test_legacy_text_is_classified_not_trusted():
    assert (
        talk_apiserver.outcome_from_text("the agent run cancelled: by user").outcome == "cancelled"
    )
    assert talk_apiserver.outcome_from_text("the agent run failed: boom").outcome == "failed"
    assert talk_apiserver.outcome_from_text("the agent run vanished: ?").outcome == "unknown"
    ok = talk_apiserver.outcome_from_text("all good, 3 files changed")
    assert ok.succeeded and ok.output == "all good, 3 files changed"


# -- the registry: status stays two-valued, outcome carries the truth ---------


def test_finish_run_refuses_a_done_status_with_a_failing_outcome():
    run_id = talk_runs.start_run("agent", "x", lambda rid: "unused")
    _wait_terminal(run_id)
    with pytest.raises(ValueError):
        talk_runs.finish_run(run_id, "done", "x", outcome=talk_runs.OUTCOME_CANCELLED)


def test_worker_records_cancelled_as_failed_status_with_cancelled_outcome(monkeypatch):
    _fake_remote(
        monkeypatch,
        [{"status": "running"}, {"status": "cancelled", "error": "stopped", "output": "partial"}],
    )
    run_id = talk_runs.start_run(
        "agent", "triage", talk_host._api_server_worker("go", session_id=None)
    )
    run = _wait_terminal(run_id)
    assert run["status"] == "failed"
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_CANCELLED
    assert run["meta"]["remote_status"] == "cancelled"
    spoken = _spoken(run)
    assert "was cancelled" in spoken
    assert "finished" not in spoken
    assert "partial or diagnostic" in spoken
    # check_work says the same thing.
    assert "cancelled" in talk_tools.execute_talk_tool("check_work", {"run_id": run_id})


def test_worker_records_interrupted_and_the_announcement_says_so(monkeypatch):
    _fake_remote(monkeypatch, [{"status": "interrupted", "error": "gateway restart"}])
    run_id = talk_runs.start_run(
        "agent", "triage", talk_host._api_server_worker("go", session_id=None)
    )
    run = _wait_terminal(run_id)
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_INTERRUPTED
    assert "interrupted" in _spoken(run)


def test_worker_records_a_real_success_as_finished(monkeypatch):
    _fake_remote(monkeypatch, [{"status": "completed", "output": "done: 4 emails triaged"}])
    run_id = talk_runs.start_run(
        "agent", "triage", talk_host._api_server_worker("go", session_id=None)
    )
    run = _wait_terminal(run_id)
    assert run["status"] == "done"
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_SUCCESS
    assert "is done" in _spoken(run)
    assert "partial" not in _spoken(run)


def test_records_from_before_outcomes_existed_still_read_sensibly():
    assert talk_runs.run_outcome({"status": "done", "meta": {}}) == talk_runs.OUTCOME_SUCCESS
    assert talk_runs.run_outcome({"status": "failed", "meta": {}}) == talk_runs.OUTCOME_FAILED
    assert talk_runs.run_outcome({"status": "lost"}) == talk_runs.OUTCOME_UNKNOWN


# -- nested delegated work (#50) ---------------------------------------------


def test_children_seen_counts_lifecycle_events_from_the_sidecar():
    talk_approvals._note_event(7, "run_remote_7", {"event": "subagent.start", "goal": "a"})
    talk_approvals._note_event(7, "run_remote_7", {"event": "subagent.start", "goal": "b"})
    talk_approvals._note_event(
        7, "run_remote_7", {"event": "subagent.complete", "status": "completed"}
    )
    assert talk_approvals.children_seen(7) == (2, 1)
    talk_approvals.forget_children(7)
    assert talk_approvals.children_seen(7) == (0, 0)


def test_a_completed_turn_with_children_outstanding_is_incomplete_not_success(monkeypatch):
    _fake_remote(
        monkeypatch, [{"status": "completed", "output": "Review in progress, I'll report back."}]
    )
    out = talk_apiserver.run_to_outcome("go", child_counter=lambda: (2, 0))
    assert out.outcome == talk_apiserver.OUTCOME_INCOMPLETE
    assert out.children_outstanding == 2
    assert "2 helper(s) are still working" in out.speakable()
    assert not out.succeeded


def test_worker_waits_for_parked_child_results_and_synthesizes_once(monkeypatch):
    """The Bob shape: parent says 'in progress', two children land later, ONE final."""

    monkeypatch.setattr(talk_host, "CHILD_WAIT_POLL_S", 0.01)
    submissions: list[str] = []
    remote_states = {
        "run_parent": [{"status": "completed", "output": "Started two reviews, back shortly."}],
        "run_final": [
            {"status": "completed", "output": "Both reviews done: 3 urgent, 1 to draft."}
        ],
    }

    def start_run(prompt, **_kw):
        submissions.append(prompt)
        return "run_parent" if len(submissions) == 1 else "run_final"

    def get_run(rid):
        states = remote_states[rid]
        row = states.pop(0) if len(states) > 1 else states[0]
        row = dict(row)
        row["session_id"] = "sess_A"
        return row

    parked: list[dict] = []

    def session_messages(session_id, *, limit=50):
        assert session_id == "sess_A"
        return list(parked)

    monkeypatch.setattr(talk_apiserver, "start_run", start_run)
    monkeypatch.setattr(talk_apiserver, "get_run", get_run)
    monkeypatch.setattr(talk_apiserver, "session_messages", session_messages)

    # The sidecar saw two children start and none finish before the parent returned.
    counts = {"v": (2, 0)}
    monkeypatch.setattr(talk_approvals, "children_seen", lambda rid: counts["v"])

    run_id = talk_runs.start_run(
        "agent", "triage", talk_host._api_server_worker("go", session_id=None)
    )

    # While children are outstanding the job is WORKING and says why.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        run = talk_runs.get_run(run_id)
        if run and run["meta"].get("phase") == "awaiting_children":
            break
        time.sleep(0.01)
    assert run["status"] == "running"
    assert "waiting on 2 helper(s)" in talk_tools.execute_talk_tool(
        "check_work", {"run_id": run_id}
    )

    # Children land out of order on the session as delivery rows.
    parked.append(
        {"id": 11, "display_kind": "async_delegation_complete", "content": "child B result"}
    )
    time.sleep(0.05)
    assert talk_runs.get_run(run_id)["status"] == "running", "one of two is not done"
    parked.append(
        {"id": 12, "display_kind": "async_delegation_complete", "content": "child A result"}
    )
    counts["v"] = (0, 0)  # the synthesis turn spawns nothing

    run = _wait_terminal(run_id)
    assert run["status"] == "done"
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_SUCCESS
    assert run["output"].startswith("Both reviews done")
    # Exactly one follow-up run, on the same session, and it was the synthesis prompt.
    assert len(submissions) == 2
    assert submissions[1] == talk_host.CONTINUATION_PROMPT
    assert "is done" in _spoken(run)


def test_children_that_never_report_leave_an_honest_incomplete(monkeypatch):
    monkeypatch.setattr(talk_host.talk_config, "agent_timeout_s", lambda: 0.1)
    monkeypatch.setattr(talk_host, "CHILD_WAIT_POLL_S", 0.01)
    _fake_remote(
        monkeypatch, [{"status": "completed", "output": "kicked off", "session_id": "sess_B"}]
    )
    monkeypatch.setattr(talk_apiserver, "session_messages", lambda sid, *, limit=50: [])
    monkeypatch.setattr(talk_approvals, "children_seen", lambda rid: (1, 0))
    run_id = talk_runs.start_run(
        "agent", "triage", talk_host._api_server_worker("go", session_id=None)
    )
    run = _wait_terminal(run_id, timeout=4.0)
    assert run["status"] == "failed"
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_INCOMPLETE
    assert "never reported back" in run["output"]
    spoken = _spoken(run)
    assert "ended without a final result" in spoken
    assert "finished" not in spoken


def test_stop_acknowledgement_never_claims_the_outcome(monkeypatch):
    """A 2xx on stop is 'stop requested'; the outcome arrives from the poll."""

    gate = threading.Event()
    states = [{"status": "running"}]

    def get_run(_rid):
        if gate.is_set():
            return {"status": "cancelled", "error": "stopped by operator"}
        return states[0]

    monkeypatch.setattr(talk_apiserver, "start_run", lambda *a, **k: "run_remote_s")
    monkeypatch.setattr(talk_apiserver, "get_run", get_run)
    monkeypatch.setattr(talk_apiserver, "stop_run", lambda rid: None)
    run_id = talk_runs.start_run(
        "agent", "long job", talk_host._api_server_worker("go", session_id=None)
    )
    deadline = time.time() + 2.0
    while time.time() < deadline and not talk_runs.get_run(run_id)["meta"].get("api_run_id"):
        time.sleep(0.01)
    ack = talk_host.host().stop_work(str(run_id))
    assert "Sent the stop" in ack
    assert "cancelled" not in ack and "finished" not in ack
    assert talk_runs.get_run(run_id)["status"] == "running"
    gate.set()
    run = _wait_terminal(run_id)
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_CANCELLED


def test_a_detached_child_killed_by_stop_work_is_cancelled_not_failed(monkeypatch, tmp_path):
    """Live sim: 'cancel that job' on the detached lane recorded outcome=failed (exit -15)."""

    import sys as _sys

    sleeper = [_sys.executable, "-c", "import time; time.sleep(30)"]
    monkeypatch.setattr(talk_host, "agent_argv", lambda binary, task, profile: sleeper)
    monkeypatch.setattr(talk_host, "STOP_CONFIRM_WAIT_S", 3.0)
    run_id = talk_runs.start_run(
        "agent", "long job", talk_host._detached_agent_worker("go", _sys.executable)
    )
    deadline = time.time() + 3
    while time.time() < deadline and talk_runs.get_process(run_id) is None:
        time.sleep(0.02)
    assert talk_runs.get_process(run_id) is not None
    ack = talk_host.host().stop_work(str(run_id))
    assert "Stopped run" in ack or "stop" in ack.lower()
    run = _wait_terminal(run_id, timeout=6.0)
    assert run["status"] == "failed"
    assert talk_runs.run_outcome(run) == talk_runs.OUTCOME_CANCELLED
    assert "operator's request" in run["output"]
    assert "was cancelled" in _spoken(run)


def test_delivery_evidence_is_a_receipt_not_a_gate():
    """Live sim: gating the flip on acks left every result 'undelivered' and re-adopted
    on the next call. Evidence rides the run meta; the flip still happens on send."""

    import talk_delivery

    snapshot = {"audible_ms": 0, "acknowledged": False, "dequeued_ms": 480}
    assert talk_delivery.stage(dict(snapshot, injected=True)) == talk_delivery.INJECTED
    assert not talk_delivery.is_delivered(snapshot)
    acked = {"audible_ms": 480, "acknowledged": True, "ms": 480}
    assert talk_delivery.is_delivered(acked)
    assert talk_delivery.stage(acked) == talk_delivery.DELIVERED


def test_a_cancelled_run_from_an_earlier_call_is_not_adopted_as_news(monkeypatch, tmp_path):
    """Live sim: 'the work you asked for is back: nothing to share' on the next call, for a
    job the operator had cancelled on the previous one."""

    history = tmp_path / "talk-runs.jsonl"
    monkeypatch.setattr(talk_runs, "_history_path", lambda: history)
    monkeypatch.setattr(talk_runs, "_history_enabled", lambda: True)
    talk_runs.attach_owner(
        talk_session_id="ts-earlier",
        generation_id="g1",
        hermes_session_id="hs-1",
        operator="op",
        profile="p",
    )
    gate = threading.Event()

    def worker(_rid: int) -> str:
        gate.wait(5)
        return "unused"

    run_id = talk_runs.start_run("agent", "gbrain lookup", worker)
    talk_runs.finish_run(run_id, "failed", "stopped", outcome=talk_runs.OUTCOME_CANCELLED)
    gate.set()
    adopted = talk_runs.list_undelivered_for_session(
        "hs-1", operator="op", profile="p", claimant="ts-later"
    )
    assert [r["runId"] for r in adopted] == []
    assert talk_runs.get_run(run_id)["delivery"] == talk_runs.DELIVERED

