"""Receipt-aware steering of api_server runs from the phone lane (hermes-sip-live-voice#56)."""

from __future__ import annotations

import threading

import pytest

import talk_api_steer
import talk_apiserver
import talk_host
import talk_runs
import talk_tools


def _blocking(gate: threading.Event, text: str):
    """A worker that returns ``text`` once ``gate`` is set (or after 5 s)."""

    def worker(_run_id: int) -> str:
        gate.wait(5)
        return text

    return worker


@pytest.fixture(autouse=True)
def _bound(monkeypatch):
    talk_runs.reset_for_tests()
    talk_api_steer.reset_for_tests()
    talk_runs.attach_owner(
        talk_session_id="ts-test",
        generation_id="gen-test",
        hermes_session_id="sess-test",
        operator="test",
        profile=None,
    )
    monkeypatch.setattr(talk_apiserver, "steering_supported", lambda: True)
    yield
    talk_runs.detach_owner()
    talk_runs.reset_for_tests()
    talk_api_steer.reset_for_tests()


def _api_run(gate: threading.Event, *, api_run_id="r-1", session="s-1") -> int:
    run_id = talk_runs.start_run("agent", "triage", _blocking(gate, "ok"))
    talk_runs.annotate_run(
        run_id, lane=talk_host.LANE_API_SERVER, api_run_id=api_run_id, api_session_id=session
    )
    return run_id


def test_correction_reaches_the_same_api_job_once_as_queued(monkeypatch):
    posts = []
    monkeypatch.setattr(
        talk_apiserver,
        "steer_run",
        lambda rid, text: posts.append((rid, text)) or {"accepted": True},
    )
    gate = threading.Event()
    run_id = _api_run(gate)
    out = talk_tools.execute_talk_tool(
        "steer_agent", {"agent_id": str(run_id), "text": "focus on the invoices"}
    )
    assert posts == [("r-1", "focus on the invoices")]
    assert "queued" in out and "applied" not in out.split("not applied")[0].replace("Queued", "")
    (rec,) = talk_api_steer.receipts_for(run_id)
    assert rec["state"] == talk_api_steer.QUEUED and rec["evidence"] == "backend_queue_ack"
    assert rec["api_run_id"] == "r-1" and rec["text_hash"]
    assert talk_runs.get_run(run_id)["meta"]["steer_receipts"][0]["action_id"] == rec["action_id"]
    gate.set()


def test_retry_with_the_same_action_id_does_not_duplicate(monkeypatch):
    posts = []

    def _post(rid, text):
        posts.append(text)
        if len(posts) == 1:
            raise talk_apiserver.TalkApiServerError("I couldn't reach the Hermes api server")
        return {"accepted": True}

    monkeypatch.setattr(talk_apiserver, "steer_run", _post)
    gate = threading.Event()
    run_id = _api_run(gate)
    run = talk_runs.get_run(run_id)
    spoken, rec = talk_api_steer.steer(run, "use the other mailbox", action_id="act-fixed")
    assert rec["state"] == talk_api_steer.UNKNOWN and "didn't get an answer" in spoken
    _again_spoken, again = talk_api_steer.steer(run, "use the other mailbox", action_id="act-fixed")
    assert posts == ["use the other mailbox"]  # one POST, never a second
    assert again["action_id"] == "act-fixed" and again["state"] == talk_api_steer.UNKNOWN
    assert len(talk_api_steer.receipts_for(run_id)) == 1
    gate.set()


def test_completed_and_foreign_owned_jobs_are_refused(monkeypatch):
    monkeypatch.setattr(talk_apiserver, "steer_run", lambda *a, **k: pytest.fail("must not POST"))
    done = talk_runs.start_run("agent", "x", lambda _rid: "ok")
    for _ in range(50):
        if talk_runs.get_run(done)["status"] in talk_runs.TERMINAL_STATUSES:
            break
        threading.Event().wait(0.02)
    talk_runs.annotate_run(done, lane=talk_host.LANE_API_SERVER, api_run_id="r-done")
    out = talk_tools.execute_talk_tool("steer_agent", {"agent_id": str(done), "text": "more"})
    assert "already finished" in out
    gate = threading.Event()
    run_id = _api_run(gate)
    talk_runs.attach_owner(
        talk_session_id="ts-other",
        generation_id="g2",
        hermes_session_id="sess-other",
        operator="test",
        profile=None,
    )
    out = talk_tools.execute_talk_tool(
        "redirect_agent", {"agent_id": str(run_id), "text": "stop that"}
    )
    assert "different session" in out
    (rec,) = talk_api_steer.receipts_for(run_id)
    assert rec["state"] == talk_api_steer.REFUSED and rec["evidence"] == "foreign_owner"
    gate.set()


def test_run_not_accepting_steer_becomes_a_followup_on_the_same_session(monkeypatch):
    def _refuse(rid, text):
        raise talk_apiserver.SteerRefused("409", code="run_not_accepting_steer")

    monkeypatch.setattr(talk_apiserver, "steer_run", _refuse)
    gate = threading.Event()
    run_id = _api_run(gate)
    out = talk_tools.execute_talk_tool(
        "steer_agent", {"agent_id": str(run_id), "text": "also check the calendar"}
    )
    assert "queued for after this step" in out and "applied" not in out
    (rec,) = talk_api_steer.receipts_for(run_id)
    assert rec["state"] == talk_api_steer.QUEUED_FOLLOWUP
    assert talk_api_steer.pending_followups(run_id) == [rec]
    gate.set()


def test_capability_absent_queues_followup_without_posting(monkeypatch):
    monkeypatch.setattr(talk_apiserver, "steering_supported", lambda: False)
    monkeypatch.setattr(talk_apiserver, "steer_run", lambda *a, **k: pytest.fail("must not POST"))
    gate = threading.Event()
    run_id = _api_run(gate)
    spoken, rec = talk_api_steer.steer(talk_runs.get_run(run_id), "x")
    assert rec["state"] == talk_api_steer.QUEUED_FOLLOWUP and rec["evidence"] == "capability_absent"
    assert "queued for after this step" in spoken
    gate.set()


def test_followup_runs_as_next_turn_on_same_session_and_is_marked_applied(monkeypatch):
    prompts = []

    def _run(prompt, *, session_id=None, session_key=None, on_start=None, on_event=None):
        prompts.append((prompt, session_id))
        if on_start is not None:
            on_start(f"r-{len(prompts)}")
        return "done, corrected"

    monkeypatch.setattr(talk_apiserver, "run_to_completion", _run)
    monkeypatch.setattr(talk_apiserver, "steering_supported", lambda: False)
    monkeypatch.setattr(talk_apiserver, "status", lambda: type("V", (), {"available": True})())
    first = talk_apiserver.RunOutcome(
        outcome=talk_apiserver.OUTCOME_SUCCESS,
        remote_status="completed",
        output="first answer",
        error="",
        api_run_id="r-1",
        session_id="s-1",
    )
    gate = threading.Event()
    run_id = _api_run(gate)
    _spoken, rec = talk_api_steer.steer(talk_runs.get_run(run_id), "also the calendar")
    final = talk_host._apply_followups(
        run_id, first, session_key=None, on_start=lambda _id: None, on_event=None
    )
    assert final.succeeded and prompts and prompts[0][1] == "s-1"
    assert "also the calendar" in prompts[0][0]
    assert talk_api_steer.receipt(rec["action_id"])["state"] == talk_api_steer.APPLIED
    # No pending follow-ups: the outcome passes through untouched.
    assert (
        talk_host._apply_followups(run_id, first, session_key=None, on_start=None, on_event=None)
        is first
    )
    gate.set()


def test_failed_parent_supersedes_followups_instead_of_correcting_behind_the_operator():
    gate = threading.Event()
    run_id = _api_run(gate)
    talk_api_steer._record(
        {
            "action_id": "act-1",
            "run_id": run_id,
            "state": talk_api_steer.QUEUED_FOLLOWUP,
            "text": "x",
        }
    )
    cancelled = talk_apiserver.RunOutcome(
        outcome=talk_apiserver.OUTCOME_CANCELLED,
        remote_status="cancelled",
        output="",
        error="stopped",
        api_run_id="r-1",
        session_id="s-1",
    )
    out = talk_host._apply_followups(
        run_id, cancelled, session_key=None, on_start=None, on_event=None
    )
    assert out is cancelled
    assert talk_api_steer.receipt("act-1")["state"] == talk_api_steer.SUPERSEDED
    gate.set()


def test_stop_work_supersedes_open_receipts_and_detached_runs_keep_the_old_refusal(monkeypatch):
    monkeypatch.setattr(talk_apiserver, "steer_run", lambda rid, text: {"accepted": True})
    monkeypatch.setattr(talk_apiserver, "stop_run", lambda rid: None)
    gate = threading.Event()
    run_id = _api_run(gate)
    _s, rec = talk_api_steer.steer(talk_runs.get_run(run_id), "note")
    assert rec["state"] == talk_api_steer.QUEUED
    out = talk_host.host().stop_work(str(run_id))
    assert "stop" in out.lower()
    assert talk_api_steer.receipt(rec["action_id"])["state"] == talk_api_steer.SUPERSEDED
    gate.set()
    # A detached one-shot has no channel: the honest offer stands.
    gate2 = threading.Event()
    detached = talk_runs.start_run("agent", "d", _blocking(gate2, "ok"))
    out = talk_tools.execute_talk_tool("steer_agent", {"agent_id": str(detached), "text": "x"})
    assert "detached one-shot" in out
    gate2.set()


def test_steer_run_maps_http_statuses(monkeypatch):
    class _Resp:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body

        def json(self):
            return self._body

    calls = []

    def _post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return _Resp(*calls_out.pop(0))

    monkeypatch.setattr(talk_apiserver.httpx, "post", _post)
    monkeypatch.setattr(talk_apiserver.talk_config, "api_server_url", lambda: "http://h")
    monkeypatch.setattr(talk_apiserver.talk_config, "api_server_probe_timeout_s", lambda: 1)
    calls_out = [
        (200, {"object": "hermes.run.steer", "run_id": "r", "accepted": True}),
        (409, {"error": {"code": "run_not_accepting_steer"}}),
        (500, {}),
        (200, {"accepted": False}),
    ]
    assert talk_apiserver.steer_run("r", "hi")["accepted"] is True
    assert calls[0] == ("http://h/v1/runs/r/steer", {"input": "hi"})
    with pytest.raises(talk_apiserver.SteerRefused) as refused:
        talk_apiserver.steer_run("r", "hi")
    assert refused.value.code == "run_not_accepting_steer"
    with pytest.raises(talk_apiserver.TalkApiServerError):
        talk_apiserver.steer_run("r", "hi")
    with pytest.raises(talk_apiserver.TalkApiServerError):
        talk_apiserver.steer_run("r", "hi")


def test_steering_supported_reads_capabilities_and_fails_closed(monkeypatch):
    monkeypatch.undo()  # the fixture's steering_supported stub must not shadow the real read
    monkeypatch.setattr(
        talk_apiserver, "capabilities_payload", lambda: {"features": {"run_steer": True}}
    )
    assert talk_apiserver.steering_supported() is True
    monkeypatch.setattr(talk_apiserver, "capabilities_payload", lambda: {"features": {}})
    assert talk_apiserver.steering_supported() is False

    def _boom():
        raise talk_apiserver.TalkApiServerError("down")

    monkeypatch.setattr(talk_apiserver, "capabilities_payload", _boom)
    assert talk_apiserver.steering_supported() is False
