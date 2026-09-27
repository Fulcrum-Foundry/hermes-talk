"""Durable caller binding across calls (hermes-sip-live-voice#35, I11)."""

from __future__ import annotations

import os
import threading
import time

import pytest

import talk_binding
import talk_lane
import talk_runs


def _blocking(gate: threading.Event, text: str):
    """A worker that returns ``text`` once ``gate`` is set (or after 5 s)."""

    def worker(_run_id: int) -> str:
        gate.wait(5)
        return text

    return worker


@pytest.fixture(autouse=True)
def _durable(monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(talk_binding.talk_config, "state_dir", lambda: state)
    monkeypatch.setattr(talk_binding, "enabled", lambda: True)
    monkeypatch.setattr(talk_runs, "_history_path", lambda: state / "talk-runs.jsonl")
    monkeypatch.setattr(talk_runs, "_history_enabled", lambda: True)
    talk_runs.reset_for_tests()
    yield
    talk_runs.detach_owner()
    talk_runs.reset_for_tests()


def _attach(binding: talk_binding.Binding, talk_session_id: str) -> None:
    binding.note_session(talk_session_id)
    talk_runs.attach_owner(
        talk_session_id=talk_session_id,
        generation_id=f"gen-{talk_session_id}",
        hermes_session_id=binding.hermes_session_id,
        operator="sip",
        profile=None,
    )


def _wait_terminal(run_id: int) -> None:
    for _ in range(200):
        run = talk_runs.get_run(run_id)
        if run is not None and run["status"] in talk_runs.TERMINAL_STATUSES:
            return
        time.sleep(0.01)
    raise AssertionError("run never finished")


def test_binding_is_keyed_and_persisted_privately(tmp_path):
    key = talk_binding.make_key("+15550001111", "bob-prod")
    assert key == talk_binding.make_key("+15550001111", "bob-prod")
    assert key != talk_binding.make_key("+15550001111", "bob-staging")
    binding = talk_binding.for_caller(key)
    assert binding.hermes_session_id.startswith("talk-binding-")
    path = tmp_path / "state" / talk_binding.BINDINGS_DIRNAME / f"{key}.json"
    assert path.exists()
    if os.name != "nt":  # Windows has no POSIX mode bits
        assert oct(path.stat().st_mode & 0o777) == "0o600"
    again = talk_binding.for_caller(key)
    assert again.hermes_session_id == binding.hermes_session_id
    # A host-supplied durable id is adopted on first sight, kept afterwards.
    fresh = talk_binding.for_caller("other-key-0001", hermes_session_id="host-sess")
    assert fresh.hermes_session_id == "host-sess"
    assert talk_binding.for_caller("other-key-0001", hermes_session_id="x").hermes_session_id == (
        "host-sess"
    )
    with pytest.raises(ValueError):
        talk_binding.for_caller("../evil")


def test_hang_up_then_same_caller_adopts_the_exact_pending_job():
    key = talk_binding.make_key("caller-a", "dep")
    first = talk_binding.for_caller(key)
    _attach(first, "ts-call-1")
    gate = threading.Event()
    run_id = talk_runs.start_run("agent", "outlook triage", _blocking(gate, "42 mails"))
    first.note_run(run_id, label="outlook triage")
    assert first.pending_view()[0] == {
        "run_id": run_id,
        "state": talk_binding.PENDING,
        "label": "outlook triage",
        "run_status": "running",
        "outcome": talk_runs.OUTCOME_UNKNOWN,
    }
    # Hang up before completion.
    talk_runs.detach_owner()
    gate.set()
    _wait_terminal(run_id)
    # Same caller, new call: a NEW binding load, same durable id.
    second = talk_binding.for_caller(key)
    assert second.hermes_session_id == first.hermes_session_id
    assert second.talk_session_ids == ["ts-call-1"]
    _attach(second, "ts-call-2")
    owed = talk_runs.list_undelivered_for_session(
        second.hermes_session_id, operator="sip", profile=None, claimant="ts-call-2"
    )
    assert [r["runId"] for r in owed] == [run_id]
    assert owed[0]["output"] == "42 mails"
    assert talk_runs.claim_delivery(run_id, claimant="ts-call-2")
    assert talk_runs.mark_delivered(run_id, claimant="ts-call-2")
    second.mark_delivered(run_id)
    assert second.pending_view() == []
    assert talk_binding.for_caller(key).pending[str(run_id)]["state"] == talk_binding.DELIVERED


def test_another_callers_key_sees_nothing():
    key_a = talk_binding.make_key("caller-a", "dep")
    key_b = talk_binding.make_key("caller-b", "dep")
    a = talk_binding.for_caller(key_a)
    _attach(a, "ts-a")
    run_id = talk_runs.start_run("agent", "private", lambda _rid: "secret")
    a.note_run(run_id, label="private")
    _wait_terminal(run_id)
    talk_runs.detach_owner()
    b = talk_binding.for_caller(key_b)
    assert b.hermes_session_id != a.hermes_session_id
    assert b.pending_view() == [] and b.run_ids == []
    _attach(b, "ts-b")
    assert (
        talk_runs.list_undelivered_for_session(
            b.hermes_session_id, operator="sip", profile=None, claimant="ts-b"
        )
        == []
    )
    # And a caller with NO durable id still gets nothing (the check is intact).
    assert talk_runs.list_undelivered_for_session(None, operator="sip", profile=None) == []


def test_gateway_restart_yields_interrupted_never_still_running():
    key = talk_binding.make_key("caller-a", "dep")
    binding = talk_binding.for_caller(key)
    _attach(binding, "ts-1")
    gate = threading.Event()
    run_id = talk_runs.start_run("agent", "long job", _blocking(gate, "late"))
    binding.note_run(run_id, label="long job")
    # Simulate the process dying: the live registry is gone, history still says running.
    talk_runs._RUNS.clear()
    assert talk_runs.resolve_run_record(run_id)["status"] == "lost"
    reloaded = talk_binding.for_caller(key)
    (row,) = reloaded.pending_view()
    assert row["state"] == talk_binding.INTERRUPTED and row["run_status"] == "lost"
    assert "running" not in {row["state"], row["outcome"]}
    assert talk_binding.for_caller(key).pending[str(run_id)]["state"] == talk_binding.INTERRUPTED
    gate.set()


def test_after_call_none_records_nothing_pending():
    key = talk_binding.make_key("caller-a", "dep")
    binding = talk_binding.for_caller(key, after_call=talk_binding.AFTER_CALL_NONE)
    _attach(binding, "ts-1")
    run_id = talk_runs.start_run("agent", "x", lambda _rid: "y")
    binding.note_run(run_id)
    assert binding.run_ids == [run_id] and binding.pending == {}
    with pytest.raises(ValueError):
        talk_binding.for_caller(key, after_call="sms")


def test_lane_policy_defaults_are_backward_compatible():
    policy = talk_lane.LanePolicy()
    assert policy.binding_key is None and policy.after_call == "retrievable"
    receipt = policy.receipt()
    assert receipt["binding"] is False and receipt["after_call"] == "retrievable"
    phone = talk_lane.LanePolicy(name="phone", binding_key="b-abc12345", after_call="none")
    assert phone.receipt()["binding"] is True and phone.receipt()["after_call"] == "none"
    assert "b-abc12345" not in str(phone.receipt()), "the key itself never rides the receipt"
