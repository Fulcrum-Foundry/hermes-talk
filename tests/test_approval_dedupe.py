"""Approval dedupe and delivery evidence (hermes-sip-live-voice#52 I04, #35 I12)."""

from __future__ import annotations

import hashlib
import json
import time

import pytest

import talk_approvals
import talk_delivery
import talk_host
import talk_lane
import talk_runs


@pytest.fixture(autouse=True)
def clean():
    talk_approvals.reset_for_tests()
    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()
    yield
    talk_approvals.reset_for_tests()
    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()


def _args_hash(args: dict) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12]


# -- already_requested (the replay-suite seam) -------------------------------------


def test_identical_read_only_request_is_redundant():
    first = {
        "request_id": "a1",
        "action": "mock_triage",
        "read_only": True,
        "args_hash": _args_hash({"mailbox": "inbox"}),
    }
    second = {**first, "request_id": "a2"}
    assert talk_approvals.already_requested([first], second) is True


def test_changed_args_other_action_or_consequential_are_never_redundant():
    base = {"action": "mock_triage", "read_only": True, "args_hash": "aaaaaaaaaaaa"}
    assert not talk_approvals.already_requested([base], {**base, "args_hash": "bbbbbbbbbbbb"})
    assert not talk_approvals.already_requested([base], {**base, "action": "send_mail"})
    send = {"action": "send_mail", "read_only": False, "args_hash": "cccccccccccc"}
    assert not talk_approvals.already_requested([send], dict(send))
    assert not talk_approvals.already_requested([base], {**base, "read_only": False})
    assert not talk_approvals.already_requested([], base)
    assert not talk_approvals.already_requested([{**base, "outcome": "deny"}], dict(base))
    assert not talk_approvals.already_requested([base], None)


def test_identity_is_derived_from_host_event_fields_when_hash_is_absent():
    event = {"description": "Read the mailbox", "args": {"b": 2, "a": 1}, "read_only": True}
    reordered = {"description": "Read the mailbox", "args": {"a": 1, "b": 2}, "read_only": True}
    assert talk_approvals.request_identity(event) == talk_approvals.request_identity(reordered)
    changed = {"description": "Read the mailbox", "args": {"a": 1, "b": 3}, "read_only": True}
    assert talk_approvals.request_identity(event) != talk_approvals.request_identity(changed)
    by_command = {"description": "Run a shell command", "command": "ls"}
    assert talk_approvals.request_identity(by_command)[0] == "Run a shell command"


# -- registration-time dedupe ----------------------------------------------------------


class _Loop:
    def call_soon_threadsafe(self, callback, event):
        callback(event)


def _event(**overrides) -> dict:
    event = {
        "event": "approval.request",
        "run_id": "r-1",
        "timestamp": time.time(),
        "request_id": "req-1",
        "description": "Read the mailbox",
        "args": {"mailbox": "inbox"},
        "read_only": True,
        "choices": ["once", "session", "deny"],
    }
    event.update(overrides)
    return event


def test_already_granted_read_only_request_is_auto_resolved_not_reprompted(monkeypatch):
    posted = []
    monkeypatch.setattr(
        talk_approvals.talk_apiserver,
        "respond_to_approval",
        lambda api_run_id, choice, approval_id=None: posted.append(
            (api_run_id, choice, approval_id)
        ),
    )
    prompts = []
    talk_approvals.attach_session(_Loop(), prompts.append)
    talk_approvals._note_event(7, "r-1", _event())
    assert len(prompts) == 1 and talk_approvals.has_pending(7)
    out = talk_approvals.resolve(7, "once")
    assert out.startswith("Approved")
    assert posted == [("r-1", "once", "req-1")]
    ledger = talk_approvals.requested_this_session()
    assert ledger[-1]["outcome"] == "once" and ledger[-1]["read_only"] is True
    # The identical read-only request again: answered on the operator's behalf.
    talk_approvals._note_event(7, "r-1", _event(request_id="req-2"))
    for _ in range(50):
        if len(posted) == 2:
            break
        time.sleep(0.01)
    assert len(prompts) == 1, "redundant approval prompted"
    assert posted[-1] == ("r-1", "once", "req-2")
    assert not talk_approvals.has_pending(7)
    # Changed arguments: a new request that prompts.
    talk_approvals._note_event(7, "r-1", _event(request_id="req-3", args={"mailbox": "sent"}))
    assert len(prompts) == 2 and talk_approvals.has_pending(7)


def test_consequential_and_denied_requests_always_prompt(monkeypatch):
    monkeypatch.setattr(talk_approvals.talk_apiserver, "respond_to_approval", lambda *a, **k: None)
    prompts = []
    talk_approvals.attach_session(_Loop(), prompts.append)
    send = _event(description="Send an email", args={"to": "x"}, read_only=False)
    talk_approvals._note_event(7, "r-1", send)
    talk_approvals.resolve(7, "once")
    talk_approvals._note_event(7, "r-1", {**send, "request_id": "req-2"})
    assert len(prompts) == 2 and talk_approvals.has_pending(7)
    talk_approvals.resolve(7, "deny")
    # A read-only request that was DENIED does not make its repeat redundant.
    talk_approvals._note_event(8, "r-2", _event(request_id="q-1"))
    talk_approvals.resolve(8, "deny")
    talk_approvals._note_event(8, "r-2", _event(request_id="q-2"))
    assert len(prompts) == 4 and talk_approvals.has_pending(8)


def test_ledger_clears_with_the_session():
    talk_approvals.attach_session(_Loop(), lambda e: None)
    talk_approvals._note_event(7, "r-1", _event())
    assert talk_approvals.requested_this_session()
    talk_approvals.detach_session()
    assert talk_approvals.requested_this_session() == []


# -- is_delivered --------------------------------------------------------------------


def test_delivery_requires_a_transport_ack_never_mere_injection():
    assert not talk_delivery.is_delivered({"injected": True})
    assert not talk_delivery.is_delivered({"injected": True, "dequeued_ms": 400, "played_ms": 400})
    assert not talk_delivery.is_delivered({"injected": True, "interrupted": True, "audible_ms": 0})
    assert not talk_delivery.is_delivered(
        {"acknowledged": True, "interrupted": True, "audible_ms": 0}
    )
    assert talk_delivery.is_delivered({"acked": True})
    assert talk_delivery.is_delivered({"acknowledged": True, "audible_ms": 0})
    assert talk_delivery.is_delivered({"injected": True, "audible_ms": 200})
    assert not talk_delivery.is_delivered({"audible_ms": True})  # a bool is not a measurement
    assert not talk_delivery.is_delivered(None)
    assert talk_delivery.stage({"injected": True}) == talk_delivery.INJECTED
    assert talk_delivery.stage({"queued": True}) == talk_delivery.QUEUED
    assert talk_delivery.stage({"acked": True, "audible_ms": 900}) == talk_delivery.DELIVERED
    assert talk_delivery.stage({}) == talk_delivery.STORED


def test_replay_notice_shape_interrupted_is_not_delivered():
    # tests/replay/runner.Notice.__dict__ after a barge-in at 500 ms.
    notice = {
        "item": "n-a",
        "job": "job-a",
        "kind": "result",
        "ms": 2000,
        "injected": True,
        "audible_ms": 0,
        "dequeued_ms": 600,
        "acknowledged": False,
        "interrupted": True,
    }
    assert not talk_delivery.is_delivered(notice)
    notice.update(audible_ms=2000, acknowledged=True, interrupted=False)
    assert talk_delivery.is_delivered(notice)


def test_on_sent_with_evidence_is_fail_closed():
    flips = []
    hook = talk_delivery.on_sent_with_evidence(lambda: flips.append(1), lambda: {"injected": True})
    hook()
    assert flips == []
    hook = talk_delivery.on_sent_with_evidence(lambda: flips.append(1), lambda: {"acked": True})
    hook()
    assert flips == [1]

    def _boom():
        raise RuntimeError("no transport")

    hook = talk_delivery.on_sent_with_evidence(lambda: flips.append(2), _boom)
    hook()
    assert flips == [1]
    assert talk_delivery.on_sent_with_evidence(None, lambda: {}) is None
    assert talk_delivery.delivery_record(run_id=3, acked=True, audible_ms=10) == {
        "run_id": 3,
        "injected": False,
        "acked": True,
        "audible_ms": 10,
        "interrupted": False,
    }


def test_lane_policy_delivery_evidence_default_is_off():
    assert talk_lane.LanePolicy().delivery_evidence is None
    assert talk_lane.LanePolicy().receipt()["delivery_evidence"] is False
    policy = talk_lane.LanePolicy(delivery_evidence=lambda rid: {"acked": True})
    assert policy.receipt()["delivery_evidence"] is True
