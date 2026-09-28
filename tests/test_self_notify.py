"""Consent owner self-notify (Talk 0.25, #67): a message to the caller needs no okay."""

from __future__ import annotations

import time

import pytest

import talk_approvals
import talk_host
import talk_identity
import talk_lane
import talk_runs


@pytest.fixture(autouse=True)
def clean():
    talk_approvals.reset_for_tests()
    talk_lane.detach_policy()
    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()
    yield
    talk_approvals.reset_for_tests()
    talk_lane.detach_policy()
    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()


CALLER = "+1 (317) 555-0100"


def _send(**overrides) -> dict:
    request = {
        "event": "approval.request",
        "run_id": "r-1",
        "timestamp": time.time(),
        "request_id": "req-1",
        "description": "Send an SMS",
        "args": {"to": "+13175550100", "body": "your summary"},
        "read_only": False,
        "choices": ["once", "session", "deny"],
    }
    request.update(overrides)
    return request


# -- normalize_handle / request_destination / is_self_notify --------------------------


def test_handles_normalize_to_one_comparable_form():
    n = talk_approvals.normalize_handle
    assert n("+1 (317) 555-0100") == n("317-555-0100") == n("tel:13175550100") == "3175550100"
    assert n("+44 20 7946 0958") == "442079460958"
    assert n(" @Dustin ") == "@dustin"
    assert n("") == "" and n(None) == ""


def test_destination_is_read_from_the_field_or_the_usual_arg_names():
    assert talk_approvals.request_destination(_send()) == "+13175550100"
    assert talk_approvals.request_destination({"destination": "@d", "args": {"to": "x"}}) == "@d"
    assert talk_approvals.request_destination({"args": {"recipient": "@d"}}) == "@d"
    assert talk_approvals.request_destination({"args": {"to": ["a", "b"]}}) is None
    assert talk_approvals.request_destination({"args": {"to": ["a"]}}) == "a"
    assert talk_approvals.request_destination({"args": {"body": "hi"}}) is None
    assert talk_approvals.request_destination("nope") is None


def test_self_notify_is_a_message_send_to_the_bound_caller_only():
    assert talk_approvals.is_self_notify(_send(), CALLER)
    assert talk_approvals.is_self_notify(_send(description="Text the operator"), "3175550100")
    assert talk_approvals.is_self_notify(_send(kind="message_send", description="x"), CALLER)
    # A different number, a group, no caller, a non-send action: never.
    assert not talk_approvals.is_self_notify(_send(args={"to": "+13175550101"}), CALLER)
    assert not talk_approvals.is_self_notify(
        _send(args={"to": ["+13175550100", "+13175550101"]}), CALLER
    )
    assert not talk_approvals.is_self_notify(_send(), None)
    assert not talk_approvals.is_self_notify(_send(), "")
    assert not talk_approvals.is_self_notify(_send(description="Delete the mailbox"), CALLER)
    assert not talk_approvals.is_self_notify(_send(args={"body": "hi"}), CALLER)
    assert not talk_approvals.is_self_notify(None, CALLER)
    # Opaque binding keys match by equality, not by handle shape.
    assert talk_approvals.is_self_notify(
        {"description": "send message", "binding_key": "b1"}, "b1"
    )
    assert not talk_approvals.is_self_notify(
        {"description": "send message", "binding_key": "b2"}, "b1"
    )


# -- the approval path auto-grants ----------------------------------------------------


class _Loop:
    def call_soon_threadsafe(self, callback, event):
        callback(event)


def _wait(predicate, tries=100):
    for _ in range(tries):
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_text_to_the_caller_is_granted_without_a_prompt(monkeypatch):
    posted = []
    monkeypatch.setattr(
        talk_approvals.talk_apiserver,
        "respond_to_approval",
        lambda api_run_id, choice, approval_id=None: posted.append(
            (api_run_id, choice, approval_id)
        ),
    )
    notes = []
    monkeypatch.setattr(talk_approvals, "_annotate", lambda rid, outcome: notes.append(outcome))
    talk_lane.attach_policy(talk_lane.LanePolicy(name="phone", caller_handle=CALLER))
    prompts = []
    talk_approvals.attach_session(_Loop(), prompts.append)
    event = _send()
    talk_approvals._note_event(7, "r-1", event)
    assert _wait(lambda: len(posted) == 1)
    assert posted == [("r-1", "once", "req-1")]
    assert prompts == [] and not talk_approvals.has_pending(7)
    assert event["destination"] == "+13175550100"
    ledger = talk_approvals.requested_this_session()
    assert ledger[-1]["outcome"] == "once" and ledger[-1]["reason"] == "self_notify"
    assert ledger[-1]["destination"] == "+13175550100"
    assert notes and "self_notify" in notes[-1]
    # It is granted every time — a consequential send is never deduped, but
    # a self-notify is never asked either.
    talk_approvals._note_event(7, "r-1", _send(request_id="req-2"))
    assert _wait(lambda: len(posted) == 2)
    assert prompts == []


def test_a_text_to_anyone_else_still_prompts(monkeypatch):
    monkeypatch.setattr(talk_approvals.talk_apiserver, "respond_to_approval", lambda *a, **k: None)
    talk_lane.attach_policy(talk_lane.LanePolicy(name="phone", caller_handle=CALLER))
    prompts = []
    talk_approvals.attach_session(_Loop(), prompts.append)
    talk_approvals._note_event(7, "r-1", _send(args={"to": "+13175550199", "body": "hi"}))
    assert len(prompts) == 1 and talk_approvals.has_pending(7)
    # Without a bound caller handle nothing is self: the prompt stands.
    talk_lane.detach_policy()
    talk_approvals._note_event(8, "r-2", _send(request_id="req-9"))
    assert len(prompts) == 2 and talk_approvals.has_pending(8)


def test_self_notify_never_widens_past_what_the_host_offered(monkeypatch):
    monkeypatch.setattr(talk_approvals.talk_apiserver, "respond_to_approval", lambda *a, **k: None)
    talk_lane.attach_policy(talk_lane.LanePolicy(name="phone", caller_handle=CALLER))
    prompts = []
    talk_approvals.attach_session(_Loop(), prompts.append)
    talk_approvals._note_event(7, "r-1", _send(choices=["deny"]))
    assert len(prompts) == 1 and talk_approvals.has_pending(7)


def test_consent_policy_says_so_and_lane_receipt_carries_the_flags():
    assert (
        "Texting or messaging the person you are talking to, on their own channel, "
        "needs no approval" in talk_identity.CONSENT_POLICY
    )
    receipt = talk_lane.LanePolicy(caller_handle=CALLER, caller_name="D").receipt()
    assert receipt["caller_handle"] is True and receipt["caller_named"] is True
    assert CALLER not in str(receipt)
