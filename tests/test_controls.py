"""Conversational controls (hermes-sip-live-voice#57): hold, resume, cancel_job, verbosity.

Five intents that used to share the word "stop". The tests pin the contract
each tool has with the model: hold is idempotent and silent, resume clears
both hold and a deferral, cancel_job is explicit and only ever a REQUEST, and
verbosity is a speech-only switch the preamble frames that way.
"""

from __future__ import annotations

import pytest

import talk_announce
import talk_cli
import talk_controls
import talk_host
import talk_identity
import talk_lane
import talk_operator_auth
import talk_tools


@pytest.fixture(autouse=True)
def _controls():
    talk_controls.reset_for_tests()
    talk_announce.reset_for_tests()
    talk_tools.register_lane_handlers(None)
    talk_controls.attach_session()
    yield
    talk_controls.reset_for_tests()
    talk_announce.reset_for_tests()
    talk_tools.register_lane_handlers(None)


def test_the_controls_are_advertised_and_classified():
    names = [t["name"] for t in talk_tools.default_talk_tools()]
    for name in ("hold", "resume", "cancel_job", "set_verbosity", "defer_updates", "get_result"):
        assert name in names
    assert "cancel_job" in talk_operator_auth.MUTATING_TALK_TOOLS
    for name in ("hold", "resume", "set_verbosity", "defer_updates", "get_result"):
        assert name in talk_operator_auth.READ_ONLY_TALK_TOOLS


def test_hold_is_idempotent_and_tells_the_model_to_stay_silent():
    first = talk_tools.execute_talk_tool("hold", {})
    assert "say nothing" in first.lower()
    assert talk_controls.is_holding()
    second = talk_tools.execute_talk_tool("hold", {})
    assert "already on hold" in second.lower()
    assert "stay silent" in second.lower()
    # Never a cancel: no host call was made.
    assert talk_controls.snapshot()["hold"] is True


def test_hold_suppresses_routine_speech_but_keeps_listening():
    scheduler = talk_announce.Scheduler("deferred")
    talk_tools.execute_talk_tool("hold", {})
    assert talk_announce.BLOCK_HOLD in scheduler.blockers(talk_announce.KIND_ROUTINE)
    # The microphone is untouched: the controls module never pauses input.
    assert talk_controls.snapshot()["caller_speaking"] is False
    # Approval questions still pass the hold gate (only the caller's own
    # speech and an in-flight answer block them).
    assert scheduler.may_speak(talk_announce.KIND_APPROVAL)


def test_resume_leaves_hold_and_lifts_a_deferral_and_names_ready_results():
    scheduler = talk_announce.Scheduler("deferred")
    talk_announce.attach_session(scheduler)
    scheduler.park_completion(7, ["cmd"], None)
    talk_tools.execute_talk_tool("hold", {})
    talk_tools.execute_talk_tool("defer_updates", {})
    assert talk_controls.is_topic_deferred()
    out = talk_tools.execute_talk_tool("resume", {})
    assert not talk_controls.is_holding()
    assert not talk_controls.is_topic_deferred()
    assert "ready:" in out.lower() and "by name, never by number" in out
    assert "7" not in out, "the caller hears labels, never run numbers"
    scheduler.release_all()
    assert talk_tools.execute_talk_tool("resume", {}) == "Nothing was on hold; carry on."


def test_cancel_job_is_explicit_and_only_a_request(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class _Host:
        def stop_work(self, target, reason=None):
            calls.append((target, reason))
            return "stop sent, receipt pending."

    monkeypatch.setattr(talk_host, "host", lambda: _Host())
    out = talk_tools.execute_talk_tool("cancel_job", {"run_id": 12, "reason": "wrong repo"})
    assert calls == [("12", "wrong repo")]
    assert out.startswith("Stop requested for run 12")
    assert "final outcome will be announced" in out
    assert "cancelled" not in out.lower().replace("stop requested", "")


def test_cancel_job_without_a_run_number_asks_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(
        talk_host, "host", lambda: (_ for _ in ()).throw(AssertionError("must not stop"))
    )
    out = talk_tools.execute_talk_tool("cancel_job", {})
    assert "ask which one" in out


def test_cancel_job_passes_a_host_refusal_through_verbatim(monkeypatch):
    class _Host:
        def stop_work(self, target, reason=None):
            return "run 3 already finished."

    monkeypatch.setattr(talk_host, "host", lambda: _Host())
    assert talk_tools.execute_talk_tool("cancel_job", {"run_id": 3}) == "run 3 already finished."


def test_cancel_job_target_rides_the_spoken_cross_check_as_a_string():
    _digest, target = talk_operator_auth._canonical_call("cancel_job", '{"run_id": 12}')
    assert target == "12"
    _digest, none_target = talk_operator_auth._canonical_call(
        "cancel_job", '{"run_id": true}'
    )
    assert none_target is None


def test_set_verbosity_is_session_local_and_speech_only():
    out = talk_tools.execute_talk_tool("set_verbosity", {"mode": "concise"})
    assert talk_controls.verbosity() == "concise"
    assert "stay complete" in out
    assert talk_tools.execute_talk_tool("set_verbosity", {"mode": "loud"}).startswith(
        "set_verbosity needs"
    )
    talk_controls.detach_session()
    talk_controls.attach_session()
    assert talk_controls.verbosity() is None, "a new session inherits nothing"


def test_lane_verbosity_default_renders_one_line_in_the_pack():
    assert talk_cli._with_verbosity(None, None) is None
    only = talk_cli._with_verbosity(None, "concise")
    assert only == talk_cli.VERBOSITY_LINES["concise"]
    both = talk_cli._with_verbosity("pack text", "detailed") or ""
    assert both.startswith("pack text\n\n")
    assert "requested reports" not in both  # the detailed line does not repeat the caveat
    assert talk_cli._with_verbosity("pack text", "weird") == "pack text"


def test_end_call_lane_tool_marks_closing_without_touching_work():
    fired: list[dict] = []
    handlers = talk_cli._wrap_end_call({"end_call": lambda a: fired.append(a) or "bye"})
    talk_tools.register_lane_handlers(handlers)
    assert not talk_controls.is_closing()
    assert talk_tools.execute_talk_tool("end_call", {"reason": "done"}) == "bye"
    assert fired == [{"reason": "done"}]
    assert talk_controls.is_closing()
    # Closing silences ROUTINE notices only; nothing here cancels a run.
    scheduler = talk_announce.Scheduler("deferred")
    assert talk_announce.BLOCK_CLOSING in scheduler.blockers()
    assert scheduler.may_speak(talk_announce.KIND_APPROVAL)


def test_lane_policy_exposes_the_new_knobs_with_backwards_compatible_defaults():
    policy = talk_lane.coerce(None, "phone")
    assert policy.announcements == "immediate"
    assert policy.verbosity is None
    assert talk_announce.coerce_policy(policy.announcements) == talk_announce.POLICY_IMMEDIATE
    assert talk_announce.coerce_policy("DEFERRED") == talk_announce.POLICY_DEFERRED
    assert talk_announce.coerce_policy("bogus") == talk_announce.POLICY_IMMEDIATE
    receipt = talk_lane.LanePolicy(announcements="deferred", verbosity="concise").receipt()
    assert receipt["announcements"] == "deferred"
    assert receipt["verbosity"] == "concise"


def test_a_plain_stop_maps_to_no_tool_in_the_preamble():
    text = talk_identity.VOICE_PREAMBLE
    assert "means stop talking" in text
    assert "Cancelling background work happens only through cancel_job" in text


def test_controls_are_neutral_when_no_session_is_attached():
    talk_controls.detach_session()
    assert talk_controls.enter_hold() is False
    assert talk_controls.defer_topic() is False
    assert not talk_controls.is_holding()
    assert "No live voice session" in talk_tools.execute_talk_tool("hold", {})
    assert "nothing to defer" in talk_tools.execute_talk_tool("defer_updates", {})
