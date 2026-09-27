"""The lane policy hook (talk_lane.LanePolicy) a transport hands to a session.

Covers what hermes-sip-live-voice needs from Talk 0.22 (#26, #35, #51, #57):
a trusted operating pack in the prompt, spoken heartbeats off, transcript
retention separated from memory promotion, and lane-owned tools dispatched
through the normal tool contract. Every default reproduces pre-policy
behaviour exactly.
"""

from __future__ import annotations

import pytest

import talk_identity
import talk_lane
import talk_tools
import talk_transcript


@pytest.fixture(autouse=True)
def _clear_lane_handlers():
    talk_tools.register_lane_handlers(None)
    yield
    talk_tools.register_lane_handlers(None)


def test_neutral_policy_changes_nothing():
    policy = talk_lane.coerce(None, "discord")
    assert policy.name == "discord"
    assert policy.rendered_instructions() is None
    assert policy.spoken_heartbeats is None
    assert policy.memory_review is None
    assert policy.tools == ()
    receipt = policy.receipt()
    assert receipt["instructions_chars"] == 0 and receipt["tools"] == []


def test_pack_is_rendered_into_the_prompt_after_the_preamble_and_before_identity():
    pack = "You are Bob, Dustin's executive assistant. Lead with the answer."
    built = talk_identity.build_instructions(
        {"PERSONA": "Hermes persona text"}, lane="phone", lane_instructions=pack
    )
    i_pre = built.index(talk_identity.VOICE_PREAMBLE[:40])
    i_pack = built.index(talk_identity.LANE_INSTRUCTIONS_HEADER)
    i_persona = built.index("Hermes persona text")
    assert i_pre < i_pack < i_persona
    assert pack in built


def test_pack_is_capped_with_a_visible_marker_and_the_receipt_says_so():
    policy = talk_lane.LanePolicy(
        name="phone",
        instructions="x" * (talk_lane.INSTRUCTIONS_CAP + 500),
        instructions_version="v3",
    )
    rendered = policy.rendered_instructions()
    assert rendered is not None
    assert len(rendered) == talk_lane.INSTRUCTIONS_CAP
    assert rendered.endswith(talk_lane.TRUNCATION_MARKER)
    receipt = policy.receipt()
    assert receipt["instructions_truncated"] is True
    assert receipt["instructions_version"] == "v3"
    assert "x" * 50 not in str(receipt), "the receipt never carries prompt text"


def test_blank_pack_renders_nothing():
    assert talk_lane.LanePolicy(instructions="   \n").rendered_instructions() is None
    built = talk_identity.build_instructions(None, lane_instructions=None)
    assert talk_identity.LANE_INSTRUCTIONS_HEADER not in built


def test_retained_transcripts_are_kept_but_never_swept_for_memory(tmp_path):
    long = "durable context " * 12
    kept = talk_transcript.TranscriptCapture(tmp_path, memory_review=False)
    kept.append_turn("user", "remember the Friday deadline " + long)
    kept.append_turn("assistant", "noted " + long)
    kept.finish()
    assert kept.path.exists()
    assert kept.path.parent.name == talk_transcript.RETAINED_DIRNAME

    handed: list[str] = []
    talk_transcript.sweep_transcripts(
        tmp_path, run_agent=lambda prompt: handed.append(prompt) or "ok"
    )
    assert handed == [], "the retained root is not a memory-review source"
    assert kept.path.exists(), "retention means the file stays"

    reviewed = talk_transcript.TranscriptCapture(tmp_path, memory_review=True)
    reviewed.append_turn("user", "and this one may be promoted " + long)
    reviewed.append_turn("assistant", "understood " + long)
    reviewed.finish()
    talk_transcript.sweep_transcripts(
        tmp_path, run_agent=lambda prompt: handed.append(prompt) or "ok"
    )
    assert len(handed) == 1
    assert "promoted" in handed[0]


def test_lane_tools_dispatch_through_the_normal_contract_and_cannot_shadow_builtins():
    calls: list[dict] = []
    talk_tools.register_lane_handlers({"end_call": lambda args: calls.append(args) or "hanging up"})
    assert talk_tools.execute_talk_tool("end_call", {"reason": "done"}) == "hanging up"
    assert calls == [{"reason": "done"}]
    with pytest.raises(ValueError):
        talk_tools.register_lane_handlers({"check_work": lambda a: "nope"})
    # A new session's registration replaces the previous lane's set.
    talk_tools.register_lane_handlers({})
    with pytest.raises(talk_tools.TalkToolError):
        talk_tools.execute_talk_tool("end_call", {})
