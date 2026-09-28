"""GPT-Live over a plain duplex audio device (the phone lane), with scripted sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from test_live_config_protocol import fake_auth

import talk_cli
import talk_identity
import talk_lane
import talk_live_phone as phone
import talk_realtime as rt
import talk_runs
from talk_live_config import LiveConfig


class Audio:
    """The talk_audio.DuplexAudio surface a phone transport implements."""

    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.played = []
        self.drained = 0
        self.started = self.stopped = False
        self.playback_pending = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read_input_chunk(self):
        return self.chunks.pop(0) if self.chunks else None

    def queue_playback(self, pcm, item_id=None):
        self.played.append((pcm, item_id))
        self.playback_pending = True

    def drain_playback(self):
        self.drained += 1
        self.playback_pending = False
        return None, 0


class ScriptedSession:
    """A LiveRealtimeSession stand-in: records commands, replays scripted events."""

    def __init__(self, events, *, fail_connect=False):
        self.events = list(events)
        self.sent = []
        self.setup = None
        self.closed = False
        self.fail_connect = fail_connect
        self.state = rt.SessionState.NEW
        self._queue = asyncio.Queue()

    async def connect(self, setup):
        if self.fail_connect:
            raise rt.RealtimeSessionError("GPT-Live connection failed")
        self.setup = setup
        self.state = rt.SessionState.CONNECTED
        for event in self.events:
            self._queue.put_nowait(event)

    async def send(self, commands):
        self.sent.extend(commands)
        for command in commands:
            if isinstance(command, rt.SubmitDelegationResult):
                # The provider answers a delegation result by speaking.
                self._queue.put_nowait(rt.OutputAudio(b"\x00\x01" * 4, item_id="after-delegation"))
            if isinstance(command, rt.AppendLiveContext):
                self._queue.put_nowait(rt.SessionTerminated(rt.SessionState.CLOSED))

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self._queue.get()
        return event

    async def close(self):
        self.closed = True


@dataclass
class Host:
    prompts: list

    def identity_sections(self):
        return {"PERSONA": "Friendly operator"}

    def run_agent(self, prompt, background=True, **_):
        self.prompts.append(prompt)
        return "WORK_STARTED #7 kind=api_server — on it."


@pytest.fixture
def live_lane(monkeypatch):
    config = LiveConfig("subscription")
    lane = talk_cli.ProviderLane(
        "live", fake_auth("subscription"), config.model, config.voice, config
    )
    monkeypatch.setattr(talk_cli, "resolve_provider_lane", lambda: lane)
    monkeypatch.setattr(
        talk_runs,
        "get_run",
        lambda run_id: {
            "runId": run_id,
            "status": "done",
            "output": "Dustin has two meetings.",
            "label": "calendar",
            "outcome": talk_runs.OUTCOME_SUCCESS,
        },
    )
    monkeypatch.setattr(talk_runs, "claim_delivery", lambda run_id, claimant: True)
    monkeypatch.setattr(talk_runs, "mark_delivered", lambda run_id, claimant: True)
    monkeypatch.setattr(phone, "WATCH_POLL_S", 0.01)
    return lane


def policy(**kw):
    return talk_lane.LanePolicy(
        name="phone",
        instructions="You are Bob, Dustin's assistant.",
        instructions_version="v1",
        spoken_heartbeats=False,
        memory_review=False,
        tools=({"type": "function", "name": "end_call", "parameters": {"type": "object"}},),
        handlers={"end_call": lambda a: "bye"},
        **kw,
    )


def test_live_phone_session_audio_delegation_and_capture(live_lane, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    audio = Audio(chunks=[b"\x01\x02" * 240])
    session = ScriptedSession(
        [
            rt.SessionReady("live_test"),
            rt.OutputAudio(b"\x10\x00" * 8, item_id="greet"),
            rt.Transcript(
                rt.TranscriptRole.USER,
                "what's on my calendar",
                True,
                rt.TranscriptProvenance.INPUT_AUDIO,
                finality="turn",
            ),
            rt.DelegationRequested("del-1", target="client"),
        ]
    )
    host = Host(prompts=[])
    turns = []
    monkeypatch.setattr(
        phone.talk_transcript, "TranscriptCapture", lambda home, **kw: _Capture(turns, kw)
    )
    monkeypatch.setattr(phone.talk_transcript, "sweep_transcripts", lambda home, *a, **k: None)

    code = asyncio.run(
        phone.run_live_phone_session(
            audio,
            lane_policy=policy(),
            session_factory=lambda auth: session,
            host=host,
        )
    )
    assert code == 0
    # Session setup: Live model/voice, the lane pack in the prompt, NO function tools.
    assert (session.setup.model, session.setup.voice) == ("gpt-live-1-codex", "cove")
    assert session.setup.tools == ()
    assert "Operating policy for this call" in session.setup.instructions
    assert "You are Bob, Dustin's assistant." in session.setup.instructions
    assert "end_call" not in session.setup.instructions
    assert talk_identity.advertised_tool_names(session.setup.instructions) == ()
    # Audio both ways through the device.
    assert any(
        isinstance(c, rt.AppendInputAudio) and c.data == b"\x01\x02" * 240 for c in session.sent
    )
    assert [item for _, item in audio.played] == ["greet", "after-delegation"]
    # Delegation reached the host with the captured operator turn, answered on the delegation id,
    # and the started run's result was appended as Live context.
    assert host.prompts == ["what's on my calendar"]
    results = [c for c in session.sent if isinstance(c, rt.SubmitDelegationResult)]
    assert (
        results and results[0].delegation_id == "del-1" and "WORK_STARTED #7" in results[0].content
    )
    contexts = [c for c in session.sent if isinstance(c, rt.AppendLiveContext)]
    assert contexts and "The calendar work is done." in contexts[0].content
    assert "Background run" not in contexts[0].content
    assert "Dustin has two meetings." in contexts[0].content
    # Transcript captured with the lane's retention switch; device torn down.
    assert turns == [("user", "what's on my calendar")]
    assert _Capture.seen_kwargs == {"memory_review": False}
    assert audio.started and audio.stopped and session.closed


class _Capture:
    seen_kwargs = None

    def __init__(self, turns, kwargs):
        self.turns = turns
        _Capture.seen_kwargs = kwargs
        self.finished = False

    def append_turn(self, role, text):
        self.turns.append((role, text))

    def finish(self):
        self.finished = True


def test_barge_in_clears_local_playback_and_asks_live_to_stop(live_lane, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    audio = Audio()
    session = ScriptedSession(
        [
            rt.SessionReady("live_test"),
            rt.OutputAudio(b"\x10\x00" * 8, item_id="long"),
            rt.Transcript(
                rt.TranscriptRole.USER,
                "wait",
                False,
                rt.TranscriptProvenance.INPUT_AUDIO,
                finality="delta",
            ),
            rt.SessionTerminated(rt.SessionState.CLOSED),
        ]
    )
    monkeypatch.setattr(phone.talk_transcript, "sweep_transcripts", lambda home, *a, **k: None)
    code = asyncio.run(
        phone.run_live_phone_session(
            audio,
            lane_policy=policy(),
            session_factory=lambda auth: session,
            host=Host([]),
        )
    )
    assert code == 0
    assert audio.drained == 1
    assert any(isinstance(c, rt.CancelResponse) for c in session.sent)


def test_refuses_without_a_live_lane_or_on_connect_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reasons = []
    realtime = talk_cli.ProviderLane(
        "openai", fake_auth("subscription"), "gpt-realtime-2.1", "cedar"
    )
    monkeypatch.setattr(talk_cli, "resolve_provider_lane", lambda: realtime)
    audio = Audio()
    assert (
        asyncio.run(phone.run_live_phone_session(audio, on_refusal=reasons.append, host=Host([])))
        == 1
    )
    assert reasons == [talk_cli.STARTUP_REFUSAL_CONFIGURATION] and not audio.started

    config = LiveConfig("subscription")
    lane = talk_cli.ProviderLane(
        "live", fake_auth("subscription"), config.model, config.voice, config
    )
    monkeypatch.setattr(talk_cli, "resolve_provider_lane", lambda: lane)
    session = ScriptedSession([], fail_connect=True)
    assert (
        asyncio.run(
            phone.run_live_phone_session(
                audio,
                on_refusal=reasons.append,
                session_factory=lambda auth: session,
                host=Host([]),
            )
        )
        == 1
    )
    assert reasons[-1] == talk_cli.STARTUP_REFUSAL_PROVIDER and audio.stopped


def test_result_context_frames_non_success_as_partial():
    text = phone.result_context(
        {"runId": 3, "status": "failed", "output": "boom", "outcome": talk_runs.OUTCOME_FAILED}
    )
    assert text.startswith("The work you asked for didn't go through.")
    assert "partial or diagnostic" in text and "boom" in text
    assert phone.result_context(
        {"runId": 4, "status": "done", "output": "", "outcome": talk_runs.OUTCOME_SUCCESS}
    ).startswith("The work you asked for is done. It produced nothing to show.")


def test_live_prompt_carries_lane_instructions_before_persona():
    prompt = talk_identity.build_live_instructions(
        {"PERSONA": "Friendly operator"},
        lane="phone",
        lane_instructions="  Owner: Dustin.  ",
    )
    header = prompt.index(talk_identity.LANE_INSTRUCTIONS_HEADER)
    assert prompt.index("Owner: Dustin.") > header > prompt.index(talk_identity.LIVE_PREAMBLE[:20])
    assert header < prompt.index("Friendly operator")
    assert talk_identity.LANE_INSTRUCTIONS_HEADER not in talk_identity.build_live_instructions()
