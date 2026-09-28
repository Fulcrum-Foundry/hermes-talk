"""Announcement scheduler (hermes-sip-live-voice#51): routine notices subordinate to the call.

Every scenario drives the REAL pump (:func:`talk_cli.pump_announcements`)
with a deferred :class:`talk_announce.Scheduler`, a stub relay and a
recording send seam, and asserts on what reached the wire. The acceptance
list from the issue, one test each: completion during caller speech, during
buffered assistant audio, after hold, during another topic ("later"); three
ready jobs -> one notice; "later" suppresses repeats; explicit check_work
still works; an obsolete heartbeat never plays; approvals are never coalesced
or dropped; the immediate policy is byte-for-byte 0.22 behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

import talk_announce
import talk_cli
import talk_controls
import talk_realtime
import talk_runs
import talk_tools


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(talk_cli, "ANNOUNCE_IDLE_POLL_S", 0.005)
    monkeypatch.setattr(talk_cli, "ANNOUNCE_NOTICE_POLL_S", 0.01)
    monkeypatch.setattr(talk_cli, "ANNOUNCE_STARVATION_WARN_S", 0)
    talk_controls.reset_for_tests()
    talk_announce.reset_for_tests()
    talk_controls.attach_session()
    talk_controls.note_conversation_started()  # scenarios are mid-call unless stated
    yield
    talk_controls.reset_for_tests()
    talk_announce.reset_for_tests()


class _Relay:
    def __init__(self, active=False):
        self.response_active = active
        self.last_audio_item_id = None


class _Wire:
    """Records every batch the pump wrote; ``busy`` is the speaker/answer predicate."""

    def __init__(self):
        self.batches: list[list] = []
        self.busy = False

    async def send(self, batch, *, is_announcement=False):
        if self.busy:
            return False
        self.batches.append(list(batch))
        return True

    def headlines(self) -> list[str]:
        return [
            c.text for b in self.batches for c in b if isinstance(c, talk_realtime.AddContext)
        ]


def _runs(*entries):
    table = {
        rid: {"runId": rid, "label": label, "status": status} for rid, label, status in entries
    }
    return lambda rid: table.get(rid)


def _completion(run_id: int, label: str, flips: list):
    run = {"runId": run_id, "label": label, "status": "done", "output": "the report body"}
    return talk_cli.QueuedAnnouncement(
        talk_cli.run_finished_commands(run),
        lambda: flips.append(run_id),
        kind=talk_announce.KIND_COMPLETION,
        run_id=run_id,
    )


def _progress(run_id: int, get_run):
    return talk_cli.QueuedAnnouncement(
        talk_cli.run_phase_commands(get_run(run_id), "heartbeat"),
        kind=talk_announce.KIND_PROGRESS,
        run_id=run_id,
    )


async def _settle(seconds=0.08):
    await asyncio.sleep(seconds)


def _scenario(policy, get_run=None, wire=None):
    wire = wire or _Wire()
    relay = _Relay()
    scheduler = talk_announce.Scheduler(
        policy, answer_pending=lambda: wire.busy, get_run=get_run or _runs()
    )
    talk_announce.attach_session(scheduler)
    queue: asyncio.Queue = asyncio.Queue()
    pump = asyncio.create_task(
        talk_cli.pump_announcements(
            queue, relay, None, wire.send, lambda: wire.busy, None, scheduler=scheduler
        )
    )
    return queue, wire, scheduler, pump


# -- completions never interrupt ------------------------------------------------


def test_completion_during_caller_speech_produces_no_routine_speech():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        flips: list[int] = []
        talk_controls.note_caller_speaking(True)
        await queue.put(_completion(1, "triage", flips))
        await _settle()
        during = list(wire.headlines())
        talk_controls.note_caller_speaking(False)
        await _settle()
        pump.cancel()
        return during, wire.headlines(), flips

    during, after, flips = asyncio.run(run())
    assert during == []
    assert len(after) == 1 and "ready" in after[0] and "the report body" not in after[0]
    assert flips == [1], "the notice is the delivery"


def test_completion_during_buffered_assistant_audio_waits_for_the_speaker():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        wire.busy = True  # speaker_busy / answer pending
        await queue.put(_completion(1, "triage", []))
        await _settle()
        during = list(wire.headlines())
        wire.busy = False
        await _settle()
        pump.cancel()
        return during, wire.headlines()

    during, after = asyncio.run(run())
    assert during == []
    assert len(after) == 1


def test_completion_after_hold_stays_parked_until_resume():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        talk_tools.execute_talk_tool("hold", {})
        await queue.put(_completion(1, "triage", []))
        await _settle()
        held = list(wire.headlines())
        talk_tools.execute_talk_tool("resume", {})
        await _settle()
        pump.cancel()
        return held, wire.headlines()

    held, after = asyncio.run(run())
    assert held == []
    assert len(after) == 1 and "ready" in after[0]


def test_completion_during_another_topic_is_silenced_by_later_until_asked():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        talk_tools.execute_talk_tool("defer_updates", {})
        await queue.put(_completion(1, "triage", []))
        await _settle()
        deferred = list(wire.headlines())
        # The caller asks: check_work is the boundary and the delivery.
        released = talk_announce.acknowledge()
        await _settle()
        pump.cancel()
        return deferred, released, wire.headlines()

    deferred, released, after = asyncio.run(run())
    assert deferred == []
    assert released == [1]
    assert after == [], "already delivered by the tool; never spoken a second time"


def test_closing_call_never_plays_a_last_routine_update_but_approvals_pass():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        talk_controls.request_close()
        await queue.put(_completion(1, "triage", []))
        await queue.put(_progress(1, _runs((1, "triage", "running"))))
        await queue.put(
            talk_cli.QueuedAnnouncement(
                talk_cli.approval_prompt_commands(
                    {"run_id": 1, "request": "send the email", "choices": ("once",)}
                ),
                kind=talk_announce.KIND_APPROVAL,
            )
        )
        await _settle()
        pump.cancel()
        return wire.headlines()

    spoken = asyncio.run(run())
    assert len(spoken) == 1 and "needs an okay" in spoken[0]


# -- coalescing and repeats -----------------------------------------------------


def test_three_ready_jobs_produce_one_notice():
    async def run():
        get_run = _runs((1, "triage", "done"), (2, "outlook", "done"), (3, "deploy", "done"))
        queue, wire, _scheduler, pump = _scenario("deferred", get_run)
        flips: list[int] = []
        talk_controls.note_caller_speaking(True)
        for rid, label in ((1, "triage"), (2, "outlook"), (3, "deploy")):
            await queue.put(_completion(rid, label, flips))
        await _settle()
        talk_controls.note_caller_speaking(False)
        await _settle()
        pump.cancel()
        return wire.headlines(), flips

    spoken, flips = asyncio.run(run())
    assert len(spoken) == 1
    assert "3 things you asked for are ready" in spoken[0]
    assert "run 1" not in spoken[0] and "job" not in spoken[0]
    for label in ("triage", "outlook", "deploy"):
        assert label in spoken[0]
    assert "ONE short sentence" in spoken[0]
    assert sorted(flips) == [1, 2, 3]


def test_later_suppresses_repeats_until_asked_or_resumed():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        await queue.put(_completion(1, "triage", []))
        await _settle()
        first = len(wire.headlines())
        # Offered once; the caller says later.
        talk_tools.execute_talk_tool("defer_updates", {})
        await _settle(0.1)
        still = len(wire.headlines())
        # Resume lifts the deferral and re-arms the offer.
        talk_tools.execute_talk_tool("resume", {})
        await _settle()
        pump.cancel()
        return first, still, len(wire.headlines())

    first, still, after_resume = asyncio.run(run())
    assert first == 1
    assert still == 1, "no repeat while deferred"
    assert after_resume == 2, "re-offered once at the boundary, not on a timer"


def test_an_offered_notice_is_not_repeated_on_a_timer():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "triage", "done")))
        await queue.put(_completion(1, "triage", []))
        await _settle(0.15)
        pump.cancel()
        return wire.headlines()

    assert len(asyncio.run(run())) == 1


def test_explicit_check_work_still_works_and_releases_parked_results(monkeypatch):
    run = {"runId": 4, "kind": "agent", "label": "triage", "status": "done", "output": "all clear",
           "meta": {"outcome": "success"}, "ts": 0}
    monkeypatch.setattr(talk_runs, "get_run", lambda rid: run if rid == 4 else None)
    scheduler = talk_announce.Scheduler("deferred")
    talk_announce.attach_session(scheduler)
    flips: list[int] = []
    scheduler.park_completion(4, ["cmd"], lambda: flips.append(4))
    out = talk_tools.execute_talk_tool("check_work", {"run_id": 4})
    assert "all clear" in out
    assert flips == [4]
    assert scheduler.ready_ids() == []


# -- revalidation -----------------------------------------------------------------


def test_an_obsolete_heartbeat_never_plays_after_completion():
    async def run():
        state = {"status": "running"}
        get_run = lambda rid: {"runId": rid, "label": "triage", "status": state["status"]}  # noqa: E731
        queue, wire, _scheduler, pump = _scenario("deferred", get_run)
        wire.busy = True
        await queue.put(_progress(1, get_run))
        await _settle()
        state["status"] = "done"  # the run finished while the heartbeat waited
        wire.busy = False
        await _settle()
        pump.cancel()
        return wire.headlines()

    assert asyncio.run(run()) == []


def test_a_live_heartbeat_still_plays_under_the_deferred_policy():
    async def run():
        get_run = _runs((1, "triage", "running"))
        queue, wire, _scheduler, pump = _scenario("deferred", get_run)
        await queue.put(_progress(1, get_run))
        await _settle()
        pump.cancel()
        return wire.headlines()

    spoken = asyncio.run(run())
    assert len(spoken) == 1 and "still going" in spoken[0]


# -- approvals are actionable events ----------------------------------------------


def test_approvals_are_never_coalesced_or_dropped_but_wait_for_the_caller():
    async def run():
        queue, wire, _scheduler, pump = _scenario("deferred", _runs((1, "a", "running")))
        talk_controls.note_caller_speaking(True)
        talk_tools.execute_talk_tool("hold", {})
        for rid in (1, 2):
            await queue.put(
                talk_cli.QueuedAnnouncement(
                    talk_cli.approval_prompt_commands(
                        {"run_id": rid, "request": f"request {rid}", "choices": ("once",)}
                    ),
                    kind=talk_announce.KIND_APPROVAL,
                )
            )
        await _settle()
        while_speaking = list(wire.headlines())
        talk_controls.note_caller_speaking(False)  # still on hold
        await _settle()
        pump.cancel()
        return while_speaking, wire.headlines()

    while_speaking, after = asyncio.run(run())
    assert while_speaking == []
    assert len(after) == 2
    assert "request 1" in after[0] and "request 2" in after[1]


# -- the immediate policy is 0.22 ----------------------------------------------------


def test_immediate_policy_speaks_every_completion_as_before():
    async def run():
        queue, wire, _scheduler, pump = _scenario("immediate", _runs((1, "triage", "done")))
        flips: list[int] = []
        talk_controls.note_caller_speaking(True)
        talk_tools.execute_talk_tool("hold", {})
        await queue.put(_completion(1, "triage", flips))
        await queue.put(_completion(2, "outlook", flips))
        await _settle()
        pump.cancel()
        return wire.headlines(), flips

    spoken, flips = asyncio.run(run())
    assert len(spoken) == 2
    assert "the report body" in spoken[0]
    assert flips == [1, 2]


def test_a_session_without_a_scheduler_behaves_as_0_22():
    async def run():
        wire = _Wire()
        queue: asyncio.Queue = asyncio.Queue()
        pump = asyncio.create_task(
            talk_cli.pump_announcements(queue, _Relay(), None, wire.send, lambda: wire.busy)
        )
        talk_controls.note_caller_speaking(True)
        await queue.put(_completion(1, "triage", []))
        await queue.put(talk_cli.landed_note_commands("sa-0-aaaa"))
        await _settle()
        pump.cancel()
        return wire.headlines()

    spoken = asyncio.run(run())
    assert len(spoken) == 2


def test_blockers_name_every_state_for_diagnostics():
    scheduler = talk_announce.Scheduler("deferred", answer_pending=lambda: True)
    talk_controls.note_caller_speaking(True)
    talk_controls.enter_hold()
    talk_controls.defer_topic()
    talk_controls.request_close()
    assert set(scheduler.blockers()) == {
        talk_announce.BLOCK_ANSWER_PENDING,
        talk_announce.BLOCK_CALLER_SPEAKING,
        talk_announce.BLOCK_HOLD,
        talk_announce.BLOCK_TOPIC_DEFERRED,
        talk_announce.BLOCK_CLOSING,
    }
    assert set(scheduler.blockers(talk_announce.KIND_APPROVAL)) == {
        talk_announce.BLOCK_ANSWER_PENDING,
        talk_announce.BLOCK_CALLER_SPEAKING,
    }
    immediate = talk_announce.Scheduler("immediate", answer_pending=lambda: False)
    assert immediate.blockers() == []


def test_a_ready_record_adopted_at_connect_waits_for_the_caller_to_speak():
    """Reconnect adoption parks results before anyone has spoken; the caller opens, not the
    notice (hermes-sip-live-voice#51). A finalized user turn lifts the block."""

    async def run():
        talk_controls.attach_session()  # fresh call: nobody has spoken yet
        get_run = _runs((9, "weather", "done"))
        queue, wire, _scheduler, pump = _scenario("deferred", get_run)
        flips: list[int] = []
        await queue.put(_completion(9, "weather", flips))
        await _settle(0.6)
        before = list(wire.headlines())
        talk_controls.note_conversation_started()
        await _settle(0.6)
        pump.cancel()
        return before, wire.headlines()

    before, after = asyncio.run(run())
    assert before == [], "spoke a ready record before the caller said anything"
    assert any("ready" in h.lower() for h in after), after
