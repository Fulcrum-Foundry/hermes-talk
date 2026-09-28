"""Filler only when waiting, one reply per turn (#69).

The model must not speak before a tool call; Talk itself says one short
filler only when a call has been out longer than ``FILLER_AFTER_S``. Every
scenario drives the REAL :class:`talk_cli.ToolResponseCoordinator` with a
:class:`talk_filler.FillerTimer` on a fake clock, and asserts on what reached
the wire and in which order.
"""

from __future__ import annotations

import asyncio

import pytest

import talk_cli
import talk_filler
import talk_lane
import talk_realtime


class _Clock:
    """A sleep seam the test advances by hand; nothing waits on real time."""

    def __init__(self) -> None:
        self.now = 0.0
        self._waiters: list[tuple[float, asyncio.Future]] = []

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((self.now + seconds, fut))
        await fut

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        due = [w for w in self._waiters if w[0] <= self.now]
        self._waiters = [w for w in self._waiters if w[0] > self.now]
        for _at, fut in due:
            if not fut.done():
                fut.set_result(None)
        for _ in range(5):
            await asyncio.sleep(0)


class _Relay:
    """A tool relay whose call completes when the test releases it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def handle_tool_call_async(self, event):
        await self.release.wait()
        return [talk_realtime.SubmitToolResult(call_id=event["call_id"], output="42")]

    def tool_queue_full_commands(self, _event):
        raise AssertionError("queue should admit the call")


def _kinds(batch) -> list[str]:
    return [type(c).__name__ for c in batch]


def _is_filler(batch) -> bool:
    return any(
        isinstance(c, talk_realtime.AddContext) and "Say exactly" in c.text for c in batch
    )


async def _drive(after_s, *, release_at: float, spoke_first: bool = False, wire_busy=None):
    """One turn: a tool call at t=0, released at ``release_at``; returns sent batches."""

    clock = _Clock()
    sent: list[list] = []
    busy = {"value": False}

    async def send(batch, *, is_announcement=False):
        if is_announcement and busy["value"]:
            return False
        sent.append(list(batch))
        return True

    timer = talk_filler.FillerTimer(
        lambda commands: send(commands, is_announcement=True),
        after_s=after_s,
        busy=lambda: busy["value"],
        sleep=clock.sleep,
    )
    ledger = talk_filler.ReplyLedger()
    late: list = []
    relay = _Relay()
    coordinator = talk_cli.ToolResponseCoordinator(
        relay,
        send,
        max_pending=1,
        provider_neutral=True,
        filler=timer,
        reply_ledger=ledger,
        on_late_reply=lambda turn, cont: late.append(turn),
    )
    worker = asyncio.create_task(coordinator.run())
    ledger.note_started("resp-1")
    if spoke_first:
        ledger.note_spoke("resp-1")
    coordinator.admit({"call_id": "call_1", "response_id": "resp-1", "name": "check_work"})
    await coordinator.response_done()
    await asyncio.sleep(0)
    if wire_busy is not None:
        busy["value"] = wire_busy
    step = 0.25
    elapsed = 0.0
    while elapsed < release_at:
        await clock.advance(step)
        elapsed += step
    relay.release.set()
    await asyncio.wait_for(coordinator.join(), 1)
    # Past the threshold with the answer already in: the timer must stay quiet.
    await clock.advance(10)
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)
    return sent, late, timer


def test_a_fast_tool_gets_no_filler_at_all():
    sent, late, timer = asyncio.run(_drive(1.5, release_at=0.5))
    assert len(sent) == 1 and not _is_filler(sent[0])
    assert _kinds(sent[0]) == ["SubmitToolResult", "StartResponse"]
    assert timer.sent == 0 and late == []


def test_a_slow_tool_gets_exactly_one_filler_after_the_threshold_then_the_answer():
    sent, _late, timer = asyncio.run(_drive(1.5, release_at=4.0))
    assert len(sent) == 2
    assert _is_filler(sent[0]), "the filler goes out first, once the wait is real"
    assert talk_filler.FILLERS[0] in sent[0][0].text
    assert _kinds(sent[0]) == ["AddContext", "StartResponse", "RemoveContext"]
    start = next(c for c in sent[0] if isinstance(c, talk_realtime.StartResponse))
    assert start.allow_tools is False and start.metadata.get(talk_filler.FILLER_METADATA_KEY)
    assert not _is_filler(sent[1]) and _kinds(sent[1]) == ["SubmitToolResult", "StartResponse"]
    assert timer.sent == 1


def test_the_threshold_is_one_and_a_half_seconds_by_default():
    assert talk_lane.FILLER_AFTER_S == 1.5
    assert talk_filler.FILLER_AFTER_S == 1.5
    assert talk_lane.LanePolicy().filler_after_s == 1.5
    assert talk_lane.LanePolicy(filler_after_s=None).receipt()["filler_after_s"] is None
    # Released just under the threshold: silence. Just over: one filler.
    sent, _l, _t = asyncio.run(_drive(1.5, release_at=1.25))
    assert len(sent) == 1
    sent, _l, _t = asyncio.run(_drive(1.5, release_at=1.75))
    assert len(sent) == 2 and _is_filler(sent[0])


def test_none_disables_the_filler_however_long_the_tool_takes():
    sent, _l, timer = asyncio.run(_drive(None, release_at=30.0))
    assert not timer.enabled
    assert len(sent) == 1 and not _is_filler(sent[0])


def test_a_lane_override_moves_the_threshold():
    sent, _l, _t = asyncio.run(_drive(5.0, release_at=3.0))
    assert len(sent) == 1, "under a 5s lane threshold a 3s wait is silent"
    sent, _l, _t = asyncio.run(_drive(0.5, release_at=1.0))
    assert len(sent) == 2 and _is_filler(sent[0])


def test_the_filler_never_lands_over_an_open_response():
    # The wire stays busy for the whole wait: the filler is skipped, not spoken late.
    sent, _l, timer = asyncio.run(_drive(1.5, release_at=3.0, wire_busy=True))
    assert not any(_is_filler(b) for b in sent)
    assert timer.sent == 0


def test_exactly_one_reply_per_turn_when_the_tool_returns_after_the_answer():
    """The model already answered (spoke) before its tool call returned: the
    result is submitted WITHOUT a continuation, and the follow-up is handed to
    the announcement path for a pause — never a second immediate reply."""

    sent, late, _t = asyncio.run(_drive(1.5, release_at=0.5, spoke_first=True))
    assert len(sent) == 1
    assert _kinds(sent[0]) == ["SubmitToolResult"], "no StartResponse: no second reply"
    assert late == ["resp-1"]


def test_a_turn_that_has_not_spoken_yet_is_continued_normally():
    sent, late, _t = asyncio.run(_drive(1.5, release_at=0.5, spoke_first=False))
    assert _kinds(sent[0]) == ["SubmitToolResult", "StartResponse"]
    assert late == []


def test_reply_ledger_ignores_filler_responses_and_unnamed_turns():
    ledger = talk_filler.ReplyLedger()
    ledger.note_started("fill-1", {talk_filler.FILLER_METADATA_KEY: "1"})
    ledger.note_spoke("fill-1")
    assert not ledger.replied("fill-1"), "'one moment' is not the turn's reply"
    ledger.note_started("resp-2")
    assert not ledger.replied("resp-2")
    ledger.note_spoke("resp-2")
    assert ledger.replied("resp-2")
    assert not ledger.replied(object()) and not ledger.replied(None)


def test_late_reply_commands_are_a_tools_off_self_deleting_notice():
    batch = talk_cli.late_reply_commands()
    assert _kinds(batch) == ["AddContext", "StartResponse", "RemoveContext"]
    assert batch[1].allow_tools is False
    assert "one short sentence" in batch[0].text and "say nothing" in batch[0].text


def test_the_coordinator_without_the_seams_is_the_pre_0_25_coordinator():
    async def scenario():
        sent: list = []

        async def send(batch):
            sent.extend(batch)

        relay = _Relay()
        relay.release.set()
        coordinator = talk_cli.ToolResponseCoordinator(
            relay, send, max_pending=1, provider_neutral=True
        )
        worker = asyncio.create_task(coordinator.run())
        coordinator.admit({"call_id": "c", "response_id": "r"})
        await coordinator.response_done()
        await asyncio.wait_for(coordinator.join(), 1)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        return sent

    assert _kinds(asyncio.run(scenario())) == ["SubmitToolResult", "StartResponse"]


@pytest.mark.parametrize("phrase", talk_filler.FILLERS)
def test_fillers_are_short_plain_sentences(phrase):
    assert len(phrase.split()) <= 4 and phrase.endswith(".")
