"""Filler only when waiting, and one reply per turn (#69).

A live handset review heard "Let me check on that..." before every instant
tool call, and once heard two complete replies to one question. Both come
from the model narrating its own machinery. The rule this module makes
enforceable: the model never speaks before a tool call. If a tool call is
still out after :data:`talk_lane.FILLER_AFTER_S`, TALK says one short filler
("Give me a second.") — once per turn — through the same self-deleting,
tools-off command shape every out-of-band injection uses. A tool that
returns inside the window is answered with no filler at all.

The second half is the :class:`ReplyLedger`: which responses have actually
produced speech. When a tool result comes back for a response that already
spoke a reply, the continuation is not a second immediate reply — the
session hands it to the announcement pump, which speaks it at a pause.

Everything here is loop-side and synchronous except the armed timer task;
the seams (``send``, ``busy``, ``sleep``) are injected so a test can drive
the clock without a provider.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable

try:
    from . import talk_lane, talk_realtime
except ImportError:  # pragma: no cover - flat-module fallback (Hermes file-path load)
    import talk_lane
    import talk_realtime

_log = logging.getLogger(__name__)

#: Module default; :class:`talk_lane.LanePolicy.filler_after_s` overrides per lane.
FILLER_AFTER_S = talk_lane.FILLER_AFTER_S
#: What Talk says while a tool call is out. Short, plain, alternated per use.
FILLERS = ("Give me a second.", "One moment.")
#: The filler waits for the wire to be free (the response that made the call
#: must finish first); past this bound it is skipped rather than spoken late.
FILLER_WIRE_WAIT_S = 5.0
FILLER_WIRE_POLL_S = 0.05
#: Response metadata that marks a filler response, so the reply ledger never
#: mistakes "One moment." for the turn's reply.
FILLER_METADATA_KEY = "talk_filler"
_MAX_TURNS = 64


def filler_commands(index: int = 0) -> list[talk_realtime.RealtimeCommand]:
    """The commands that make the model say ONE filler sentence and nothing else."""

    phrase = FILLERS[index % len(FILLERS)]
    item_id = f"talkfill{uuid.uuid4().hex[:20]}"
    return [
        talk_realtime.AddContext(
            item_id=item_id,
            text=(
                f'The tool call is still running. Say exactly "{phrase}" and nothing '
                "else — no explanation, no guess at the answer."
            ),
        ),
        talk_realtime.StartResponse(allow_tools=False, metadata={FILLER_METADATA_KEY: "1"}),
        talk_realtime.RemoveContext(item_id=item_id),
    ]


class FillerTimer:
    """Speak one filler per turn, only once a tool call has been out too long.

    ``send`` writes a command batch to the wire (the session's
    ``send_outgoing``). ``busy`` says whether a response is open or pending;
    the filler waits for it to clear so it never collides with the response
    that made the call. ``after_s`` ``None`` (or ``<= 0``) disables the timer
    entirely — every method is then a no-op.
    """

    def __init__(
        self,
        send: Callable[[list], Awaitable[object]],
        *,
        after_s: float | None = FILLER_AFTER_S,
        busy: Callable[[], bool] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.after_s = after_s if isinstance(after_s, (int, float)) and after_s > 0 else None
        self._send = send
        self._busy = busy or (lambda: False)
        self._sleep = sleep or asyncio.sleep
        self._task: asyncio.Task | None = None
        self._turn: object = None
        #: Turns that already got their one filler (or whose tool returned).
        self._done: OrderedDict[object, None] = OrderedDict()
        #: Turns whose filler is on the wire: the continuation must wait for it.
        self._speaking: set = set()
        self.sent = 0

    @property
    def enabled(self) -> bool:
        return self.after_s is not None

    @property
    def armed(self) -> bool:
        return self._task is not None and not self._task.done()

    def tool_started(self, turn_id: object) -> None:
        """A tool call went out for ``turn_id``: start the one clock for that turn."""

        if not self.enabled or turn_id in self._done:
            return
        if self.armed and self._turn == turn_id:
            return
        self._cancel()
        self._turn = turn_id
        self._task = asyncio.get_running_loop().create_task(self._arm(turn_id))

    def tool_finished(self, turn_id: object) -> None:
        """The tool returned: no filler for this turn unless it already went out."""

        self._remember(turn_id)
        if self._turn == turn_id:
            self._cancel()

    def cancel(self) -> None:
        """Session teardown or discard: nothing pending may speak."""

        self._cancel()

    async def settle(self, turn_id: object) -> None:
        """Wait (bounded) for this turn's filler to finish before the continuation."""

        if turn_id not in self._speaking:
            return
        waited = 0.0
        while self._busy() and waited < FILLER_WIRE_WAIT_S:
            await self._sleep(FILLER_WIRE_POLL_S)
            waited += FILLER_WIRE_POLL_S
        self._speaking.discard(turn_id)

    def _cancel(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()

    def _remember(self, turn_id: object) -> None:
        self._done[turn_id] = None
        while len(self._done) > _MAX_TURNS:
            self._done.popitem(last=False)

    async def _arm(self, turn_id: object) -> None:
        assert self.after_s is not None
        await self._sleep(self.after_s)
        # The response that made the call must finish before another may
        # start; a tool that returns during this wait cancels us.
        waited = 0.0
        while self._busy():
            if waited >= FILLER_WIRE_WAIT_S:
                return
            await self._sleep(FILLER_WIRE_POLL_S)
            waited += FILLER_WIRE_POLL_S
        self._remember(turn_id)
        self._speaking.add(turn_id)
        try:
            accepted = await self._send(filler_commands(self.sent))
        except Exception as exc:  # noqa: BLE001 — a filler must never end the call
            self._speaking.discard(turn_id)
            _log.debug("filler not sent: %s", type(exc).__name__)
            return
        if accepted is False:
            # The wire declined (a response opened in the gap): silence, not a
            # late "one moment" over the answer.
            self._speaking.discard(turn_id)
            return
        self.sent += 1


class ReplyLedger:
    """Which responses produced speech, so a turn is answered exactly once.

    ``note_started`` records a response and whether it is a filler (from its
    metadata); ``note_spoke`` marks it as having produced audio or a final
    assistant transcript. ``replied(response_id)`` is then the question the
    tool coordinator asks before continuing a response: a response that
    already spoke gets its tool follow-up at a pause, not as a second reply.
    """

    def __init__(self) -> None:
        self._spoke: OrderedDict[str, bool] = OrderedDict()
        self._fillers: set[str] = set()

    def note_started(self, response_id: str | None, metadata=None) -> None:
        if not response_id:
            return
        if isinstance(metadata, dict) and metadata.get(FILLER_METADATA_KEY):
            self._fillers.add(response_id)
        self._spoke.setdefault(response_id, False)
        while len(self._spoke) > _MAX_TURNS:
            old, _ = self._spoke.popitem(last=False)
            self._fillers.discard(old)

    def note_spoke(self, response_id: str | None) -> None:
        if not response_id or response_id in self._fillers:
            return
        self._spoke[response_id] = True

    def is_filler(self, response_id: str | None) -> bool:
        return bool(response_id) and response_id in self._fillers

    def replied(self, response_id: object) -> bool:
        return isinstance(response_id, str) and bool(self._spoke.get(response_id, False))


__all__ = [
    "FILLERS",
    "FILLER_AFTER_S",
    "FILLER_METADATA_KEY",
    "FillerTimer",
    "ReplyLedger",
    "filler_commands",
]
