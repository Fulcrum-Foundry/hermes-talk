"""Announcement scheduler — routine notices subordinate to the live conversation.

hermes-sip-live-voice#51: Talk already serialized announcements and waited for
the model to go idle (:func:`talk_cli.pump_announcements`), but "idle" only
meant the wire. It did not know the CALLER was speaking, that a hold was on,
that the caller had said "later", or that the call was closing — so a routine
"still working" or a completion landed inside the caller's own sentence, after
"hold on", and once after "hang up". This module is the missing half of that
gate: the conversation-state predicates, the ready ledger that completions
become instead of speech, and the one coalesced notice.

Two policies, chosen by :class:`talk_lane.LanePolicy.announcements`:

- ``"immediate"`` (default): today's behaviour. Every batch waits only for the
  wire/speaker to go idle; nothing is deferred or coalesced. Sessions with no
  policy behave exactly as 0.22.0.
- ``"deferred"``: routine batches (completions, heartbeats, phases) are held
  while the caller speaks, an answer is pending, a hold is on, the call is
  closing, or the caller deferred the topic. Completions are parked as READY
  RECORDS; at a natural pause ONE short notice offers everything ready.

Approval prompts and outcomes are never routine: they pass the pause gate
(no speaking over the caller) but are never coalesced or dropped, and the
closing state does not silence them — an unanswered approval denies itself.

Revalidation: a queued heartbeat/phase is re-checked against the run's
current status right before it is spoken, so an obsolete "still working"
never plays after the run finished.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

try:
    from . import talk_brief, talk_controls, talk_realtime, talk_runs
except ImportError:  # pragma: no cover - flat-module fallback (Hermes file-path load)
    import talk_brief
    import talk_controls
    import talk_realtime
    import talk_runs

_log = logging.getLogger(__name__)

POLICY_IMMEDIATE = "immediate"
POLICY_DEFERRED = "deferred"
POLICIES = frozenset({POLICY_IMMEDIATE, POLICY_DEFERRED})

#: Kinds a queued batch may declare. ``ROUTINE`` is the default for anything
#: unlabelled, so a transport that never learned the kinds is treated as the
#: cautious case (deferrable), never as urgent.
KIND_ROUTINE = "routine"
KIND_COMPLETION = "completion"
KIND_PROGRESS = "progress"
KIND_APPROVAL = "approval"
KIND_CONTROL = "control"  # microphone flips, landed notes: caller-triggered receipts

#: Why a batch is being held right now, for the diagnostics log.
BLOCK_CALLER_SPEAKING = "caller_speaking"
BLOCK_ANSWER_PENDING = "answer_pending"
BLOCK_HOLD = "hold"
BLOCK_NOT_STARTED = "conversation_not_started"
BLOCK_CLOSING = "closing"
BLOCK_TOPIC_DEFERRED = "topic_deferred"

#: Bound on the labels named in one coalesced notice; the rest is "and N more".
NOTICE_LABELS = 3
_LABEL_CHARS = 60


def coerce_policy(value: str | None) -> str:
    """The announcement policy for a session; unknown or absent → immediate (today)."""

    if isinstance(value, str) and value.strip().lower() in POLICIES:
        return value.strip().lower()
    return POLICY_IMMEDIATE


class Scheduler:
    """Conversation-state gate for the announcement pump.

    ``answer_pending`` is the session's own predicate (response active,
    continuation pending, tool outputs owed, speaker busy) — the pump already
    has it; this class only adds the caller/hold/closing/deferral states from
    :mod:`talk_controls` and the ready ledger. Everything is synchronous and
    cheap: the pump polls it every :data:`talk_cli.ANNOUNCE_IDLE_POLL_S`.
    """

    def __init__(
        self,
        policy: str | None = None,
        *,
        answer_pending: Callable[[], bool] | None = None,
        get_run: Callable[[int], dict | None] | None = None,
    ) -> None:
        self.policy = coerce_policy(policy)
        self._answer_pending = answer_pending or (lambda: False)
        self._get_run = get_run or talk_runs.get_run
        #: Completions parked instead of spoken: run_id -> (commands, on_sent).
        self._ready: dict[int, tuple[Any, Any]] = {}
        self._ready_order: list[int] = []
        #: Ready records already offered in a notice; "later" re-arms them.
        self._offered: set[int] = set()
        #: Tool handlers (daemon pool) and the pump (loop) both touch the ledger.
        self._lock = threading.Lock()

    # -- state ---------------------------------------------------------------

    @property
    def deferred(self) -> bool:
        return self.policy == POLICY_DEFERRED

    def blockers(self, kind: str = KIND_ROUTINE) -> list[str]:
        """Why a batch of ``kind`` may not be spoken right now (empty = go)."""

        found: list[str] = []
        if self._answer_pending():
            found.append(BLOCK_ANSWER_PENDING)
        if not self.deferred:
            return found
        if talk_controls.is_caller_speaking():
            found.append(BLOCK_CALLER_SPEAKING)
        if kind in (KIND_APPROVAL, KIND_CONTROL):
            # Actionable or caller-triggered: never silenced by hold, closing
            # or a topic deferral — only by the caller's own speech and an
            # in-flight answer.
            return found
        if not talk_controls.conversation_started():
            # A ready record from an EARLIER call (adopted at connect) must
            # not be the first thing the caller hears; the caller opens.
            found.append(BLOCK_NOT_STARTED)
        if talk_controls.is_holding():
            found.append(BLOCK_HOLD)
        if talk_controls.is_closing():
            found.append(BLOCK_CLOSING)
        if talk_controls.is_topic_deferred():
            found.append(BLOCK_TOPIC_DEFERRED)
        return found

    def may_speak(self, kind: str = KIND_ROUTINE) -> bool:
        return not self.blockers(kind)

    # -- ready ledger --------------------------------------------------------

    def park_completion(self, run_id: int, commands: Any, on_sent: Any = None) -> None:
        """A completion becomes a ready record instead of speech (deferred policy)."""

        with self._lock:
            if run_id not in self._ready:
                self._ready_order.append(run_id)
            self._ready[run_id] = (commands, on_sent)

    def ready_ids(self) -> list[int]:
        with self._lock:
            return list(self._ready_order)

    def pending_notice(self) -> bool:
        """Whether there is something ready the caller has not yet been offered."""

        with self._lock:
            return any(rid not in self._offered for rid in self._ready_order)

    def take_ready(self, run_id: int) -> tuple[Any, Any] | None:
        """Hand a parked completion back (the caller asked for it) and forget it."""

        with self._lock:
            entry = self._ready.pop(run_id, None)
            if entry is not None:
                self._ready_order.remove(run_id)
                self._offered.discard(run_id)
            return entry

    def release_all(self) -> list[tuple[int, Any, Any]]:
        """Every parked completion, in order, cleared — for an explicit status request."""

        with self._lock:
            released = [(rid, *self._ready[rid]) for rid in self._ready_order]
            self._ready.clear()
            self._ready_order.clear()
            self._offered.clear()
            return released

    def rearm(self) -> None:
        """The boundary passed (hold left, "later" lifted): ready results may be offered again."""

        with self._lock:
            self._offered.clear()

    def notice_commands(self) -> list[talk_realtime.RealtimeCommand]:
        """ONE short optional notice for everything ready and not yet offered.

        Composed from plugin-owned words and run labels only — no output text
        rides the notice (the artifact stays in the ledger, hermes-sip-live-
        voice#55). The delivery flips for the parked completions fire when
        this notice is sent; the caller retrieves the content via
        ``get_result`` / ``check_work``.
        """

        with self._lock:
            fresh = [rid for rid in self._ready_order if rid not in self._offered]
            self._offered.update(fresh)
        if not fresh:
            return []
        parts: list[str] = []
        ids: list[str] = []
        for rid in fresh[:NOTICE_LABELS]:
            run = self._get_run(rid) or {}
            label = talk_brief.spoken_label(run.get("label"))
            outcome = talk_runs.run_outcome(run)
            verb = "is done" if outcome == talk_runs.OUTCOME_SUCCESS else f"ended ({outcome})"
            parts.append((f"the {label} work" if label else "something you asked for") + f" {verb}")
            ids.append(str(rid))
        more = len(fresh) - len(parts)
        listing = "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
        count = len(fresh)
        noun = "one thing you asked for is" if count == 1 else f"{count} things you asked for are"
        headline = (
            f"Natural pause: {noun} ready — {listing}. Offer this in ONE short "
            "sentence and ask whether they want it now; if they say later, drop it "
            f"until asked. Do not read any result. (run_ids {', '.join(ids)} — for your "
            "tool calls only; never say the numbers aloud.)"
        )
        return _notice(headline)

    def notice_on_sent(self) -> Callable[[], None]:
        """The delivery flips of every parked completion the notice covered."""

        with self._lock:
            flips = [on_sent for rid, (_, on_sent) in self._ready.items() if rid in self._offered]

        def fire() -> None:
            for flip in flips:
                if flip is None:
                    continue
                try:
                    flip()
                except Exception as exc:  # noqa: BLE001 — a flip must never take down the pump
                    _log.debug("ready notice delivery flip failed: %s", type(exc).__name__)

        return fire

    # -- revalidation --------------------------------------------------------

    def still_valid(self, kind: str, run_id: int | None) -> bool:
        """A queued progress/heartbeat notice is obsolete once its run is terminal.

        Completions and approvals are always valid here: the terminal
        announcement IS the truth, and an approval closes itself.
        """

        if kind != KIND_PROGRESS or run_id is None:
            return True
        run = self._get_run(run_id)
        if run is None:
            return False
        return run.get("status") not in talk_runs.TERMINAL_STATUSES

    def acknowledge(self, run_id: int | None = None) -> list[int]:
        """The caller asked (check_work / get_result): parked completions are delivered.

        The tool output carried the result, so the two-phase delivery flips
        fire here and the parked speech is dropped — saying it again after
        the caller just heard it would be the interruption this module
        exists to stop. A topic deferral ends at the same moment: asking IS
        the boundary. Returns the run ids released.
        """

        talk_controls.clear_topic_deferral()
        if run_id is not None:
            entry = self.take_ready(run_id)
            released = [(run_id, *entry)] if entry is not None else []
        else:
            released = self.release_all()
        for rid, _commands, on_sent in released:
            if on_sent is None:
                continue
            try:
                on_sent()
            except Exception as exc:  # noqa: BLE001 — a flip must never fail the tool
                _log.debug("acknowledged delivery flip failed for %s: %s", rid, type(exc).__name__)
        return [rid for rid, _c, _s in released]


#: The live session's scheduler, so tool handlers on the relay's daemon pool
#: (check_work, get_result, defer_updates) reach the same ready ledger the
#: pump drains. Same attach/detach contract as talk_pause: last attach wins,
#: nothing attached means every call is a no-op.
_CURRENT: Scheduler | None = None


def attach_session(scheduler: Scheduler | None) -> None:
    global _CURRENT
    _CURRENT = scheduler


def detach_session() -> None:
    global _CURRENT
    _CURRENT = None


def current() -> Scheduler | None:
    return _CURRENT


def acknowledge(run_id: int | None = None) -> list[int]:
    """Module-level convenience for tool handlers; no session → nothing to release."""

    scheduler = _CURRENT
    if scheduler is None:
        talk_controls.clear_topic_deferral()
        return []
    return scheduler.acknowledge(run_id)


def reset_for_tests() -> None:
    detach_session()


def _notice(headline: str) -> list[talk_realtime.RealtimeCommand]:
    """The same self-deleting, tools-off shape every out-of-band injection uses."""

    import uuid

    item_id = f"talkann{uuid.uuid4().hex[:20]}"
    return [
        talk_realtime.AddContext(item_id=item_id, text=headline),
        talk_realtime.StartResponse(allow_tools=False),
        talk_realtime.RemoveContext(item_id=item_id),
    ]


__all__ = [
    "BLOCK_ANSWER_PENDING",
    "BLOCK_CALLER_SPEAKING",
    "BLOCK_CLOSING",
    "BLOCK_HOLD",
    "BLOCK_TOPIC_DEFERRED",
    "KIND_APPROVAL",
    "KIND_COMPLETION",
    "KIND_CONTROL",
    "KIND_PROGRESS",
    "KIND_ROUTINE",
    "NOTICE_LABELS",
    "POLICIES",
    "POLICY_DEFERRED",
    "POLICY_IMMEDIATE",
    "Scheduler",
    "acknowledge",
    "attach_session",
    "coerce_policy",
    "current",
    "detach_session",
    "reset_for_tests",
]
