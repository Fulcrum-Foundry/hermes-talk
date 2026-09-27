"""Conversational controls — hold, resume, verbosity, closing, topic deferral.

One live voice session per process, one set of controls (hermes-sip-live-voice#57,
#51). The observed failure this module answers: a caller said "hold on" three
times and got three acknowledgments, "stop" was one word away from cancelling a
costly job, and a hangup was followed by one more routine status update. Those
are FIVE different intents that used to share one verb, so they are given
separate state here and separate tools in :mod:`talk_tools`:

- ``hold``: the model produces no speech and routine announcements are
  suppressed, but the microphone keeps listening so "continue" works.
- ``resume``: leaves hold (and clears a topic deferral).
- ``defer_updates``: "later" — routine notices stay parked until the caller
  asks or the hold/resume boundary; the conversation itself continues.
- ``set_verbosity``: concise vs detailed SPOKEN depth. A session-local prompt
  instruction only; never the completeness of an artifact or a brief.
- closing: an end-call request was made (:func:`request_close`); nothing
  routine may play after it.

Plain "stop" is deliberately NOT here: it means "stop speaking" and the relay's
barge-in path already does that. Cancelling work is ``cancel_job``, an explicit
tool that names a run (reuses ``talk_host.stop_work``).

Same one-at-a-time contract as :mod:`talk_pause`: last attach wins, every
reader takes the lock, and while nothing is attached every state reads as
neutral so the announcement gate never blocks on a session that is gone. The
attaching session's ``on_change`` fires OUTSIDE the lock; marshalling onto the
loop is the session's business (tool handlers run on the relay's daemon pool).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

_log = logging.getLogger(__name__)

VERBOSITY_CONCISE = "concise"
VERBOSITY_DETAILED = "detailed"
VERBOSITIES = frozenset({VERBOSITY_CONCISE, VERBOSITY_DETAILED})

#: What changed, passed to ``on_change(kind, active)``.
CHANGE_HOLD = "hold"
CHANGE_TOPIC_DEFERRED = "topic_deferred"
CHANGE_CLOSING = "closing"
CHANGE_VERBOSITY = "verbosity"

_LOCK = threading.Lock()
_ATTACHED = False
_ON_CHANGE: Callable[[str, object], None] | None = None
_HOLD = False
_STARTED = False
_TOPIC_DEFERRED = False
_CLOSING = False
_CALLER_SPEAKING = False
_VERBOSITY: str | None = None


def attach_session(on_change: Callable[[str, object], None] | None = None) -> None:
    """Bind the live session. Every state starts neutral: a new call inherits nothing."""

    global _ATTACHED, _ON_CHANGE, _HOLD, _TOPIC_DEFERRED, _CLOSING, _CALLER_SPEAKING
    global _VERBOSITY, _STARTED
    with _LOCK:
        _ATTACHED = True
        _ON_CHANGE = on_change
        _HOLD = False
        _STARTED = False
        _TOPIC_DEFERRED = False
        _CLOSING = False
        _CALLER_SPEAKING = False
        _VERBOSITY = None


def detach_session() -> None:
    global _ATTACHED, _ON_CHANGE, _HOLD, _TOPIC_DEFERRED, _CLOSING, _CALLER_SPEAKING
    global _VERBOSITY, _STARTED
    with _LOCK:
        _ATTACHED = False
        _ON_CHANGE = None
        _HOLD = False
        _STARTED = False
        _TOPIC_DEFERRED = False
        _CLOSING = False
        _CALLER_SPEAKING = False
        _VERBOSITY = None


def _notify(kind: str, value: object) -> None:
    with _LOCK:
        callback = _ON_CHANGE
    if callback is None:
        return
    try:
        callback(kind, value)
    except Exception as exc:  # noqa: BLE001 — a receipt must never undo the state change
        _log.debug("controls change callback failed: %s: %s", type(exc).__name__, exc)


# -- hold / resume -------------------------------------------------------------


def enter_hold() -> bool:
    """Enter hold. Returns True when this call made the transition (False = already held).

    Idempotent on purpose: the second "hold on" must produce nothing, not a
    second acknowledgment.
    """

    global _HOLD
    with _LOCK:
        if not _ATTACHED or _HOLD:
            return False
        _HOLD = True
    _notify(CHANGE_HOLD, True)
    return True


def leave_hold() -> bool:
    """Leave hold and clear a topic deferral (the caller is back; both boundaries pass)."""

    global _HOLD, _TOPIC_DEFERRED
    with _LOCK:
        changed = _HOLD or _TOPIC_DEFERRED
        _HOLD = False
        _TOPIC_DEFERRED = False
    if changed:
        _notify(CHANGE_HOLD, False)
    return changed


def is_holding() -> bool:
    with _LOCK:
        return _HOLD


# -- topic deferral ("later") ---------------------------------------------------


def defer_topic() -> bool:
    global _TOPIC_DEFERRED
    with _LOCK:
        if not _ATTACHED or _TOPIC_DEFERRED:
            return False
        _TOPIC_DEFERRED = True
    _notify(CHANGE_TOPIC_DEFERRED, True)
    return True


def clear_topic_deferral() -> bool:
    """The caller asked (check_work / get_result) — the deferral boundary is reached."""

    global _TOPIC_DEFERRED
    with _LOCK:
        if not _TOPIC_DEFERRED:
            return False
        _TOPIC_DEFERRED = False
    _notify(CHANGE_TOPIC_DEFERRED, False)
    return True


def is_topic_deferred() -> bool:
    with _LOCK:
        return _TOPIC_DEFERRED


# -- closing -------------------------------------------------------------------


def request_close() -> bool:
    """Mark the call as closing. The lane calls this from its end-call handler.

    Public seam for transports: ``talk_cli`` also wraps a lane tool literally
    named ``end_call`` so a lane that forgets still gets the gate. Background
    work is NOT touched — closing the call never cancels authorized jobs.
    """

    global _CLOSING
    with _LOCK:
        if _CLOSING:
            return False
        _CLOSING = True
    _notify(CHANGE_CLOSING, True)
    return True


def is_closing() -> bool:
    with _LOCK:
        return _CLOSING


# -- caller speaking -----------------------------------------------------------



def note_conversation_started() -> None:
    """The caller has spoken (a final user turn) or heard a real reply."""

    global _STARTED
    with _LOCK:
        _STARTED = True


def conversation_started() -> bool:
    with _LOCK:
        return _STARTED


def note_caller_speaking(active: bool) -> None:
    """Track VAD speech_started/stopped so a notice never starts over the caller."""

    global _CALLER_SPEAKING
    with _LOCK:
        _CALLER_SPEAKING = bool(active)


def is_caller_speaking() -> bool:
    with _LOCK:
        return _CALLER_SPEAKING


# -- verbosity -----------------------------------------------------------------


def set_verbosity(mode: str) -> str:
    """Set the spoken-depth mode. Returns the normalized mode; raises on an unknown one."""

    global _VERBOSITY
    normalized = str(mode or "").strip().lower()
    if normalized not in VERBOSITIES:
        raise ValueError(f"unknown verbosity mode: {mode!r}")
    with _LOCK:
        _VERBOSITY = normalized
    _notify(CHANGE_VERBOSITY, normalized)
    return normalized


def verbosity() -> str | None:
    """The session's mode, or ``None`` when the caller never set one (lane default applies)."""

    with _LOCK:
        return _VERBOSITY


def snapshot() -> dict:
    """Diagnostics view of every state, under one lock read."""

    with _LOCK:
        return {
            "attached": _ATTACHED,
            "hold": _HOLD,
            "topic_deferred": _TOPIC_DEFERRED,
            "closing": _CLOSING,
            "caller_speaking": _CALLER_SPEAKING,
            "verbosity": _VERBOSITY,
        }


def reset_for_tests() -> None:
    detach_session()


__all__ = [
    "CHANGE_CLOSING",
    "CHANGE_HOLD",
    "CHANGE_TOPIC_DEFERRED",
    "CHANGE_VERBOSITY",
    "VERBOSITIES",
    "VERBOSITY_CONCISE",
    "VERBOSITY_DETAILED",
    "attach_session",
    "clear_topic_deferral",
    "conversation_started",
    "defer_topic",
    "detach_session",
    "enter_hold",
    "is_caller_speaking",
    "is_closing",
    "is_holding",
    "is_topic_deferred",
    "leave_hold",
    "note_caller_speaking",
    "note_conversation_started",
    "request_close",
    "reset_for_tests",
    "set_verbosity",
    "snapshot",
    "verbosity",
]
