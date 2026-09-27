"""What "delivered" may mean for a spoken notice (hermes-sip-live-voice#35, I12).

Talk's two-phase claim flips a result to ``delivered`` at the announcement
pump's post-send point: the batch was handed to the provider wire. That is
the strongest evidence THIS process has, and it is the right flip for a
transport that offers nothing better. But it is not proof the caller heard
anything — an injected notice can be barged in on at 200 ms of audio.

:func:`is_delivered` is the stricter predicate for transports that DO have
acknowledgments (SIP's Twilio mark acks measure ``audible_ms``): a notice is
delivered only when the transport acknowledged it (``acked: True``) or some
audio is proven audible (``audible_ms > 0``). ``injected: True`` alone —
the item was written to the wire — is never enough. Consumers that hold acks
call this before flipping their own ledgers; Talk's own post-send flip is
unchanged, so the immediate lane behaves exactly as before.

:func:`delivery_record` is the shape a transport hands back to Talk when it
does have evidence; :func:`on_sent_with_evidence` wraps an existing
post-send flip so it runs only when that evidence passes the predicate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

_log = logging.getLogger(__name__)

STORED = "stored"  # result is in the ledger
QUEUED = "queued"  # notice is in the announcement queue
INJECTED = "injected"  # notice was written to the wire
SPOKEN = "spoken"  # transport acknowledged audible playback
DELIVERED = "delivered"  # the result counts as delivered
STAGES = (STORED, QUEUED, INJECTED, SPOKEN, DELIVERED)


def is_delivered(notice: dict[str, Any] | None) -> bool:
    """Delivered only on transport evidence: ``acked``/``acknowledged`` True, or ``audible_ms > 0``.

    ``injected``/``dequeued``/``played_ms`` never count: those describe the
    sender's side of the wire. An ``interrupted`` notice with zero audible
    audio is not delivered even if acknowledged in the transport's own sense
    (the mark for cleared audio arrives after a clear).
    """

    if not isinstance(notice, dict):
        return False
    audible = notice.get("audible_ms")
    numeric = isinstance(audible, (int, float)) and not isinstance(audible, bool)
    audible_ms = audible if numeric else 0
    if audible_ms > 0:
        return True
    acked = notice.get("acked")
    if acked is None:
        acked = notice.get("acknowledged")
    if acked is True:
        # An ack with no measured audio is still an ack — unless the transport
        # tells us the item was cut before anything played.
        return not (notice.get("interrupted") is True and "audible_ms" in notice)
    return False


def stage(notice: dict[str, Any] | None) -> str:
    """The furthest proven stage for a notice dict (for receipts, never for the flip)."""

    if is_delivered(notice):
        return DELIVERED
    if isinstance(notice, dict):
        if notice.get("audible_ms") or notice.get("acked") or notice.get("acknowledged"):
            return SPOKEN
        if notice.get("injected") is True:
            return INJECTED
        if notice.get("queued") is True:
            return QUEUED
    return STORED


def delivery_record(
    *,
    run_id: int | None,
    injected: bool = False,
    acked: bool = False,
    audible_ms: int = 0,
    interrupted: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "injected": bool(injected),
        "acked": bool(acked),
        "audible_ms": int(audible_ms),
        "interrupted": bool(interrupted),
    }


def on_sent_with_evidence(
    on_sent: Callable[[], Any] | None, evidence: Callable[[], dict[str, Any] | None]
) -> Callable[[], Any] | None:
    """Wrap a post-send flip so it fires only when ``evidence()`` passes :func:`is_delivered`.

    For lanes whose transport reports acks. The wrapped hook is fail-closed:
    an evidence read that raises, or that says "not delivered", leaves the
    result claimed-but-undelivered (re-adoptable), never falsely delivered.
    """

    if on_sent is None:
        return None

    def hook() -> Any:
        try:
            notice = evidence()
        except Exception as exc:  # noqa: BLE001 — no evidence means no flip
            _log.debug("delivery evidence unavailable: %s: %s", type(exc).__name__, exc)
            return None
        if not is_delivered(notice):
            return None
        return on_sent()

    return hook


__all__ = [
    "DELIVERED",
    "INJECTED",
    "QUEUED",
    "SPOKEN",
    "STAGES",
    "STORED",
    "delivery_record",
    "is_delivered",
    "on_sent_with_evidence",
    "stage",
]
