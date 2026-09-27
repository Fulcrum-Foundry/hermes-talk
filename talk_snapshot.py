"""Session-scoped snapshots of the live call transcript (hermes-sip-live-voice#53).

A delegate never sees the call. Before this module the voice model had to
paraphrase the conversation into the brief, and the paraphrase was where
"the full context" quietly became a three-line thematic list. A
:class:`Snapshot` is the auditable alternative: the finalized turns of ONE
:class:`~talk_transcript.TranscriptCapture`, with turn ids, timestamps, a
boundary, and an explicit completeness verdict. It is data about what was
said, never an instruction: :mod:`talk_brief` quotes it as transcription
inside a trust frame.

Scope is enforced by construction. A builder is created over one capture
and can read only that capture's in-memory ring; a handle names the
capture's locally minted ``snapshot_session_id`` and is refused for any
other capture. Nothing here reads files or other sessions.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

COMPLETE = "complete"
PARTIAL_PREFIX = "partial:"
UNAVAILABLE_LINE = "call transcript not available for this brief"


class SnapshotUnavailable(RuntimeError):
    """No capture is bound, or the requested capture is not this builder's."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What one call said, as of a boundary, with a completeness verdict."""

    session_id: str
    turn_range: tuple[str | None, str | None]
    turns: list[dict[str, Any]] = field(default_factory=list)
    completeness: str = COMPLETE
    captured_at: float = 0.0

    @property
    def complete(self) -> bool:
        return self.completeness == COMPLETE

    @property
    def turn_ids(self) -> list[str]:
        return [str(turn["id"]) for turn in self.turns]

    @property
    def chars(self) -> int:
        return sum(len(str(turn.get("text") or "")) for turn in self.turns)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["turn_range"] = list(self.turn_range)
        return data

    def ref(self) -> str:
        """Opaque handle text for the brief: names the session and the boundary only."""

        first, last = self.turn_range
        return f"snapshot:{self.session_id[:12]}:{first or '-'}..{last or '-'}"


def _clean_turn(turn: dict) -> dict[str, Any] | None:
    turn_id = turn.get("id")
    role = turn.get("role")
    text = turn.get("text")
    if not isinstance(turn_id, str) or role not in {"user", "assistant"}:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    ts = turn.get("ts")
    return {
        "id": turn_id,
        "role": role,
        "text": text,
        "ts": float(ts) if isinstance(ts, (int, float)) else None,
    }


class SnapshotBuilder:
    """Builds snapshots over exactly one capture.

    ``capture`` must expose ``turns() -> list[dict]`` (id/role/text/ts),
    ``turns_dropped`` and ``snapshot_session_id`` — the 0.24
    :class:`~talk_transcript.TranscriptCapture` does. Older doubles that lack
    the ring are reported as unavailable rather than guessed at.
    """

    def __init__(self, capture: Any) -> None:
        self._capture = capture

    @property
    def session_id(self) -> str | None:
        value = getattr(self._capture, "snapshot_session_id", None)
        return value if isinstance(value, str) and value else None

    def available(self) -> bool:
        return (
            self._capture is not None
            and self.session_id is not None
            and callable(getattr(self._capture, "turns", None))
        )

    def build(
        self,
        boundary_turn_id: str | None = None,
        *,
        last_n: int | None = None,
        session_id: str | None = None,
    ) -> Snapshot:
        """Snapshot every finalized turn up to and including ``boundary_turn_id``.

        ``last_n`` keeps only the newest N turns within the boundary (the
        "recent" delegation default); the verdict then reads
        ``partial:recent_window``. ``session_id``, when given, must equal this
        builder's capture id — a handle for another capture is refused.
        """

        if not self.available():
            raise SnapshotUnavailable(UNAVAILABLE_LINE)
        own = self.session_id
        if session_id is not None and session_id != own:
            raise SnapshotUnavailable("snapshot handle names a different session")
        raw = self._capture.turns()
        turns = [cleaned for cleaned in (_clean_turn(t) for t in raw) if cleaned is not None]
        reasons: list[str] = []
        if boundary_turn_id is not None:
            ids = [turn["id"] for turn in turns]
            if boundary_turn_id not in ids:
                raise SnapshotUnavailable(f"boundary turn {boundary_turn_id!r} is not in this call")
            turns = turns[: ids.index(boundary_turn_id) + 1]
        dropped = getattr(self._capture, "turns_dropped", 0)
        if isinstance(dropped, int) and dropped > 0:
            reasons.append(f"ring_overflow:{dropped}_earliest_turns_lost")
        if last_n is not None and last_n >= 0 and len(turns) > last_n:
            turns = turns[len(turns) - last_n :]
            reasons.append("recent_window")
        completeness = COMPLETE if not reasons else PARTIAL_PREFIX + ",".join(reasons)
        first = turns[0]["id"] if turns else None
        last = turns[-1]["id"] if turns else None
        return Snapshot(
            session_id=str(own),
            turn_range=(first, last),
            turns=turns,
            completeness=completeness,
            captured_at=time.time(),
        )


# -- session binding -----------------------------------------------------------
# One builder per process, bound by the session that owns the capture (the
# same place talk_runs.attach_owner is called). Tool handlers read it; a
# detached session reads None and the brief carries the narrow unavailable
# line instead of a paraphrase.

_BUILDER: SnapshotBuilder | None = None


def attach_capture(capture: Any | None) -> None:
    global _BUILDER
    _BUILDER = SnapshotBuilder(capture) if capture is not None else None


def detach_capture() -> None:
    attach_capture(None)


def current_builder() -> SnapshotBuilder | None:
    return _BUILDER


def current_snapshot(
    boundary_turn_id: str | None = None, *, last_n: int | None = None
) -> Snapshot | None:
    """The bound session's snapshot, or ``None`` when no capture is bound/usable."""

    builder = _BUILDER
    if builder is None or not builder.available():
        return None
    try:
        return builder.build(boundary_turn_id, last_n=last_n)
    except SnapshotUnavailable:
        return None


def reset_for_tests() -> None:
    detach_capture()


__all__ = [
    "COMPLETE",
    "PARTIAL_PREFIX",
    "UNAVAILABLE_LINE",
    "Snapshot",
    "SnapshotBuilder",
    "SnapshotUnavailable",
    "attach_capture",
    "current_builder",
    "current_snapshot",
    "detach_capture",
    "reset_for_tests",
]
