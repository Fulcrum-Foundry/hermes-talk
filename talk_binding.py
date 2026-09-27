"""Durable phone-conversation binding across calls (hermes-sip-live-voice#35, I11).

A phone connection is transient: every call mints a new Talk session id, and
the reconnect-adoption path (:func:`talk_runs.list_undelivered_for_session`)
rightly refuses any run whose ticket lacks a DURABLE Hermes session id — a
new transient id must not adopt everything on the host. Before this module
that meant a caller who hung up mid-job and called back could never be
handed the exact pending result, because the phone lane usually has no
Hermes context to supply that id.

A :class:`Binding` is the durable record for ONE caller on ONE deployment,
keyed by an opaque ``binding_key`` the transport derives (SIP passes a hash
of caller + deployment). It persists under ``state/talk-bindings/`` and
holds the ``hermes_session_id`` that owns the caller's work (minted here
when the host supplies none), every Talk session id seen, the run and
result ids, and the deliveries still pending. On the next call with the
same key the session ATTACHES with that id — the adoption check is not
weakened; it is satisfied — and the exact pending results are adopted
through the existing ticket path. Another key never sees the record.

Gateway restart: a run still marked running in history with no live
counterpart is what :mod:`talk_runs` already reports as ``lost``;
:meth:`Binding.pending_view` labels those ``interrupted``, never "still
running".

After-call delivery policy (:attr:`talk_lane.LanePolicy.after_call`):
``"retrievable"`` (default) keeps results in the ledger and the binding for
the next call or the desktop; ``"none"`` records nothing pending for the
caller. No external delivery (text, callback) is ever started here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from . import talk_config, talk_runs
except ImportError:  # pragma: no cover - flat-module fallback
    import talk_config
    import talk_runs

_log = logging.getLogger(__name__)

BINDINGS_DIRNAME = "talk-bindings"
AFTER_CALL_RETRIEVABLE = "retrievable"
AFTER_CALL_NONE = "none"
AFTER_CALL_POLICIES = (AFTER_CALL_RETRIEVABLE, AFTER_CALL_NONE)
PENDING = "pending"
INTERRUPTED = "interrupted"
DELIVERED = "delivered"
_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_MAX_IDS = 200
_LOCK = threading.RLock()


def enabled() -> bool:
    """Inert under pytest unless a test opts in (same rule as the run history tee)."""

    return "PYTEST_CURRENT_TEST" not in os.environ


def make_key(caller_key: str, deployment: str) -> str:
    """The transport-side helper: a stable opaque key for (caller, deployment)."""

    digest = hashlib.sha256(f"{deployment}\x00{caller_key}".encode()).hexdigest()
    return f"b-{digest[:40]}"


def bindings_dir() -> Path:
    path = talk_config.state_dir() / BINDINGS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)
    return path


def _path(key: str) -> Path:
    if not _KEY_RE.match(key):
        raise ValueError("binding key must be an opaque token (8-128 [A-Za-z0-9._:-])")
    return bindings_dir() / f"{key}.json"


@dataclass(slots=True)
class Binding:
    key: str
    hermes_session_id: str
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    talk_session_ids: list[str] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    result_ids: list[int] = field(default_factory=list)
    #: run_id -> {"state": pending|interrupted|delivered, "ts": float, "label": str}
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    after_call: str = AFTER_CALL_RETRIEVABLE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- mutation (each persists) -------------------------------------------

    def note_session(self, talk_session_id: str) -> None:
        with _LOCK:
            if talk_session_id not in self.talk_session_ids:
                self.talk_session_ids.append(talk_session_id)
                del self.talk_session_ids[:-_MAX_IDS]
            self._save()

    def note_run(self, run_id: int, *, label: str = "") -> None:
        with _LOCK:
            if run_id not in self.run_ids:
                self.run_ids.append(int(run_id))
                del self.run_ids[:-_MAX_IDS]
            if self.after_call == AFTER_CALL_RETRIEVABLE:
                self.pending[str(run_id)] = {"state": PENDING, "ts": time.time(), "label": label}
            self._save()

    def note_result(self, run_id: int) -> None:
        with _LOCK:
            if run_id not in self.result_ids:
                self.result_ids.append(int(run_id))
                del self.result_ids[:-_MAX_IDS]
            self._save()

    def mark_delivered(self, run_id: int) -> None:
        with _LOCK:
            row = self.pending.get(str(run_id))
            if row is not None:
                row["state"] = DELIVERED
                row["ts"] = time.time()
            self._save()

    def mark_interrupted(self, run_id: int) -> None:
        with _LOCK:
            row = self.pending.get(str(run_id))
            if row is not None and row.get("state") == PENDING:
                row["state"] = INTERRUPTED
                row["ts"] = time.time()
            self._save()

    # -- views -----------------------------------------------------------------

    def pending_view(self) -> list[dict[str, Any]]:
        """Pending rows reconciled against the run registry/history.

        A run the registry reports as ``lost`` (history says running, no live
        worker: the process that owned it is gone) is labelled
        ``interrupted`` here and persisted so, never "still running".
        """

        out = []
        with _LOCK:
            for raw_id, row in list(self.pending.items()):
                if row.get("state") == DELIVERED:
                    continue
                run_id = int(raw_id)
                record = talk_runs.resolve_run_record(run_id)
                status = record.get("status") if isinstance(record, dict) else None
                state = row.get("state")
                if status == "lost" and state == PENDING:
                    row["state"] = state = INTERRUPTED
                    row["ts"] = time.time()
                out.append(
                    {
                        "run_id": run_id,
                        "state": state,
                        "label": row.get("label") or (record or {}).get("label") or "",
                        "run_status": status,
                        "outcome": talk_runs.run_outcome(record) if record else None,
                    }
                )
            self._save()
        return out

    # -- persistence ------------------------------------------------------------

    def _save(self) -> None:
        if not enabled():
            return
        self.updated = time.time()
        try:
            path = _path(self.key)
            tmp = path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 — a binding write must never break a call
            _log.warning("talk binding %s could not be persisted: %s", self.key[:12], exc)


def _load(key: str) -> Binding | None:
    path = _path(key)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("key") != key:
        return None
    try:
        return Binding(
            key=key,
            hermes_session_id=str(raw["hermes_session_id"]),
            created=float(raw.get("created") or time.time()),
            updated=float(raw.get("updated") or time.time()),
            talk_session_ids=[str(x) for x in raw.get("talk_session_ids") or []],
            run_ids=[int(x) for x in raw.get("run_ids") or []],
            result_ids=[int(x) for x in raw.get("result_ids") or []],
            pending={
                str(k): dict(v)
                for k, v in (raw.get("pending") or {}).items()
                if isinstance(v, dict)
            },
            after_call=(
                str(raw["after_call"])
                if raw.get("after_call") in AFTER_CALL_POLICIES
                else AFTER_CALL_RETRIEVABLE
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def for_caller(
    caller_key: str,
    deployment: str | None = None,
    *,
    hermes_session_id: str | None = None,
    after_call: str = AFTER_CALL_RETRIEVABLE,
) -> Binding:
    """Load or create THE binding for this caller on this deployment.

    ``caller_key`` may already be an opaque binding key (SIP passes one);
    with ``deployment`` given it is hashed with it first. ``hermes_session_id``
    is the host's durable id when the lane has one — it becomes the binding's
    on first sight; an existing binding keeps its own so work accepted under
    it stays adoptable. Without either, an id is minted and persisted here.
    """

    key = make_key(caller_key, deployment) if deployment is not None else str(caller_key)
    if after_call not in AFTER_CALL_POLICIES:
        raise ValueError(f"after_call must be one of {', '.join(AFTER_CALL_POLICIES)}")
    with _LOCK:
        binding = _load(key)
        if binding is None:
            binding = Binding(
                key=key,
                hermes_session_id=hermes_session_id or f"talk-binding-{uuid.uuid4().hex}",
                after_call=after_call,
            )
        binding.after_call = after_call
        binding._save()
        return binding


__all__ = [
    "AFTER_CALL_NONE",
    "AFTER_CALL_POLICIES",
    "AFTER_CALL_RETRIEVABLE",
    "BINDINGS_DIRNAME",
    "DELIVERED",
    "INTERRUPTED",
    "PENDING",
    "Binding",
    "bindings_dir",
    "enabled",
    "for_caller",
    "make_key",
]
