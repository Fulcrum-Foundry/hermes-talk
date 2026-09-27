"""Receipt-aware steering of api_server runs from the phone lane (hermes-sip-live-voice#56).

Before this module a run NUMBER (the api_server and detached lanes) was
routed straight to ``talk_host._unsteerable_run``: "I can't pass it notes,
but I can stop it and restart." The dashboard already had capability-checked,
origin-linked steering (:mod:`talk_run_control`); the phone lane lacked the
integration. This module is that adapter, kept deliberately narrower than the
dashboard's: no origin receipts, just exact job ownership, one durable
action id per correction, and truthful receipt states.

Contract
--------
* **Ownership** — only a run whose acceptance ticket names THIS Talk session
  (or the same durable Hermes session) may be steered. A foreign-owned or
  finished job is refused in one clear sentence.
* **Capability** — when the host advertises ``run_steer``, the correction is
  ``POST``ed to ``/v1/runs/{id}/steer``; a 2xx is *queue admission* and the
  receipt says ``queued``, never "applied". When the host lacks the
  capability, or the run is between turns and will not accept steer input,
  the correction is queued as a **follow-up on the same api session**: the
  worker runs it as the next turn after the current one completes, and the
  spoken text says "queued for after this step".
* **Idempotency** — every correction has an action id. Retrying the same id
  (a lost acknowledgment) returns the existing receipt and never sends twice.
* **Receipts** — ``queued`` / ``queued_followup`` / ``applied`` /
  ``superseded`` / ``refused`` / ``unknown``. Applied is claimed only for a
  follow-up turn that actually ran. Stopping the job flips its open receipts
  to superseded. Stopping speech is unrelated and never reaches here.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from typing import Any

try:
    from . import talk_apiserver, talk_runs
except ImportError:  # pragma: no cover - flat-module fallback
    import talk_apiserver
    import talk_runs

_log = logging.getLogger(__name__)

QUEUED = "queued"
QUEUED_FOLLOWUP = "queued_followup"
APPLIED = "applied"
SUPERSEDED = "superseded"
REFUSED = "refused"
UNKNOWN = "unknown"
STATES = frozenset({QUEUED, QUEUED_FOLLOWUP, APPLIED, SUPERSEDED, REFUSED, UNKNOWN})
OPEN_STATES = frozenset({QUEUED, QUEUED_FOLLOWUP, UNKNOWN})

LANE_API_SERVER = "api-server"
MAX_TEXT_CHARS = 4_000
_MAX_RECEIPTS_PER_RUN = 32

_LOCK = threading.Lock()
#: action_id -> receipt dict
_RECEIPTS: dict[str, dict[str, Any]] = {}
#: run_id -> ordered action ids
_BY_RUN: dict[int, list[str]] = {}

FOLLOWUP_PROMPT = (
    "The operator sent this correction while your previous turn was running; it "
    "arrived too late to change that turn. Apply it now to the same task, using the "
    "results already in this session's history. Do not start unrelated work.\n\n"
    "CORRECTION:\n{text}"
)


def new_action_id() -> str:
    return f"act-{uuid.uuid4().hex[:16]}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _owned(run: dict) -> bool:
    ticket = run.get("ticket")
    owner = talk_runs.current_owner()
    if not isinstance(ticket, dict) or owner is None:
        return False
    if ticket.get("talkSessionId") == owner.get("talkSessionId"):
        return True
    durable = owner.get("hermesSessionId")
    return bool(durable) and ticket.get("hermesSessionId") == durable


def _record(receipt: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        _RECEIPTS[receipt["action_id"]] = receipt
        ids = _BY_RUN.setdefault(int(receipt["run_id"]), [])
        if receipt["action_id"] not in ids:
            ids.append(receipt["action_id"])
            if len(ids) > _MAX_RECEIPTS_PER_RUN:
                dropped = ids.pop(0)
                _RECEIPTS.pop(dropped, None)
    _persist(receipt["run_id"])
    return dict(receipt)


def _persist(run_id: int) -> None:
    """Tee the run's receipts onto its record so a later check_work can read them."""

    try:
        talk_runs.annotate_run(run_id, tee=True, steer_receipts=receipts_for(run_id))
    except Exception:  # noqa: BLE001 — a receipt must never cost the run
        _log.debug("steer receipt annotation failed for run %s", run_id, exc_info=True)


def receipts_for(run_id: int) -> list[dict[str, Any]]:
    with _LOCK:
        return [dict(_RECEIPTS[aid]) for aid in _BY_RUN.get(int(run_id), []) if aid in _RECEIPTS]


def receipt(action_id: str) -> dict[str, Any] | None:
    with _LOCK:
        found = _RECEIPTS.get(action_id)
        return dict(found) if found is not None else None


def _set_state(action_id: str, state: str, **fields: Any) -> dict[str, Any]:
    with _LOCK:
        found = _RECEIPTS.get(action_id)
        if found is None:
            return {"action_id": action_id, "state": state, **fields}
        found["state"] = state
        found["updated"] = time.time()
        found.update(fields)
        run_id = int(found["run_id"])
        view = dict(found)
    _persist(run_id)
    return view


# -- the verb -------------------------------------------------------------------


def _spoken(receipt: dict[str, Any], run_label: int) -> str:
    state = receipt["state"]
    if state == QUEUED:
        return (
            f"Passed your correction to run {run_label} — the server queued it for the "
            "agent's next step. Queued means admitted, not applied yet; I'll report "
            "what it does with it."
        )
    if state == QUEUED_FOLLOWUP:
        return (
            f"Run {run_label} can't take input mid-step here, so your correction is queued "
            "for after this step: it runs as the next turn on the same job, not as new "
            "work. I'll tell you when that turn finishes."
        )
    if state == UNKNOWN:
        return (
            f"I sent the correction to run {run_label} but didn't get an answer. I'm keeping "
            "the same action so a retry can't duplicate it; ask me again in a moment."
        )
    return receipt.get("detail") or f"Run {run_label}: {state}."


def steer(
    run: dict,
    text: str,
    *,
    action_id: str | None = None,
    mode: str = "steer",
) -> tuple[str, dict[str, Any]]:
    """Route one correction to an api_server run. Returns (spoken text, receipt).

    ``run`` is a :mod:`talk_runs` snapshot. ``action_id`` is the durable id
    of THIS correction; pass the same id again after a lost ack and the
    existing receipt comes back with no second submission.
    """

    text = (text or "").strip()[:MAX_TEXT_CHARS]
    run_id = int(run["runId"])
    action_id = action_id or new_action_id()
    existing = receipt(action_id)
    if existing is not None:
        return _spoken(existing, run_id), existing
    base = {
        "action_id": action_id,
        "run_id": run_id,
        "api_run_id": None,
        "api_session_id": None,
        "mode": mode,
        "text_hash": _hash(text),
        "text": text,
        "state": UNKNOWN,
        "evidence": None,
        "ts": time.time(),
        "updated": time.time(),
    }
    if not text:
        return "I need the correction itself before I can pass it along.", base
    if run.get("status") in talk_runs.TERMINAL_STATUSES:
        detail = f"Run {run_id} already finished, so there's nothing left to correct."
        return detail, _record(
            {**base, "state": REFUSED, "evidence": "run_finished", "detail": detail}
        )
    if not _owned(run):
        detail = (
            f"Run {run_id} belongs to a different session, so I can't change it from this call."
        )
        return detail, _record(
            {**base, "state": REFUSED, "evidence": "foreign_owner", "detail": detail}
        )
    raw_meta = run.get("meta")
    meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
    if meta.get("lane") != LANE_API_SERVER:
        detail = f"Run {run_id} isn't an api-server job, so this correction path doesn't apply."
        return detail, _record(
            {**base, "state": REFUSED, "evidence": "wrong_lane", "detail": detail}
        )
    api_run_id = meta.get("api_run_id")
    base["api_run_id"] = api_run_id if isinstance(api_run_id, str) else None
    base["api_session_id"] = meta.get("api_session_id")
    if not base["api_run_id"]:
        detail = f"I can't reach run {run_id} yet — the api server hasn't told me its run id."
        return detail, _record(
            {**base, "state": REFUSED, "evidence": "no_api_run_id", "detail": detail}
        )

    if not talk_apiserver.steering_supported():
        rec = _record({**base, "state": QUEUED_FOLLOWUP, "evidence": "capability_absent"})
        return _spoken(rec, run_id), rec
    # Reserve the action BEFORE the POST so a crash mid-flight leaves an
    # unknown receipt (retry reads it back) rather than a second submission.
    rec = _record({**base, "state": UNKNOWN, "evidence": "reserved_before_post"})
    try:
        talk_apiserver.steer_run(base["api_run_id"], text)
    except talk_apiserver.SteerRefused as exc:
        # Not running right now (between turns, waiting on an approval): the
        # correction becomes the next turn on the same session instead.
        rec = _set_state(action_id, QUEUED_FOLLOWUP, evidence=exc.code or "run_not_accepting_steer")
        return _spoken(rec, run_id), rec
    except talk_apiserver.TalkApiServerError as exc:
        rec = _set_state(action_id, UNKNOWN, evidence=f"transport:{exc}"[:120])
        return _spoken(rec, run_id), rec
    rec = _set_state(action_id, QUEUED, evidence="backend_queue_ack")
    return _spoken(rec, run_id), rec


# -- lifecycle hooks (called by the worker and stop_work) ------------------------


def pending_followups(run_id: int) -> list[dict[str, Any]]:
    return [r for r in receipts_for(run_id) if r["state"] == QUEUED_FOLLOWUP]


def followup_prompt(receipts: list[dict[str, Any]]) -> str:
    text = "\n\n".join(str(r.get("text") or "") for r in receipts)
    return FOLLOWUP_PROMPT.format(text=text)


def mark_applied(action_ids: list[str], *, evidence: str) -> None:
    for aid in action_ids:
        _set_state(aid, APPLIED, evidence=evidence)


def supersede_open(run_id: int, *, evidence: str) -> list[str]:
    """Flip every open receipt for ``run_id`` to superseded (stop, interruption)."""

    flipped = []
    for rec in receipts_for(run_id):
        if rec["state"] in OPEN_STATES:
            _set_state(rec["action_id"], SUPERSEDED, evidence=evidence)
            flipped.append(rec["action_id"])
    return flipped


def reset_for_tests() -> None:
    with _LOCK:
        _RECEIPTS.clear()
        _BY_RUN.clear()


__all__ = [
    "APPLIED",
    "OPEN_STATES",
    "QUEUED",
    "QUEUED_FOLLOWUP",
    "REFUSED",
    "STATES",
    "SUPERSEDED",
    "UNKNOWN",
    "followup_prompt",
    "mark_applied",
    "new_action_id",
    "pending_followups",
    "receipt",
    "receipts_for",
    "reset_for_tests",
    "steer",
    "supersede_open",
]
