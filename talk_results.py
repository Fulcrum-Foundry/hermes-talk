"""Result ledger — the full artifact of every finished run, by exact job (hermes-sip-live-voice#55).

The announcement path is deliberately lossy: a finished run is spoken from a
bounded tail inside a self-deleting context item, so the raw report never
lingers with system priority (:func:`talk_cli._announcement_commands`). That
containment is right, but it left the model with nothing to retrieve when the
caller asked "what did the second one say?" — and it reached for whichever
summary it remembered, which in the reviewed call was the OLDER run's.

This module is the canonical retrieval path. One record per run, written at
the terminal transition (:func:`talk_runs.finish_run` calls :func:`record`),
holding the whole untruncated output in a 0600 file under
``$HERMES_HOME/state/talk-results/<run_id>.txt`` and a small JSON index
beside it. Retrieval is by exact run id or by label (+ order: "the second
one"), paged, and every page is framed as UNTRUSTED data.

Supersession: when a newer run with the same label completes, the older
record is marked ``superseded_by`` so a stale brief can never silently answer
the revised question — retrieving it says so first.

Fail-open everywhere: a ledger that cannot write must never fail the run.
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
from pathlib import Path

try:
    from . import talk_config, talk_runs
except ImportError:  # pragma: no cover - flat-module fallback (Hermes file-path load)
    import talk_config
    import talk_runs

_log = logging.getLogger(__name__)

RESULTS_DIRNAME = "talk-results"
INDEX_FILENAME = "index.json"
#: One retrieval page. Sized for speech: the model summarizes a page, it does
#: not read it, and a caller who wants the rest says "more".
PAGE_CHARS = 1_500
#: Bound on the spoken summary stored in the record. The artifact is the file.
SPOKEN_SUMMARY_CHARS = 600
#: How many records the index keeps. Old ones drop with their files.
MAX_RECORDS = 200

#: Word → position for "the second one" style references.
_ORDINALS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "last": -1,
    "latest": -1,
    "newest": -1,
    "most recent": -1,
    "recent": -1,
    "earlier": 1,
    "older": 1,
    "previous": 1,
    "original": 1,
    "earliest": 1,
}

_LOCK = threading.RLock()


def enabled() -> bool:
    """Inert under pytest unless a test opts in (same rule as ``talk_runs._history_enabled``).

    Every suite that finishes a run would otherwise write into the operator's
    real Hermes home. Ledger tests monkeypatch this to ``lambda: True`` with a
    repointed ``HERMES_HOME``.
    """

    return "PYTEST_CURRENT_TEST" not in os.environ


class Ambiguous(Exception):
    """More than one record matched and no order was given; the caller must ask."""

    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates
        super().__init__(f"{len(candidates)} results match")


def results_dir() -> Path:
    path = talk_config.state_dir() / RESULTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)
    return path


def _index_path() -> Path:
    return results_dir() / INDEX_FILENAME


def _read_index() -> list[dict]:
    try:
        raw = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [entry for entry in raw if isinstance(entry, dict)] if isinstance(raw, list) else []


def _write_index(records: list[dict]) -> None:
    path = _index_path()
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(records, handle)
    os.replace(tmp, path)


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def brief_version(prompt: str | None) -> str:
    """A short stable hash of the delegated brief, so two runs of the same label are told apart."""

    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()[:12]


def _spoken_summary(output: str, outcome: str) -> str:
    text = " ".join(str(output or "").split())
    if not text:
        return "" if outcome == talk_runs.OUTCOME_SUCCESS else f"({outcome}, no output)"
    if len(text) <= SPOKEN_SUMMARY_CHARS:
        return text
    return text[: SPOKEN_SUMMARY_CHARS - 1].rstrip() + "…"


def record(run_id: int, run: dict, *, prompt: str | None = None) -> dict | None:
    """Persist the result record for a terminal run. Returns it, or ``None`` on failure.

    Called by :func:`talk_runs.finish_run` after its terminal tee. The full
    output is written FIRST so the index never points at a file that is not
    there. A previous record for the same label (an older run) is marked
    superseded by this one.
    """

    if not enabled():
        return None
    try:
        raw_meta = run.get("meta")
        meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
        output = str(run.get("output") or "")
        outcome = talk_runs.run_outcome(run)
        label = str(run.get("label") or "").strip()
        with _LOCK:
            directory = results_dir()
            _write_private(directory / f"{int(run_id)}.txt", output)
            brief = meta.get("brief_version")
            if not isinstance(brief, str) or not brief:
                brief = brief_version(prompt if prompt is not None else label)
            entry = {
                "job_label": label,
                "run_id": int(run_id),
                "api_run_id": str(meta.get("api_run_id") or ""),
                "api_session_id": str(meta.get("api_session_id") or ""),
                "brief_version": brief,
                "outcome": outcome,
                "completed_at": float(run.get("updated") or time.time()),
                "spoken_summary": _spoken_summary(output, outcome),
                "full_output_path": str(directory / f"{int(run_id)}.txt"),
                "output_chars": len(output),
                "superseded_by": None,
            }
            records = [r for r in _read_index() if r.get("run_id") != entry["run_id"]]
            # Supersession follows REQUEST order (run id), not completion
            # order: a later-requested job of the same label is the current
            # one even when it finished first (hermes-sip-live-voice#55).
            for other in records:
                if not label or other.get("job_label") != label:
                    continue
                other_id = int(other.get("run_id", 0))
                if other_id < entry["run_id"] and other.get("superseded_by") is None:
                    other["superseded_by"] = entry["run_id"]
                elif other_id > entry["run_id"] and (
                    entry["superseded_by"] is None or int(entry["superseded_by"]) > other_id
                ):
                    entry["superseded_by"] = other_id
            records.append(entry)
            records.sort(key=lambda r: int(r.get("run_id", 0)))
            dropped = records[:-MAX_RECORDS] if len(records) > MAX_RECORDS else []
            records = records[-MAX_RECORDS:]
            _write_index(records)
            for gone in dropped:
                with contextlib.suppress(OSError):
                    Path(str(gone.get("full_output_path") or "")).unlink(missing_ok=True)
        return entry
    except Exception as exc:  # noqa: BLE001 — the ledger never fails the run
        _log.warning("result ledger write failed for run %s: %s", run_id, type(exc).__name__)
        return None


def list_records() -> list[dict]:
    if not enabled():
        return []
    with _LOCK:
        return _read_index()


def get_record(run_id: int) -> dict | None:
    for entry in list_records():
        if entry.get("run_id") == int(run_id):
            return entry
    return None


def ready_records(since_run_id: int | None = None) -> list[dict]:
    """Records not yet superseded, newest last — the scheduler's coalescing input."""

    return [
        r
        for r in list_records()
        if r.get("superseded_by") is None
        and (since_run_id is None or int(r.get("run_id", 0)) > since_run_id)
    ]


def read_output(run_id: int, *, offset: int = 0, limit: int = PAGE_CHARS) -> tuple[str, int, int]:
    """One page of the full artifact: ``(page, total_chars, next_offset)``.

    ``next_offset == total_chars`` means this was the last page.
    """

    entry = get_record(run_id)
    if entry is None:
        raise KeyError(run_id)
    text = Path(str(entry["full_output_path"])).read_text(encoding="utf-8")
    start = max(0, int(offset or 0))
    end = min(len(text), start + max(1, int(limit or PAGE_CHARS)))
    return text[start:end], len(text), end


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _parse_ordinal(reference: str) -> tuple[int | None, str]:
    """Split "the second outlook one" into (2, "outlook one")."""

    lowered = reference.lower()
    for word, position in sorted(_ORDINALS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            lowered = re.sub(rf"\b{re.escape(word)}\b", " ", lowered)
            return position, lowered
    return None, lowered


def resolve(reference: str | int | None) -> dict:
    """Find the ONE record a spoken reference means.

    Accepts a run number, a label fragment, or an ordinal phrase ("the second
    one", "the latest triage"). Ordinals count in REQUEST order (run id), so
    "the second one" is the second job the caller asked for even when it
    finished first. Raises :class:`KeyError` for no match and
    :class:`Ambiguous` when several match and no order disambiguates —
    the tool then asks rather than guessing (ANTI_GUESS_RULE).
    """

    records = list_records()
    if reference is None or (isinstance(reference, str) and not reference.strip()):
        if len(records) == 1:
            return records[0]
        if not records:
            raise KeyError("no results")
        raise Ambiguous(records)
    if isinstance(reference, int) or (isinstance(reference, str) and reference.strip().isdigit()):
        entry = get_record(int(reference))
        if entry is None:
            raise KeyError(reference)
        return entry
    text = str(reference).strip()
    match = re.fullmatch(r"(?:run\s*#?\s*)(\d+)", text, flags=re.IGNORECASE)
    if match:
        entry = get_record(int(match.group(1)))
        if entry is None:
            raise KeyError(reference)
        return entry
    position, remainder = _parse_ordinal(text)
    wanted = _tokens(remainder) - {"one", "the", "run", "job", "result", "report", "that", "this"}
    if wanted:
        scored = []
        for entry in records:
            have = _tokens(str(entry.get("job_label") or ""))
            overlap = len(wanted & have)
            if overlap:
                scored.append((overlap, entry))
        if not scored:
            raise KeyError(reference)
        best = max(score for score, _ in scored)
        candidates = [entry for score, entry in scored if score == best]
    else:
        candidates = list(records)
    if not candidates:
        raise KeyError(reference)
    if position is not None:
        if position == -1:
            return candidates[-1]
        if position <= len(candidates):
            return candidates[position - 1]
        raise KeyError(reference)
    if len(candidates) == 1:
        return candidates[0]
    raise Ambiguous(candidates)


def describe(entry: dict) -> str:
    """One spoken line naming a record: number, label, outcome, staleness."""

    label = str(entry.get("job_label") or "").strip() or "unlabelled"
    line = f"run {entry.get('run_id')} ({label}) {entry.get('outcome')}"
    if entry.get("superseded_by") is not None:
        line += f" — SUPERSEDED by run {entry['superseded_by']}, treat as stale"
    return line


def reset_for_tests() -> None:
    if not enabled():
        return
    with _LOCK:
        try:
            directory = results_dir()
        except Exception:  # noqa: BLE001
            return
        for path in directory.glob("*"):
            with contextlib.suppress(OSError):
                path.unlink()


__all__ = [
    "INDEX_FILENAME",
    "MAX_RECORDS",
    "PAGE_CHARS",
    "RESULTS_DIRNAME",
    "SPOKEN_SUMMARY_CHARS",
    "Ambiguous",
    "brief_version",
    "describe",
    "enabled",
    "get_record",
    "list_records",
    "read_output",
    "ready_records",
    "record",
    "reset_for_tests",
    "resolve",
    "results_dir",
]
