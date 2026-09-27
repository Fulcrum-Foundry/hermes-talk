"""Tier-2 agent lane — a real Hermes agent reached over the api_server platform.

Hermes's api_server gateway platform (``gateway/platforms/api_server.py``)
exposes a server-side ``AIAgent`` to any process on the box. That is what lets
a voice session running OUTSIDE an agent — the dashboard tab, a standalone
``hermes talk`` — still reach real tools instead of announcing a fallback.

**Why /v1/runs and not a synchronous chat endpoint.** ``POST /v1/runs``
(api_server.py:4159) answers ``202 {"run_id": …}`` the instant the run is
queued, and the result is collected later from ``GET /v1/runs/{id}``
(:4470, ``output`` on success / ``error`` on failure). The synchronous
alternatives hold an HTTP connection open for the entire agent loop:
``/v1/chat/completions`` is one call but blocks until the agent finishes, and
``POST /api/sessions/{id}/chat`` (:1867) blocks AND 404s unless a session was
created first — two round trips plus a persistent session row in the
operator's SessionDB for what is one lookup. A handle you can walk away from
is worth more here than a response you must wait for.

**Why nothing in this module blocks a voice turn.** The relay executes tools
SYNCHRONOUSLY on the event loop (``talk_relay.handle_event`` is called inline
from ``talk_cli``'s receive loop), so a tool that waits is a microphone that
stops. An agent run takes seconds to minutes. Therefore the run lane is
handed to a :mod:`talk_runs` worker thread and the caller gets the
``WORK_STARTED`` receipt immediately — the same machinery, and the same
already-built watchers on both surfaces, that background delegation uses. The
answer is SPOKEN when it lands. A bounded "wait a few seconds first" hybrid
was considered and rejected: any wait at all is dead air on a live call,
because the wait happens on the loop that carries the audio.

**Why :func:`status` never blocks either.** The obvious shortcut is to let the
availability probe wait once, on the theory that a closed loopback port
refuses instantly. That is not true everywhere: on a box whose firewall DROPS
instead of RSTs, connecting to a closed ``127.0.0.1`` port burns the entire
timeout (measured here — 8642, 9999, and 45999 all took the full budget while
an open port answered in 15ms). So :func:`status` answers from cache and
schedules a background refresh, and the ONE function that waits for the
network says so in its name: :func:`warm`, which belongs at session start,
never in a turn.

Probe/status/warm never raise — a bad lane answers False, not an exception.
The run and catalog reads (start_run, get_run, stop_run, list_skills,
list_toolsets, capabilities_payload, health_detailed) DO raise
TalkApiServerError with speakable text; callers on a tool handler must catch it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

try:
    from . import talk_config
except ImportError:  # pragma: no cover - flat-module fallback (Hermes file-path load)
    import talk_config

_log = logging.getLogger(__name__)

CAPABILITIES_PATH = "/v1/capabilities"
RUNS_PATH = "/v1/runs"
#: Read-only catalog surfaces. These answer "what does this install HAVE",
#: which is a different question from :data:`CAPABILITIES_PATH`'s "may I submit
#: a run" — see :func:`capabilities_payload` on why both read the same path.
SKILLS_PATH = "/v1/skills"
TOOLSETS_PATH = "/v1/toolsets"
HEALTH_DETAILED_PATH = "/health/detailed"

#: api_server run statuses that will never change again (api_server.py:4377-4404).
#: ``interrupted`` is stamped by the host when the gateway shuts a live run
#: down (api_server_runs._finish); it is terminal and NOT a success.
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
#: Statuses the host uses while a run is still alive. Anything outside this
#: set AND outside :data:`TERMINAL_RUN_STATUSES` is a status this Talk does
#: not know; it is treated as terminal-unknown rather than polled forever
#: (hermes-sip-live-voice#49), and it can never be announced as success.
LIVE_RUN_STATUSES = frozenset({"queued", "running", "waiting_for_approval", "stopping"})

#: Outcome categories: what the caller may truthfully SAY about the run.
OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_INTERRUPTED = "interrupted"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_UNKNOWN = "unknown"
OUTCOME_INCOMPLETE = "incomplete"  # host turn ended, delegated children still running
_REMOTE_TO_OUTCOME = {
    "completed": OUTCOME_SUCCESS,
    "failed": OUTCOME_FAILED,
    "cancelled": OUTCOME_CANCELLED,
    "interrupted": OUTCOME_INTERRUPTED,
}

#: Why the lane is unusable, in the words the model should say.
REASON_OK = "ok"
REASON_ABSENT = "absent"
REASON_UNAUTHORIZED = "unauthorized"
REASON_ERROR = "error"

MAX_OUTPUT_CHARS = 4_000


class TalkApiServerError(Exception):
    """The api_server was reachable but the run could not be completed."""


@dataclass(frozen=True, slots=True)
class ApiServerStatus:
    """One probe verdict. ``detail`` is written to be said out loud."""

    available: bool
    reason: str
    detail: str


#: Cached verdict + when it was taken. Guarded because tool calls arrive from
#: the event loop, run workers, and (in the dashboard) a thread pool.
_STATUS_LOCK = threading.Lock()
_STATUS: ApiServerStatus | None = None
_STATUS_AT: float = 0.0
_REFRESHING = False


INERT_DETAIL = "the Hermes api server lane is switched off in this process"
CHECKING_DETAIL = "I haven't finished checking whether the Hermes api server is up"


def _lane_enabled() -> bool:
    """Inert under pytest unless a test explicitly opts in.

    Same guard, and the same reason, as ``talk_runs._history_enabled``: suites
    exercise the host chain transitively without ever meaning to touch this
    lane, and an always-on probe would have every one of them dial whatever is
    listening on port 8642 of the machine running CI. Tests that DO mean to
    exercise the lane monkeypatch this to ``lambda: True``.
    """

    return "PYTEST_CURRENT_TEST" not in os.environ


def _auth_headers(session_key: str | None = None) -> dict:
    """Bearer credential, plus the optional memory-scoping key.

    ``session_key`` is threaded only by the run-SUBMISSION call site: the
    read and control routes (``probe``, ``get_run``, ``stop_run``) address a
    run that already exists and have nothing to scope. Defaulting it to
    ``None`` keeps all four of those call sites byte-identical to before.
    """

    headers: dict = {}
    key = talk_config.api_server_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if session_key:
        headers["X-Hermes-Session-Key"] = session_key
    return headers


def probe() -> ApiServerStatus:
    """Ask the api_server whether it is there and whether we may use it.

    ``/v1/capabilities`` is authenticated (api_server.py:1455), which is the
    point: a 401 means the server is UP and our key is wrong, and that is a
    different sentence than "not running" — an operator who hears "not
    reachable" when the real problem is a mismatched key goes looking in the
    wrong place.
    """

    url = talk_config.api_server_url() + CAPABILITIES_PATH
    try:
        response = httpx.get(
            url,
            headers=_auth_headers(),
            timeout=talk_config.api_server_probe_timeout_s(),
        )
    except httpx.HTTPError as exc:
        return ApiServerStatus(
            available=False,
            reason=REASON_ABSENT,
            detail=(
                "the Hermes api server isn't reachable "
                f"({type(exc).__name__}) — set a strong API_SERVER_KEY "
                "(16+ chars) in the gateway environment and restart it; the "
                "platform enables itself when a usable key exists"
            ),
        )
    if response.status_code == 401:
        return ApiServerStatus(
            available=False,
            reason=REASON_UNAUTHORIZED,
            detail=(
                "the Hermes api server is running but rejected my key — set "
                "TALK_API_SERVER_KEY (or API_SERVER_KEY) to match the gateway"
            ),
        )
    if response.status_code != 200:
        return ApiServerStatus(
            available=False,
            reason=REASON_ERROR,
            detail=(
                f"the Hermes api server answered {response.status_code} when I "
                "asked what it supports"
            ),
        )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    features = payload.get("features") if isinstance(payload, dict) else None
    if isinstance(features, dict) and features.get("run_submission") is False:
        return ApiServerStatus(
            available=False,
            reason=REASON_ERROR,
            detail="the Hermes api server is running but does not accept runs",
        )
    return ApiServerStatus(available=True, reason=REASON_OK, detail="Hermes api server")


def _refresh_in_background() -> None:
    """Re-probe off the hot path so a warm caller never waits on the network."""

    global _REFRESHING
    with _STATUS_LOCK:
        if _REFRESHING:
            return
        _REFRESHING = True

    def worker() -> None:
        global _REFRESHING
        try:
            verdict = probe()
        except Exception as exc:  # noqa: BLE001 — a probe must never escape
            _log.warning("api server probe failed: %s: %s", type(exc).__name__, exc)
            verdict = ApiServerStatus(
                available=False,
                reason=REASON_ERROR,
                detail=f"I couldn't check the Hermes api server ({type(exc).__name__})",
            )
        finally:
            with _STATUS_LOCK:
                _REFRESHING = False
        # Stored even on failure: a cold cache that never fills would re-probe
        # on every single turn, which is the stampede this cache exists to stop.
        _store(verdict)

    threading.Thread(target=worker, name="talk-apiserver-probe", daemon=True).start()


def _store(verdict: ApiServerStatus) -> None:
    global _STATUS, _STATUS_AT
    with _STATUS_LOCK:
        _STATUS = verdict
        _STATUS_AT = time.monotonic()


def warm() -> ApiServerStatus:
    """Probe now, cache the verdict, and return it. **BLOCKS.**

    The one function here that waits for the network on the caller's thread.
    Call it where waiting is free — a session start, a dashboard status route
    already on a worker thread — so that :func:`status` is warm before the
    first spoken question arrives. Never call it from a tool handler.
    """

    if not _lane_enabled():
        return ApiServerStatus(available=False, reason=REASON_ABSENT, detail=INERT_DETAIL)
    try:
        verdict = probe()
    except Exception as exc:  # noqa: BLE001 — availability is never fatal
        _log.warning("api server probe failed: %s: %s", type(exc).__name__, exc)
        verdict = ApiServerStatus(
            available=False,
            reason=REASON_ERROR,
            detail=f"I couldn't check the Hermes api server ({type(exc).__name__})",
        )
    _store(verdict)
    return verdict


def warm_in_background() -> None:
    """Kick :func:`warm` off the caller's thread. Fire and forget."""

    _refresh_in_background()


def status() -> ApiServerStatus:
    """The current verdict. NEVER waits for the network.

    A cold cache schedules a probe and answers "still checking" — unavailable,
    because a lane we cannot vouch for must not be claimed. A stale cache
    answers with the last verdict and refreshes behind it. Either way the
    caller returns in microseconds, which is the only thing that matters on a
    thread that is also carrying audio.
    """

    if not _lane_enabled():
        return ApiServerStatus(available=False, reason=REASON_ABSENT, detail=INERT_DETAIL)
    with _STATUS_LOCK:
        cached, taken_at = _STATUS, _STATUS_AT
    if cached is None:
        _refresh_in_background()
        return ApiServerStatus(
            available=False, reason=REASON_ABSENT, detail=CHECKING_DETAIL
        )
    if time.monotonic() - taken_at >= talk_config.api_server_probe_ttl_s():
        _refresh_in_background()
    return cached


def is_available() -> bool:
    """True when a run submitted right now would be accepted."""

    return status().available


def start_run(
    prompt: str, *, session_id: str | None = None, session_key: str | None = None
) -> str:
    """POST /v1/runs. Returns the run id; raises on anything else.

    ``session_id`` names an EXISTING remote conversation to continue;
    ``session_key`` is the operator's stable scope for the memory a run may
    read and write, and survives the ``/clear`` that ends a session_id.
    """

    body: dict = {"input": prompt}
    if session_id:
        body["session_id"] = session_id
    try:
        response = httpx.post(
            talk_config.api_server_url() + RUNS_PATH,
            json=body,
            headers=_auth_headers(session_key),
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I couldn't reach the Hermes api server ({type(exc).__name__})"
        ) from exc
    # 202 is the documented success (api_server.py:4464-4468); accept any 2xx
    # so a future 200 does not read as a failure.
    if response.status_code // 100 != 2:
        raise TalkApiServerError(
            f"the Hermes api server refused the run ({response.status_code}): "
            f"{response.text[:200] or 'no detail'}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TalkApiServerError("the Hermes api server returned a non-JSON run") from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise TalkApiServerError("the Hermes api server returned a run with no id")
    return run_id


def get_run(run_id: str) -> dict:
    """GET /v1/runs/{id}. Returns the status object; raises on anything else."""

    try:
        response = httpx.get(
            f"{talk_config.api_server_url()}{RUNS_PATH}/{run_id}",
            headers=_auth_headers(),
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I lost contact with the Hermes api server ({type(exc).__name__})"
        ) from exc
    if response.status_code != 200:
        raise TalkApiServerError(
            f"the Hermes api server answered {response.status_code} for that run"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TalkApiServerError("the Hermes api server returned a non-JSON status") from exc
    if not isinstance(payload, dict):
        raise TalkApiServerError("the Hermes api server returned an invalid status")
    return payload


def _get_json(path: str, what: str) -> Any:
    """GET ``path`` and return its decoded JSON. Raises on anything else.

    Factored out because four catalog reads want byte-identical failure
    handling; ``what`` is the noun that lands in the spoken error, so an
    operator hears which read failed rather than a bare path.
    """

    try:
        response = httpx.get(
            f"{talk_config.api_server_url()}{path}",
            headers=_auth_headers(),
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I lost contact with the Hermes api server ({type(exc).__name__})"
        ) from exc
    if response.status_code != 200:
        raise TalkApiServerError(
            f"the Hermes api server answered {response.status_code} when I asked "
            f"about {what}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise TalkApiServerError(
            f"the Hermes api server returned a non-JSON {what} response"
        ) from exc


def _listing(path: str, key: str, what: str) -> list[dict]:
    """One catalog listing, accepting a bare list or a ``{key: [...]}`` envelope.

    Both shapes are parsed because this repo has no Hermes gateway source to
    pin the envelope against — only the documented route. Guessing one shape
    and raising on the other would turn a cosmetic difference into a dead
    feature; anything that is NEITHER shape still raises, because an
    unrecognized payload must not read as an empty-but-successful catalog.
    """

    payload = _get_json(path, what)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get(key), list):
        entries = payload[key]
    else:
        raise TalkApiServerError(
            f"the Hermes api server returned an unrecognized {what} payload"
        )
    return [entry for entry in entries if isinstance(entry, dict)]


def list_skills() -> list[dict]:
    """GET /v1/skills — the skills this Hermes install actually has."""

    return _listing(SKILLS_PATH, "skills", "skills")


def list_toolsets() -> list[dict]:
    """GET /v1/toolsets — resolved toolsets, each with its own enabled/configured state."""

    return _listing(TOOLSETS_PATH, "toolsets", "toolsets")


def capabilities_payload() -> dict:
    """GET /v1/capabilities as the RAW feature document.

    Same path :func:`probe` dials, deliberately kept as a separate function:
    ``probe`` collapses the whole document into one yes/no the rest of the
    codebase already depends on, and widening its return type would ripple
    into ``talk_host.agent_lane`` and ``talk_status``. This one answers the
    other question — what the flags actually say.
    """

    payload = _get_json(CAPABILITIES_PATH, "capabilities")
    if not isinstance(payload, dict):
        raise TalkApiServerError("the Hermes api server returned invalid capabilities")
    return payload


def health_detailed() -> dict:
    """GET /health/detailed — live run/delegation counters."""

    payload = _get_json(HEALTH_DETAILED_PATH, "health")
    if not isinstance(payload, dict):
        raise TalkApiServerError("the Hermes api server returned invalid health")
    return payload


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What ONE api_server run actually came to (hermes-sip-live-voice#49).

    ``outcome`` is the category the caller may speak from; ``remote_status``
    is the host's own last word, kept verbatim so a status this Talk does not
    understand is visible rather than laundered into a category. ``output``
    is the text the host returned (clipped), present for success AND for a
    partial result on a failed/cancelled run. ``error`` is the host's reason
    when it gave one. ``children_started``/``children_finished`` count the
    ``subagent.start``/``subagent.complete`` events seen on the run's own
    stream; when the first exceeds the second at the host's ``completed``,
    the host turn ended with delegated work still running and the outcome is
    :data:`OUTCOME_INCOMPLETE`, not success (hermes-sip-live-voice#50).
    """

    outcome: str
    remote_status: str
    output: str
    error: str
    api_run_id: str
    session_id: str
    children_started: int = 0
    children_finished: int = 0

    @property
    def succeeded(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS

    @property
    def children_outstanding(self) -> int:
        return max(0, self.children_started - self.children_finished)

    def speakable(self) -> str:
        """The one sentence a worker may return when it is asked for text."""

        if self.outcome == OUTCOME_SUCCESS:
            return self.output or "the agent finished without saying anything"
        if self.outcome == OUTCOME_INCOMPLETE:
            head = self.output or "the agent's turn ended"
            why = self.error or (
                f"{self.children_outstanding} delegated task(s) are still running; "
                "the final result is not in yet"
            )
            return f"{head} — {why}"
        detail = self.error or "no reason given"
        if self.outcome == OUTCOME_TIMEOUT:
            return (
                "the agent run is still going after its whole time budget — "
                "it may still finish, but I stopped waiting"
            )
        if self.outcome == OUTCOME_UNKNOWN:
            return (
                f"the agent run ended in a state I don't recognize ({self.remote_status}): {detail}"
            )
        return f"the agent run {self.remote_status}: {detail}"


def _outcome_for(remote_status: str) -> str:
    return _REMOTE_TO_OUTCOME.get(remote_status, OUTCOME_UNKNOWN)


def _clip(text: Any) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    return s.strip()[:MAX_OUTPUT_CHARS]


def run_to_outcome(
    prompt: str,
    *,
    session_id: str | None = None,
    session_key: str | None = None,
    on_start=None,
    on_event=None,
    child_counter=None,
) -> RunOutcome:
    """Run one agent turn and return a typed :class:`RunOutcome`.

    BLOCKS until the run terminates or the budget expires, so this belongs on
    a :mod:`talk_runs` worker thread and nowhere else. The deadline is the
    same ``TALK_AGENT_TIMEOUT_S`` that bounds a detached spawn — one budget
    for "how long may work run", whichever lane carries it.

    ``on_event`` is the progress tap (hermes-talk#33): it is handed each
    poll's raw status payload — which carries the host-stamped ``last_event``
    — so the caller can project bounded phases onto its own bookkeeping. Like
    ``on_start``, it is pure telemetry: suppressed on error, never consulted
    for the answer, and never evidence the run finished. The payload arrives
    BEFORE the terminal branch, so the terminal ``last_event`` is seen too.

    ``child_counter`` is an optional zero-arg callable returning
    ``(started, finished)`` subagent counts observed on the run's own event
    stream (the approval sidecar reads it). It is consulted once, at the
    host's terminal status, to tell a genuinely finished turn from one that
    merely returned while its children were still working.

    Never raises for a run the host reported on; a run this Talk stopped
    waiting for is :data:`OUTCOME_TIMEOUT`. Transport failures still raise
    :class:`TalkApiServerError`.
    """

    run_id = start_run(prompt, session_id=session_id, session_key=session_key)
    if on_start is not None:
        with contextlib.suppress(Exception):  # a bookkeeping hook is never fatal
            on_start(run_id)
    poll = talk_config.api_server_poll_s()
    deadline = time.monotonic() + talk_config.agent_timeout_s()
    run: dict = {}
    while time.monotonic() < deadline:
        time.sleep(poll)
        run = get_run(run_id)
        if on_event is not None:
            try:
                on_event(run)
            except Exception:  # noqa: BLE001 — telemetry, never the run's fate
                _log.debug("on_event progress tap failed", exc_info=True)
        state = str(run.get("status") or "")
        if state in LIVE_RUN_STATUSES:
            continue
        if state not in TERMINAL_RUN_STATUSES:
            _log.warning(
                "api run %s reported an unknown status %r; not waiting further", run_id, state
            )
        started = finished = 0
        if child_counter is not None:
            with contextlib.suppress(Exception):
                started, finished = child_counter()
        outcome = _outcome_for(state)
        if outcome == OUTCOME_SUCCESS and started > finished:
            outcome = OUTCOME_INCOMPLETE
        return RunOutcome(
            outcome=outcome,
            remote_status=state,
            output=_clip(run.get("output")),
            error=_clip(run.get("error")),
            api_run_id=str(run_id),
            session_id=str(run.get("session_id") or session_id or ""),
            children_started=started,
            children_finished=finished,
        )
    return RunOutcome(
        outcome=OUTCOME_TIMEOUT,
        remote_status=str(run.get("status") or "running"),
        output=_clip(run.get("output")),
        error="time budget exhausted while the run was still live",
        api_run_id=str(run_id),
        session_id=str(run.get("session_id") or session_id or ""),
    )


def run_to_completion(
    prompt: str,
    *,
    session_id: str | None = None,
    session_key: str | None = None,
    on_start=None,
    on_event=None,
) -> str:
    """String-returning wrapper over :func:`run_to_outcome`.

    Kept for callers that only need speakable text (the transcript memory
    flush). Its documented failure shapes are unchanged: failed/cancelled
    runs return text starting ``"the agent run "``; a timeout raises
    :class:`TalkApiServerError`. New code should call :func:`run_to_outcome`
    and branch on ``outcome`` instead of parsing prose.
    """

    result = run_to_outcome(
        prompt, session_id=session_id, session_key=session_key, on_start=on_start, on_event=on_event
    )
    if result.outcome == OUTCOME_TIMEOUT:
        raise TalkApiServerError(result.speakable())
    return result.speakable()[:MAX_OUTPUT_CHARS]


_ORIGINAL_RUN_TO_COMPLETION = run_to_completion
_LEGACY_TEXT_PREFIX = "the agent run "


def outcome_from_text(text: str, *, api_run_id: str = "", session_id: str = "") -> RunOutcome:
    """Classify a legacy string from :func:`run_to_completion` into a RunOutcome.

    The string contract had exactly one failure shape, ``"the agent run
    <status>: <detail>"``; anything else was the answer. This is the ONLY
    place that prose is parsed, and it exists so a caller (or a test double)
    that still speaks the string contract cannot make a cancelled or failed
    run read as success.
    """

    text = (text or "").strip()
    if text.startswith(_LEGACY_TEXT_PREFIX):
        rest = text[len(_LEGACY_TEXT_PREFIX) :]
        status, _, detail = rest.partition(":")
        status = status.strip()
        outcome = _outcome_for(status) if status in TERMINAL_RUN_STATUSES else OUTCOME_UNKNOWN
        if outcome == OUTCOME_SUCCESS:
            outcome = OUTCOME_UNKNOWN  # "the agent run completed: ..." was never emitted
        return RunOutcome(
            outcome=outcome,
            remote_status=status or "unknown",
            output="",
            error=detail.strip() or "no reason given",
            api_run_id=api_run_id,
            session_id=session_id,
        )
    return RunOutcome(
        outcome=OUTCOME_SUCCESS,
        remote_status="completed",
        output=text[:MAX_OUTPUT_CHARS],
        error="",
        api_run_id=api_run_id,
        session_id=session_id,
    )


def run_for_worker(
    prompt: str,
    *,
    session_id: str | None = None,
    session_key: str | None = None,
    on_start=None,
    on_event=None,
    child_counter=None,
) -> RunOutcome:
    """The worker's entry point: typed in production, tolerant of a swapped string seam.

    If something replaced :func:`run_to_completion` on this module (the
    suite's fakes do; a downstream fork might), that replacement is the
    lane's authority and its string is classified through
    :func:`outcome_from_text`. Otherwise the typed poll runs directly.
    """

    if run_to_completion is not _ORIGINAL_RUN_TO_COMPLETION:
        remote: dict[str, str] = {}

        def _capture(api_run_id: str) -> None:
            remote["id"] = str(api_run_id)
            if on_start is not None:
                on_start(api_run_id)

        text = run_to_completion(
            prompt,
            session_id=session_id,
            session_key=session_key,
            on_start=_capture,
            on_event=on_event,
        )
        started = finished = 0
        if child_counter is not None:
            with contextlib.suppress(Exception):
                started, finished = child_counter()
        result = outcome_from_text(
            text, api_run_id=remote.get("id", ""), session_id=session_id or ""
        )
        if result.succeeded and started > finished:
            result = RunOutcome(
                outcome=OUTCOME_INCOMPLETE,
                remote_status=result.remote_status,
                output=result.output,
                error="",
                api_run_id=result.api_run_id,
                session_id=result.session_id,
                children_started=started,
                children_finished=finished,
            )
        return result
    return run_to_outcome(
        prompt,
        session_id=session_id,
        session_key=session_key,
        on_start=on_start,
        on_event=on_event,
        child_counter=child_counter,
    )


SESSIONS_PATH = "/api/sessions"
#: The host's row kind for a detached child result parked on an api_server
#: session between client turns (gateway/wake.py ``persist_delegation_delivery``).
DELEGATION_DELIVERY_KIND = "async_delegation_complete"


def session_messages(session_id: str, *, limit: int = 50) -> list[dict]:
    """``GET /api/sessions/{id}/messages`` — the newest ``limit`` rows, oldest first.

    Read-only. Raises :class:`TalkApiServerError` on any failure. Used to
    see whether the host has parked a delegated child's result on the
    session after the parent run returned (hermes-sip-live-voice#50); the
    row's ``display_kind`` is :data:`DELEGATION_DELIVERY_KIND`.
    """

    try:
        response = httpx.get(
            f"{talk_config.api_server_url()}{SESSIONS_PATH}/{session_id}/messages",
            params={"limit": str(limit), "order": "latest"},
            headers=_auth_headers(),
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I lost contact with the Hermes api server ({type(exc).__name__})"
        ) from exc
    if response.status_code != 200:
        raise TalkApiServerError(
            f"the Hermes api server refused the session read ({response.status_code})"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TalkApiServerError("the Hermes api server returned non-JSON messages") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []


def pending_child_deliveries(session_id: str, *, after_message_id: int | None = None) -> list[dict]:
    """Delivery rows parked on ``session_id`` newer than ``after_message_id``.

    Each row is a child result the host has stored for the NEXT run on this
    session to read; none has been synthesized into an answer yet. Empty list
    when nothing is parked. Raises on transport failure.
    """

    rows = []
    for m in session_messages(session_id):
        if m.get("display_kind") != DELEGATION_DELIVERY_KIND:
            continue
        mid = m.get("id")
        if after_message_id is not None and isinstance(mid, int) and mid <= after_message_id:
            continue
        rows.append(m)
    return rows


def latest_message_id(session_id: str) -> int | None:
    rows = session_messages(session_id, limit=1)
    mid = rows[-1].get("id") if rows else None
    return mid if isinstance(mid, int) else None


def stop_run(run_id: str) -> None:
    """POST /v1/runs/{id}/stop — hard-interrupt a running api_server agent.

    The lane's ONE lifecycle verb (api_server.py routes: create/get/events/
    approval/stop). Raises :class:`TalkApiServerError` with speakable text on
    any failure; returns None on any 2xx.
    """

    try:
        response = httpx.post(
            f"{talk_config.api_server_url()}{RUNS_PATH}/{run_id}/stop",
            headers=_auth_headers(),
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I couldn't reach the Hermes api server ({type(exc).__name__})"
        ) from exc
    if response.status_code // 100 != 2:
        raise TalkApiServerError(
            f"the Hermes api server refused the stop ({response.status_code})"
        )


class SteerRefused(TalkApiServerError):
    """The host answered 4xx: the run is not accepting steer input right now."""

    def __init__(self, message: str, *, code: str | None = None, status: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def steer_run(run_id: str, text: str) -> dict:
    """POST /v1/runs/{id}/steer — queue guidance into a RUNNING api_server agent.

    The host's contract (api_server_runs._handle_steer_run): 200 with
    ``accepted: true`` means the text entered the agent's steer QUEUE — it
    is read at the agent's next step, never applied on the spot; 409
    (``run_not_accepting_steer`` / ``steer_not_accepted``) means the run is
    not running or refused the text. Raises :class:`SteerRefused` for the
    4xx family and :class:`TalkApiServerError` for transport/5xx.
    """

    try:
        response = httpx.post(
            f"{talk_config.api_server_url()}{RUNS_PATH}/{run_id}/steer",
            headers=_auth_headers(),
            json={"input": text},
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I couldn't reach the Hermes api server ({type(exc).__name__})"
        ) from exc
    if response.status_code // 100 == 4:
        code = None
        with contextlib.suppress(Exception):
            body = response.json()
            error = body.get("error") if isinstance(body, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
        raise SteerRefused(
            f"the Hermes api server refused the steer ({response.status_code})",
            code=code,
            status=response.status_code,
        )
    if response.status_code // 100 != 2:
        raise TalkApiServerError(
            f"the Hermes api server failed the steer ({response.status_code})"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TalkApiServerError(
            "the Hermes api server returned an unreadable steer receipt"
        ) from exc
    if not isinstance(payload, dict) or payload.get("accepted") is not True:
        raise TalkApiServerError("the Hermes api server did not confirm the steer")
    return payload


def steering_supported() -> bool:
    """Whether the host advertises ``run_steer`` in /v1/capabilities. False on any doubt."""

    try:
        features = capabilities_payload().get("features")
    except Exception:  # noqa: BLE001 — unknown reads as unsupported
        return False
    return isinstance(features, dict) and features.get("run_steer") is True


class ApprovalGoneError(TalkApiServerError):
    """The host answered 409: no pending approval remains to resolve.

    Distinct from a transport failure on purpose: the caller CLEARS its
    pending record on this verdict but keeps it (retryable) on a network
    failure.
    """


#: Read-idle bound for the run-events SSE stream. The host emits a keepalive
#: comment every 30s (api_server.py ``_handle_run_events``), so a healthy
#: stream never idles this long; a dead one breaks here instead of hanging a
#: daemon thread forever.
SSE_READ_IDLE_S = 45.0


def stream_run_events(run_id: str, on_event, *, idle_timeout_s: float = SSE_READ_IDLE_S) -> None:
    """Read ``GET /v1/runs/{id}/events`` (SSE) and hand each event to ``on_event``.

    BLOCKS for the life of the run — this belongs on a dedicated daemon
    thread, never the voice loop. The host's frames are bare
    ``data: {json}`` lines (the event name rides INSIDE the payload's
    ``event`` key) plus ``:`` keepalive comments. Returns when the server
    closes the stream (run terminal) or the read idles out. Never raises: a
    dead stream degrades to the host's own approval timeout, and the poll
    loop in :func:`run_to_completion` remains the run's completion
    authority — one reconnect attempt is deliberately NOT made, because a
    re-subscribe re-reads nothing the run already finished saying.

    Inert under pytest unless a test opts in — the same guard and the same
    reason as :func:`status`: run workers spawn this sidecar transitively,
    and a suite faking the rest of the lane must never dial port 8642 for
    real.
    """

    if not _lane_enabled():
        return
    url = f"{talk_config.api_server_url()}{RUNS_PATH}/{run_id}/events"
    timeout = httpx.Timeout(
        talk_config.api_server_probe_timeout_s() * 4,
        read=idle_timeout_s,
    )
    try:
        with httpx.stream("GET", url, headers=_auth_headers(), timeout=timeout) as response:
            if response.status_code != 200:
                _log.debug(
                    "run events stream for %s refused (%s)", run_id, response.status_code
                )
                return
            data_lines: list[str] = []
            for line in response.iter_lines():
                if line:
                    if line.startswith(":"):
                        continue  # keepalive / stream-closed comment
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    continue
                if not data_lines:
                    continue
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue  # a torn frame costs itself, never the stream
                if not isinstance(event, dict):
                    continue
                try:
                    on_event(event)
                except Exception:  # noqa: BLE001 — a consumer must never kill the stream
                    _log.debug("run event consumer failed for %s", run_id, exc_info=True)
    except httpx.HTTPError as exc:
        _log.debug(
            "run events stream ended for %s: %s: %s", run_id, type(exc).__name__, exc
        )


def respond_to_approval(
    run_id: str, choice: str, *, approval_id: str | None = None
) -> dict:
    """POST ``/v1/runs/{id}/approval`` — resolve the run's pending approval.

    Returns the decoded payload on 2xx (the host's receipt carries
    ``resolved``, the count it unblocked). Raises :class:`ApprovalGoneError`
    on the host's 409s (no active approval session / nothing pending) and
    :class:`TalkApiServerError` with speakable text on any other failure.

    ``approval_id`` is the request's own id from the ``approval.request``
    event: a host that supports exact routing (the field is ``request_id``
    on the wire) resolves THAT request instead of FIFO-popping the oldest;
    hosts that predate the field ignore it.

    The choice is NOT re-validated here — the narrowing of what voice may
    grant lives in :mod:`talk_approvals`, the one choke point every caller
    goes through.
    """

    body: dict = {"choice": choice}
    if approval_id:
        body["request_id"] = approval_id
    try:
        response = httpx.post(
            f"{talk_config.api_server_url()}{RUNS_PATH}/{run_id}/approval",
            json=body,
            headers=_auth_headers(),
            timeout=talk_config.api_server_probe_timeout_s() * 4,
        )
    except httpx.HTTPError as exc:
        raise TalkApiServerError(
            f"I couldn't reach the Hermes api server ({type(exc).__name__})"
        ) from exc
    if response.status_code == 409:
        raise ApprovalGoneError(
            "the host says that approval was already answered or expired"
        )
    if response.status_code // 100 != 2:
        raise TalkApiServerError(
            f"the Hermes api server refused the approval answer ({response.status_code})"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TalkApiServerError(
            "the Hermes api server returned a non-JSON approval receipt"
        ) from exc
    return payload if isinstance(payload, dict) else {}


def reset_for_tests() -> None:
    """Clear the cached verdict between tests (never called in production)."""

    global _STATUS, _STATUS_AT, _REFRESHING
    with _STATUS_LOCK:
        _STATUS = None
        _STATUS_AT = 0.0
        _REFRESHING = False


__all__ = [
    "CAPABILITIES_PATH",
    "CHECKING_DETAIL",
    "DELEGATION_DELIVERY_KIND",
    "HEALTH_DETAILED_PATH",
    "INERT_DETAIL",
    "LIVE_RUN_STATUSES",
    "MAX_OUTPUT_CHARS",
    "OUTCOME_CANCELLED",
    "OUTCOME_FAILED",
    "OUTCOME_INCOMPLETE",
    "OUTCOME_INTERRUPTED",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIMEOUT",
    "OUTCOME_UNKNOWN",
    "REASON_ABSENT",
    "REASON_ERROR",
    "REASON_OK",
    "REASON_UNAUTHORIZED",
    "RUNS_PATH",
    "SESSIONS_PATH",
    "SKILLS_PATH",
    "SSE_READ_IDLE_S",
    "TERMINAL_RUN_STATUSES",
    "TOOLSETS_PATH",
    "ApiServerStatus",
    "ApprovalGoneError",
    "RunOutcome",
    "TalkApiServerError",
    "capabilities_payload",
    "get_run",
    "health_detailed",
    "is_available",
    "latest_message_id",
    "list_skills",
    "list_toolsets",
    "outcome_from_text",
    "pending_child_deliveries",
    "probe",
    "reset_for_tests",
    "respond_to_approval",
    "run_for_worker",
    "run_to_completion",
    "run_to_outcome",
    "session_messages",
    "start_run",
    "status",
    "stop_run",
    "stream_run_events",
    "warm",
    "warm_in_background",
]
