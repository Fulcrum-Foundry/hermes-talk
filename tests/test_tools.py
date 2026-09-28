"""Tool surface — the speakable-error contract and off-host degradation."""

from __future__ import annotations

import json
import threading
import time

import pytest

import talk_capabilities
import talk_host
import talk_operator_auth
import talk_runs
import talk_tools
import talk_vault


def _wait_terminal(run_id: int, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = talk_runs.get_run(run_id)
        if run and run["status"] in talk_runs.TERMINAL_STATUSES:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished")


@pytest.fixture(autouse=True)
def unbound_ctx(monkeypatch):
    """Every test starts detached from Hermes unless it says otherwise.

    ``hermes_binary`` is neutralized too: this box has a real ``hermes`` on
    PATH, and a tool test must never spawn one. Backend-chain coverage lives
    in test_delegation.py, where the subprocess is replaced explicitly.
    """

    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()
    # Runs are refused without a bound return route (hermes-talk#35), so the
    # suite attaches one. Tests that assert the REFUSAL detach it explicitly.
    talk_runs.attach_owner(
        talk_session_id="ts-test",
        generation_id="gen-test",
        hermes_session_id="sess-test",
        operator="test",
        profile=None,
    )
    monkeypatch.setattr(talk_host, "hermes_binary", lambda: None)
    # Vault availability is a property of the BOX, so leaving it unpinned
    # would make the advertised tool list depend on whether the machine
    # running the suite happens to have a memory provider installed.
    monkeypatch.setattr(talk_vault, "available", lambda: False)
    talk_vault.reset()
    yield
    talk_vault.reset()
    talk_host.bind_ctx(None)
    talk_runs.reset_for_tests()


_BASE_TOOLS = [
    "search_memory",
    "delegate_task",
    "check_work",
    "list_agents",
    "steer_agent",
    "redirect_agent",
    "stop_work",
    "resolve_approval",
    "talk_status",
    "talk_capabilities",
    "hold",
    "resume",
    "cancel_job",
    "set_verbosity",
    "defer_updates",
    "deliver_when_done",
    "get_result",
]


def test_default_tools_are_fresh_copies():
    first = talk_tools.default_talk_tools()
    first[0]["name"] = "mutated"
    assert [tool["name"] for tool in talk_tools.default_talk_tools()] == _BASE_TOOLS


def test_the_pause_tool_is_advertised_only_when_the_session_asks_for_it():
    """pause_voice_input (hermes-talk#100) rides on ``pausable``, which the
    session sets only when the operator has a guaranteed way to resume; the
    default is the safe direction — no pause a key or command cannot undo."""

    assert [tool["name"] for tool in talk_tools.default_talk_tools(pausable=True)] == [
        *_BASE_TOOLS,
        "pause_voice_input",
    ]
    assert [tool["name"] for tool in talk_tools.default_talk_tools(pausable=False)] == _BASE_TOOLS


def test_the_vault_tool_is_advertised_only_when_it_can_be_served(monkeypatch):
    """Advertising a lookup that cannot run is the same defect as the
    provider block this plugin stopped passing through — the model calls it,
    the relay says the tool does not exist, and the call stalls on nothing."""

    monkeypatch.setattr(talk_vault, "available", lambda: True)
    names = [tool["name"] for tool in talk_tools.default_talk_tools()]

    assert names == [*_BASE_TOOLS[:1], "search_vault", *_BASE_TOOLS[1:]]


def test_an_erroring_availability_check_costs_only_the_vault_tool(monkeypatch):
    def boom():
        raise RuntimeError("provider import exploded")

    monkeypatch.setattr(talk_vault, "available", boom)

    # A session that starts without one tool beats a session that never
    # starts, so this degrades rather than raising into the mint path.
    assert [tool["name"] for tool in talk_tools.default_talk_tools()] == _BASE_TOOLS


def test_every_advertised_tool_has_a_handler(monkeypatch):
    """Checked with the CONDITIONAL tool advertised too. With vault
    availability pinned off (the fixture default) this test would never see
    search_vault — and a tool advertised with no handler raises TalkToolError
    at the relay, which the model hears as "that tool isn't available" in the
    middle of a live call."""

    monkeypatch.setattr(talk_vault, "available", lambda: True)
    names = {tool["name"] for tool in talk_tools.default_talk_tools(pausable=True)}
    assert "search_vault" in names
    assert "pause_voice_input" in names

    for tool in talk_tools.default_talk_tools(pausable=True):
        assert tool["name"] in talk_tools._HANDLERS
        assert tool["type"] == "function"
        assert tool["parameters"]["type"] == "object"


def test_no_handler_is_orphaned():
    """The other direction: a handler with no schema is dead code the model
    can never reach."""

    advertised = {tool["name"] for tool in talk_tools.default_talk_tools()}
    advertised.add("search_vault")  # conditional, absent when unservable
    advertised.add("pause_voice_input")  # conditional, absent without a way to resume

    assert set(talk_tools._HANDLERS) == advertised


def test_every_tool_is_explicitly_classified_read_only_or_mutating():
    read_only = talk_operator_auth.READ_ONLY_TALK_TOOLS
    mutating = talk_operator_auth.MUTATING_TALK_TOOLS

    assert read_only.isdisjoint(mutating)
    assert set(talk_tools._HANDLERS) == read_only | mutating


def test_unknown_tool_raises():
    # A name the model was never given is a client bug, not a call failure —
    # it is the one case that escapes as an exception.
    with pytest.raises(talk_tools.TalkToolError, match="launch_missiles"):
        talk_tools.execute_talk_tool("launch_missiles", {})


def test_handler_failure_returns_speakable_text(monkeypatch):
    def boom(_arguments):
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(talk_tools._HANDLERS, "talk_status", boom)

    result = talk_tools.execute_talk_tool("talk_status", {})

    assert result.startswith("talk_status failed: RuntimeError: disk on fire")


def test_output_is_bounded(monkeypatch):
    monkeypatch.setitem(talk_tools._HANDLERS, "talk_status", lambda _a: "x" * 99_999)
    assert len(talk_tools.execute_talk_tool("talk_status", {})) == talk_tools.MAX_OUTPUT_CHARS


def test_empty_output_still_says_something(monkeypatch):
    monkeypatch.setitem(talk_tools._HANDLERS, "talk_status", lambda _a: "")
    assert talk_tools.execute_talk_tool("talk_status", {}) == "(no output)"


def test_talk_status_reports_state(monkeypatch):
    monkeypatch.delenv("TALK_VOICE", raising=False)
    monkeypatch.setenv("TALK_MODEL", "gpt-realtime-2.1")
    talk_tools.REGISTRATION_FAILURES.clear()

    status = json.loads(talk_tools.execute_talk_tool("talk_status", {}))

    assert status["model"] == "gpt-realtime-2.1"
    assert status["voice"] == "cedar"
    assert status["attached_to_hermes"] is False
    assert isinstance(status["audio_available"], bool)
    assert status["legacy_lane"] == "legacy-provider-executor"
    assert status["legacy_session"] == {
        "scope": "limited provider-owned session",
        "full_parity_command": "/talk core join",
    }
    assert status["transcript"] == {
        "current_call": "temporary local capture",
        "after_close": "handed off for durable-memory review",
        "archive": "not live searchable or user-facing",
        "core_persistence": "separate canonical session path",
    }
    assert status["core_realtime"]["contract"] == "api-v2-input-only"
    assert isinstance(status["core_realtime"]["contract_available"], bool)
    assert isinstance(status["core_realtime"]["provider_available"], bool)
    assert status["core_realtime"]["registration"] == "unsupported-optional"
    assert "registration_failures" not in status


def _snapshot(**overrides) -> talk_capabilities.CatalogSnapshot:
    fields = {
        "source": talk_capabilities.SOURCE_IN_PROCESS,
        "skills": ({"name": "web_search"},),
        "toolsets": (
            {"name": "browser", "enabled": True, "configured": True, "tools": ["open"]},
        ),
        "capabilities": {"run_approval": True},
        "health": {"active_runs": 1},
        "detail": "the Hermes agent I'm attached to",
    }
    fields.update(overrides)
    return talk_capabilities.CatalogSnapshot(**fields)


def test_talk_capabilities_reports_the_snapshot(monkeypatch):
    monkeypatch.setattr(talk_capabilities, "status", lambda: _snapshot())

    catalog = json.loads(talk_tools.execute_talk_tool("talk_capabilities", {}))

    assert catalog["source"] == talk_capabilities.SOURCE_IN_PROCESS
    assert catalog["skills"] == [{"name": "web_search"}]
    assert catalog["capabilities"] == {"run_approval": True}
    assert catalog["health"] == {"active_runs": 1}


def test_talk_capabilities_passes_disabled_toolsets_through(monkeypatch):
    """A disabled toolset is REPORTED, not filtered: the model has to be able
    to say "installed but not usable", and it cannot say what it never saw."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            toolsets=({"name": "email", "enabled": False, "configured": False},)
        ),
    )

    catalog = json.loads(talk_tools.execute_talk_tool("talk_capabilities", {}))

    assert catalog["toolsets"] == [
        {"name": "email", "enabled": False, "configured": False}
    ]


def test_talk_capabilities_redacts_secret_shaped_values(monkeypatch):
    """Upstream payloads are not this process's text, and this one is spoken
    aloud and lands in a transcript."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            toolsets=(
                {"name": "email", "config": {"token": "sk-abcdefgh12345678"}},
            ),
            capabilities={"webhook": "https://x.test/xoxb-abcdefgh12345678"},
        ),
    )

    rendered = talk_tools.execute_talk_tool("talk_capabilities", {})

    assert "sk-abcdefgh12345678" not in rendered
    assert "xoxb-abcdefgh12345678" not in rendered
    assert rendered.count("<redacted-secret>") == 2


def test_talk_capabilities_stays_parseable_when_the_catalog_is_huge(monkeypatch):
    """execute_talk_tool bounds by TAIL TRUNCATION, so an oversized payload
    would otherwise reach the model as JSON cut off mid-object."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            skills=tuple(
                {"name": f"skill_{index}", "description": "x" * 200}
                for index in range(200)
            ),
            toolsets=tuple(
                {"name": f"toolset_{index}", "enabled": True, "configured": False}
                for index in range(60)
            ),
        ),
    )

    rendered = talk_tools.execute_talk_tool("talk_capabilities", {})
    catalog = json.loads(rendered)  # the assertion that matters: still parses

    assert len(rendered) <= talk_tools.MAX_OUTPUT_CHARS
    assert catalog["skills"][0] == {"name": "skill_0"}
    assert len(catalog["skills"]) == talk_tools.MAX_CATALOG_ENTRIES
    assert catalog["skills_omitted"] == 200 - talk_tools.MAX_CATALOG_ENTRIES
    assert catalog["toolsets_omitted"] == 60 - talk_tools.MAX_CATALOG_ENTRIES
    # The flags that decide usability survive the compaction; the prose does not.
    assert catalog["toolsets"][0] == {
        "name": "toolset_0",
        "enabled": True,
        "configured": False,
    }


def test_a_disabled_skill_is_not_presented_as_available_after_compaction(monkeypatch):
    """Skills compact like toolsets: name PLUS usability flags. A disabled
    skill flattened to a bare name would read as plainly available, breaking
    "missing/disabled tools are not advertised"."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            skills=tuple(
                {
                    "name": f"skill_{index}",
                    "description": "x" * 200,
                    "enabled": index != 0,
                }
                for index in range(200)
            ),
        ),
    )

    catalog = json.loads(talk_tools.execute_talk_tool("talk_capabilities", {}))

    assert catalog["skills"][0] == {"name": "skill_0", "enabled": False}
    assert catalog["skills"][1] == {"name": "skill_1", "enabled": True}


def test_absurdly_long_names_still_yield_parseable_json_under_the_bound(monkeypatch):
    """Upstream-controlled names can blow past MAX_OUTPUT_CHARS even at the
    deepest compaction tier — the handler must degrade to a minimal summary
    rather than let execute_talk_tool's tail truncation tear the JSON."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            skills=tuple({"name": "s" * 5_000} for _ in range(50)),
            toolsets=tuple(
                {"name": "t" * 5_000, "enabled": True} for _ in range(10)
            ),
        ),
    )

    rendered = talk_tools.execute_talk_tool("talk_capabilities", {})
    catalog = json.loads(rendered)  # the assertion that matters: still parses

    assert len(rendered) <= talk_tools.MAX_OUTPUT_CHARS
    assert catalog["source"] == talk_capabilities.SOURCE_IN_PROCESS
    assert catalog["skills_count"] == 50
    assert catalog["toolsets_count"] == 10
    assert "too large" in catalog["detail"]


def test_catalog_entries_without_a_name_shaped_key_compact_to_unnamed(monkeypatch):
    """An upstream entry missing name/id/slug still renders, not KeyErrors."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            skills=tuple({"description": "x" * 200} for _ in range(60)),
        ),
    )

    catalog = json.loads(talk_tools.execute_talk_tool("talk_capabilities", {}))

    assert catalog["skills"][0] == {"name": "unnamed"}


def test_talk_capabilities_omits_capabilities_when_still_oversized_after_compaction(
    monkeypatch,
):
    """A huge `capabilities` document alone can push the payload back over
    budget even after skills/toolsets are compacted — must not silently
    reach execute_talk_tool's tail truncation and come out torn mid-object."""

    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: _snapshot(
            capabilities={
                "features": {f"flag_{index}": "x" * 200 for index in range(50)}
            },
        ),
    )

    rendered = talk_tools.execute_talk_tool("talk_capabilities", {})
    catalog = json.loads(rendered)  # the assertion that matters: still parses

    assert len(rendered) <= talk_tools.MAX_OUTPUT_CHARS
    assert catalog["capabilities"] == {}
    assert catalog["capabilities_omitted"] is True
    assert catalog["detail"].endswith("capabilities omitted")


def test_talk_capabilities_says_when_it_could_not_read_the_catalog(monkeypatch):
    monkeypatch.setattr(
        talk_capabilities,
        "status",
        lambda: talk_capabilities._empty(talk_capabilities.CHECKING_DETAIL),
    )

    catalog = json.loads(talk_tools.execute_talk_tool("talk_capabilities", {}))

    assert catalog["source"] == talk_capabilities.SOURCE_NONE
    assert catalog["detail"] == talk_capabilities.CHECKING_DETAIL
    assert catalog["skills"] == []


def test_talk_capabilities_is_read_only():
    """The authority boundary, stated at the tool rather than only in the
    generic classification sweep: a catalog read must never be a way to act."""

    assert "talk_capabilities" in talk_operator_auth.READ_ONLY_TALK_TOOLS
    assert "talk_capabilities" not in talk_operator_auth.MUTATING_TALK_TOOLS


def test_talk_status_surfaces_registration_failures():
    talk_tools.REGISTRATION_FAILURES.append("tts provider: ValueError: nope")
    try:
        status = json.loads(talk_tools.execute_talk_tool("talk_status", {}))
        assert status["registration_failures"] == ["tts provider: ValueError: nope"]
    finally:
        talk_tools.REGISTRATION_FAILURES.clear()


def test_talk_status_survives_an_unusable_voice(monkeypatch):
    monkeypatch.setenv("TALK_VOICE", "not-a-voice")
    status = json.loads(talk_tools.execute_talk_tool("talk_status", {}))
    assert status["voice"].startswith("unusable")


def test_search_memory_degrades_without_a_host():
    result = talk_tools.execute_talk_tool("search_memory", {"query": "the deploy"})
    assert "memory isn't available" in result
    assert "Traceback" not in result


def test_delegate_task_degrades_with_no_agent_loop_and_no_binary():
    result = talk_tools.execute_talk_tool("delegate_task", {"task": "ship it"})
    assert "can't hand off work" in result
    assert "WORK_STARTED" not in result


def test_search_memory_needs_a_query():
    assert "needs something to look for" in talk_tools.execute_talk_tool("search_memory", {})


def test_delegate_task_needs_a_task():
    assert "needs a task" in talk_tools.execute_talk_tool("delegate_task", {"task": "  "})


def test_check_work_on_an_empty_registry():
    assert "Nothing is running" in talk_tools.execute_talk_tool("check_work", {})


def test_check_work_lists_a_running_run():
    gate = threading.Event()
    run_id = talk_runs.start_run("agent", "audit the site", lambda _rid: gate.wait(3) or "ok")

    result = talk_tools.execute_talk_tool("check_work", {})

    assert f"run {run_id} (agent) running" in result
    gate.set()


def test_check_work_lists_a_finished_run():
    run_id = talk_runs.start_run("agent", "audit", lambda _rid: "the index is rebuilt")
    _wait_terminal(run_id)

    result = talk_tools.execute_talk_tool("check_work", {})
    assert f"run {run_id} (agent) finished" in result  # the outcome, not the lifecycle word
    assert f"check_work with run_id {run_id}" in result
    assert "the index is rebuilt" not in result


def test_search_memory_schema_says_ask_rather_than_guess_on_an_ambiguous_match():
    """The WORKING section carries this rule too, but only when a plugin
    context is bound. The schema ships on every lane, so this is the copy
    that reaches a standalone or dashboard session — the ones with no screen
    and no operator watching a guess go by."""

    schema = next(
        tool for tool in talk_tools.default_talk_tools() if tool["name"] == "search_memory"
    )

    assert "ask which one before acting on it" in schema["description"]
    assert "from remembered context" in schema["description"]


def test_check_work_schema_directs_specific_bounded_finished_output_retrieval():
    schema = next(tool for tool in talk_tools.default_talk_tools() if tool["name"] == "check_work")

    assert "finished run_id" in schema["description"]
    assert "output" in schema["parameters"]["properties"]["run_id"]["description"]


def test_check_work_by_id_speaks_the_output():
    run_id = talk_runs.start_run("agent", "audit", lambda _rid: "the index is rebuilt")
    _wait_terminal(run_id)

    result = talk_tools.execute_talk_tool("check_work", {"run_id": run_id})

    assert "the index is rebuilt" in result
    assert "audit" in result


def test_check_work_by_unknown_id():
    out = talk_tools.execute_talk_tool("check_work", {"run_id": 4242})
    assert "don't have work numbered" in out


def test_check_work_rejects_a_non_numeric_id():
    assert "needs a run number" in talk_tools.execute_talk_tool("check_work", {"run_id": "soon"})


# -- #71: hide delivered and prior-call work -------------------------------------


def test_check_work_groups_working_and_ready_not_yet_shared():
    gate = threading.Event()
    running = talk_runs.start_run("agent", "audit the site", lambda _rid: gate.wait(3) or "ok")
    finished = talk_runs.start_run("agent", "triage", lambda _rid: "all clear")
    _wait_terminal(finished)
    try:
        result = talk_tools.execute_talk_tool("check_work", {})
    finally:
        gate.set()
    assert result.startswith("Working: ")
    assert "Ready, not yet shared: " in result
    assert f"run {running} (agent) running" in result
    assert f"run {finished} (agent) finished" in result


def test_check_work_hides_a_result_already_delivered_to_the_caller():
    run_id = talk_runs.start_run("agent", "triage", lambda _rid: "all clear")
    _wait_terminal(run_id)
    assert talk_runs.claim_delivery(run_id, claimant="ts-test")
    assert talk_runs.mark_delivered(run_id, claimant="ts-test")

    result = talk_tools.execute_talk_tool("check_work", {})

    assert f"run {run_id}" not in result
    assert "waiting to be shared" in result
    # Asked for by number it is still readable — hiding is about the listing.
    assert "all clear" in talk_tools.execute_talk_tool("check_work", {"run_id": run_id})


def test_check_work_hides_a_result_claimed_by_an_earlier_call_but_lists_our_own_claim():
    older = talk_runs.start_run("agent", "outlook", lambda _rid: "sunny")
    ours = talk_runs.start_run("agent", "triage", lambda _rid: "all clear")
    _wait_terminal(older)
    _wait_terminal(ours)
    assert talk_runs.claim_delivery(older, claimant="ts-previous-call")
    assert talk_runs.claim_delivery(ours, claimant="ts-test")

    result = talk_tools.execute_talk_tool("check_work", {})

    assert f"run {older}" not in result, "an earlier call was speaking it; not ours to re-list"
    assert f"run {ours} (agent) finished" in result, "our own in-flight claim is still unshared"
    assert talk_runs.shared_with_caller(talk_runs.get_run(older))
    assert not talk_runs.shared_with_caller(talk_runs.get_run(ours))


def test_check_work_never_lists_a_run_replaced_by_a_widened_one():
    gate = threading.Event()
    original = talk_runs.start_run("agent", "audit", lambda _rid: gate.wait(3) or "x")
    talk_runs.annotate_run(original, replaced_by=original + 1)
    try:
        result = talk_tools.execute_talk_tool("check_work", {})
    finally:
        gate.set()
    assert f"run {original}" not in result


def test_check_work_description_says_talk_tracks_what_was_shared():
    schema = next(tool for tool in talk_tools.default_talk_tools() if tool["name"] == "check_work")
    text = schema["description"]
    assert "You track what has been shared" in text
    assert "never ask the caller which one to open" in text
    assert "volunteer unshared results at a pause" in text
    assert "'working'" in text and "'ready, not yet shared'" in text


# -- #72: steer by replace ----------------------------------------------------------


class _FakeProcess:
    """A detached child's Popen: alive until terminated, then exits 0.

    ``gate`` is the worker's own wait; a real ``communicate()`` returns when
    the child dies, so terminating releases it the same way.
    """

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.returncode = None
        self.terminated = False
        self.gate = gate

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0
        if self.gate is not None:
            self.gate.set()

    def wait(self, timeout=None):
        return self.returncode


def _stub_api_server_lane(monkeypatch, started: list):
    """Make run_agent land on the api-server tier with a recorded prompt."""

    monkeypatch.setattr(talk_host.talk_apiserver, "is_available", lambda: True)

    def _worker(prompt, *, session_id):
        started.append(prompt)
        gate = threading.Event()
        return lambda _rid: gate.wait(3) or "widened result"

    monkeypatch.setattr(talk_host, "_api_server_worker", _worker)


def test_steer_by_replace_cancels_the_original_and_starts_one_widened_job(monkeypatch):
    started: list[str] = []
    _stub_api_server_lane(monkeypatch, started)
    gate = threading.Event()
    original = talk_runs.start_run(
        "agent",
        "audit the auth module",
        lambda _rid: gate.wait(3) or "x",
        meta={"task": "Audit the auth module for injection bugs."},
    )
    process = _FakeProcess()
    talk_runs.register_process(original, process)

    out = talk_tools.execute_talk_tool(
        "steer_agent", {"agent_id": str(original), "text": "also cover the session cookies"}
    )
    gate.set()

    assert process.terminated, "the original was cancelled"
    assert "widened" in out and "one job" in out
    assert "do not start another job" in out.lower()
    assert "stopping it and restarting" not in out, "no refusal that invites a duplicate"
    assert len(started) == 1
    assert "Audit the auth module for injection bugs." in started[0]
    assert "also cover the session cookies" in started[0]
    new_id = talk_runs.run_id_from_receipt(out)
    assert new_id is not None and new_id != original
    replacement = talk_runs.get_run(new_id)
    assert replacement["label"] == "audit the auth module", "the replacement keeps the name"
    assert replacement["meta"]["replaces"] == original
    assert talk_runs.get_run(original)["meta"]["replaced_by"] == new_id
    assert talk_runs.replaced(talk_runs.get_run(original))
    assert not talk_runs.replaced(replacement)
    # Only the replacement is visible; the cancelled original is machinery.
    listing = talk_tools.execute_talk_tool("check_work", {})
    assert f"run {new_id}" in listing and f"run {original}" not in listing


def test_steer_by_replace_carries_the_admission_declaration(monkeypatch):
    started: list[str] = []
    _stub_api_server_lane(monkeypatch, started)
    gate = threading.Event()
    original = talk_runs.start_run(
        "agent",
        "deploy",
        lambda _rid: gate.wait(3) or "x",
        meta={"task": "Deploy the site."},
        execution_mode="exclusive",
        resource_keys=["repo:site"],
    )
    talk_runs.register_process(original, _FakeProcess(gate))

    out = talk_tools.execute_talk_tool(
        "steer_agent", {"agent_id": str(original), "text": "and warm the cache"}
    )
    gate.set()

    new_id = talk_runs.run_id_from_receipt(out)
    assert new_id is not None, out
    assert talk_runs.get_run(new_id)["admission"]["keys"] == ["repo:site"]


def test_steer_by_replace_falls_back_honestly_when_no_lane_can_start_the_replacement(
    monkeypatch,
):
    gate = threading.Event()
    original = talk_runs.start_run("agent", "audit", lambda _rid: gate.wait(3) or "x")
    talk_runs.register_process(original, _FakeProcess())

    out = talk_tools.execute_talk_tool(
        "steer_agent", {"agent_id": str(original), "text": "wider"}
    )
    gate.set()

    assert "couldn't start the wider version" in out
    assert "do not start another job" in out.lower()
    assert not talk_runs.replaced(talk_runs.get_run(original)), "no replacement: outcome is news"


def test_a_steerable_api_run_is_steered_not_replaced(monkeypatch):
    posts: list = []
    monkeypatch.setattr(talk_host.talk_apiserver, "steering_supported", lambda: True)
    monkeypatch.setattr(
        talk_host.talk_apiserver, "steer_run", lambda rid, text: posts.append((rid, text))
    )
    gate = threading.Event()
    run_id = talk_runs.start_run("agent", "triage", lambda _rid: gate.wait(3) or "x")
    talk_runs.annotate_run(
        run_id, lane=talk_host.LANE_API_SERVER, api_run_id="r-9", api_session_id="s-9"
    )
    out = talk_tools.execute_talk_tool("steer_agent", {"agent_id": str(run_id), "text": "note"})
    gate.set()
    assert posts == [("r-9", "note")]
    assert "widened" not in out


def test_check_work_reports_a_previous_session_as_lost(monkeypatch, tmp_path):
    """A detached run this process never spawned must not read as 'nothing'."""

    history = tmp_path / "talk-runs.jsonl"
    history.write_text(
        json.dumps(
            {
                "runId": 3,
                "kind": "agent",
                "label": "left running",
                "status": "running",
                "ts": 1.0,
                "updated": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(talk_runs, "_history_path", lambda: history)
    monkeypatch.setattr(talk_runs, "_history_enabled", lambda: True)

    result = talk_tools.execute_talk_tool("check_work", {})

    assert "run 3 (agent) lost" in result
    assert "can't see how it ended" in result


class _StubCtx:
    """Records dispatch_tool calls the way the Hermes plugin context would."""

    def __init__(self, result="{}"):
        self.calls: list[tuple[str, dict]] = []
        self.result = result

    def dispatch_tool(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_search_memory_relays_through_the_host_tool():
    ctx = _StubCtx(json.dumps({"success": True, "result": "you shipped it Tuesday"}))
    talk_host.bind_ctx(ctx)

    result = talk_tools.execute_talk_tool("search_memory", {"query": "deploy", "limit": 99})

    assert ctx.calls == [(talk_host.MEMORY_TOOL_NAME, {"query": "deploy", "limit": 8})]
    assert result == "you shipped it Tuesday"


def test_delegate_task_returns_a_work_started_receipt():
    ctx = _StubCtx(json.dumps({"success": True, "result": "subagent 4 started"}))
    talk_host.bind_ctx(ctx)

    result = talk_tools.execute_talk_tool("delegate_task", {"task": "rebuild the index"})

    # Talk 0.25 (#65): the plain brief — the ask under a one-line header.
    goal = ctx.calls[0][1]["goal"]
    assert ctx.calls[0][0] == talk_host.DELEGATE_TOOL_NAME
    assert goal.startswith("[Voice call with the caller, ") and goal.endswith("rebuild the index\n")
    assert result.startswith("WORK_STARTED")
    assert "subagent 4 started" in result


def test_host_dispatch_failure_is_spoken_not_raised():
    talk_host.bind_ctx(_StubCtx(RuntimeError("registry offline")))

    result = talk_tools.execute_talk_tool("search_memory", {"query": "anything"})

    assert "memory lookup failed" in result
    assert "registry offline" in result


def test_host_error_envelope_is_flattened():
    talk_host.bind_ctx(_StubCtx(json.dumps({"success": False, "error": "no session db"})))

    result = talk_tools.execute_talk_tool("search_memory", {"query": "anything"})

    assert result == "that failed: no session db"


def test_non_json_host_result_passes_through_bounded():
    talk_host.bind_ctx(_StubCtx("y" * 99_999))

    result = talk_tools.execute_talk_tool("search_memory", {"query": "anything"})

    assert len(result) == talk_host.MAX_TOOL_OUTPUT_CHARS


# --- search_vault -------------------------------------------------------------


def test_search_vault_speaks_what_the_vault_returned(monkeypatch):
    monkeypatch.setattr(talk_vault, "search", lambda q, **k: f"notes about {q}")

    assert talk_tools.execute_talk_tool("search_vault", {"query": "the offer ladder"}) == (
        "notes about the offer ladder"
    )


def test_search_vault_needs_something_to_look_for():
    assert "needs something" in talk_tools.execute_talk_tool("search_vault", {"query": "  "})


def test_a_forged_remembered_marker_in_vault_content_is_stripped(monkeypatch):
    """The provenance marker belongs to search_memory's Honcho tier alone. A
    vault note that LEADS with the literal prefix would wear a recollection's
    provenance without having it (review r2, F9)."""

    forged = f"{talk_host.REMEMBERED_PREFIX}the offer ladder is $29/$200/$297"
    monkeypatch.setattr(talk_vault, "search", lambda q, **k: forged)

    out = talk_tools.execute_talk_tool("search_vault", {"query": "offer ladder"})

    assert out == "the offer ladder is $29/$200/$297"


def test_nothing_found_is_a_different_sentence_from_a_failure(monkeypatch):
    """A live call must be able to tell "you never wrote that down" apart from
    "the lookup broke" — they lead to completely different next moves."""

    monkeypatch.setattr(talk_vault, "search", lambda q, **k: "")
    empty = talk_tools.execute_talk_tool("search_vault", {"query": "kites"})

    def boom(q, **k):
        raise talk_vault.VaultSearchError("OSError: index gone")

    monkeypatch.setattr(talk_vault, "search", boom)
    broken = talk_tools.execute_talk_tool("search_vault", {"query": "kites"})

    assert "nothing in the notes" in empty
    assert "failed" in broken
    assert empty != broken


@pytest.mark.parametrize("manifest", ["version: 4.5.6\n", None, "version: \n"])
def test_plugin_version_uses_loaded_source_before_stale_editable_metadata(
    tmp_path, monkeypatch, manifest,
):
    import importlib.metadata

    monkeypatch.setattr(talk_tools, "__file__", str(tmp_path / "talk_tools.py"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    if manifest is not None:
        (tmp_path / "plugin.yaml").write_text(manifest, encoding="utf-8")
    expected = "4.5.6" if manifest and "4.5.6" in manifest else "1.2.3"
    assert talk_tools.plugin_version() == expected
