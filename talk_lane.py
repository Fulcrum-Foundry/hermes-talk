"""Per-lane operating policy a transport hands to :func:`talk_cli.run_talk_session`.

A lane (terminal, Discord, dashboard, phone) is more than a where-am-I
sentence. The phone lane in particular needs to inject a trusted operating
pack (owner, assistant name, delivery rules; hermes-sip-live-voice#26),
switch off routine spoken heartbeats (#51), keep transcript retention apart
from durable-memory promotion (#35), and offer lane-specific controls such as
``end_call`` (#57). Before this module every one of those was either a
one-line ``LANE_LINES`` entry monkey-patched from outside or impossible.

:class:`LanePolicy` is the whole contract. Everything is optional and every
default reproduces the pre-policy behaviour exactly, so a lane that passes
``None`` (or nothing) runs as it always did. The session reads the policy in
ONE place per concern; consumers never reach back into the transport.

Trust boundary: ``instructions`` is TRUSTED operating policy from the host
process that owns the lane (it is rendered by code, never by the model or a
caller). It is placed ahead of host identity sections and behind the fixed
voice preamble; the preamble's safety rules still come first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: Bound on the injected operating pack. Larger packs are truncated with a
#: visible marker so a diagnostics read can tell the pack was cut.
INSTRUCTIONS_CAP = 6_000
TRUNCATION_MARKER = "\n[operating pack truncated at cap]"
#: Seconds a tool call may run in silence before Talk speaks one short filler
#: (#69). Lives here, not in talk_cli, so the policy default can name it.
FILLER_AFTER_S = 1.5


@dataclass(frozen=True, slots=True)
class LanePolicy:
    """What one transport asks of the session it starts.

    ``name``: the lane label for the where-am-I line and diagnostics.
    ``instructions``: rendered operating pack (trusted), or ``None``.
    ``instructions_version``: opaque version string echoed in receipts.
    ``spoken_heartbeats``: whether routine "still working" milestones may be
        spoken. ``None`` = the lane's historical default (on).
    ``memory_review``: whether the post-call transcript is handed to the
        durable-memory review. ``None`` = historical default (on).
    ``tools``: extra function-tool schemas the lane exposes to the model
        (each a dict with ``name``/``description``/``parameters``) paired
        with ``handlers`` by name. The session advertises them alongside its
        own; unknown names are refused by the normal tool contract.
    ``on_end_call``: called when the model invokes the lane's end-call tool
        (if the lane exposed one). The lane owns the physical teardown.
    ``announcements``: how routine background notices reach the caller
        (hermes-sip-live-voice#51). ``"immediate"`` (default) is the
        pre-0.23 behaviour: spoken as soon as the wire is idle.
        ``"deferred"`` parks completions as ready records and offers ONE
        coalesced notice at a natural pause; routine speech is suppressed
        while the caller speaks, during hold, after "later", and once the
        call is closing. Approval questions are never deferred past the pause.
        ``"immediate_segue"`` (#68) gates the same way but SPEAKS each result
        at the next pause with a short transition and no "now or later?"
        question; "later" still parks.
    ``filler_after_s``: how long a tool call may run in silence before Talk
        itself speaks one short filler ("Give me a second.") (#69). Default
        :data:`FILLER_AFTER_S`; ``None`` disables the filler for this lane.
        The model never speaks before a tool call; only Talk fills a wait
        that is actually long.
    ``verbosity``: the lane's default spoken depth (``"concise"`` /
        ``"detailed"``), or ``None`` for the preamble's own default. The
        caller can flip it per session with ``set_verbosity`` (#57).
    ``manifest``: free-form producer facts (what was included, omitted,
        truncated) for the diagnostics receipt. Never read by the model.
    ``binding_key``: opaque durable-conversation key for this caller on this
        deployment (hermes-sip-live-voice#35, I11; SIP passes a hash of
        caller + deployment). With a key, the session attaches under the
        binding's durable Hermes session id and adopts the caller's exact
        pending results; ``None`` (default) binds nothing and behaves as
        every release before 0.24.
    ``after_call``: what happens to results that finish after the caller is
        gone. ``"retrievable"`` (default) keeps them in the ledger and the
        binding for the next call; ``"none"`` records nothing pending. No
        external delivery is ever started by Talk.
    ``delivery_evidence``: optional ``callable(run_id) -> dict`` a transport
        with acknowledgments supplies (``acked``/``audible_ms``; see
        :mod:`talk_delivery`). When set, the transport's acknowledged state at
        send time is recorded on the run as ``meta.delivery`` (a receipt for
        ``check_work`` and the ledger). It never gates the exactly-once
        delivered flip, which happens on send on every lane.
    """

    name: str = "cli"
    instructions: str | None = None
    instructions_version: str | None = None
    spoken_heartbeats: bool | None = None
    memory_review: bool | None = None
    tools: tuple[dict, ...] = ()
    handlers: dict[str, Callable[[dict], Any]] = field(default_factory=dict)
    on_end_call: Callable[[], Any] | None = None
    announcements: str = "immediate"
    filler_after_s: float | None = FILLER_AFTER_S
    verbosity: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    binding_key: str | None = None
    after_call: str = "retrievable"
    delivery_evidence: Callable[[int], Any] | None = None

    def rendered_instructions(self) -> str | None:
        """The pack as it will be placed in the prompt: stripped, capped, marked when cut."""

        if not self.instructions:
            return None
        text = self.instructions.strip()
        if not text:
            return None
        if len(text) > INSTRUCTIONS_CAP:
            return text[: INSTRUCTIONS_CAP - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
        return text

    @property
    def truncated(self) -> bool:
        rendered = self.rendered_instructions()
        return bool(rendered and rendered.endswith(TRUNCATION_MARKER))

    def receipt(self) -> dict[str, Any]:
        """Diagnostics view: what this session actually ran with. No secrets, no prompt text."""

        rendered = self.rendered_instructions()
        return {
            "lane": self.name,
            "instructions_chars": len(rendered) if rendered else 0,
            "instructions_version": self.instructions_version,
            "instructions_truncated": self.truncated,
            "spoken_heartbeats": self.spoken_heartbeats,
            "memory_review": self.memory_review,
            "announcements": self.announcements,
            "filler_after_s": self.filler_after_s,
            "verbosity": self.verbosity,
            "binding": bool(self.binding_key),
            "after_call": self.after_call,
            "delivery_evidence": self.delivery_evidence is not None,
            "tools": [t.get("name") for t in self.tools if isinstance(t, dict)],
            "manifest": dict(self.manifest),
        }


def coerce(policy: LanePolicy | None, lane: str | None) -> LanePolicy:
    """A policy for every session: the given one, or the lane's neutral default."""

    if policy is not None:
        return policy
    return LanePolicy(name=str(lane or "cli"))


__all__ = ["FILLER_AFTER_S", "INSTRUCTIONS_CAP", "TRUNCATION_MARKER", "LanePolicy", "coerce"]
