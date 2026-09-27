"""Exact target resolution for heard names (hermes-sip-live-voice#54).

A phone caller says "the hermes sip live voice repo" and speech-to-text
hands the model "hermes sip life voice". Before this module the model
either insisted on an exact identifier the caller could not reasonably
spell aloud, or silently normalized the corrupted token into whatever
matched best — and the delegate then reviewed an unrelated public project.

:func:`resolve_target` matches the heard phrase against an alias catalog
built from what is actually installed here — Hermes plugins
(``$HERMES_HOME/plugins/*/plugin.yaml``), local git repositories under the
configured roots (``TALK_REPO_ROOTS``; default ``$HERMES_HOME/repos`` and
``~/repos`` when they exist), and a static alias map
(``TALK_TARGET_ALIASES``, ``alias=name;alias2=name2``). Every answer keeps
the raw heard phrase beside the resolved name and the evidence that linked
them. Two candidates within the margin => ``resolved=None`` with the
alternatives listed, so the model asks ONE discriminating question instead
of picking the closest match (``talk_identity.ANTI_GUESS_RULE``).

:func:`resolve` is the narrow module-level seam the SIP replay suite calls:
the resolved string, or ``None`` on anything uncertain.
"""

from __future__ import annotations

import configparser
import difflib
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from . import talk_config
except ImportError:  # pragma: no cover - flat-module fallback
    import talk_config

_log = logging.getLogger(__name__)

#: A candidate below this similarity is not a match at all.
THRESHOLD = 0.72
#: Two candidates whose scores differ by less than this are ambiguous.
MARGIN = 0.08
REPO_ROOTS_ENV = "TALK_REPO_ROOTS"
ALIASES_ENV = "TALK_TARGET_ALIASES"
_MAX_CATALOG = 400
_SPLIT_RE = re.compile(r"[\s\-_./:]+")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One thing a heard phrase may name, with where that knowledge came from."""

    name: str
    kind: str  # plugin | repo | alias | given
    aliases: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Resolution:
    heard: str
    resolved: str | None
    evidence: dict[str, Any]
    confidence: float
    alternatives: list[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.resolved is None and bool(self.alternatives)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def question(self) -> str | None:
        """The one discriminating question to ask, or ``None`` when nothing to ask."""

        if not self.ambiguous:
            return None
        options = " or ".join(self.alternatives[:3])
        return f"Did you mean {options}?"


def _norm(text: str) -> str:
    return " ".join(part for part in _SPLIT_RE.split(str(text or "").lower()) if part)


def _tokens(text: str) -> list[str]:
    return _norm(text).split()


def _score(heard: str, alias: str) -> float:
    """Sequence similarity on normalized text, blended with token-level matching.

    Token matching absorbs speech corruption inside one word ("life" for
    "live") without letting a two-word overlap masquerade as a match.
    """

    a, b = _norm(heard), _norm(alias)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = _tokens(a), _tokens(b)
    matched = 0.0
    for token in ta:
        best = max((difflib.SequenceMatcher(None, token, other).ratio() for other in tb), default=0)
        matched += best if best >= 0.75 else 0.0
    coverage = matched / max(len(ta), len(tb))
    return round(0.5 * seq + 0.5 * coverage, 4)


# -- catalog ------------------------------------------------------------------


def _plugin_candidates(home: Path) -> list[Candidate]:
    out: list[Candidate] = []
    root = home / "plugins"
    if not root.is_dir():
        return out
    for manifest in sorted(root.glob("*/plugin.yaml")):
        name = homepage = None
        try:
            for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip("'\"")
                elif line.startswith("homepage:"):
                    homepage = line.split(":", 1)[1].strip().strip("'\"")
        except OSError:
            continue
        name = name or manifest.parent.name
        aliases = {name, manifest.parent.name}
        if homepage:
            aliases.add(homepage.rstrip("/").rsplit("/", 1)[-1])
        out.append(
            Candidate(
                name=name,
                kind="plugin",
                aliases=tuple(sorted(a for a in aliases if a)),
                evidence={"manifest": str(manifest), "homepage": homepage},
            )
        )
    return out


def _origin_remote(repo: Path) -> str | None:
    config = repo / ".git" / "config"
    if not config.is_file():
        return None
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read(config, encoding="utf-8")
        return parser.get('remote "origin"', "url", fallback=None)
    except (configparser.Error, OSError):
        return None


def repo_roots() -> list[Path]:
    raw = (os.environ.get(REPO_ROOTS_ENV) or "").strip()
    if raw:
        return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]
    defaults = [talk_config.get_hermes_home() / "repos", Path.home() / "repos"]
    return [root for root in defaults if root.is_dir()]


def _repo_candidates(roots: list[Path]) -> list[Candidate]:
    out: list[Candidate] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            children = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for repo in children:
            if not (repo / ".git").exists():
                continue
            remote = _origin_remote(repo)
            aliases = {repo.name}
            if remote:
                tail = remote.rstrip("/").rsplit("/", 1)[-1]
                aliases.add(tail[:-4] if tail.endswith(".git") else tail)
            out.append(
                Candidate(
                    name=repo.name,
                    kind="repo",
                    aliases=tuple(sorted(aliases)),
                    evidence={"path": str(repo), "origin": remote},
                )
            )
    return out


def static_aliases() -> dict[str, str]:
    raw = (os.environ.get(ALIASES_ENV) or "").strip()
    out: dict[str, str] = {}
    for pair in raw.split(";"):
        if "=" in pair:
            alias, name = pair.split("=", 1)
            if alias.strip() and name.strip():
                out[alias.strip()] = name.strip()
    return out


def catalog(extra: list[str] | None = None) -> list[Candidate]:
    """Installed plugins + local repos + static aliases (+ caller-given names)."""

    found: list[Candidate] = []
    try:
        home = talk_config.get_hermes_home()
        found.extend(_plugin_candidates(home))
        found.extend(_repo_candidates(repo_roots()))
    except Exception as exc:  # noqa: BLE001 — a thin catalog, never a dead resolver
        _log.debug("target catalog scan failed: %s: %s", type(exc).__name__, exc)
    by_name: dict[str, Candidate] = {}
    for cand in found:
        by_name.setdefault(cand.name, cand)
    for alias, name in static_aliases().items():
        existing = by_name.get(name)
        if existing is not None:
            by_name[name] = Candidate(
                existing.name, existing.kind, (*existing.aliases, alias), existing.evidence
            )
        else:
            by_name[name] = Candidate(name, "alias", (name, alias), {"source": ALIASES_ENV})
    for name in extra or ():
        if isinstance(name, str) and name.strip() and name not in by_name:
            by_name[name] = Candidate(name, "given", (name,), {"source": "candidates"})
    return list(by_name.values())[:_MAX_CATALOG]


# -- resolution ---------------------------------------------------------------


def _best_alias(heard: str, cand: Candidate) -> tuple[float, str]:
    best, via = 0.0, cand.name
    for alias in (cand.name, *cand.aliases):
        score = _score(heard, alias)
        if score > best:
            best, via = score, alias
    return best, via


def resolve_target(heard: str, candidates: list[str] | None = None) -> Resolution:
    """Resolve ``heard`` against the catalog (plus ``candidates``); never guess."""

    heard = str(heard or "").strip()
    if not heard:
        return Resolution(heard, None, {"reason": "empty"}, 0.0)
    if candidates is not None:
        pool = catalog(list(candidates))
        # Given candidates constrain the answer: the installed catalog only
        # adds evidence for names the caller already put on the table.
        allowed = {c for c in candidates if isinstance(c, str)}
        pool = [c for c in pool if c.name in allowed or set(c.aliases) & allowed]
    else:
        pool = catalog()
    scored = sorted(
        ((*_best_alias(heard, cand), cand) for cand in pool), key=lambda t: t[0], reverse=True
    )
    if not scored or scored[0][0] < THRESHOLD:
        near = [cand.name for score, _via, cand in scored[:3] if score >= THRESHOLD - 0.15]
        return Resolution(
            heard,
            None,
            {"reason": "no_match", "best_score": scored[0][0] if scored else 0.0},
            scored[0][0] if scored else 0.0,
            near,
        )
    top_score, via, top = scored[0]
    rivals = [cand for score, _v, cand in scored[1:] if top_score - score < MARGIN]
    if rivals:
        return Resolution(
            heard,
            None,
            {"reason": "ambiguous", "margin": MARGIN, "top_score": top_score},
            top_score,
            [top.name, *(r.name for r in rivals)],
        )
    return Resolution(
        heard,
        top.name,
        {"kind": top.kind, "matched_alias": via, **top.evidence},
        top_score,
        [cand.name for score, _v, cand in scored[1:3] if score >= THRESHOLD - 0.15],
    )


def resolve(heard: str, candidates: list[str] | None = None) -> str | None:
    """Narrow seam: the exact resolved name, or ``None`` for anything uncertain."""

    return resolve_target(heard, candidates).resolved


__all__ = [
    "ALIASES_ENV",
    "MARGIN",
    "REPO_ROOTS_ENV",
    "THRESHOLD",
    "Candidate",
    "Resolution",
    "catalog",
    "repo_roots",
    "resolve",
    "resolve_target",
    "static_aliases",
]
