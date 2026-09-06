"""Timeline annotations: load, validate, and select curated events for a chart.

The curated list lives in ``data/reference/events.yaml`` so that every flag drawn
on a public chart is version-controlled and reviewable. Validation is strict and
fails the build rather than the browser: a mistyped date or an over-long label is
a curation bug, and shipping it costs more credibility than the annotation buys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.common.taxonomy import canonical

#: Categories of event. Anything else is a typo in the curated file.
VALID_KINDS: frozenset[str] = frozenset(
    {"paper", "model", "release", "policy", "infrastructure", "taxonomy"}
)

#: Longest on-chart flag text that fits without overlapping its neighbours.
MAX_SHORT_LENGTH = 18

MIN_WEIGHT = 1
MAX_WEIGHT = 5

_DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

_REQUIRED_FIELDS = (
    "date",
    "label",
    "short",
    "url",
    "scope",
    "weight",
    "kind",
    "description",
    "on_arxiv",
)


@dataclass(frozen=True)
class Event:
    """A dated annotation drawn on a trend chart.

    Attributes:
        date: Month of the arXiv v1 posting, as ``YYYY-MM``
        label: Full name, shown on hover
        short: On-chart flag text, at most MAX_SHORT_LENGTH characters
        url: Primary source for the claim
        scope: Categories or terms the event is relevant to; empty means global
        weight: Importance from MIN_WEIGHT to MAX_WEIGHT, used to break collisions
        kind: One of VALID_KINDS
        description: One sentence of context
        on_arxiv: Whether the work itself was posted to arXiv
    """

    date: str
    label: str
    short: str
    url: str
    scope: tuple[str, ...]
    weight: int
    kind: str
    description: str
    on_arxiv: bool


def load_events(path: str | Path) -> list[Event]:
    """Load and validate the curated event list.

    Args:
        path: Path to a YAML file holding a list of event mappings

    Returns:
        Events in file order, each one fully validated

    Raises:
        ValueError: If the file is not a list, or any entry fails validation.
            The message names the offending entry by label, or by position when
            the label itself is missing.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of events, got {type(raw).__name__}")

    return [_parse_event(entry, position) for position, entry in enumerate(raw)]


def to_artifact(events: list[Event]) -> dict:
    """Convert events to a JSON-ready structure for the static frontend.

    Args:
        events: Events to serialise

    Returns:
        Dict with a ``count`` and an ``events`` list sorted by date, then by
        descending weight so that same-month collisions arrive pre-ranked
    """
    ordered = sorted(events, key=lambda event: (event.date, -event.weight, event.label))

    return {
        "count": len(ordered),
        "events": [
            {
                "date": event.date,
                "label": event.label,
                "short": event.short,
                "url": event.url,
                "scope": list(event.scope),
                "weight": event.weight,
                "kind": event.kind,
                "description": event.description,
                "on_arxiv": event.on_arxiv,
            }
            for event in ordered
        ],
    }


def select_for_scope(
    events: list[Event],
    scope: str | list[str],
    limit: int = 4,
) -> list[Event]:
    """Pick the events worth annotating on one series.

    An event qualifies when it is global (empty scope) or when its scope
    intersects the active series. Intersection resolves category aliases and
    treats an archive as containing its subcategories, so an event scoped to
    ``q-bio`` annotates a ``q-bio.PE`` series and vice versa.

    Only weight and scope decide the outcome here. Pixel collision is the
    frontend's problem, because it depends on the rendered axis width.

    Args:
        events: Candidate events, typically the whole curated list
        scope: The active series, as one category/term or a list of them
        limit: Maximum number of events to return

    Returns:
        At most ``limit`` events, most important first, ties broken by date
    """
    if limit <= 0:
        return []

    active = [scope] if isinstance(scope, str) else list(scope)
    matches = [event for event in events if _matches_scope(event, active)]

    matches.sort(key=lambda event: (-event.weight, event.date, event.label))
    return matches[:limit]


def _parse_event(entry: object, position: int) -> Event:
    """Validate one raw YAML mapping and build an Event from it.

    Args:
        entry: Raw value decoded from the YAML list
        position: Index in the file, used to name the entry when it has no label

    Returns:
        The validated event

    Raises:
        ValueError: If any field is missing, mistyped, or out of range
    """
    if not isinstance(entry, dict):
        raise ValueError(f"event #{position}: expected a mapping, got {type(entry).__name__}")

    name = entry.get("label") or f"event #{position}"

    missing = [field for field in _REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"{name}: missing required field(s) {', '.join(missing)}")

    date = entry["date"]
    if not isinstance(date, str) or not _DATE_PATTERN.match(date):
        raise ValueError(f"{name}: date must be YYYY-MM with a real month, got {date!r}")

    for field in ("label", "short", "url", "kind", "description"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise ValueError(f"{name}: {field} must be a non-empty string, got {entry[field]!r}")

    short = entry["short"]
    if len(short) > MAX_SHORT_LENGTH:
        raise ValueError(
            f"{name}: short is {len(short)} chars, at most {MAX_SHORT_LENGTH} fit on the chart"
        )

    kind = entry["kind"]
    if kind not in VALID_KINDS:
        raise ValueError(f"{name}: unknown kind {kind!r}, expected one of {sorted(VALID_KINDS)}")

    weight = entry["weight"]
    # bool is an int subclass, so reject it explicitly rather than reading True as weight 1.
    if isinstance(weight, bool) or not isinstance(weight, int):
        raise ValueError(f"{name}: weight must be an int, got {weight!r}")
    if not MIN_WEIGHT <= weight <= MAX_WEIGHT:
        raise ValueError(f"{name}: weight {weight} is outside {MIN_WEIGHT}-{MAX_WEIGHT}")

    scope = entry["scope"]
    if not isinstance(scope, list) or not all(isinstance(term, str) for term in scope):
        raise ValueError(f"{name}: scope must be a list of strings, got {scope!r}")
    # A blank term is not global; it just matches nothing, so the flag would
    # silently vanish from every chart instead of failing the build.
    if not all(term.strip() for term in scope):
        raise ValueError(f"{name}: scope terms must be non-empty, got {scope!r}")

    on_arxiv = entry["on_arxiv"]
    if not isinstance(on_arxiv, bool):
        raise ValueError(f"{name}: on_arxiv must be true or false, got {on_arxiv!r}")

    return Event(
        date=date,
        label=entry["label"],
        short=short,
        url=entry["url"],
        scope=tuple(scope),
        weight=weight,
        kind=kind,
        description=entry["description"],
        on_arxiv=on_arxiv,
    )


def _matches_scope(event: Event, active: list[str]) -> bool:
    """Report whether an event should annotate the given series.

    Args:
        event: The candidate event
        active: Categories or terms of the series being drawn

    Returns:
        True if the event is global or its scope intersects the active terms
    """
    if not event.scope:
        return True

    return any(
        _terms_overlap(event_term, active_term)
        for event_term in event.scope
        for active_term in active
    )


def _terms_overlap(event_term: str, active_term: str) -> bool:
    """Report whether two scope terms refer to overlapping literature.

    Args:
        event_term: A term from an event's scope
        active_term: A term describing the active series

    Returns:
        True if the terms are equal after alias resolution, or if one is the
        parent archive of the other
    """
    left = canonical(event_term)
    right = canonical(active_term)

    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")
