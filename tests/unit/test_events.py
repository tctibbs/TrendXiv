"""Unit tests for the timeline annotation layer.

The parametrised tests over the shipped YAML are the point of this file: they are
what catches a curation typo before it reaches a public chart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.pipeline.events import (
    MAX_SHORT_LENGTH,
    VALID_KINDS,
    Event,
    load_events,
    select_for_scope,
    to_artifact,
)

EVENTS_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "events.yaml"

RAW_EVENTS: list[dict[str, Any]] = yaml.safe_load(EVENTS_PATH.read_text(encoding="utf-8"))

#: New-style (1706.03762) and legacy (astro-ph/9805201) abs URLs.
_NEW_STYLE_ID = re.compile(r"arxiv\.org/abs/(\d{2})(\d{2})\.\d{4,5}")
_OLD_STYLE_ID = re.compile(r"arxiv\.org/abs/[a-z-]+(?:\.[A-Z]{2})?/(\d{2})(\d{2})\d{3}")


def arxiv_id_month(url: str) -> str | None:
    """Extract the posting month encoded in an arXiv abs URL.

    Both identifier schemes embed the v1 submission month, so the URL is an
    independent check on the curated date.

    Args:
        url: A source URL, arXiv or otherwise

    Returns:
        The month as ``YYYY-MM``, or None if the URL is not an arXiv abs link
    """
    match = _NEW_STYLE_ID.search(url) or _OLD_STYLE_ID.search(url)
    if match is None:
        return None

    year_2digit, month = match.groups()
    century = 1900 if int(year_2digit) >= 91 else 2000
    return f"{century + int(year_2digit)}-{month}"


def labels_of(entries: list[dict[str, Any]]) -> list[str]:
    """Name parametrised cases so a failure identifies the offending event.

    Args:
        entries: Raw event mappings

    Returns:
        One id per entry
    """
    return [entry.get("label", f"entry-{i}") for i, entry in enumerate(entries)]


ARXIV_URL_EVENTS = [entry for entry in RAW_EVENTS if arxiv_id_month(entry["url"])]

OFF_ARXIV_EVENTS = [entry for entry in RAW_EVENTS if not entry["on_arxiv"]]

VALID_ENTRY: dict[str, Any] = {
    "date": "2017-06",
    "label": "Attention Is All You Need",
    "short": "Attention",
    "url": "https://arxiv.org/abs/1706.03762",
    "scope": ["cs.LG", "cs.CL"],
    "weight": 5,
    "kind": "paper",
    "description": "The Transformer.",
    "on_arxiv": True,
}


def write_events(tmp_path: Path, entries: Any) -> Path:
    """Write a YAML event file for a test and return its path.

    Args:
        tmp_path: pytest temporary directory
        entries: Any structure to serialise, valid or not

    Returns:
        Path to the written file
    """
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def entry_with(**overrides: Any) -> dict[str, Any]:
    """Build a copy of the known-good entry with fields replaced.

    Args:
        **overrides: Fields to override

    Returns:
        A single-event mapping
    """
    return {**VALID_ENTRY, **overrides}


def make_event(**overrides: Any) -> Event:
    """Build an Event directly, bypassing YAML.

    Args:
        **overrides: Fields to override on the known-good event

    Returns:
        The constructed event
    """
    fields = entry_with(**overrides)
    fields["scope"] = tuple(fields["scope"])
    return Event(**fields)


# --- The shipped curation ------------------------------------------------------


@pytest.mark.unit
def test_shipped_file_loads() -> None:
    events = load_events(EVENTS_PATH)

    assert len(events) == len(RAW_EVENTS)
    assert all(isinstance(event, Event) for event in events)


@pytest.mark.unit
@pytest.mark.parametrize("entry", RAW_EVENTS, ids=labels_of(RAW_EVENTS))
def test_shipped_entry_validates(entry: dict[str, Any], tmp_path: Path) -> None:
    assert load_events(write_events(tmp_path, [entry]))


@pytest.mark.unit
@pytest.mark.parametrize("entry", ARXIV_URL_EVENTS, ids=labels_of(ARXIV_URL_EVENTS))
def test_shipped_arxiv_url_month_matches_date(entry: dict[str, Any]) -> None:
    """An arXiv identifier encodes its v1 month, so a date typo shows up here."""
    assert entry["date"] == arxiv_id_month(entry["url"])
    assert entry["on_arxiv"] is True


@pytest.mark.unit
@pytest.mark.parametrize("entry", OFF_ARXIV_EVENTS, ids=labels_of(OFF_ARXIV_EVENTS))
def test_shipped_off_arxiv_entry_has_no_arxiv_url(entry: dict[str, Any]) -> None:
    """An on_arxiv:false flag must not link to an arXiv posting."""
    assert "arxiv.org/abs/" not in entry["url"]


@pytest.mark.unit
def test_shipped_file_size_is_curated() -> None:
    assert 35 <= len(RAW_EVENTS) <= 55


@pytest.mark.unit
def test_shipped_file_spans_the_whole_corpus() -> None:
    """A CS-only, last-five-years timeline would undercut the all-of-arXiv framing."""
    dates = sorted(entry["date"] for entry in RAW_EVENTS)

    assert dates[0] < "2000-01"
    assert dates[-1] > "2024-01"
    assert len({date[:3] for date in dates}) >= 4  # at least four decades represented


@pytest.mark.unit
def test_shipped_file_covers_non_cs_fields() -> None:
    scopes = {term for entry in RAW_EVENTS for term in entry["scope"]}
    archives = {term.split(".")[0] for term in scopes}

    assert {"math", "astro-ph", "hep-ex", "hep-th", "q-bio", "cond-mat", "quant-ph"} <= archives


@pytest.mark.unit
def test_shipped_file_uses_every_kind() -> None:
    assert {entry["kind"] for entry in RAW_EVENTS} == VALID_KINDS


@pytest.mark.unit
def test_shipped_file_has_no_duplicates() -> None:
    keys = [(entry["date"], entry["label"]) for entry in RAW_EVENTS]

    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_shipped_file_flags_alexnet_as_off_arxiv() -> None:
    """AlexNet was never posted to arXiv; claiming otherwise is the trap this layer avoids."""
    alexnet = next(entry for entry in RAW_EVENTS if "AlexNet" in entry["label"])

    assert alexnet["on_arxiv"] is False
    assert alexnet["date"] == "2012-10"


@pytest.mark.unit
def test_shipped_on_arxiv_entries_all_cite_an_abs_url() -> None:
    """Every claimed arXiv posting must be checkable against the identifier."""
    on_arxiv = [entry for entry in RAW_EVENTS if entry["on_arxiv"]]

    assert on_arxiv
    assert all(arxiv_id_month(entry["url"]) for entry in on_arxiv)


@pytest.mark.unit
def test_shipped_file_includes_global_events() -> None:
    assert any(entry["scope"] == [] for entry in RAW_EVENTS)


# --- Validation ----------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"date": "2017-6"}, "date must be YYYY-MM"),
        ({"date": "2017-13"}, "date must be YYYY-MM"),
        ({"date": "June 2017"}, "date must be YYYY-MM"),
        ({"date": 201706}, "date must be YYYY-MM"),
        ({"short": "Attention Is All You Need"}, "at most 18"),
        ({"weight": 0}, "outside 1-5"),
        ({"weight": 6}, "outside 1-5"),
        ({"weight": "5"}, "weight must be an int"),
        ({"weight": True}, "weight must be an int"),
        ({"kind": "blogpost"}, "unknown kind"),
        ({"scope": "cs.LG"}, "scope must be a list"),
        ({"scope": ["cs.LG", 7]}, "scope must be a list"),
        ({"scope": ["cs.LG", "  "]}, "scope terms must be non-empty"),
        ({"on_arxiv": "yes"}, "on_arxiv must be true or false"),
        ({"label": ""}, "label must be a non-empty string"),
        ({"url": "   "}, "url must be a non-empty string"),
        ({"description": None}, "description must be a non-empty string"),
    ],
)
def test_invalid_field_is_rejected(
    overrides: dict[str, Any], expected: str, tmp_path: Path
) -> None:
    path = write_events(tmp_path, [entry_with(**overrides)])

    with pytest.raises(ValueError, match=re.escape(expected)):
        load_events(path)


@pytest.mark.unit
@pytest.mark.parametrize("field", sorted(VALID_ENTRY))
def test_missing_field_is_rejected(field: str, tmp_path: Path) -> None:
    entry = entry_with()
    del entry[field]
    path = write_events(tmp_path, [entry])

    with pytest.raises(ValueError, match=f"missing required field.*{field}"):
        load_events(path)


@pytest.mark.unit
def test_error_names_the_offending_entry(tmp_path: Path) -> None:
    path = write_events(tmp_path, [VALID_ENTRY, entry_with(label="Bad Event", weight=9)])

    with pytest.raises(ValueError, match="Bad Event"):
        load_events(path)


@pytest.mark.unit
def test_error_names_position_when_label_is_missing(tmp_path: Path) -> None:
    entry = entry_with()
    del entry["label"]
    path = write_events(tmp_path, [VALID_ENTRY, entry])

    with pytest.raises(ValueError, match="event #1"):
        load_events(path)


@pytest.mark.unit
def test_non_list_file_is_rejected(tmp_path: Path) -> None:
    path = write_events(tmp_path, {"events": [VALID_ENTRY]})

    with pytest.raises(ValueError, match="expected a list of events"):
        load_events(path)


@pytest.mark.unit
def test_non_mapping_entry_is_rejected(tmp_path: Path) -> None:
    path = write_events(tmp_path, ["Attention Is All You Need"])

    with pytest.raises(ValueError, match="event #0: expected a mapping"):
        load_events(path)


@pytest.mark.unit
def test_empty_file_loads_to_no_events(tmp_path: Path) -> None:
    path = write_events(tmp_path, [])

    assert load_events(path) == []


@pytest.mark.unit
def test_valid_entry_round_trips(tmp_path: Path) -> None:
    path = write_events(tmp_path, [VALID_ENTRY])

    (event,) = load_events(path)

    assert event.date == "2017-06"
    assert event.scope == ("cs.LG", "cs.CL")
    assert event.weight == 5
    assert event.on_arxiv is True


@pytest.mark.unit
def test_short_of_exactly_the_limit_is_accepted(tmp_path: Path) -> None:
    path = write_events(tmp_path, [entry_with(short="x" * MAX_SHORT_LENGTH)])

    assert load_events(path)


# --- select_for_scope ----------------------------------------------------------


@pytest.mark.unit
def test_select_returns_heaviest_first() -> None:
    events = [
        make_event(label="light", weight=2),
        make_event(label="heavy", weight=5),
        make_event(label="middle", weight=3),
    ]

    selected = select_for_scope(events, "cs.LG")

    assert [event.label for event in selected] == ["heavy", "middle", "light"]


@pytest.mark.unit
def test_select_breaks_weight_ties_by_date() -> None:
    # Labels run against date order so that sorting by label alone cannot pass.
    events = [
        make_event(label="a later event", date="2020-01"),
        make_event(label="z earlier event", date="2014-06"),
    ]

    selected = select_for_scope(events, "cs.LG")

    assert [event.label for event in selected] == ["z earlier event", "a later event"]


@pytest.mark.unit
def test_select_respects_the_limit() -> None:
    events = [make_event(label=f"event {i}", weight=5 - (i % 5)) for i in range(20)]

    assert len(select_for_scope(events, "cs.LG", limit=4)) == 4
    assert len(select_for_scope(events, "cs.LG", limit=1)) == 1


@pytest.mark.unit
def test_select_keeps_the_heaviest_when_limited() -> None:
    events = [make_event(label=f"w{weight}", weight=weight) for weight in (1, 2, 3, 4, 5)]

    selected = select_for_scope(events, "cs.LG", limit=2)

    assert [event.weight for event in selected] == [5, 4]


@pytest.mark.unit
def test_select_excludes_unrelated_scopes() -> None:
    events = [
        make_event(label="ml", scope=["cs.LG"]),
        make_event(label="astro", scope=["astro-ph.CO"]),
    ]

    selected = select_for_scope(events, "astro-ph.CO")

    assert [event.label for event in selected] == ["astro"]


@pytest.mark.unit
def test_select_always_includes_global_events() -> None:
    events = [
        make_event(label="global", scope=[], weight=1),
        make_event(label="ml", scope=["cs.LG"], weight=5),
    ]

    selected = select_for_scope(events, "math.NT")

    assert [event.label for event in selected] == ["global"]


@pytest.mark.unit
def test_select_accepts_multiple_active_terms() -> None:
    events = [
        make_event(label="ml", scope=["cs.LG"]),
        make_event(label="astro", scope=["astro-ph.CO"]),
        make_event(label="math", scope=["math.NT"]),
    ]

    selected = select_for_scope(events, ["cs.LG", "math.NT"])

    assert {event.label for event in selected} == {"ml", "math"}


@pytest.mark.unit
def test_select_matches_an_archive_against_its_subcategories() -> None:
    events = [make_event(label="q-bio event", scope=["q-bio"])]

    assert select_for_scope(events, "q-bio.PE")
    assert select_for_scope([make_event(scope=["q-bio.PE"])], "q-bio")


@pytest.mark.unit
def test_select_resolves_category_aliases() -> None:
    """cs.SY is canonicalised to eess.SY, so an event scoped to either matches both."""
    events = [make_event(label="control", scope=["eess.SY"])]

    assert select_for_scope(events, "cs.SY")


@pytest.mark.unit
def test_select_does_not_match_a_prefix_that_is_not_an_archive() -> None:
    """cs.LG must not match cs.LGX; only a dot boundary counts."""
    events = [make_event(scope=["cs.LG"])]

    assert select_for_scope(events, "cs.LGX") == []


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, -1])
def test_select_with_no_room_returns_nothing(limit: int) -> None:
    assert select_for_scope([make_event()], "cs.LG", limit=limit) == []


@pytest.mark.unit
def test_select_on_the_shipped_file() -> None:
    events = load_events(EVENTS_PATH)

    selected = select_for_scope(events, "cs.CL", limit=4)

    assert len(selected) == 4
    assert all(event.weight >= selected[-1].weight for event in selected)


# --- to_artifact ---------------------------------------------------------------


@pytest.mark.unit
def test_artifact_is_sorted_by_date() -> None:
    events = [
        make_event(label="late", date="2022-11"),
        make_event(label="early", date="1991-08"),
        make_event(label="middle", date="2017-06"),
    ]

    artifact = to_artifact(events)

    assert [event["date"] for event in artifact["events"]] == ["1991-08", "2017-06", "2022-11"]


@pytest.mark.unit
def test_artifact_orders_same_month_by_weight() -> None:
    # Labels run against weight order so that sorting by label alone cannot pass.
    events = [
        make_event(label="a minor event", date="2007-04", weight=2),
        make_event(label="z major event", date="2007-04", weight=5),
    ]

    artifact = to_artifact(events)

    assert [event["label"] for event in artifact["events"]] == ["z major event", "a minor event"]


@pytest.mark.unit
def test_artifact_reports_the_count() -> None:
    artifact = to_artifact([make_event(label="a"), make_event(label="b")])

    assert artifact["count"] == 2


@pytest.mark.unit
def test_artifact_of_nothing_is_empty() -> None:
    assert to_artifact([]) == {"count": 0, "events": []}


@pytest.mark.unit
def test_artifact_is_json_serialisable() -> None:
    artifact = to_artifact(load_events(EVENTS_PATH))

    restored = json.loads(json.dumps(artifact))

    assert restored["count"] == len(RAW_EVENTS)
    assert isinstance(restored["events"][0]["scope"], list)


@pytest.mark.unit
def test_artifact_preserves_every_field() -> None:
    (entry,) = to_artifact([make_event()])["events"]

    assert entry == VALID_ENTRY
