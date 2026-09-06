"""Tests for human-readable category naming."""

import pytest

from src.common.categories import (
    CATEGORY_NAMES,
    LEGACY_ARCHIVES,
    SHORT_NAMES,
    display_name,
    short_name,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("cs.LG", "Machine Learning"),
        ("cs.CV", "Computer Vision and Pattern Recognition"),
        ("hep-th", "High Energy Physics - Theory"),
        ("astro-ph", "Astrophysics"),
    ],
)
def test_display_name_matches_the_official_taxonomy(code, expected):
    assert display_name(code) == expected


@pytest.mark.unit
def test_two_categories_share_a_name_so_the_code_cannot_be_dropped():
    """cs.LG and stat.ML are both officially "Machine Learning".

    This is the whole reason the interface pairs a name with its code rather
    than replacing one with the other.
    """
    assert display_name("cs.LG") == display_name("stat.ML")
    assert display_name("cs.LG") == "Machine Learning"


@pytest.mark.unit
def test_unknown_codes_fall_back_to_the_code():
    assert display_name("xx.ZZ") == "xx.ZZ"
    assert short_name("xx.ZZ") == "xx.ZZ"


@pytest.mark.unit
def test_short_names_fall_back_to_the_full_name():
    """A category without an abbreviation still gets a readable label."""
    assert short_name("math.AG") == display_name("math.AG")


@pytest.mark.unit
def test_short_names_stay_short_enough_for_a_chart_label():
    for code in SHORT_NAMES:
        assert len(short_name(code)) <= 24, code


@pytest.mark.unit
def test_short_names_are_abbreviations_not_renamings():
    """A reader who knows the code must still recognise the field.

    Every abbreviation has to share a meaningful word with the official name, or
    be a well-known initialism for it.
    """
    initialisms = {"cs.AI", "cs.HC", "hep-ex", "hep-lat", "hep-ph", "hep-th", "cs.DS", "cs.LO"}

    def words(text: str) -> set[str]:
        return set(text.lower().replace(",", "").replace("-", " ").replace(".", "").split())

    for code, short in SHORT_NAMES.items():
        if code in initialisms:
            continue
        official = words(CATEGORY_NAMES[code])
        # A word matches if it appears in the official name or abbreviates one of
        # its words, so "Astro" counts against "Astrophysical".
        shared = any(
            any(word == full or full.startswith(word) for full in official)
            for word in words(short)
        )
        assert shared, (code, short, sorted(official))


@pytest.mark.unit
def test_pre_split_archives_are_named_and_flagged():
    """These codes hold papers filed before the archive was subdivided."""
    for code in LEGACY_ARCHIVES:
        assert code in CATEGORY_NAMES
        assert display_name(code) != code


@pytest.mark.unit
def test_every_short_name_belongs_to_a_known_category():
    assert set(SHORT_NAMES) <= set(CATEGORY_NAMES)
