"""Tests for category alias resolution and grouping."""

import pytest

from src.common.taxonomy import CATEGORY_ALIASES, canonical, group_of


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("stat.TH", "math.ST"),
        ("cs.SY", "eess.SY"),
        ("q-fin.EC", "econ.GN"),
        ("cmp-lg", "cs.CL"),
    ],
)
def test_known_aliases_resolve(alias, expected):
    """These codes return zero papers from arXiv because it rewrites them retroactively."""
    assert canonical(alias) == expected


@pytest.mark.unit
def test_canonical_is_idempotent():
    """Resolving twice must not move a code again, or counts would be merged twice."""
    for alias in CATEGORY_ALIASES:
        once = canonical(alias)
        assert canonical(once) == once


@pytest.mark.unit
def test_no_alias_target_is_itself_an_alias():
    """A two-hop alias chain would silently drop a category's papers."""
    for target in CATEGORY_ALIASES.values():
        assert target not in CATEGORY_ALIASES


@pytest.mark.unit
@pytest.mark.parametrize(
    ("code", "group"),
    [("cs.LG", "cs"), ("hep-th", "hep"), ("nucl-ex", "nucl"),
     ("astro-ph.GA", "astro-ph"), ("quant-ph", "quant-ph"), ("math-ph", "math-ph")],
)
def test_group_of(code, group):
    assert group_of(code) == group
