"""arXiv category taxonomy: aliases, display names, and dated structural events.

Two facts drive this module:

1. arXiv canonicalises certain category codes *retroactively*. A query for the
   non-canonical member of an alias pair returns zero papers for all time, so a
   naive tool renders a permanently flat line labelled identically to a real one.

2. Categories are born, split, and renamed. A category's first months are a
   cliff, not a trend, because existing papers are never reclassified into it.
   Series are truncated before ``valid_from`` and the birth is annotated.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Non-canonical code -> canonical code. arXiv rewrites these in returned
#: metadata, so the left-hand side yields zero results for every date range.
CATEGORY_ALIASES: dict[str, str] = {
    "cs.SY": "eess.SY",
    "stat.TH": "math.ST",
    "q-fin.EC": "econ.GN",
    "cs.NA": "math.NA",
    "cs.IT": "math.IT",
    "math.MP": "math-ph",
    # Legacy archive-level renames.
    "solv-int": "nlin.SI",
    "patt-sol": "nlin.PS",
    "adap-org": "nlin.AO",
    "chao-dyn": "nlin.CD",
    "comp-gas": "nlin.CG",
    "mtrl-th": "cond-mat.mtrl-sci",
    "supr-con": "cond-mat.supr-con",
    "funct-an": "math.FA",
    "alg-geom": "math.AG",
    "dg-ga": "math.DG",
    "q-alg": "math.QA",
    "cmp-lg": "cs.CL",
    "acc-phys": "physics.acc-ph",
    "ao-sci": "physics.ao-ph",
    "atom-ph": "physics.atom-ph",
    "bayes-an": "physics.data-an",
    "chem-ph": "physics.chem-ph",
    "plasm-ph": "physics.plasm-ph",
}


def canonical(code: str) -> str:
    """Resolve a category code to its canonical form."""
    return CATEGORY_ALIASES.get(code, code)


@dataclass(frozen=True)
class CategoryEvent:
    """A dated structural change to the arXiv taxonomy."""

    code: str
    date: str
    kind: str  # created | split_from | renamed | aliased_to
    note: str
    source_url: str = "https://info.arxiv.org/new/index.html"


#: Dated births and splits. Series are truncated before these dates, and the
#: discontinuity is drawn as an annotation rather than left to read as growth.
CATEGORY_EVENTS: tuple[CategoryEvent, ...] = (
    CategoryEvent("astro-ph.CO", "2009-01", "split_from",
                  "astro-ph split into six subcategories; prior papers not reclassified"),
    CategoryEvent("astro-ph.GA", "2009-01", "split_from", "astro-ph subdivision"),
    CategoryEvent("astro-ph.EP", "2009-01", "split_from", "astro-ph subdivision"),
    CategoryEvent("astro-ph.HE", "2009-01", "split_from", "astro-ph subdivision"),
    CategoryEvent("astro-ph.IM", "2009-01", "split_from", "astro-ph subdivision"),
    CategoryEvent("astro-ph.SR", "2009-01", "split_from", "astro-ph subdivision"),
    CategoryEvent("eess.SP", "2017-09", "created", "eess archive launched"),
    CategoryEvent("eess.IV", "2017-09", "created", "eess archive launched"),
    CategoryEvent("eess.AS", "2017-09", "created", "eess archive launched"),
    CategoryEvent("eess.SY", "2017-09", "created", "eess archive launched; absorbed cs.SY"),
    CategoryEvent("econ.EM", "2017-09", "created", "econ archive launched"),
    CategoryEvent("econ.GN", "2019-09", "created", "absorbed q-fin.EC"),
    CategoryEvent("econ.TH", "2017-09", "created", "econ archive launched"),
    CategoryEvent("q-fin.CP", "2008-12", "created", "q-fin archive launched"),
    CategoryEvent("stat.ML", "2007-04", "created", "stat archive launched"),
    CategoryEvent("cs.LG", "1998-03", "created", "cs archive expansion"),
)

#: Top-level groups used for the landing-page landscape view.
GROUPS: dict[str, str] = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "hep": "High Energy Physics",
    "gr-qc": "General Relativity",
    "quant-ph": "Quantum Physics",
    "nucl": "Nuclear",
    "nlin": "Nonlinear Sciences",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "eess": "Electrical Engineering",
    "econ": "Economics",
    "math-ph": "Mathematical Physics",
}


def group_of(code: str) -> str:
    """Return the top-level group key for a canonical category code."""
    if code.startswith(("hep-", "nucl-")):
        return code.split("-")[0]
    if "." in code:
        return code.split(".")[0]
    return code
