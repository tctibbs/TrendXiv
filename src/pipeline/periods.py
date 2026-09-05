"""The shared monthly period axis.

Every series in every artifact is a dense array aligned to one axis, so the
frontend never ships dates alongside values. Index 0 is ``AXIS_START``.
"""

from __future__ import annotations

from datetime import date

#: arXiv's first month of operation. The snapshot contains a handful of
#: backdated submissions stamped as early as 1986; they are real metadata but
#: they are not arXiv activity, so the public axis starts here.
AXIS_START = "1991-07"


def month_range(start: str, end: str) -> list[str]:
    """Return every ``YYYY-MM`` label from start to end inclusive.

    Args:
        start: First month, ``YYYY-MM``.
        end: Last month, ``YYYY-MM``.

    Returns:
        Ordered list of month labels.
    """
    sy, sm = (int(part) for part in start.split("-"))
    ey, em = (int(part) for part in end.split("-"))
    out: list[str] = []
    year, month = sy, sm
    while (year, month) <= (ey, em):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def previous_month(period: str) -> str:
    """Return the month label immediately before ``period``."""
    year, month = (int(part) for part in period.split("-"))
    return f"{year - 1:04d}-12" if month == 1 else f"{year:04d}-{month - 1:02d}"


def completeness(max_submission: date) -> tuple[str, str]:
    """Derive the provisional boundary from the snapshot's latest submission.

    A snapshot taken mid-month contains a partial final month. Plotting it
    unmarked produces an apparent collapse -- arXiv's own 2026-09 stands at
    4,203 against 31,173 for 2026-08 -- and a trailing moving average smooths
    that cliff into a *believable* decline.

    Args:
        max_submission: Latest v1 submission date present in the snapshot.

    Returns:
        Tuple of (data_complete_through, provisional_from) month labels.
    """
    provisional_from = f"{max_submission.year:04d}-{max_submission.month:02d}"
    return previous_month(provisional_from), provisional_from
