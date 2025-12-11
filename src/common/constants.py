"""arXiv category definitions and constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    """arXiv category with display name."""

    code: str
    name: str
    parent: str | None = None


ARXIV_CATEGORIES: dict[str, Category] = {
    # Computer Science - Machine Learning & AI
    "cs.LG": Category("cs.LG", "Machine Learning", "cs"),
    "cs.AI": Category("cs.AI", "Artificial Intelligence", "cs"),
    "cs.CL": Category("cs.CL", "Computation & Language (NLP)", "cs"),
    "cs.CV": Category("cs.CV", "Computer Vision", "cs"),
    "cs.NE": Category("cs.NE", "Neural & Evolutionary Computing", "cs"),
    "cs.RO": Category("cs.RO", "Robotics", "cs"),
    "cs.IR": Category("cs.IR", "Information Retrieval", "cs"),
    # Computer Science - Systems & Theory
    "cs.DC": Category("cs.DC", "Distributed Computing", "cs"),
    "cs.SE": Category("cs.SE", "Software Engineering", "cs"),
    "cs.PL": Category("cs.PL", "Programming Languages", "cs"),
    "cs.CR": Category("cs.CR", "Cryptography & Security", "cs"),
    "cs.DS": Category("cs.DS", "Data Structures & Algorithms", "cs"),
    # Statistics
    "stat.ML": Category("stat.ML", "Machine Learning (Statistics)", "stat"),
    "stat.TH": Category("stat.TH", "Statistics Theory", "stat"),
    "stat.ME": Category("stat.ME", "Methodology", "stat"),
    # Mathematics
    "math.OC": Category("math.OC", "Optimization & Control", "math"),
    "math.ST": Category("math.ST", "Statistics Theory", "math"),
    "math.NA": Category("math.NA", "Numerical Analysis", "math"),
    # Electrical Engineering
    "eess.SP": Category("eess.SP", "Signal Processing", "eess"),
    "eess.IV": Category("eess.IV", "Image & Video Processing", "eess"),
    # Quantitative Biology & Finance
    "q-bio.QM": Category("q-bio.QM", "Quantitative Methods (Biology)", "q-bio"),
    "q-fin.ST": Category("q-fin.ST", "Statistical Finance", "q-fin"),
    # Physics
    "physics.comp-ph": Category("physics.comp-ph", "Computational Physics", "physics"),
}

PARENT_CATEGORIES: dict[str, str] = {
    "cs": "Computer Science",
    "stat": "Statistics",
    "math": "Mathematics",
    "eess": "Electrical Engineering",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "physics": "Physics",
}

DEFAULT_CATEGORIES: list[str] = [
    "cs.LG",
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "stat.ML",
]

CHART_COLORS: list[str] = [
    "#636EFA",  # Blue
    "#EF553B",  # Red
    "#00CC96",  # Green
    "#AB63FA",  # Purple
    "#FFA15A",  # Orange
    "#19D3F3",  # Cyan
    "#FF6692",  # Pink
    "#B6E880",  # Light Green
    "#FF97FF",  # Magenta
    "#FECB52",  # Yellow
]

SMOOTHING_OPTIONS: dict[str, int] = {
    "None": 0,
    "3-month": 3,
    "6-month": 6,
}
