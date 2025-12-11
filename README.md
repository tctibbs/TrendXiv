# TrendXiv

Google Trends for arXiv - visualize publication trends across scientific categories and keywords.

## Overview

TrendXiv is a Streamlit dashboard that lets you explore and compare publication trends on arXiv. Track the rise of Machine Learning, compare NLP vs Computer Vision, or search for specific keywords like "transformer" or "diffusion" to see their growth over time.

## Features

- **Category Trends**: Compare publication volume across arXiv categories (cs.LG, cs.AI, cs.CL, etc.)
- **Keyword Search**: Track mentions of specific terms in paper titles and abstracts
- **Normalization**: Toggle between absolute counts and relative share (%)
- **Smoothing**: Apply 3-month or 6-month moving averages to reduce noise
- **Multiple Chart Types**: Line, area, and bar charts
- **Data Persistence**: SQLite database caches fetched papers locally

## Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/TrendXiv.git
cd TrendXiv

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Usage

### Running Locally

```bash
streamlit run src/main.py
```

The app will open in your browser at `http://localhost:8501`.

### First Time Setup

1. Select categories from the sidebar (default: cs.LG, cs.AI, cs.CL)
2. Click "Fetch Latest Data" to download papers from arXiv
3. Wait for the data to load (this may take a few minutes)
4. Explore the trends!

### Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `TRENDXIV_DATABASE_PATH` | `data/trendxiv.db` | SQLite database location |
| `TRENDXIV_ARXIV_DELAY_SECONDS` | `3.0` | Delay between API requests |
| `TRENDXIV_DEFAULT_LOOKBACK_YEARS` | `5` | Default date range |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

### Linting

```bash
# Check code style
ruff check src tests

# Auto-fix issues
ruff check --fix src tests

# Format code
ruff format src tests
```

### Project Structure

```
TrendXiv/
├── src/
│   ├── api/           # arXiv API client
│   ├── data/          # Data processing (aggregation, normalization)
│   ├── repository/    # Database access layer
│   ├── visualization/ # Plotly chart builders
│   ├── common/        # Config and constants
│   └── main.py        # Streamlit application
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── data/              # SQLite database (gitignored)
└── pyproject.toml     # Project configuration
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set the main file path to `src/main.py`
5. Deploy!

Note: On Streamlit Cloud, the database will reset on each deployment. For persistent data, consider using a cloud database.

## Data Sources

- **arXiv API**: https://info.arxiv.org/help/api/index.html
- Rate limit: 3 seconds between requests (enforced automatically)
- Data updates: arXiv updates once daily

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request
