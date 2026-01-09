# NBAMind

NBA player similarity and explanation system based on representation-level features from nba_api.
The goal is to move beyond raw box-score neighbors and build a style vector that captures
shooting geometry, touch/playmaking texture, and defensive activity, then explain similarity
with transparent feature drivers.

## Repository layout

```
.
├── api/                 # FastAPI service (future)
├── app/                 # Streamlit UI (future)
├── data/
│   ├── raw/             # Cached nba_api JSON payloads
│   ├── processed/       # Cleaned feature tables (parquet/csv)
│   └── external/        # Manual mappings or lookups
├── notebooks/           # Exploration and ablations
├── src/
│   └── nbamind/
│       ├── data/        # Ingestion and caching
│       ├── features/    # Feature engineering
│       ├── models/      # Similarity models
│       └── explanation/ # SHAP and narrative output
├── pyproject.toml
└── README.md
```

## Quickstart

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
    NOTE: if this gives issues, run this and repeat: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
pip install -e .
```

## Fetching data

The nba_api fetcher handles rate limiting, retries, and simple disk caching.
Raw payloads are stored under `data/raw/<EndpointName>/`.

```python
from nba_api.stats.endpoints import commonplayerinfo
from nbamind.data.fetcher import NBAApiFetcher

fetcher = NBAApiFetcher()
lebron = fetcher.fetch(commonplayerinfo, {"player_id": 2544})
```
