# Joke Recommender (Dissertation 2025/26)

A personalised joke recommender system.  
Current baseline: **TF-IDF (content-based)** with Top-K recommendations and evaluation metrics.

## Project Structure

- `src/joke_reco/` — reusable library code (models, metrics, data prep helpers)
- `src/scripts/` — runnable scripts (like “main” entry points)
- `data/raw/` — place datasets here (ignored by git)
- `data/processed/` — generated outputs (ignored by git)
- `notebooks/` — optional notebooks / experiments (if used)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt