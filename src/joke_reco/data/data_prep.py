from __future__ import annotations

"""
Data preparation helpers for the Jester dataset.

Purpose:
- Load raw Jester ratings data
- Load raw joke text data
- Apply basic cleaning to the ratings data
- Save processed outputs for later stages of the pipeline
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Helper: load raw Jester interaction data
# ---------------------------------------------------------
def load_jester_edges(edges_csv: Path) -> pd.DataFrame:
    """
    Load the long-format Jester ratings / edges CSV.

    Expected columns usually look like:
    - user_id
    - joke_id
    - rating
    """
    df = pd.read_csv(edges_csv)
    return df


# ---------------------------------------------------------
# Helper: load raw Jester joke text data
# ---------------------------------------------------------
def load_jester_jokes(jokes_csv: Path) -> pd.DataFrame:
    """
    Load the jokes text CSV.

    Expected columns usually look like:
    - joke_id
    - joke_text
    """
    df = pd.read_csv(jokes_csv)
    return df


# ---------------------------------------------------------
# Helper: apply basic cleaning to ratings data
# ---------------------------------------------------------
def basic_clean_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleanup to the interaction data.

    Current cleaning step:
    - remove rows with missing values

    This keeps the dataset cleaner before saving it
    as the processed interaction file.
    """
    edges = edges.dropna().copy()
    return edges


# ---------------------------------------------------------
# Helper: save dataframe to CSV
# ---------------------------------------------------------
def save_df(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a dataframe to CSV.

    If the output folder does not exist yet,
    create it first.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)