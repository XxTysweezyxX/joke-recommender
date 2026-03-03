from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_jester_edges(edges_csv: Path) -> pd.DataFrame:
    """
    Load the long-format Jester ratings/edges CSV.
    Expected columns usually look like: user_id, joke_id, rating (names may vary).
    """
    df = pd.read_csv(edges_csv)
    return df


def load_jester_jokes(jokes_csv: Path) -> pd.DataFrame:
    """
    Load jokes text CSV (e.g., joke_id + joke_text).
    """
    df = pd.read_csv(jokes_csv)
    return df


def basic_clean_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleanup: drop missing values, ensure ids are ints, etc.
    (We will adapt this to match your exact columns from the notebook.)
    """
    edges = edges.dropna().copy()
    return edges


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a dataframe to CSV, creating directories if needed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)