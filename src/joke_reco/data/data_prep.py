from __future__ import annotations

"""
Prepares the Jester dataset for the recommendation pipeline.
Loads raw ratings and joke text, applies basic cleaning, and saves processed CSV files.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# 1. Raw ratings loading
# Loads the raw Jester interaction data from CSV.
# ---------------------------------------------------------
def load_jester_edges(edges_csv: Path) -> pd.DataFrame:
    # Load the raw ratings file
    df = pd.read_csv(edges_csv)

    return df


# ---------------------------------------------------------
# 2. Raw joke text loading
# Loads the raw Jester joke text data from CSV.
# ---------------------------------------------------------
def load_jester_jokes(jokes_csv: Path) -> pd.DataFrame:
    # Load the raw joke text file
    df = pd.read_csv(jokes_csv)

    return df


# ---------------------------------------------------------
# 3. Ratings cleaning
# Applies basic cleaning to the interaction data.
# ---------------------------------------------------------
def basic_clean_edges(edges: pd.DataFrame) -> pd.DataFrame:
    # Remove rows with missing values
    edges = edges.dropna().copy()

    return edges


# ---------------------------------------------------------
# 4. CSV saving helper
# Saves a dataframe to CSV and creates the folder if needed.
# ---------------------------------------------------------
def save_df(df: pd.DataFrame, out_path: Path) -> None:
    # Create the output folder if it does not exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataframe to CSV
    df.to_csv(out_path, index=False)