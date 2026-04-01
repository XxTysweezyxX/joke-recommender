from joke_reco.paths import RAW_DIR, PROCESSED_DIR
from joke_reco.data.data_prep import (
    load_jester_edges,
    load_jester_jokes,
    basic_clean_edges,
    save_df,
)

"""
Prepare the Jester dataset for the rest of the project.

Purpose:
- Load the raw Jester interaction file
- Load the raw Jester joke text file
- Apply basic cleaning to the interaction data
- Save cleaned outputs into the processed data folder
"""


# ---------------------------------------------------------
# Main: prepare and save processed Jester files
# ---------------------------------------------------------
def main() -> None:
    """
    Main preprocessing runner for the Jester dataset.

    This script:
    1) loads the raw ratings file
    2) loads the raw joke text file
    3) applies basic cleaning to the ratings data
    4) saves both processed files for later stages
    """

    # Raw input file paths
    # Update these filenames if your raw files are named differently
    edges_path = RAW_DIR / "jester_edges_long.csv"
    jokes_path = RAW_DIR / "jester_jokes.csv"

    # Load raw interaction data and raw joke text
    edges = load_jester_edges(edges_path)
    jokes = load_jester_jokes(jokes_path)

    # Apply basic cleaning to the interaction data
    edges_clean = basic_clean_edges(edges)

    # Save processed interaction data and processed joke text
    save_df(edges_clean, PROCESSED_DIR / "jester_edges_clean.csv")
    save_df(jokes, PROCESSED_DIR / "jester_jokes_clean.csv")

    # Print saved file locations so I can confirm the script worked
    print("Saved:")
    print(" -", PROCESSED_DIR / "jester_edges_clean.csv")
    print(" -", PROCESSED_DIR / "jester_jokes_clean.csv")


if __name__ == "__main__":
    main()