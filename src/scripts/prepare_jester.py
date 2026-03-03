from joke_reco.paths import RAW_DIR, PROCESSED_DIR
from joke_reco.data.data_prep import load_jester_edges, load_jester_jokes, basic_clean_edges, save_df


def main() -> None:
    # Update these filenames if yours differ
    edges_path = RAW_DIR / "jester_edges_long.csv"
    jokes_path = RAW_DIR / "jester_jokes.csv"

    edges = load_jester_edges(edges_path)
    jokes = load_jester_jokes(jokes_path)

    edges_clean = basic_clean_edges(edges)

    save_df(edges_clean, PROCESSED_DIR / "jester_edges_clean.csv")
    save_df(jokes, PROCESSED_DIR / "jester_jokes_clean.csv")

    print("Saved:")
    print(" -", PROCESSED_DIR / "jester_edges_clean.csv")
    print(" -", PROCESSED_DIR / "jester_jokes_clean.csv")


if __name__ == "__main__":
    main()