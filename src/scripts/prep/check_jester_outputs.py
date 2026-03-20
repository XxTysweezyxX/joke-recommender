import pandas as pd
from joke_reco.paths import PROCESSED_DIR

def main():
    edges = pd.read_csv(PROCESSED_DIR / "jester_edges_clean.csv")
    jokes = pd.read_csv(PROCESSED_DIR / "jester_jokes_clean.csv")

    print("EDGES shape:", edges.shape)
    print("EDGES columns:", list(edges.columns))
    print(edges.head(3))

    print("\nJOKES shape:", jokes.shape)
    print("JOKES columns:", list(jokes.columns))
    print(jokes.head(3))

if __name__ == "__main__":
    main()
