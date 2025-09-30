import pandas as pd
from src.data_loader import load_processed


def avg_features_by_genre(df: pd.DataFrame, features=None) -> pd.DataFrame:
    # Compute average audio features per genre.
    if features is None:
        features = [
            "danceability", "energy", "valence", "tempo",
            "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness"
        ]
    avg_df = df.groupby("track_genre")[features].mean()
    # Sort by the first feature in the list (requested feature)
    return avg_df.sort_values(features[0], ascending=False)


def top_genres(df: pd.DataFrame, n=10) -> pd.DataFrame:
    # Get top N genres by track count overall, return a DataFrame compatible with Seaborn
    counts = df["track_genre"].value_counts().head(n)
    tg_df = counts.reset_index()
    tg_df.columns = ["track_genre", "count"]  # rename columns correctly
    return tg_df


if __name__ == "__main__":
    df = load_processed()

    print("Top Genres:")
    print(top_genres(df, n=10))

    print("\nAverage Features by Genre:")
    print(avg_features_by_genre(df).head())
