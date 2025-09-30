import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.analysis import avg_features_by_genre, top_genres
from src.data_loader import load_processed

FIG_PATH = "reports/figures"


def save_fig(fig, filename: str):
    # Save figure to reports/figures/ folder
    os.makedirs(FIG_PATH, exist_ok=True)
    filepath = os.path.join(FIG_PATH, filename)
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Saved figure: {filepath}")


def plot_top_genres(df: pd.DataFrame, n: int = 10):
    # Plot bar chart of top N genres by track count
    tg = top_genres(df, n=n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="count", y="track_genre", data=tg, ax=ax, palette="rocket")
    ax.set_title(f"Top {n} Genres by Track Count")
    save_fig(fig, "top_genres.png")


def plot_avg_features_by_genre(df: pd.DataFrame, feature: str = "danceability", top_n: int = 10):
    # Plot bar chart of average feature value by genre
    avg_features = avg_features_by_genre(df, features=[feature])
    avg_features_sorted = avg_features.sort_values(feature, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=feature,
        y=avg_features_sorted.index,  # genre index
        data=avg_features_sorted,
        ax=ax,
        palette="viridis"
    )
    ax.set_title(f"Top {top_n} Genres by Average {feature.capitalize()}")
    save_fig(fig, f"avg_{feature}_by_genre.png")


if __name__ == "__main__":
    df = load_processed()
    plot_top_genres(df, n=10)
    plot_avg_features_by_genre(df, feature="danceability", top_n=10)
