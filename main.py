from src.data_loader import load_processed
from src.visualization import (
    plot_top_genres,
    plot_avg_features_by_genre,
)


def run_pipeline():
    print("Loading processed dataset...")
    df = load_processed()
    print(f"Dataset loaded with shape {df.shape}")

    print("\nGenerating visualizations...")

    plot_top_genres(df, n=10)
    plot_avg_features_by_genre(df, feature="danceability", top_n=10)
    plot_avg_features_by_genre(df, feature="tempo", top_n=10)
    plot_avg_features_by_genre(df, feature="energy", top_n=10)

    print("\nPipeline finished! Check reports/figures/ for saved charts.")


if __name__ == "__main__":
    run_pipeline()
