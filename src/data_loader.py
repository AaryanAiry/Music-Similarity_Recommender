import os
import pandas as pd


RAW_PATH = "data/raw/spotify_tracks.csv"
PROCESSED_PATH = "data/processed/spotify_tracks_clean.csv"


def load_raw():

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
   
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

   
    df = df.drop_duplicates()

    df = df.dropna(subset=["track_id", "track_genre"])

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df["track_genre"] = df["track_genre"].str.strip().str.lower()

    return df


def save_processed(df: pd.DataFrame):

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned dataset saved to {PROCESSED_PATH}")


def load_processed() -> pd.DataFrame:

    if os.path.exists(PROCESSED_PATH):
        print("Loading already processed dataset...")
        return pd.read_csv(PROCESSED_PATH)
    
    print("Processed dataset not found. Cleaning raw data...")
    raw_df = load_raw()
    clean_df = clean_data(raw_df)
    save_processed(clean_df)
    return clean_df


if __name__ == "__main__":
    df = load_processed()
    print(df.head())
    print(df.shape)
