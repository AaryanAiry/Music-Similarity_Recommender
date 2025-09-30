import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

class SongRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [
            "danceability", "energy", "valence", "tempo",
            "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness"
        ]

        # Preprocess features
        self.features = self.df[self.feature_cols].fillna(0)
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)

        # Fit NearestNeighbors model
        self.nn_model = NearestNeighbors(n_neighbors=6, metric="cosine")  # 6 because first is the song itself
        self.nn_model.fit(self.features_scaled)

        # Map song name to index
        self.song_to_index = pd.Series(self.df.index, index=self.df["track_name"].str.lower()).to_dict()

    def get_similar_songs(self, song_name: str, n: int = 5):
        """
        Returns a DataFrame of the top `n` most similar songs to `song_name`.
        """
        idx = self.song_to_index.get(song_name.lower())
        if idx is None:
            raise ValueError(f"Song '{song_name}' not found in dataset.")

        distances, indices = self.nn_model.kneighbors([self.features_scaled[idx]], n_neighbors=n+1)
        similar_indices = indices[0][1:]  # exclude the song itself
        similar_songs = self.df.iloc[similar_indices][["track_name", "artists", "track_genre"]].copy()
        similar_songs["similarity_score"] = 1 - distances[0][1:]  # cosine similarity
        return similar_songs.reset_index(drop=True)
