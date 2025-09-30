import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import unidecode

class SongRecommender:
    def __init__(self, csv_path: str):
        # Load dataset
        self.df = pd.read_csv(csv_path)

        # Normalize track and artist names for matching
        self.df['track_name_clean'] = self.df['track_name'].apply(lambda x: unidecode.unidecode(str(x).lower().strip()))
        self.df['artist_name_clean'] = self.df['artists'].apply(lambda x: unidecode.unidecode(str(x).lower().strip()))

        # Features used for similarity
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
        self.nn_model = NearestNeighbors(n_neighbors=11, metric="cosine")  # extra neighbors to account for duplicates
        self.nn_model.fit(self.features_scaled)

        # Map cleaned song name to index (keep list for multiple matches)
        self.song_to_index = self.df.groupby('track_name_clean').apply(lambda x: x.index.tolist()).to_dict()

    def get_similar_songs(self, song_input: str, n: int = 5):
        """
        Returns a DataFrame of top `n` most similar songs to the given song_input.
        Supports optional 'Song Name by Artist'.
        """
        song_input_clean = unidecode.unidecode(song_input.lower().strip())

        # Check for "by" to separate artist
        if ' by ' in song_input_clean:
            song_part, artist_part = song_input_clean.split(' by ')
            song_part = song_part.strip()
            artist_part = artist_part.strip()
            # Filter dataset by song and artist
            matches = self.df[
                (self.df['track_name_clean'] == song_part) &
                (self.df['artist_name_clean'] == artist_part)
            ]
        else:
            # Only song name
            indices = self.song_to_index.get(song_input_clean)
            if indices is None:
                raise ValueError(f"Song '{song_input}' not found in dataset.")
            matches = self.df.loc[indices]

        if matches.empty:
            raise ValueError(f"Song '{song_input}' not found in dataset.")

        # Take the first match if multiple exist
        idx = matches.index[0]

        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors([self.features_scaled[idx]], n_neighbors=n*2)  # extra to remove duplicates
        similar_indices = indices[0][1:]  # exclude the song itself

        # Prepare similar songs DataFrame
        similar_songs = self.df.iloc[similar_indices][["track_name", "artists", "track_genre"]].copy()
        similar_songs["similarity_score"] = 1 - distances[0][1:]

        # Remove duplicates based on track + artist
        similar_songs = similar_songs.drop_duplicates(subset=['track_name', 'artists']).head(n)

        # Add display name column
        similar_songs['display_name'] = similar_songs['track_name'] + " by " + similar_songs['artists']

        return similar_songs.reset_index(drop=True)
