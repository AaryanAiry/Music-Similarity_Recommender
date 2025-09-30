import streamlit as st
from recommender.model import SongRecommender
from recommender.utils import format_song_list

# Load recommender
recommender = SongRecommender("data/processed/spotify_tracks_clean.csv")

st.title("🎵 Spotify Song Similarity Recommender")
st.write("Enter a song name and get top 5 most similar tracks.")

song_input = st.text_input("Song Name")
n_results = st.slider("Number of results", 1, 10, 5)

if st.button("Get Similar Songs"):
    if song_input.strip() == "":
        st.warning("Please enter a song name.")
    else:
        try:
            similar_df = recommender.get_similar_songs(song_input, n=n_results)
            st.subheader(f"Top {n_results} songs similar to '{song_input}'")
            st.text(format_song_list(similar_df, max_items=n_results))
        except ValueError as e:
            st.error(str(e))
