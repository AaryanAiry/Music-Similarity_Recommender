from typing import List

def format_song_list(df, max_items: int = 5) -> str:
    """
    Returns a formatted string of songs for display.
    """
    output = []
    for i, row in df.head(max_items).iterrows():
        output.append(f"{row['track_name']} by {row['artists']} (Genre: {row['track_genre']}) | Score: {row['similarity_score']:.3f}")
    return "\n".join(output)
