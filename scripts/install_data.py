from datasets import load_dataset

dataset = load_dataset("maharshipandya/spotify-tracks-dataset")
df = dataset["train"].to_pandas()

# save into your project folder
df.to_csv("data/raw/spotify_tracks.csv", index=False)
