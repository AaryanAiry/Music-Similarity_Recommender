import pandas as pd
import time
import json
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from spotipy.exceptions import SpotifyException


df = pd.read_csv("data/processed/spotify_tracks_clean.csv")


cache_file = "data/processed/spotify_year_cache.json"
try:
    with open(cache_file, "r") as f:
        year_cache = json.load(f)
except FileNotFoundError:
    year_cache = {}

client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"

sp = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

def get_release_year_safe(track_id):
    if track_id in year_cache:
        return year_cache[track_id]
    try:
        track = sp.track(track_id)
        release_date = track["album"]["release_date"]
        year = int(release_date.split("-")[0])
        year_cache[track_id] = year
        return year
    except SpotifyException as e:
        if e.http_status == 429:  # Rate limit hit
            retry_after = int(e.headers.get("Retry-After", 5))
            print(f"Rate limit hit. Sleeping {retry_after} seconds...")
            time.sleep(retry_after)
            return get_release_year_safe(track_id)
        else:
            year_cache[track_id] = None
            return None
    except:
        year_cache[track_id] = None
        return None

cooldown = 0.12  
years = []

for track_id in tqdm(df["track_id"]):
    year = get_release_year_safe(track_id)
    years.append(year)
    time.sleep(cooldown)


with open(cache_file, "w") as f:
    json.dump(year_cache, f, indent=2)


df["year"] = years
df.to_csv("data/processed/spotify_tracks_with_year.csv", index=False)
print(" Year column added and CSV saved.")
