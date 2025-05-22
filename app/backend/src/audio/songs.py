import os
import json
import subprocess
import pandas as pd
from utils.constants import CACHE_DIR, SONGS_DIR, SONGS_DATASET_FILE, SONGS_METADATA_FILE, SPOTIFY_PLAYLIST_URL
from utils.types import ServerState, current_state
from utils.logger_config import logger

def initialize_songs():
  current_state['state'] = ServerState.DOWNLOADING_SONGS
  download_songs()
  current_state['state'] = ServerState.DOWNLOADING_METADATA 
  download_metadata()
  current_state['state'] = ServerState.PROCESSING_SONGS
  process_songs()

def download_songs():
  if len(os.listdir(SONGS_DIR)) == 0:
    logger.info("The songs directory is empty. Downloading songs...")

    # Download songs
    subprocess.run(
      ["spotdl", SPOTIFY_PLAYLIST_URL, "--bitrate", "96k"],
      check=True,
      cwd=SONGS_DIR,
    )
  else:
    logger.info("The songs directory is not empty. Skipping download.")

def download_metadata():
  if not os.path.exists(SONGS_METADATA_FILE) or os.path.getsize(SONGS_METADATA_FILE) == 0:
    logger.info("The songs metadata file does not exist or its empty. Downloading metadata...")

    # Download metadata
    subprocess.run(
      ["spotdl", "save", SPOTIFY_PLAYLIST_URL, "--save-file", SONGS_METADATA_FILE.split("/")[-1]], 
      check=True,
      cwd=CACHE_DIR,
    )
  else:
    logger.info("The songs metadata file already exists. Skipping download.")

def process_songs():
  if not os.path.exists(SONGS_DATASET_FILE):
    logger.info("The songs dataset file does not exist. Processing songs...")

    with open(SONGS_METADATA_FILE, "r", encoding="utf-8") as f:
      metadata = json.load(f)

    data = []

    for i, song in enumerate(os.listdir(SONGS_DIR)):
      song_name = os.path.splitext(song)[0]
      song_ext = os.path.splitext(song)[1]
      song_path = os.path.join(SONGS_DIR, song) 

      if (song_ext != ".mp3" and song_ext != ".m4a"):
        logger.warning(f"Skipping {song} due to unsupported file extension.")
        continue  

      song_data = next((item for item in metadata if item["name"] in song_name), None)  

      if song_data is None:
        logger.warning(f"Song data not found for {song_name}. Skipping...")
        continue

      output_path = (f"{i}{song_ext}")

      data.append({
        "id": i,
        "path": output_path,  
        "name": song_data["name"],
        "artist": song_data["artist"],
        "genres": song_data["genres"],
      })

      # Rename the song file
      os.rename(
        song_path,
        os.path.join(SONGS_DIR, output_path),
      )
    
    df = pd.DataFrame(data)
    df.to_csv(SONGS_DATASET_FILE, index=False)
  else:
    logger.info("The songs dataset file already exists. Skipping processing.")

if __name__ == "__main__":
  initialize_songs()