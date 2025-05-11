import librosa
import os
import subprocess
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from utils.types import CACHE_PATH, ServerState

# Must be a valid Spotify public playlist url
SPOTIFY_PLAYLIST_URL = os.getenv("SPOTIFY_PLAYLIST_URL")
SONGS_DIR = f"{CACHE_PATH}/raw_songs"
SPECTOGRAMS_DIR = f"{CACHE_PATH}/spectograms"
DATASET_FILE = f"{CACHE_PATH}/songs.csv"

if not SPOTIFY_PLAYLIST_URL:
    raise ValueError("SPOTIFY_PLAYLIST_URL must be set in the environment variables")

def initialize_songs(current_state):
  current_state['state'] = ServerState.DOWNLOADING_SONGS
  download_songs()
  current_state['state'] = ServerState.DOWNLOADING_METADATA
  download_metadata()
  current_state['state'] = ServerState.PROCESSING_SONGS
  transform_songs()

def download_songs():
  """
  Download songs from the Spotify playlist URL using spotdl(https://github.com/spotDL/spotify-downloader)
  """

  logging.info("Checking if songs are already downloaded...")

  # Check if spotdl is installed
  try:
      import spotdl
  except ImportError:
      raise ImportError("spotdl is not installed. Please install it using 'pip install spotdl'")

  # Check if the songs directory exists, if not create it
  if not os.path.exists(SONGS_DIR):
    os.makedirs(SONGS_DIR)
    logging.info("Songs directory does not exist. Creating it...")
  
  # Check if the songs directory is empty
  if not os.listdir(SONGS_DIR):
    logging.info("Songs directory is empty. Downloading songs...")

    # Download songs from the playlist URL
    subprocess.run(
      ["spotdl", SPOTIFY_PLAYLIST_URL],
      check=True,
      cwd=SONGS_DIR,
    )

    logging.info("Songs downloaded successfully.")
  else:
    logging.info("Songs directory already exists. Skipping download...")
  
def download_metadata():
  """
  Download metadata from the Spotify playlist URL using spotdl(
  """

  if (not os.path.exists(f"{SONGS_DIR}/metadata.json")):
    logging.info("Metadata file does not exist. Downloading metadata...")
    subprocess.run(
      ["spotdl", "save", SPOTIFY_PLAYLIST_URL, "--save-file", "songs_metadata.spotdl"],
      check=True,
      cwd=CACHE_PATH,
    )
  else:
    logging.info("Metadata already exists. Skipping download...")

def process_and_save(chunk, sr, chunk_name, aug_name, song_name):
  S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=512, hop_length=512, n_mels=64)
  S_db = librosa.power_to_db(S, ref=np.max)
  S_db = (S_db - np.mean(S_db)) / np.std(S_db)
  file_name = f"{chunk_name}_{aug_name}.npz"
  file_path = os.path.join(SPECTOGRAMS_DIR, file_name)
  np.savez_compressed(file_path, S_db=S_db)  # Guardamos el espectrograma comprimido
  return {
    "file_name": file_name,
    "song_name": song_name,
    "chunk_id": chunk_name,
    "aug_name": aug_name
  }

def process_song(song):
  song_path = os.path.join(SONGS_DIR, song)
  song_name = os.path.splitext(song)[0]
  y, sr = librosa.load(song_path, sr=None)

  records = []

  # Separate the song into intervals
  chunk_duration = 3
  samples_per_chunk = chunk_duration * sr

  for i in range(0, len(y), samples_per_chunk):
    transformed_song_name = song_name.replace(" ", "_").lower()
    chunk_name = f"{transformed_song_name}_{i // samples_per_chunk}"
    chunk = y[i:i+samples_per_chunk]

    if len(chunk) == samples_per_chunk:
      # Original Chunk
      records.append(process_and_save(chunk, sr, chunk_name, "original", song_name))

      # Pitch Shift
      pitch_steps = np.random.randint(-2, 2)
      y_pitch = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=pitch_steps)
      records.append(process_and_save(y_pitch, sr, chunk_name, "pitch_shift", song_name))

      # Random Gain
      random_gain = np.random.uniform(0.5, 1.5)
      y_gain = chunk * random_gain
      records.append(process_and_save(y_gain, sr, chunk_name, "gain", song_name))

      # Random Noise
      noise = np.random.normal(0, 0.005, chunk.shape)
      y_noise = chunk + noise
      records.append(process_and_save(y_noise, sr, chunk_name, "noise", song_name))

  return records

def transform_songs():
  logging.info("Transforming songs into spectograms...")

  songs = os.listdir(SONGS_DIR)
  if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0:
    logging.info("Dataset already exists. Skipping transformation...")
    return

  os.makedirs(SPECTOGRAMS_DIR, exist_ok=True)

  # Filter out non-audio files
  songs = [s for s in songs if s.endswith(('.mp3', '.wav'))]

  records = []
  with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_song, songs))
    for r in results:
      records.extend(r)

  df = pd.DataFrame(records)
  df.to_csv(DATASET_FILE, index=False)

  logging.info("Songs transformed into spectograms successfully.")