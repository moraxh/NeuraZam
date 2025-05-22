import os
import torch

CACHE_DIR = "./cache"
SONGS_DIR = f"{CACHE_DIR}/songs"
SEGMENTS_DIR = f"{CACHE_DIR}/segments"  
SPECTOGRAMS_DIR = f"{CACHE_DIR}/spectograms"

ASSETS_DIR = "./assets"
BACKGROUND_NOISE_DIR = f"{ASSETS_DIR}/background_noises"

SONGS_STATS_FILE = f"{CACHE_DIR}/songs_stats.json"
SONGS_METADATA_FILE = f"{CACHE_DIR}/songs_metadata.spotdl"
SONGS_DATASET_FILE = f"{CACHE_DIR}/songs_info.csv"

DATASET_FILE = f"{CACHE_DIR}/dataset.csv"
EMBEDDINGS_FILE = f"{CACHE_DIR}/embeddings.npz" 
EMBEDDINGS_PLOT_FILE = f"{CACHE_DIR}/embeddings_plot.png"
TRAINED_MODEL_FILE = f"{CACHE_DIR}/trained_model.pt"

# SPOTIFY_PLAYLIST_URL=os.getenv("SPOTIFY_PLAYLIST_URL", "https://open.spotify.com/playlist/1Y0Qk1K1DEMXeKgvjjnN7m?si=80a2a297dded480b")
SPOTIFY_PLAYLIST_URL=os.getenv("SPOTIFY_PLAYLIST_URL", "https://open.spotify.com/playlist/34NbomaTu7YuOYnky8nLXL?si=4bf54104cf4c480c")

TARGET_SAMPLE_RATE = 32000
SEGMENT_DURATION = 3 # seconds

# CNN model parameters  
BATCH_SIZE = 64
NUM_WORKERS = 2
SHUFFLE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create needed directories
def create_directories():
  os.makedirs(CACHE_DIR, exist_ok=True)
  os.makedirs(SONGS_DIR, exist_ok=True)
  os.makedirs(SPECTOGRAMS_DIR, exist_ok=True)
  os.makedirs(SEGMENTS_DIR, exist_ok=True)
  os.makedirs(ASSETS_DIR, exist_ok=True)  
  os.makedirs(BACKGROUND_NOISE_DIR, exist_ok=True)  

  if (len(os.listdir(BACKGROUND_NOISE_DIR)) == 0):
    raise Exception(f"Background noises not found in {BACKGROUND_NOISE_DIR}.")
  
create_directories()