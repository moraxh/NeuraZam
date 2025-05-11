import os
from enum import Enum

# Create cache path if not exists
CACHE_PATH = "cache"
os.makedirs(CACHE_PATH, exist_ok=True)

class ServerState(str, Enum):
  LOADING_SERVER = "loading_server",
  DOWNLOADING_SONGS = "downloading_songs",
  DOWNLOADING_METADATA = "downloading_metadata",
  PROCESSING_SONGS = "processing_songs",
  LOADING_MODEL = "loading_model",
  TRAINING_MODEL = "training_model",
  READY = "ready"
