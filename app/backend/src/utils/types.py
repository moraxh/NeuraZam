from enum import Enum

class ServerState(str, Enum):
  LOADING_SERVER = "loading_server",
  DOWNLOADING_SONGS = "downloading_songs",
  DOWNLOADING_METADATA = "downloading_metadata",
  PROCESSING_SONGS = "processing_songs",
  EXTRACTING_FEATURES = "extracting_features",
  AUGMENTING_SONGS = "augmenting_songs",
  LOADING_MODEL = "loading_model",
  TRAINING_MODEL = "training_model",
  STORING_EMBEDDINGS = "storing_embeddings",
  READY = "ready"

current_state = {
  'state': ServerState.LOADING_SERVER,
  'data': []
}