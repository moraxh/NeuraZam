import gc
import os
import torch
import logging
import subprocess
import torchaudio
import numpy as np
import pandas as pd
from utils.types import CACHE_PATH, ServerState
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch_audiomentations import AddBackgroundNoise, AddColoredNoise, Gain, PitchShift, Shift

device = "cuda" if torch.cuda.is_available() else "cpu"

# Must be a valid Spotify public playlist url
BACKGROUND_NOISE_DIR = "assets/background_noises"
SPOTIFY_PLAYLIST_URL = os.getenv("SPOTIFY_PLAYLIST_URL", "https://open.spotify.com/playlist/1Y0Qk1K1DEMXeKgvjjnN7m?si=9fe441ab6e51466d")
METADATA_FILE = f"{CACHE_PATH}/songs_metadata.spotdl"
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

  if (not os.path.exists(METADATA_FILE)):
    logging.info("Metadata file does not exist. Downloading metadata...")
    subprocess.run(
      ["spotdl", "save", SPOTIFY_PLAYLIST_URL, "--save-file", METADATA_FILE],
      check=True,
    )
  else:
    logging.info("Metadata already exists. Skipping download...")

def process_and_save(mel_spec_transform, segment, song_name, chunk_id, aug_name):
  with torch.no_grad():
    mel_spec = mel_spec_transform(segment) # Get mel spectrogram  

  spec_numpy = mel_spec.squeeze().cpu().numpy()

  file_name = f"{song_name}_{chunk_id}_{aug_name}.npz"
  file_name = file_name.replace(" ", "_").replace(":", "_").replace("/", "_").lower()

  # Store the spectrogram as a compressed numpy file
  np.savez_compressed(
    os.path.join(SPECTOGRAMS_DIR, file_name),
    data=spec_numpy,
  )

  return {
    "file_name": file_name,
    "song_name": song_name,
    "chunk_id": chunk_id,
    "aug_name": aug_name,
  }

def get_augmentations(sample_rate=44100):
   return {
    "background_noise": AddBackgroundNoise(
      p=1.0,
      background_paths=BACKGROUND_NOISE_DIR,
      min_snr_in_db=10,
      max_snr_in_db=20,
      output_type="dict"
    ),  
    "colored_noise": AddColoredNoise(
      p=1.0,
      min_snr_in_db=10,
      max_snr_in_db=20,
      output_type="dict"
    ),
    "gain": Gain(
      p=1.0,  
      min_gain_in_db=-10,
      max_gain_in_db=20,
      output_type="dict"
    ),
    "pitch_shift": PitchShift(
      p=1.0,
      min_transpose_semitones=-2,
      max_transpose_semitones=2,
      sample_rate=sample_rate,
      output_type="dict"
    ),
  }

def process_song(song_file):
  song_path = os.path.join(SONGS_DIR, song_file)
  song_name = os.path.splitext(song_file)[0]

  # Load audio
  waveform, sample_rate = torchaudio.load(song_path)
  waveform = waveform.mean(dim=0, keepdim=True)  # Mono
  waveform = waveform.unsqueeze(0)  # [B, C, T] => [1, 1, T]
  waveform = waveform / waveform.abs().max() # Normalize
  waveform = waveform.to(device)  

  # Function to convert waveform to mel spectrogram
  mel_spec_transform = torch.nn.Sequential(
    MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64),
    AmplitudeToDB()
  )
  mel_spec_transform = torch.jit.script(mel_spec_transform.to(device))  # Compile the model

  augmentations = get_augmentations(sample_rate)
  
  segment_duration = 3  # seconds
  segment_samples = segment_duration * sample_rate
  total_samples = waveform.shape[2]

  segments = []

  for start in range(0, total_samples, segment_samples):
    chunk_id = start // segment_samples

    if start + segment_samples > total_samples:
      continue

    end = start + segment_samples

    # Original Segment
    segment = waveform[:, :, start:end]
    segments.append(
      process_and_save(mel_spec_transform, segment, song_name, chunk_id, aug_name="original")
    )

    # Augmented Segments
    for aug_name, augmenter in augmentations.items(): 
      augmented = augmenter(segment.cpu(), sample_rate=sample_rate)['samples']
      augmented = augmented / augmented.abs().max() # Normalize
      augmented = augmented.to(device)  # Move to GPU

      segments.append(
        process_and_save(mel_spec_transform, augmented, song_name, chunk_id, aug_name)
      )
    
  # Free up memory
  torch.cuda.empty_cache()
  gc.collect()

  return segments 

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
  for i, song in enumerate(songs):
    print(f"Processing song {i+1}/{len(songs)}: {song}")
    records.extend(process_song(song))

  df = pd.DataFrame(records)
  df.to_csv(DATASET_FILE, index=False)

  logging.info("Songs transformed into spectograms successfully.")