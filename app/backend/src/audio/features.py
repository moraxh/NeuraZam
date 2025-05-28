import os
import json
import torch
import torchaudio
import numpy as np
import pandas as pd 
from tqdm import tqdm
from utils.logger_config import logger
from utils.types import ServerState, current_state
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch_audiomentations import AddBackgroundNoise, AddColoredNoise, Gain, PitchShift, ApplyImpulseResponse, BandPassFilter
from utils.constants import BACKGROUND_NOISE_DIR, TARGET_SAMPLE_RATE, SONGS_DIR, SPECTOGRAMS_DIR, SEGMENTS_DIR, SEGMENT_DURATION, DATASET_FILE, SONGS_STATS_FILE, DEVICE

mel_spec_transform = torch.nn.Sequential(
  MelSpectrogram(n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=8000),
  AmplitudeToDB(),
).to(DEVICE)  # Move to GPU if available

augmentations = {
    "background_noise": AddBackgroundNoise(
      p=1,
      background_paths=BACKGROUND_NOISE_DIR,
      min_snr_in_db=5,
      max_snr_in_db=15,
      output_type="dict"
    ),  
    "colored_noise": AddColoredNoise(
      p=1,
      min_snr_in_db=5,
      max_snr_in_db=15,
      output_type="dict"
    ),
    "gain": Gain(
      p=1,  
      min_gain_in_db=-15,
     max_gain_in_db=0,
      output_type="dict"
    ),
    "pitch_shift": PitchShift(
      p=1,
      min_transpose_semitones=-2,
      max_transpose_semitones=2,
      sample_rate=TARGET_SAMPLE_RATE,
      output_type="dict"
    ),
  }

def get_spectogram(waveform, as_numpy=True):
  with torch.no_grad(): 
    mel_spec = mel_spec_transform(waveform)
  mel_spec = mel_spec.to(DEVICE)  # Move to GPU if available  

  if as_numpy:
    return mel_spec.cpu().detach().numpy()

  return mel_spec 

def get_waveform_n_sr_from_file(file_path):
  waveform, sr = torchaudio.load(file_path, normalize=True)

  if sr != TARGET_SAMPLE_RATE:  
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
    waveform = resampler(waveform)

  waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
  waveform = waveform.unsqueeze(0)  
  waveform = waveform.to(DEVICE)  # Move to GPU if available

  return waveform, TARGET_SAMPLE_RATE

def get_global_mean_std(files=os.listdir(SONGS_DIR)):
  if (os.path.exists(SONGS_STATS_FILE)):
    with open(SONGS_STATS_FILE, "r") as f:
      stats = json.load(f)
      return stats["mean"], stats["std"]

  sum_ = 0.0
  sum_sq = 0.0
  count = 0

  for file in files:
    file_path = os.path.join(SONGS_DIR, file) 
    waveform, sr = get_waveform_n_sr_from_file(file_path) 
    spec_np = get_spectogram(waveform, as_numpy=False)

    vals = spec_np.flatten()

    sum_ += vals.sum()
    sum_sq += (vals ** 2).sum()
    count += vals.numel()

  mean_global = sum_ / count
  var_global  = sum_sq / count - mean_global**2
  std_global  = torch.sqrt(var_global)

  # Save the stats
  stats = {
    "mean": float(mean_global.cpu().detach().numpy()),
    "std": float(std_global.cpu().detach().numpy())
  }

  with open(SONGS_STATS_FILE, "w") as f:
    json.dump(stats, f)

  return mean_global.cpu().detach().numpy(), std_global.cpu().detach().numpy()

def save_spectogram(segment, song_id, chunk_id, aug_name, variation=0):
    file_name = f"{song_id}_{chunk_id}_{aug_name}_{variation}.npz"

    # Store as compressed file
    np.savez_compressed(
      os.path.join(SPECTOGRAMS_DIR, file_name),
      data=segment
    )

    return {
      "song_id": song_id,
      "file_name": file_name,
      "chunk_id": chunk_id,
      "aug_name": aug_name,
      "variation": variation,
    }

def extract_features():
  global_mean, global_std = get_global_mean_std(os.listdir(SONGS_DIR))

  if not (len(os.listdir(SPECTOGRAMS_DIR)) == 0 or len(os.listdir(SEGMENTS_DIR)) == 0):
    logger.info("Founded spectograms, skiping it")
    return

  logger.info("Not founded spectograms, extracting them...")

  records = []

  for song in tqdm(os.listdir(SONGS_DIR), desc="Extracting features of the songs", unit="song"):
    song_path = os.path.join(SONGS_DIR, song)
    song_id = os.path.splitext(song)[0]

    waveform, sr = get_waveform_n_sr_from_file(song_path)

    segments_samples = int(SEGMENT_DURATION * sr)
    total_samples = waveform.shape[2]

    hop_length = segments_samples // 2
    chunk_id = 0
    for start in range(0, total_samples - segments_samples + 1, hop_length):

      if start + segments_samples > total_samples:
        break

      end = start + segments_samples
      segment = waveform[:, :, start:end]

      rms = torch.sqrt(torch.mean(segment ** 2))
      if rms < 0.01:
        logger.warning(f"Skipping segment {chunk_id} of song {song_id} due to low RMS value.")
        continue

      with torch.no_grad():
        mel_spec = get_spectogram(segment)

      # Normalize using the global mean & st
      mel_spec = (mel_spec - global_mean) / global_std

      segment_file = f"{song_id}_{chunk_id}.wav"

      # Save the segment
      torchaudio.save(
        os.path.join(SEGMENTS_DIR, segment_file),
        segment.cpu().squeeze(0),
        sr,
      )

      records.append(
        save_spectogram(mel_spec, song_id, chunk_id, "original")
      )

      chunk_id += 1
    
  # Save the dataframe
  df = pd.DataFrame(records)
  df.to_csv(DATASET_FILE, index=False)

def augment_data():
  global_mean, global_std = get_global_mean_std(os.listdir(SONGS_DIR))
  df = pd.read_csv(DATASET_FILE)

  if not (df["aug_name"] == "original").all():
    logger.info("The dataset already has augmented data, skipping augmentation.")
    return

  logger.info("The dataset only has original data, applying augmentation.")

  # Get the number of clases per song_id
  logger.info(df["song_id"].value_counts())

  number_of_samples_target = df["song_id"].value_counts().max() * 1.7

  logger.info(f"Target number of samples: {number_of_samples_target}")

  unique_song_ids = df["song_id"].unique()
  
  for song_id in tqdm(unique_song_ids, desc="Augmenting data", unit="song"):
    song_df = df[df["song_id"] == song_id]

    while df[df["song_id"] == song_id].shape[0] < number_of_samples_target:
      # Get a random sample of the original data
      # TODO: CHeck that the sample is original 
      sample = song_df.sample(n=1, random_state=np.random.randint(0, 10000)).iloc[0]
      segment_file = f"{song_id}_{sample['chunk_id']}.wav"
      file_path = os.path.join(SEGMENTS_DIR, segment_file)

      # Apply a random augmentation
      aug_name = np.random.choice(list(augmentations.keys()))
      augment = augmentations[aug_name]

      waveform, sr = get_waveform_n_sr_from_file(file_path)
      augmented_segment = augment(waveform, sample_rate=TARGET_SAMPLE_RATE)['samples']
      mel_spec = get_spectogram(augmented_segment)
      mel_spec = (mel_spec - global_mean) / global_std  

      # Check if its a variation
      duplicated = df[
        (df["song_id"] == song_id) & 
        (df["chunk_id"] == sample["chunk_id"]) &
        (df["aug_name"] == aug_name)
      ]

      if duplicated.shape[0] > 0:
        variation_num = duplicated["variation"].max() + 1
      else:
        variation_num = 0

      record = save_spectogram(
        mel_spec,
        song_id,
        sample["chunk_id"],
        aug_name,
        variation=variation_num 
      )

      # Append the record to the dataframe
      df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

  df.to_csv(DATASET_FILE, index=False)
  logger.info(df["song_id"].value_counts())

def initialize_extract_features():
  current_state['state'] = ServerState.EXTRACTING_FEATURES
  extract_features()
  current_state['state'] = ServerState.AUGMENTING_SONGS
  augment_data()