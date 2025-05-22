import os
import torch
import numpy as np
from utils.constants import SPECTOGRAMS_DIR
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

class SpectogramDataset(Dataset):
  def __init__(self, spectograms_dir=SPECTOGRAMS_DIR, transform=None):
    self.spectograms_files = os.listdir(SPECTOGRAMS_DIR)
    self.spectograms_dir = spectograms_dir
    self.transform = transform

  def __len__(self):
    return len(self.spectograms_files)
  
  def __getitem__(self, idx):
    file_name = self.spectograms_files[idx]
    file_path = os.path.join(self.spectograms_dir, file_name)

    mel_spec = np.load(file_path)["data"] 

    if mel_spec.ndim == 4:
      mel_spec = mel_spec.squeeze(0).squeeze(0)
    elif mel_spec.ndim == 3:
      mel_spec = mel_spec.squeeze(0) 
        
    mel_spec = mel_spec[np.newaxis, ...]

    label = int(file_name.split("_")[0])  # Get the song id from the file name 

    if self.transform:
      mel_spec = self.transform(mel_spec) 

    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int64)

    return mel_spec, label

def get_subsets_spectograms():
  dataset = SpectogramDataset()

  idx = list(range(len(dataset)))
  train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

  train_subset = Subset(dataset, train_idx)   
  test_subset  = Subset(dataset, test_idx)

  return train_subset, test_subset