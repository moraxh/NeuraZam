import logging
import json
import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from utils.songs import DATASET_FILE, SPECTOGRAMS_DIR, CACHE_PATH
from collections import defaultdict

CLASS_MAP_CACHE = f"{CACHE_PATH}/class_map.json"

class NormalizeTransform:
    def __call__(self, sample):
        return (sample - sample.min()) / (sample.max() - sample.min())

class SpectogramDataset(Dataset):
  def __init__(self, csv_file, spectogram_dir, transform=None):
    self.spectogram_data = pd.read_csv(csv_file)
    self.spectogram_dir = spectogram_dir
    self.transform = transform
    self.transform = transform or NormalizeTransform()
    self.labels = self.spectogram_data['song_name'].values
    self.classes_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    # Save class mapping to a JSON file
    with open(CLASS_MAP_CACHE, 'w') as f:
      json.dump(self.classes_to_idx, f)
  
  def __len__(self):
    return len(self.spectogram_data)

  def __getitem__(self, idx):
    file_name = self.spectogram_data.iloc[idx]['file_name']
    file_path = os.path.join(self.spectogram_dir, file_name)

    spectogram = np.load(file_path)["S_db"]

    spectogram_tensor = torch.tensor(spectogram, dtype=torch.float32)

    label = self.spectogram_data.iloc[idx]['song_name']
    label_tensor = torch.tensor(self.classes_to_idx[label], dtype=torch.long)

    if self.transform:
      spectogram_tensor = self.transform(spectogram_tensor)
    
    return spectogram_tensor, label_tensor

class TripletDataset(Dataset):
  def __init__(self, base_dataset):
    self.base_dataset = base_dataset

    if isinstance(base_dataset, Subset):
      self.labels = [base_dataset.dataset.labels[i] for i in base_dataset.indices]
    else:
      self.labels = base_dataset.labels

    self.label_to_indices = self._group_indices_by_label()
    self.label_to_indices_keys = list(self.label_to_indices.keys())

  def _group_indices_by_label(self):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(self.labels):
      label_to_indices[label].append(idx)
    return label_to_indices

  def __getitem__(self, index):
    anchor, anchor_label = self.base_dataset[index]

    # Select positive
    positive_index = index
    while positive_index == index:
      key = self.label_to_indices_keys[anchor_label.item()]
      positive_index = random.choice(self.label_to_indices[key])
    positive, _ = self.base_dataset[positive_index]

    # Select negative
    negative_label = anchor_label.item()
    while negative_label == anchor_label.item():
      negative_label = random.choice(list(self.label_to_indices.keys()))
    negative_index = random.choice(self.label_to_indices[negative_label])
    negative, _ = self.base_dataset[negative_index]

    return anchor, positive, negative

  def __len__(self):
    return len(self.base_dataset)

def get_spectograms():
  # Load the dataset
  dataset = SpectogramDataset(csv_file=DATASET_FILE, spectogram_dir=SPECTOGRAMS_DIR)

  inx = list(range(len(dataset)))
  train_idx, test_idx = train_test_split(inx, test_size=0.2, random_state=42, stratify=dataset.spectogram_data['song_name'])

  train_subset = Subset(dataset, train_idx)
  test_subset = Subset(dataset, test_idx)

  train_dataset = TripletDataset(train_subset)
  test_dataset = TripletDataset(test_subset)

  return train_dataset, test_dataset