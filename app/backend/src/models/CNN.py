import os
import time
import json
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.cuda import is_available
from utils.songs import CACHE_PATH
from utils.types import ServerState
from datasets import get_spectograms
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pytorch_metric_learning import miners, losses

TRAINED_MODEL_PATH = f"{CACHE_PATH}/model.pt"
TRAINED_MODEL_INFO_PATH = f"{CACHE_PATH}/model_info.json"

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 2
SHUFFLE = True

def get_device():
  return 'cuda' if is_available() else 'cpu'

device = get_device()
logging.info(f"Using device: {device}")

class CNN(nn.Module):
  def __init__(self, output_size=256):
    super(CNN, self).__init__()

    self.conv_block = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.MaxPool2d(2),
      
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.MaxPool2d(2),

      nn.AdaptiveAvgPool2d((4, 4))
    )

    self.embedding = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 4 * 4, 256),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(256, output_size),
    )

    self.scaler = GradScaler(device)

    self.history = {'train_loss': [], 'val_loss': []}
    self.training_ETA = 0.0
    self.total_epochs = 0
    self.current_epoch = 0
    self.is_model_trained = False
  
  def forward(self, x):
    x = x.unsqueeze(1)

    x = self.conv_block(x)
    x = self.embedding(x)
    x = nn.functional.normalize(x, p=2, dim=1) # L2 normalization

    return x

  def fit(self, train_loader, test_loader, epochs=50, learning_rate=0.001, patience=7):
    self.total_epochs = epochs
    training_epoch_duration = []

    loss_function = losses.TripletMarginLoss(margin=0.5)
    miner = miners.TripletMarginMiner(margin=0.5, type_of_triplets="semi-hard")

    optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
      self.train()
      epoch_loss = 0.0
      start_time = time.time()
      self.current_epoch = epoch

      for spectogram, label in train_loader:
        spectogram = spectogram.to(device, non_blocking=True)
        labels = label.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device):
          embeddings = self(spectogram)
          hard_triplets = miner(embeddings, labels)
          loss = loss_function(embeddings, labels, hard_triplets)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        epoch_loss += loss.item()
      
      # Calculate the average loss for the epoch
      avg_train_loss = epoch_loss / len(train_loader)
      self.history['train_loss'].append(avg_train_loss)

      # Validation
      val_loss = 0.0
      self.eval()
      with torch.no_grad():
        for spectogram, labels in test_loader:
          spectogram = spectogram.to(device, non_blocking=True)
          labels = labels.to(device, non_blocking=True)

          with autocast(device_type=device):
            embeddings = self(spectogram)
            hard_triplets = miner(embeddings, labels)
            loss = loss_function(embeddings, labels, hard_triplets)
          val_loss += loss.item()
      
      avg_val_loss = val_loss / len(test_loader)
      self.history['val_loss'].append(avg_val_loss)
      scheduler.step(avg_val_loss)

      # Early stopping
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = self.state_dict()    
        patience_counter = 0
      else:
        patience_counter += 1
        if patience_counter >= patience:
          logging.info(f"Early stopping at epoch {epoch}/{epochs}")
          break

      # Calculate the training ETA
      elapsed_time = time.time() - start_time
      training_epoch_duration.append(elapsed_time)
      self.training_ETA = (sum(training_epoch_duration) / (self.current_epoch + 1)) * (self.total_epochs - self.current_epoch)

      logging.info(f"Epoch {self.current_epoch}/{self.total_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - ETA: {self.training_ETA:.2f}s")
    
    if best_model_state:
      self.load_state_dict(best_model_state)
      logging.info(f"Best model loaded from epoch {self.current_epoch}/{self.total_epochs}")
    self.is_model_trained = True

  def predict(self, X):
    self.eval()
    
    if isinstance(X, list):
      spectrograms = torch.stack(X)
    elif spectrograms.ndim == 2:
      spectrograms = X.unsqueeze(0)

    spectrograms = spectrograms.to(device)

    with torch.no_grad(), autocast(device_type=device):
      embeddings = self(spectrograms)

    return embeddings.cpu()

  def save_trained_model(self):
    # Save the model
    torch.save(self.state_dict(), TRAINED_MODEL_PATH)

    logging.info(f"Model saved to {TRAINED_MODEL_PATH}")

    # Save the training info
    training_info = {
      'total_epochs': self.total_epochs,
      'current_epoch': self.current_epoch,
      'train_loss': self.history['train_loss'],
      'val_loss': self.history['val_loss']
    }
    
    with open(TRAINED_MODEL_INFO_PATH, 'w') as f:
      json.dump(training_info, f)

    logging.info(f"Model info saved to {TRAINED_MODEL_INFO_PATH}")

  def load_trained_model(self):
    # Load the model
    map_location = torch.device(get_device())
    self.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=map_location))
    self.eval()
    self.is_model_trained = True

    # Load the training info
    with open(TRAINED_MODEL_INFO_PATH, 'r') as f:
      training_info = json.load(f)

    self.total_epochs = training_info['total_epochs']
    self.current_epoch = training_info['current_epoch']
    self.history['train_loss'] = training_info['train_loss']
    self.history['val_loss'] = training_info['val_loss']

    logging.info(f"Model loaded from {TRAINED_MODEL_PATH}")
    logging.info(f"Model info loaded from {TRAINED_MODEL_INFO_PATH}")

  def get_training_progress(self):
    return {
        'is_model_trained': self.is_model_trained,
        'ETA': self.training_ETA,
        'current_epoch': self.current_epoch,
        'total_epochs': self.total_epochs,
        'train_loss': self.history['train_loss'],
        'val_loss': self.history['val_loss'],
    }

def initialize_model(current_state):
  logging.info(f"Initializing model...")
  current_state['state'] = ServerState.LOADING_MODEL

  model = CNN().to(device)

  if torch.__version__ >= '2.0':
    model = torch.compile(model)

  return model

def train_model(model, current_state):
  train_ds, test_ds = get_spectograms()

  # Create DataLoaders for train and test sets
  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

  # Check if the model is already trained
  if (os.path.exists(TRAINED_MODEL_PATH) and os.path.exists(TRAINED_MODEL_INFO_PATH)):
    logging.info(f"Model already trained. Loading model...")
    model.load_trained_model()
  else:
    current_state['state'] = ServerState.TRAINING_MODEL
    logging.info(f"Model not trained. Training model with {device}...")
    model.fit(train_loader=train_loader, test_loader=test_loader, epochs=100, learning_rate=0.0001)
    model.save_trained_model()
    logging.info(f"Model trained and saved.")