import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.cuda import is_available
from torch.utils.data import DataLoader
from datasets import get_spectograms
from torch.amp import GradScaler, autocast
from utils.songs import CACHE_PATH

TRAINED_MODEL_PATH = f"{CACHE_PATH}/model.pt"
TRAINED_MODEL_INFO_PATH = f"{CACHE_PATH}/model_info.json"

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 4
SHUFFLE = True

def get_device():
  return 'cuda' if is_available() else 'cpu'

device = get_device()
logging.info(f"Using device: {device}")

class CNN(nn.Module):
  def __init__(self, output_size):
    super(CNN, self).__init__()

    self.conv_block = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2),
      
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.AdaptiveAvgPool2d((4, 4))
    )

    self.embedding = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 4 * 4, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
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

    return x

  def fit(self, train_loader, test_loader, epochs=50, learning_rate=0.001):
    self.total_epochs = epochs
    loss_function = nn.TripletMarginLoss(margin=0.5, p=2)

    training_epoch_duration = []

    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    for epoch in range(epochs):
      start_time = time.time()
      self.current_epoch = epoch + 1
      self.train()
      epoch_loss = 0.0

      for anchor, positive, negative in train_loader:
        
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device):
          anchor_embed = self(anchor)
          positive_embed = self(positive)
          negative_embed = self(negative)
          loss = loss_function(anchor_embed, positive_embed, negative_embed)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        epoch_loss += loss.item()
      
      # Calculate the average loss for the epoch
      train_loss = epoch_loss / len(train_loader)
      self.history['train_loss'].append(train_loss)

      # Validation
      self.eval()
      val_loss = 0.0
      with torch.no_grad():
        for anchor, positive, negative in test_loader:
          anchor = anchor.to(device, non_blocking=True)
          positive = positive.to(device, non_blocking=True)
          negative = negative.to(device, non_blocking=True)

          with autocast(device_type=device):
            anchor_embed = self(anchor)
            positive_embed = self(positive)
            negative_embed = self(negative)
            loss = loss_function(anchor_embed, positive_embed, negative_embed)

          val_loss += loss.item()
      
      val_loss /= len(test_loader)
      self.history['val_loss'].append(val_loss)

      # Calculate the training ETA
      elapsed_time = time.time() - start_time
      training_epoch_duration.append(elapsed_time)
      self.training_ETA = (sum(training_epoch_duration) / (self.current_epoch + 1)) * (self.total_epochs - self.current_epoch)

      logging.info(f"Epoch {self.current_epoch}/{self.total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - ETA: {self.training_ETA:.2f}s")
    
    self.is_model_trained = True

  def predict(self, X):
    pass

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

def initialize_model():
  logging.info(f"Initializing model...")

  # Create DataLoaders for train and test sets
  train_ds, test_ds = get_spectograms()
  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

  # Get the number of classes
  num_classes = len(train_ds.base_dataset.dataset.classes_to_idx)

  model = CNN(num_classes).to(device)

  if torch.__version__ >= '2.0':
    model = torch.compile(model)

  # Start training
  logging.info(f"Starting training using {device}...")
  model.fit(train_loader=train_loader, test_loader=test_loader, epochs=10, learning_rate=0.0001)