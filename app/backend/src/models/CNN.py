import os
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.logger_config import logger
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.types import ServerState, current_state
from models.datasets import get_subsets_spectograms
from utils.constants import DEVICE, TRAINED_MODEL_FILE, BATCH_SIZE, SHUFFLE, NUM_WORKERS, DEVICE

class CNN(nn.Module):
  def __init__(self, n_classes):
    super(CNN, self).__init__()

    # TODO: Try different kernel sizes and strides  
    self.conv_block = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.MaxPool2d((2, 1)),
      
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((4, 4)),
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(256 * 4 * 4, 1024),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(1024, n_classes),
    )

    self.scaler = GradScaler(DEVICE)

    self.history = {'train_loss': [], 'val_loss': []}
    self.training_ETA = 0.0
    self.total_epochs = 0
    self.current_epoch = 0
    self.is_model_trained = False
  
  def forward(self, x):
    if x.dim() > 4:
      x = x.squeeze(1)

    x = self.conv_block(x)
    x = self.classifier(x)

    return x

  def fit(self, train_loader, test_loader, epochs=50, learning_rate=0.001, patience=10):
    self.total_epochs = epochs
    training_epoch_duration = []

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
      self.train()
      epoch_loss = 0.0
      correct = 0
      total = 0
      start_time = time.time()
      self.current_epoch = epoch

      for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast(DEVICE):
          outputs = self(inputs)
          loss = loss_function(outputs, labels)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
      
      # Calculate the average loss for the epoch
      train_acc = correct / total
      avg_train_loss = epoch_loss / len(train_loader)
      self.history["train_loss"].append(avg_train_loss)

      # Validation
      val_loss = 0.0
      correct = 0
      total = 0
      self.eval()
      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs = inputs.to(DEVICE, non_blocking=True)
          labels = labels.to(DEVICE, non_blocking=True)

          with autocast(DEVICE):
            outputs = self(inputs)
            loss = loss_function(outputs, labels)

          val_loss += loss.item()
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()
      
      val_acc = correct / total
      avg_val_loss = val_loss / len(test_loader)
      self.history["val_loss"].append(avg_val_loss)
      scheduler.step(avg_val_loss)

      # Early stopping
      if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = self.state_dict()    
        patience_counter = 0
      else:
        patience_counter += 1
        if patience_counter >= patience:
          logger.info(f"Early stopping at epoch {epoch}/{epochs}")
          break

      # Calculate the training ETA
      elapsed_time = time.time() - start_time
      training_epoch_duration.append(elapsed_time)
      self.training_ETA = (sum(training_epoch_duration) / (self.current_epoch + 1)) * (self.total_epochs - self.current_epoch)

      print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, ETA: {self.training_ETA}")
    
    if best_model_state:
      self.load_state_dict(best_model_state)
      logger.info(f"Best model loaded from epoch {self.current_epoch}/{self.total_epochs}")
    self.is_model_trained = True

  def predict(self, X):
    self.eval()

    X = torch.as_tensor(X, dtype=torch.float32)

    if isinstance(X, list):
        X = torch.stack(X)
    elif X.ndim == 2:
        X = X.unsqueeze(0)

    X = X.to(DEVICE)

    with torch.no_grad(), autocast(DEVICE):
      pred = self(X)

    return pred.cpu()

  def save_trained_model(self):
    torch.save(self.state_dict(), TRAINED_MODEL_FILE)
    logger.info(f"Model saved to {TRAINED_MODEL_FILE}")

  def load_trained_model(self):
    map_location = torch.device(DEVICE)
    self.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=map_location))
    self.eval()
    self.is_model_trained = True
  
  def get_training_progress(self):
    return {
        'is_model_trained': self.is_model_trained,
        'ETA': self.training_ETA,
        'current_epoch': self.current_epoch,
        'total_epochs': self.total_epochs,
        'train_loss': self.history['train_loss'],
        'val_loss': self.history['val_loss'],
    }
  
def initialize_model() :
  global model

  train_ds, test_ds = get_subsets_spectograms()

  # Create DataLoaders for train and test sets
  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True)

  n_classes = len(list(set(train_loader.dataset.dataset.classes) | set(test_loader.dataset.dataset.classes)))
  logger.info(f"Num of classes: {n_classes}")

  model = CNN(n_classes).to(DEVICE)
  if torch.__version__ >= '2.0':
    model = torch.compile(model)

  train_model(train_loader, test_loader)
  current_state['state'] = ServerState.READY
  logger.info(f"Model initialized and ready to use.")

def train_model(train_loader, test_loader):
  current_state['state'] = ServerState.LOADING_MODEL
  if (os.path.exists(TRAINED_MODEL_FILE)):
    logger.info(f"Model already trained. Loading model...")
    model.load_trained_model()
    return

  current_state['state'] = ServerState.TRAINING_MODEL
  logger.info(f"Model not trained. Training model with {DEVICE}...")
  model.fit(train_loader=train_loader, test_loader=test_loader, epochs=50, learning_rate=0.001)
  model.save_trained_model()
  logger.info(f"Model trained and saved.")

def get_model():
  global model
  return model

model = None