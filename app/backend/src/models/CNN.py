import os
import time
import torch 
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from utils.logger_config import logger
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from utils.types import ServerState, current_state
from models.datasets import get_subsets_spectograms
from pytorch_metric_learning import miners, losses, distances
from utils.constants import DEVICE, EMBEDDINGS_PLOT_FILE, TRAINED_MODEL_FILE, EMBEDDINGS_FILE, BATCH_SIZE, SHUFFLE, NUM_WORKERS, DEVICE

class CNN(nn.Module):
  def __init__(self, output_size=256):
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

    self.embedding = nn.Sequential(
      nn.Flatten(),
      nn.Linear(256 * 4 * 4, 1024),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(1024, output_size),
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
    x = self.embedding(x)

    return x

  def fit(self, train_loader, test_loader, epochs=50, learning_rate=0.001, patience=10):
    self.total_epochs = epochs
    training_epoch_duration = []

    loss_function = losses.TripletMarginLoss(margin=0.5, distance=distances.CosineSimilarity())
    miner = miners.TripletMarginMiner(margin=0.5, type_of_triplets="all", distance=distances.CosineSimilarity())

    optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
      self.train()
      epoch_loss = 0.0
      start_time = time.time()
      self.current_epoch = epoch

      for spectogram, label in train_loader:
        spectogram = spectogram.to(DEVICE, non_blocking=True)
        labels = label.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=DEVICE):
          embeddings = self(spectogram)
          embeddings = F.normalize(embeddings, p=2, dim=1)
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
          spectogram = spectogram.to(DEVICE, non_blocking=True)
          labels = labels.to(DEVICE, non_blocking=True)

          with autocast(device_type=DEVICE):
            embeddings = self(spectogram)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            hard_triplets = miner(embeddings, labels)
            loss = loss_function(embeddings, labels, hard_triplets)
          val_loss += loss.item()
      
      avg_val_loss = val_loss / len(test_loader)
      self.history['val_loss'].append(avg_val_loss) 
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

      logger.info(f"Epoch {self.current_epoch}/{self.total_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - ETA: {self.training_ETA:.2f}s")
    
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

    with torch.no_grad(), autocast(device_type=DEVICE):
      emb = self(X)
      emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.cpu()

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
  
  def store_embeddings(self, train_loader, test_loader):
    # Check if the model is trained
    if not self.is_model_trained:
      raise Exception("Model is not trained. Please train the model before storing embeddings.")

    # Store the embeddings
    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []

    # Store the train embeddings
    for spectogram, label in train_loader:
      spectogram = spectogram.to(DEVICE, non_blocking=True)
      labels = label.to(DEVICE, non_blocking=True)

      with torch.no_grad(), autocast(device_type=DEVICE):
        embeddings = self.predict(spectogram)
      
      train_embeddings.append(embeddings.cpu())
      train_labels.append(labels.cpu())
    
    # Store the test embeddings 
    for spectogram, label in test_loader:
      spectogram = spectogram.to(DEVICE, non_blocking=True)
      labels = label.to(DEVICE, non_blocking=True)

      with torch.no_grad(), autocast(device_type=DEVICE):
        embeddings = self.predict(spectogram)
      
      test_embeddings.append(embeddings.cpu())
      test_labels.append(labels.cpu())

    # Concatenate the embeddings and labels
    train_embeddings = torch.cat(train_embeddings)
    train_labels = torch.cat(train_labels)
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)

    embeddings = torch.cat([train_embeddings, test_embeddings])
    labels = torch.cat([train_labels, test_labels])

    logger.info(f"Shape: {embeddings.shape}")
    logger.info(f"Mean por dimensión: {embeddings.mean(axis=0)[:5]}")
    logger.info(f"Std por dimensión: {embeddings.std(axis=0)[:5]}")
    logger.info(f"Varianza total: {embeddings.var()}")

    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    np.savez_compressed(EMBEDDINGS_FILE, embeddings=emb_np, labels=labels_np)

    # Calculate the silhouette score for train and test sets
    train_score = silhouette_score(train_embeddings.numpy(), train_labels.numpy())
    test_score = silhouette_score(test_embeddings.numpy(), test_labels.numpy())
    logger.info(f"Train Silhouette: {train_score:.4f}, Test Silhouette: {test_score:.4f}")

    # Plot
    tsne = TSNE( n_components=2, perplexity=30, learning_rate='auto', init='pca', max_iter=1000, random_state=42)

    test_embeddings_scaled = normalize(test_embeddings, norm='l2')
    test_embeddings_2d = tsne.fit_transform(test_embeddings_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_labels, cmap='tab10')
    plt.title("t-SNE de los embeddings")
    plt.tight_layout()
    plt.savefig(EMBEDDINGS_PLOT_FILE)
    plt.close()

def initialize_model() :
  train_ds, test_ds = get_subsets_spectograms()

  # Create DataLoaders for train and test sets
  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True)

  train_model(train_loader, test_loader)
  store_embeddings(train_loader, test_loader)
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

def store_embeddings(train_loader, test_loader):
  current_state['state'] = ServerState.STORING_EMBEDDINGS 

  if (os.path.exists(EMBEDDINGS_FILE)):
    logger.info(f"Embeddings already stored")
    return

  logger.info(f"Embeddings not stored. Storing embeddings...")
  model.store_embeddings(train_loader=train_loader, test_loader=test_loader)
  logger.info(f"Embeddings stored.")

def get_model():
  return model

model = CNN().to(DEVICE)

if torch.__version__ >= '2.0':
  model = torch.compile(model)