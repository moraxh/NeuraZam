import numpy as np
from utils.constants import EMBEDDINGS_FILE
from sklearn.neighbors import KNeighborsClassifier

def get_KNN():
  data = np.load(EMBEDDINGS_FILE)
  emb_train = data['embeddings']
  labels_train = data['labels']

  # Initialize k-NN classifier
  knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
  knn.fit(emb_train, labels_train)

  return knn