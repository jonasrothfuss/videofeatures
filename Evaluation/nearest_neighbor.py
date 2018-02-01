from math import sqrt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import os
import pandas as pd

def nearestNeighborMatching(memory_features, memory_labels, query_features, query_labels, n_closest_matches=-1, metric='cosine'):
  '''
  finds the closest vector matches (cos_similarity) for queries in the memory
  :param memory_features: memory feature vectors - ndarray of shape (memory_size, n_feature_dim)
  :param memory_labels: labels corresponding to the feature vectors in the memory - ndarray of shape (memory_size,)
  :param query_features: query feature vectors - ndarray of shape (query_size, n_feature_dim)
  :param query_labels: labels corresponding to the feature vectors in the query - ndarray of shape (query_size,)
  :param n_closest_matches: number of neighbors to be considered - default: -1 : consider all vectors in the memory
  :param metric: distance metric to use e.g. 'cosine', 'euclidean', 'mahalanobis'
  :return: dataframe of shape (query_size, memory_size + 1) containing the labels of matched vectors

           the df looks as follows:

            pred_class_0      pred_class_1      pred_class_2  ....  true_class

            label_match1      label_match2      label_match3  ....  label_query

  '''

  ''' Preconditions and Preprocessing '''
  assert isinstance(memory_labels, list) or memory_labels.ndim == 1
  assert isinstance(query_labels, list) or query_labels.ndim == 1
  memory_labels, query_labels = np.asarray(memory_labels),  np.asarray(query_labels)

  memory_features = memory_features.reshape((memory_features.shape[0], -1))
  query_features = query_features.reshape((query_features.shape[0], -1))
  assert memory_features.shape[1] == query_features.shape[1]
  assert memory_features.shape[0] == memory_labels.shape[0] and  query_features.shape[0] == query_labels.shape[0]

  cos_distances = pairwise_distances(memory_features, query_features, metric=metric)
  # get indices of n maximum values in ndarray, reverse the list (highest is leftmost)
  indices_closest = cos_distances.argsort()[:-n_closest_matches-1:-1] if n_closest_matches > 0 else cos_distances.argsort()

  retrieved_labels_dict= dict([(i, query_labels[indices_closest[:, i]]) for i in range(query_labels.shape[0])])


  df = pd.DataFrame.from_dict(retrieved_labels_dict, orient='index')
  df.columns = ['pred_class_{}'.format(i) for i in range(memory_features.shape[0])]
  df["true_class"] = query_labels

  assert df.shape[0] == query_labels.shape[0] and (df.shape[1] == memory_labels.shape[0] + 1 or df.shape[1] == n_closest_matches + 1)
  return df

