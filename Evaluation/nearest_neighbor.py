from math import sqrt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import os
import collections
import pandas as pd
from sklearn import decomposition
import time

def nearestNeighborMatching(features, labels, memory_index, query_index, n_partitions=5, n_closest_matches=-1, metric='cosine'):
  '''
  finds the closest vector matches (cos_similarity) for queries in the memory
  :param features: ndarray or memmap to feature vectors
  :param labels: ndarray with labels -> shape (n_instances, )
  :param memory_index: list with indices for selecting the memory instances
  :param query_index: list with indices for selecting the query instances
  :param n_partitions: number of partitions for computing the distances in smaller chunks
  :param n_closest_matches: number of neighbors to be considered - default: -1 : consider all vectors in the memory
  :param metric: distance metric to use e.g. 'cosine', 'euclidean', 'mahalanobis'
  :return: dataframe of shape (query_size, memory_size + 1) containing the labels of matched vectors

           the df looks as follows:

            pred_class_0      pred_class_1      pred_class_2  ....  true_class

            label_match1      label_match2      label_match3  ....  label_query

  '''


  ''' Preconditions and Preprocessing '''
  assert not bool(set(memory_index) & set(query_index)), 'memory_index and query_index must be disjoint'
  memory_labels = np.asarray(labels)[memory_index]
  query_labels = np.asarray(labels)[query_index]

  distances = np.empty(shape=(memory_labels.shape[0], query_labels.shape[0]))

  ''' Partition memory and query features for distance computation'''

  memory_index_chunks = np.array_split(memory_index, n_partitions)
  query_index_chunks = np.array_split(query_index, n_partitions)

  i = 0
  for part, memory_index_chunk in enumerate(memory_index_chunks):
    t_start = time.time()
    j = 0
    new_i =  i + len(memory_index_chunk)
    for query_index_chunk in query_index_chunks:
      memory_features_chunk = features[memory_index_chunk].reshape((len(memory_index_chunk), -1))
      query_features_chunk = features[query_index_chunk].reshape((len(query_index_chunk), -1))

      distances_chunk = pairwise_distances(memory_features_chunk, query_features_chunk, metric=metric, n_jobs=-1)
      new_j = j + len(query_index_chunk)
      distances[i:new_i, j:new_j] = distances_chunk
      j = new_j

    assert j == distances.shape[1]
    i = new_i
    print("Computed distance partition {} of {} [Duration: {} sec]".format(part + 1, n_partitions, time.time() - t_start))

  assert i == distances.shape[0]


  # get indices of n maximum values in ndarray, reverse the list (highest is leftmost)
  indices_closest = distances.argsort(axis=0)[:n_closest_matches,] if n_closest_matches > 0 else distances.argsort(axis=0)

  retrieved_labels_dict= dict([(i, memory_labels[indices_closest[:, i]]) for i in range(query_labels.shape[0])])


  df = pd.DataFrame.from_dict(retrieved_labels_dict, orient='index')
  df.columns = ['pred_class_{}'.format(i) for i in range(len(memory_index))]
  df["true_class"] = query_labels

  assert df.shape[0] == query_labels.shape[0] and (df.shape[1] == memory_labels.shape[0] + 1 or df.shape[1] == n_closest_matches + 1)
  return df



def transform_vectors_with_inter_class_pca(df, df_2=None, class_column='category', n_components=50, return_pca_object=False):
  """
    Performs PCA on mean vectors of classes and applies transformation to all hidden_reps in df
    :param df: dataframe containing hidden vectors + metadata
    :param df_2: optional 2nd dataframe that shall be transformed with the pca computed on df
    :param class_column: column_name corresponding to class labels
    :param n_components: number of principal components
    :return: dataframe with transformed vectors, if two dfs were provided, both are returned with pca-transformed hidden_reps
    """
  print(class_column)
  assert 'hidden_repr' in df.columns and class_column in df.columns
  print(df.columns.values)
  df = df.copy()
  print(n_components)
  pca = inter_class_pca(df, class_column=class_column, n_components=n_components)
  transformed_vectors_as_matrix = pca.transform(df_col_to_matrix(df['hidden_repr']))
  df['hidden_repr'] = np.split(transformed_vectors_as_matrix, transformed_vectors_as_matrix.shape[0])
  if df_2 is not None:
    df_2 = df_2.copy()
    transformed_vectors_as_matrix = pca.transform(df_col_to_matrix(df_2['hidden_repr']))
    df_2['hidden_repr'] = np.split(transformed_vectors_as_matrix, transformed_vectors_as_matrix.shape[0])
    if  return_pca_object:
      return df, df_2, pca
    else:
      return df, df_2
  else:
    if return_pca_object:
      return df, pca
    else:
      return df


def inter_class_pca(df, class_column='category', n_components=50):
  """
  Performs PCA on mean vectors of classes
  :param df: dataframe containing hidden vectors + metadata
  :param class_column: column_name corresponding to class labels
  :param n_components: number of principal components
  :return: fitted pca sklean object
  """
  assert 'hidden_repr' in df.columns and class_column in df.columns
  mean_vector_df = mean_vectors_of_classes(df, class_column=class_column)
  pca = decomposition.PCA(n_components).fit(mean_vector_df)
  relative_variance_explained = np.sum(pca.explained_variance_) / np.sum(mean_vector_df.var(axis=0))
  print("PCA (n_components= %i: relative variance explained:" % n_components, relative_variance_explained, '\n',
        pca.explained_variance_)
  return pca


def mean_vectors_of_classes(df, class_column='shape'):
  """
  Computes mean vector for each class in class_column
  :param df: dataframe containing hidden vectors + metadata
  :param class_column: column_name corresponding to class labels
  :return: dataframe with classes as index and mean vectors for each class
  """
  assert 'hidden_repr' in df.columns and class_column in df.columns
  labels = list(df[class_column])
  values = df_col_to_matrix(df['hidden_repr'])
  vector_dict = collections.defaultdict(list)
  for label, vector in zip(labels, values):
    vector_dict[label].append(vector)

  return pd.DataFrame.from_dict(dict([(label, np.mean(vectors, axis=0)) for label, vectors in vector_dict.items()]),
                                orient='index')


def df_col_to_matrix(panda_col):
  """Converts a pandas dataframe column wherin each element in an ndarray into a 2D Matrix
  :return ndarray (2D)
  :param panda_col - panda Series wherin each element in an ndarray
  """
  panda_col = panda_col.map(lambda x: x.flatten())
  return np.vstack(panda_col)