from Evaluation import nearest_neighbor
import numpy as np


def mean_average_precision(df_nearest_neighbor, n_relevant_documents=-1):
  """
  This function computes the mean average precision (MAP) for a set of queries specified by df_query. The average precision
  scores for each query are hereby computed based on the provided base instances specified in df_base. For each query,
  the nearest neighbor instances within the base are determined and used to compute the precision.

  :param df_nearest_neighbor: dataframe containing the closest matches/neighbors
         shape (query_size, memory_size + 1) containing the labels of matched vectors

         the df looks as follows:

         pred_class_0      pred_class_1      pred_class_2  ....  true_class

         label_match1      label_match2      label_match3  ....  label_query

  :param n_relevant_documents: number of neighbors to be considered - default: -1 : consider all matches
  :return: a scalar value representing the mAP
  """
  df_pred_classes = df_nearest_neighbor.filter(like="pred_class").iloc[:, :n_relevant_documents]
  matches = df_pred_classes.isin(df_nearest_neighbor.true_class).as_matrix()

  if n_relevant_documents < 0:
    n_relevant_documents = df_pred_classes.shape[1]

  P = np.zeros(shape=matches.shape)
  for k in range(n_relevant_documents):
    P[:, k] = np.mean(matches[:, :k+1], axis=1)

  mask = np.sum(matches, axis=1) > 0
  P = P[mask, :]
  matches = matches[mask]

  return np.mean((np.sum(np.multiply(P, matches), axis=1) / np.sum(matches, axis=1)))


def precision_at_k(df_nearest_neighbor, k):
  """
    :param df_nearest_neighbor: dataframe containing the closest matches/neighbors
         shape (query_size, memory_size + 1) containing the labels of matched vectors

         the df looks as follows:

         pred_class_0      pred_class_1      pred_class_2  ....  true_class

         label_match1      label_match2      label_match3  ....  label_query

  :param k: cutoff (int) - number of nearest neighbors that shall be considered for computing the precision
  :return: a scalar value representing the precision at k
  """
  df_pred_classes = df_nearest_neighbor.filter(like="pred_class").iloc[:, :k]
  matches = df_pred_classes.isin(df_nearest_neighbor.true_class).as_matrix()

  return np.mean(matches)