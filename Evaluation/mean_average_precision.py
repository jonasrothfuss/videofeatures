from Evaluation import nearest_neighbor
import numpy as np


def compute_mean_average_precision(df_nearest_neighbor, n_relevant_documents=-1):
  """
  This function computes the mean average precision (MAP) for a set of queries specified by df_query. The average precision
  scores for each query are hereby computed based on the provided base instances specified in df_base. For each query,
  the nearest neighbor instances within the base are determined and used to compute the precision.
  :param df_base: the dataframe to be queried, must contain a 'hidden_repr' column that constitutes the hidden_representation vector
  :param df_query: the dataframe from which to query, must contain a 'hidden_repr' column
  :param n_closest_matches: number of closest matches to the query that goes into the precision score
  :return: a scalar value representing the MAP
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


if __name__ == '__main__':
  n_memory_feat, n_query_feat = 1000, 500

  memory_features = np.random.normal(size=(n_memory_feat, 3000))
  query_features = np.random.normal(size=(n_query_feat, 3000))
  memory_labels = list(np.random.randint(1, 3, size=(n_memory_feat)))
  query_labels = list(np.random.randint(1, 3, size=(n_query_feat)))

  df_nearest_neighbors = nearest_neighbor.nearestNeighborMatching(memory_features, memory_labels, query_features, query_labels)
  map = compute_mean_average_precision(df_nearest_neighbors, n_relevant_documents=-1)
  print(map)