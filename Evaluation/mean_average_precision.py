import nearest_neighbor
import pandas as pd
import numpy as np

def compute_mean_average_precision(df_base, df_query, n_closest_matches=5):
  """
  This function computes the mean average precision (MAP) for a set of queries specified by df_query. The average precision
  scores for each query are hereby computed based on the provided base instances specified in df_base. For each query,
  the nearest neighbor instances within the base are determined and used to compute the precision.
  :param df_base: the dataframe to be queried, must contain a 'hidden_repr' column that constitutes the hidden_representation vector
  :param df_query: the dataframe from which to query, must contain a 'hidden_repr' column
  :param n_closest_matches: number of closest matches to the query that goes into the precision score
  :return: a scalar value representing the MAP
  """
  assert not df_base.empty and not df_query.empty
  df = get_query_matching_table(df_base=df_base, df_query=df_query, n_closest_matches=n_closest_matches)

  df_pred_classes = df.filter(like="pred_class")
  n_relevant_documents = len(df_pred_classes.columns)
  matches = df_pred_classes.isin(df.true_class).as_matrix()

  P = np.zeros(shape=matches.shape)
  for k in range(1, n_relevant_documents):
    P[:, k] = np.mean(matches[:, :k], axis=1)

  return np.mean(np.multiply(P, matches))


def get_query_matching_table(df_base, df_query, class_column='category', n_closest_matches=5, df_true_label="true_class",
                             df_pred_label="pred_class_", df_query_label="category", df_query_id="id"):
  """
  Yields a pandas dataframe in which each row contains the n_closest_matches as a result from querying the df for every single
  hidden representation in the df dataframe. In addition, every row contains the true label and the query id.
  :param df_base: the df to be queried
  :param df_query: the df from which to query
  :return: pandas dataframe with columns ("id", "true_label", "pred_class_i" for i=1,...,n_closest_matches) and
  number of rows equal to df_query rows
  """
  assert df_base is not None and df_query is not None
  assert 'hidden_repr' in df_base.columns and class_column in df_base.columns
  assert 'hidden_repr' in df_query.columns and df_query_label in df_query.columns and df_query_id in df_query.columns

  columns = [[df_query_id + "{}".format(i), df_pred_label+"{}".format(i)] for i in range(1, n_closest_matches + 1)]
  columns = [e for entry in columns for e in entry] # flatten list in list
  columns[:0] = [df_query_id, df_true_label]

  query_matching_df = pd.DataFrame(columns=columns)
  query_matching_df.set_index(df_query_id, df_true_label)

  for hidden_repr, label, id in zip(df_query['hidden_repr'], df_query[df_query_label], df_query[df_query_id]):
    closest_vectors = nearest_neighbor.find_nearest_neighbors(df_base, hidden_repr=hidden_repr, class_column=class_column,
                                                              n_closest_matches=n_closest_matches)

    matching_results = [[tpl[2], tpl[1]] for tpl in closest_vectors]
    matching_results = sum(matching_results, [])  # flatten
    matching_results[:0] = [id, label]

    row_data = dict(zip(columns, matching_results))
    query_matching_df = query_matching_df.append(row_data, ignore_index=True)

  return query_matching_df
