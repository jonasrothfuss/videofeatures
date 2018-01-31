from math import sqrt
import numpy as np

def find_nearest_neighbors(df, query_idx=None, hidden_repr=None, class_column='shape', n_closest_matches=5):
  """
  finds the closest vector matches by cosine similarity metric for a given query vector within the df dataframe
  :param df: dataframe containing hidden vectors + metadata
  :param query_idx: index of the query vector in the df
  :param hidden_repr: query vector
  :param num_vectors: denotes how many of the closest vector matches shall be returned
  :return: list of tuples (i, cos_sim, label) correspondig to the num_vectors closest vector matches
  """
  assert 'hidden_repr' in df.columns
  assert bool(query_idx is not None) != bool(
    hidden_repr is not None), "Either query_idx or hidden_repr can be set, but not both at the same time"
  assert not query_idx or query_idx in df.index

  if query_idx:
    query_row = df.iloc[query_idx]
    query_class = query_row[class_column]
    query_v_id = query_row["video_id"]
    remaining_df = df[df.index != query_idx]
    hidden_repr = query_row['hidden_repr']
  else:
    remaining_df = df

  cos_distances = [(compute_cosine_similarity(v, hidden_repr), l, int(v_id)) for _, v, l, v_id in
                   zip(remaining_df.index, remaining_df['hidden_repr'], remaining_df[class_column],
                       remaining_df['video_id'])]
  sorted_distances = sorted(cos_distances, key=lambda tup: tup[0], reverse=True)

  if query_idx:
    sorted_distances = [tup for tup in sorted_distances if tup[2] != query_v_id]
    print([l for _, l, _ in sorted_distances[:n_closest_matches]].count(query_class), query_v_id)
  return sorted_distances[:n_closest_matches]


def compute_cosine_similarity(vector_a, vector_b):
  """
  Computes the cosine similarity for the two given vectors.
  :param vector_a:
  :param vector_b:
  :return:
  """
  assert np.shape(vector_a) == np.shape(vector_b)
  if vector_a.ndim >= 2:
    vector_a = vector_a.flatten()
  if vector_b.ndim >= 2:
    vector_b = vector_b.flatten()

  numerator = sum(a * b for a, b in zip(vector_a, vector_b))
  denominator = square_rooted(vector_a) * square_rooted(vector_b)
  return round(numerator / float(denominator), 3)

def square_rooted(x):
  return round(sqrt(sum([a * a for a in x])), 3)
