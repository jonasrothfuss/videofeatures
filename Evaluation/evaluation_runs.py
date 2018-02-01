import pandas as pd
import numpy as np
from Evaluation import nearest_neighbor
from Evaluation import performance_measures


def twenty_bn_preprocessing(path_to_pickle, use_pca=False, n_components=50, metric='cosine'):
  df = pd.read_pickle(path_to_pickle)
  LABEL_COLUMN_NAME = "category"
  FEATURES_COLUMN_NAME = "hidden_repr"


  mask = np.random.rand(len(df)) < 0.8
  query_df = df[~mask]
  base_df = df[mask]

  print("number of query samples: ", np.shape(query_df)[0])
  print("number of base samples: ", np.shape(base_df)[0])

  if use_pca:
    base_df, query_df = nearest_neighbor.transform_vectors_with_inter_class_pca(base_df, query_df, class_column='category',
                                                                                n_components=n_components)

  base_labels = list(base_df[LABEL_COLUMN_NAME].values)
  base_features = np.asarray([rep[0][0] for rep in base_df[FEATURES_COLUMN_NAME].values])

  query_labels = list(query_df[LABEL_COLUMN_NAME].values)
  query_features = np.asarray([rep[0][0] for rep in query_df[FEATURES_COLUMN_NAME].values])
  return nearest_neighbor.nearestNeighborMatching(base_features, base_labels, query_features,
                                                                  query_labels, metric=metric)


def twenty_bn_gdl_optical_flow_compute_MAP(nn_df):
  return performance_measures.mean_average_precision(nn_df, n_relevant_documents=-1)



def run_evaluation_twenty_bn(path_to_pickle, model, metric="cosine"):
  nn_df = twenty_bn_preprocessing(path_to_pickle, use_pca=False)
  MAP1 = twenty_bn_gdl_optical_flow_compute_MAP(nn_df)

  nn_df = twenty_bn_preprocessing(path_to_pickle, use_pca=True, n_components=30, metric=metric)
  MAP2 = twenty_bn_gdl_optical_flow_compute_MAP(nn_df)

  nn_df = twenty_bn_preprocessing(path_to_pickle, use_pca=True, n_components=50, metric=metric)
  MAP3 = twenty_bn_gdl_optical_flow_compute_MAP(nn_df)

  nn_df = twenty_bn_preprocessing(path_to_pickle, use_pca=True, n_components=100, metric=metric)
  MAP4 = twenty_bn_gdl_optical_flow_compute_MAP(nn_df)

  nn_df = twenty_bn_preprocessing(path_to_pickle, use_pca=True, n_components=200, metric=metric)
  MAP5 = twenty_bn_gdl_optical_flow_compute_MAP(nn_df)

  print("Model: ", model, " Metric: ", metric)
  print("mean average precision (no pca) is: ", MAP1)
  print("mean average precision (with pca, 30 components) is: ", MAP2)
  print("mean average precision (with pca, 50 components) is: ", MAP3)
  print("mean average precision (with pca, 100 components) is: ", MAP4)
  print("mean average precision (with pca, 200 components) is: ", MAP5)

def run_evaluation_twenty_bn_kfold(path_to_pickle, model, metric="cosine"):
  pass


if __name__ == '__main__':
  """ 8_20bn_gdl_optical_flow """
  path_to_pickle = "/PDFData/rothfuss/selected_trainings/8_20bn_gdl_optical_flow/valid_run/metadata_and_hidden_rep_df_08-09-17_17-00-24_valid.pickle"
  """ Cosine Metric """
  #run_evaluation_twenty_bn(path_to_pickle, model="8_20bn_gdl_optical_flow", metric="cosine")

  """ Mahanabolis Metric """
  #run_evaluation_twenty_bn(path_to_pickle, model="8_20bn_gdl_optical_flow", metric="mahalanobis")

  """ Euclidean Metric """
  #run_evaluation_twenty_bn(path_to_pickle, model="8_20bn_gdl_optical_flow", metric="euclidean")

  """ 7_20bn_mse """
  path_to_pickle = "/PDFData/rothfuss/selected_trainings/7_20bn_mse/valid_run_backup/metadata_and_hidden_rep_df_07-26-17_16-52-09_valid.pickle"
  run_evaluation_twenty_bn(path_to_pickle, model="7_20bn_mse", metric="euclidean")