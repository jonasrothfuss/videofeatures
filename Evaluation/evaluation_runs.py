import pandas as pd
import numpy as np
from Evaluation import nearest_neighbor
from Evaluation import performance_measures
import Pipeline
from joblib import Parallel, delayed

def run_evaluation(path_to_pickle, use_pca=False, n_components=50, label_column_name="category", features_column_name="hidden_repr"):
  df = pd.read_pickle(path_to_pickle)


  if use_pca:
    df = nearest_neighbor.transform_vectors_with_inter_class_pca(df, class_column='category', n_components=n_components)

  labels = list(df[label_column_name].values)
  features = np.asarray([rep[0][0] for rep in df[features_column_name].values])

  return Pipeline.evaluateMatching(feature_vectors=features, labels=labels)



if __name__ == '__main__':
  """ 8_20bn_gdl_optical_flow """
  path_to_pickle = "/PDFData/rothfuss/selected_trainings/8_20bn_gdl_optical_flow/valid_run/metadata_and_hidden_rep_df_08-09-17_17-00-24_valid.pickle"

  """ 7_20bn_mse """
  # path_to_pickle = "/PDFData/rothfuss/selected_trainings/7_20bn_mse/valid_run_backup/metadata_and_hidden_rep_df_07-26-17_16-52-09_valid.pickle"

  LABEL_COLUMN_NAME = "category"
  FEATURES_COLUMN_NAME = "hidden_repr"

  settings = [[False, 50], [True, 50], [True, 100], [True, 200]]

  dicts = Parallel(n_jobs=8)(delayed(run_evaluation)(path_to_pickle, use_pca=setting[0], n_components=setting[1],
                                        label_column_name=LABEL_COLUMN_NAME,
                                        features_column_name=FEATURES_COLUMN_NAME) for setting in settings)
  print(dicts)


