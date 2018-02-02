import pandas as pd
import numpy as np
from Evaluation import nearest_neighbor
from Pipeline import evaluateMatching
import Pipeline

""" settings for eval runs """
settings_DEM = [[False, None], [True, 50], [True, 100], [True, 200]] # use_pca, #n_pca_components


def run_evaluation(path_to_pickle, use_pca=False, n_components=50, label_column_name="category", features_column_name="hidden_repr"):
  df = pd.read_pickle(path_to_pickle)

  if use_pca:
   df = nearest_neighbor.transform_vectors_with_inter_class_pca(df, class_column=label_column_name, n_components=n_components)

  labels = list(df[label_column_name].values)
  features = np.asarray([rep[0] for rep in df[features_column_name].values])

  return evaluateMatching(feature_vectors=features, labels=labels)


def run_eval_DEM_model(path_to_pickle):
  LABEL_COLUMN_NAME = "category"
  FEATURES_COLUMN_NAME = "hidden_repr"
  dicts = []
  for setting in settings_DEM:
    logger.info('Starting to evaluate matching results. Data used: {}, pca active:{}, pca components:{}'.format(path_to_pickle, setting[0], setting[1]))
    dict = run_evaluation(path_to_pickle=path_to_pickle, use_pca=setting[0], n_components=setting[1],
                   label_column_name=LABEL_COLUMN_NAME, features_column_name=FEATURES_COLUMN_NAME)
    logger.info('Finished matching evaluation. Results: \n{}'.format(dict))
    dicts.append(dict)
  print(dicts)

def run_fisher_vector_resnet_model(path_to_features, path_to_labels):
  features = np.load(path_to_features, mmap_mode='r')

  labels = np.load(path_to_labels, mmap_mode='r')
  print(features.shape, labels.shape)

if __name__ == '__main__':
  # set up logging
  logger = Pipeline.setup_logger(logfile_name='piplineRuns.log')
  #run_eval_DEM_model("/PDFData/rothfuss/selected_trainings/8_20bn_gdl_optical_flow/valid_run/metadata_and_hidden_rep_df_08-09-17_17-00-24_valid.pickle")
  #run_eval_DEM_model("/PDFData/rothfuss/selected_trainings/7_20bn_mse/valid_run_backup/metadata_and_hidden_rep_df_07-26-17_16-52-09_valid.pickle")
  run_fisher_vector_resnet_model("/PDFData/rothfuss/fisher_vector/fv_resnet_20bn_val.npy", "/PDFData/rothfuss/fisher_vector/fv_resnet_20bn_val_labels.npy")





