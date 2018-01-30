from FeatureExtractor import SIFTFeatures, CNNFeatures, SURFFeatures
from DatasetProvider import TwentyBNDataset
from FeatureProcessing import FisherVectorGMM
import pandas as pd
import numpy as np
import os, pickle



def extractFeatures(pickle_path):
  model = SURFFeatures(n_descriptors=10)
  loader = TwentyBNDataset(batch_size=20).getDataLoader()
  model.computeFeaturesForVideoDataset(loader, pickle_path=pickle_path)

def loadFeatures(feature_df=None, feature_df_path=None):
  '''
  loads features from pd dataframe and returns them as a matrix
  :param feature_df: pandas dataframe which holds features in a column 'features'
  :param feature_df_path: path to pandas dataframe that holds features
  :return: matrix of shape (n_feature_samples, n_feature_dim)
  '''
  assert feature_df is not None or feature_df_path is not None

  #load feature df
  if feature_df_path:
    assert os.path.isfile(feature_df_path)
    feature_df = pd.read_pickle(feature_df_path)

  assert 'features' in feature_df

  #stack video features to a 2d matrix
  features = np.concatenate(feature_df['features'], axis=0)

  if features.ndim == 3: #assume only one feature vector is given -> insert dimension
    features = features.reshape((features.shape[0], features.shape[1], 1, features.shape[2]))

  assert features.ndim == 4
  return features


def loadFisherVectorGMM(pickle_path):
  assert os.path.isfile(pickle_path)
  with open(pickle_path, 'rb') as f:
    fv_gmm = pickle.load(f)
  assert isinstance(fv_gmm, FisherVectorGMM)
  return fv_gmm

def main():
  #extractFeatures()
  #features = loadFeaturesForTraining(feature_df_path='"/common/homes/students/rothfuss/Desktop/SURFfeatures_10_20bn_valid.pickle"')
  #features = loadFeaturesForTraining(feature_df_path='/common/homes/students/rothfuss/Desktop/SURF_20bn.pickle')
  features = loadFeatures(feature_df_path='/common/homes/students/rothfuss/Desktop/SURF_20bn.pickle')
  print(features.shape)
  fv_gmm = FisherVectorGMM(n_kernels=20)
  fv_gmm.fit_by_bic(features, model_dump_path='/common/homes/students/rothfuss/Desktop/SURF_20bn_model.pickle')

  #fv_gmm = loadFisherVectorGMM('/common/homes/students/rothfuss/Desktop/SURF_20bn_model.pickle')
  fv = fv_gmm.predict(features)
  print(fv)



if __name__ == '__main__':
  main()