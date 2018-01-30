from FeatureExtractor import SIFTFeatures, VGGFeatures, SURFFeatures
from DatasetProvider import TwentyBNDataset
from FeatureProcessing import FisherVectorGMM
import pandas as pd
import numpy as np
import os, pickle

ModelDumpsDir = './DataDumps/Models'
FeatureDumpsDir = './DataDumps/Features'

EXTRACTOR_TYPES = ['vgg_fc1', 'vgg_fc2', 'surf', 'sift']
DATASETS = ['20bn_val', '20bn_train', 'armar_val', 'armar_train']

def extractFeatures(extractor_type='vgg', dataset='20bn_val', batch_size=20, pickle_path=None):
  assert extractor_type in EXTRACTOR_TYPES
  assert dataset in DATASETS

  ''' 1. Choose Model '''
  if extractor_type is 'vgg_fc1':
    extractor = VGGFeatures(feature='fc1')
  elif extractor_type is 'vgg_fc2':
    extractor = VGGFeatures(feature='fc2')
  elif extractor_type is 'surf':
    extractor = SURFFeatures(n_descriptors=20)
  elif extractor_type is 'sift':
    extractor = SIFTFeatures(n_descriptors=5)
  else:
    raise NotImplementedError(extractor_type + 'is not implemented')

  ''' 2. Choose Dataset '''
  if dataset is '20bn_val':
    loader = TwentyBNDataset(batch_size=batch_size).getDataLoader()
  elif dataset is '20bn_train':
    raise NotImplementedError(dataset + 'loader is not implemented') #TODO
  elif dataset is 'armar_val':
    raise NotImplementedError(dataset + 'loader is not implemented') #TODO
  elif dataset is 'armar_train':
    raise NotImplementedError(dataset + 'loader is not implemented') #TODO

  if not pickle_path:
    pickle_path = os.path.join(FeatureDumpsDir, extractor_type + '_' + dataset)

  return extractor.computeFeaturesForVideoDataset(loader, pickle_path=pickle_path)

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
  for extractor in EXTRACTOR_TYPES:
    extractFeatures(extractor_type=extractor, dataset='20bn_val')

  #features = loadFeaturesForTraining(feature_df_path='"/common/homes/students/rothfuss/Desktop/SURFfeatures_10_20bn_valid.pickle"')
  #features = loadFeaturesForTraining(feature_df_path='/common/homes/students/rothfuss/Desktop/SURF_20bn.pickle')

  #
  # features = loadFeatures(feature_df_path='/common/homes/students/rothfuss/Desktop/SURF_20bn.pickle')
  # print(features.shape)
  # fv_gmm = FisherVectorGMM(n_kernels=20)
  # fv_gmm.fit_by_bic(features, model_dump_path='/common/homes/students/rothfuss/Desktop/SURF_20bn_model.pickle')
  #
  # #fv_gmm = loadFisherVectorGMM('/common/homes/students/rothfuss/Desktop/SURF_20bn_model.pickle')
  # fv = fv_gmm.predict(features)
  # print(fv)



if __name__ == '__main__':
  main()