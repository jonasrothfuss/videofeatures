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
  """
  Extracts features from (gulped) dataset
  :param extractor_type: str denoting the feature extractor type
  :param dataset: str denoting the dataset extractor type
  :param batch_size: number of videos per batch
  :param pickle_path: (optional) if set the features are stored as pandas df to the denoted location, else the df is dumped to the default path /DataDumps/Features
  :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels of videos
  """
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
  :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (list) of videos
  '''
  assert feature_df is not None or feature_df_path is not None

  #load feature df
  if feature_df_path:
    assert os.path.isfile(feature_df_path)
    feature_df = pd.read_pickle(feature_df_path)

  assert 'features' in feature_df and 'labels' in feature_df

  #stack video features to a 2d matrix
  features = np.concatenate(feature_df['features'], axis=0)

  labels = list(feature_df['labels'])

  if features.ndim == 3: #assume only one feature vector is given -> insert dimension
    features = features.reshape((features.shape[0], features.shape[1], 1, features.shape[2]))

  assert features.ndim == 4 and len(labels) == features.shape[0]
  return features, labels


def trainFisherVectorGMM(features, by_bic=True, model_dump_path=None, n_kernels=20):
  """
  trains a GMM for Fisher Vectors on the given features.
  :param features: features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (list) of videos
  :param by_bic: denotes whether the gmm fit is chosen based on the lowest BIC
  :param n_kernels: provide if not fitted by bic (bic-method chooses from a fixed set of n_kernels)
  :return: the fitted object of type FisherVectorGMM
  """
  fv_gmm = FisherVectorGMM(n_kernels=n_kernels)
  if by_bic:
    fv_gmm.fit_by_bic(features, model_dump_path=model_dump_path)
  else:
    fv_gmm.fit(features, model_dump_path=model_dump_path)
  assert fv_gmm.fitted

  return fv_gmm


def loadFisherVectorGMM(pickle_path):
  assert os.path.isfile(pickle_path)
  with open(pickle_path, 'rb') as f:
    fv_gmm = pickle.load(f)
  assert isinstance(fv_gmm, FisherVectorGMM)
  return fv_gmm


def computeFisherVectors(features, labels, fv_gmm, normalized=True, pickle_path=None):
  """
  :param features: features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
  :param fv_gmm: fitted FisherVectorGMM instance
  :param normalized: boolean - if true: improved fisher vectors are computed
  :return: fisher vectors - ndarray of shape (n_videos, n_frames, 2*n_kernels, n_feature_dim)
  """
  assert isinstance(fv_gmm, FisherVectorGMM)
  assert isinstance(features, np.ndarray)
  assert features.ndim == 4

  if not pickle_path:
    pickle_path = os.path.join(FeatureDumpsDir, "gmm_fisher_vectors" + ".pickle" )

  fv = fv_gmm.predict(features, normalized=normalized)

  df = pd.DataFrame(data={'labels': labels, 'features': np.vsplit(fv, features.shape[0])})
  print('Dumped feature dataframe to', pickle_path)
  df.to_pickle(pickle_path)

  return df



def loadFisherVectors(fisher_vector_path):
  '''
  loads fisher vectors from pd dataframe and returns them as a matrix
  :param feature_df: pandas dataframe which holds features in a column 'features'
  :param feature_df_path: path to pandas dataframe that holds features
  :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (list) of videos
  '''
  assert fisher_vector_path is not None
  assert os.path.isfile(fisher_vector_path)

  fisher_vectors_df = pd.read_pickle(fisher_vector_path)
  return fisher_vectors_df

  #
  # assert 'features' in feature_df and 'labels' in feature_df
  #
  # # stack video features to a 2d matrix
  # features = np.concatenate(feature_df['features'], axis=0)
  #
  # labels = list(feature_df['labels'])
  #
  # if features.ndim == 3:  # assume only one feature vector is given -> insert dimension
  #   features = features.reshape((features.shape[0], features.shape[1], 1, features.shape[2]))
  #
  # assert features.ndim == 4 and len(labels) == features.shape[0]
  # return features, labels


def main():
  #for extractor in EXTRACTOR_TYPES:
  #  extractFeatures(extractor_type=extractor, dataset='20bn_val')


  #features, labels = extractFeatures(extractor_type='vgg_fc1', dataset='20bn_val')
  features, labels = loadFeatures(feature_df_path='./DataDumps/Features/vgg_fc1_20bn_val')
  fv_gmm = trainFisherVectorGMM(features, by_bic=False)
  fisher_vectors = computeFisherVectors(features[:100], labels[:100], fv_gmm)
  fisher_df = loadFisherVectors("./DataDumps/Features/gmm_fisher_vectors.pickle")
  print(fisher_df)
  #features = loadFeatures(feature_df_path=os.path.join(FeatureDumpsDir, 'surf_20bn_val'))

  #loadFisherVectors(pickle_path)


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