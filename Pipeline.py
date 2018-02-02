from FeatureExtractor import SIFTFeatures, VGGFeatures, ResNetFeatures, SURFFeatures
from DatasetProvider import TwentyBNDataset
from FeatureProcessing import FisherVectorGMM
from Evaluation import nearestNeighborMatching, precision_at_k, mean_average_precision
import pandas as pd
import numpy as np
import os, pickle
from sklearn.model_selection import KFold
import logging
from joblib import Parallel, delayed


ModelDumpsDir = './DataDumps/Models'
FeatureDumpsDir = './DataDumps/Features'
FisherVectorDumpsDir = './DataDumps/FisherVectors'
LogDir = './DataDumps/Logs'
ResultsDir = './DataDumps/Results'

EXTRACTOR_TYPES = ['resnet', 'vgg_fc1', 'vgg_fc2', 'surf', 'sift']
EXTERNALLY_EXTRACTED_FEATURES = ['stip']
DATASETS = ['20bn_val', '20bn_train', 'armar_val', 'armar_train']

def extractFeatures(extractor_type='vgg', dataset='20bn_val', batch_size=20, feature_dump_path=None):
  """
  Extracts features from (gulped) dataset
  :param extractor_type: str denoting the feature extractor type
  :param dataset: str denoting the dataset extractor type
  :param batch_size: number of videos per batch
  :param feature_dump_path: (optional) if set the features are stored as pandas df to the denoted location, else the df is dumped to the default path /DataDumps/Features
  :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels of videos
  """
  assert extractor_type in EXTRACTOR_TYPES
  assert dataset in DATASETS

  ''' 1. Choose Model '''
  if extractor_type is 'resnet':
    extractor =  ResNetFeatures()
  elif extractor_type is 'vgg_fc1':
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

  if not feature_dump_path:
    feature_dump_path = os.path.join(FeatureDumpsDir, extractor_type + '_' + dataset + '.pickle')

  return extractor.computeFeaturesForVideoDataset(loader, pickle_path=feature_dump_path)


def loadFeatures(feature_df_path):
  '''
  loads features from pd dataframe and returns them as a matrix
  :param feature_df_path: path to pandas dataframe that holds features
  :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (list) of videos
  '''

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


def trainFisherVectorGMM(features, by_bic=True, model_dump_path=None, n_kernels=50):
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


def computeFisherVectors(features, labels, fv_gmm, normalized=True, fv_dump_path=None, batch_size=500):
  """
  :param features: features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
  :param labels: labels correspodning to features - array of shape (n_videos,)
  :param fv_gmm: fitted FisherVectorGMM instance
  :param normalized: boolean - if true: improved fisher vectors are computed
  :param fv_dump_path: file path specifying the dump location for the computed vectors
  :param batch_size: batch size for computing the fisher vactors
  :return: (fv, labels) fisher vectors - ndarray of shape (n_videos, n_frames, 2*n_kernels, n_feature_dim)
  """
  assert isinstance(fv_gmm, FisherVectorGMM)
  assert isinstance(features, np.ndarray)
  assert features.ndim == 4

  labels = np.asarray(labels)
  assert labels.ndim ==1
  assert features.shape[0] == labels.shape[0]


  # compute Fisher vectors in batches
  n_instances = features.shape[0]
  n_batches = n_instances // batch_size
  split_idx = [i*batch_size for i in range(1,n_batches+1)]
  batches = np.split(features, split_idx)

  fv_batches = []

  for i, batch in enumerate(batches):
    print('Compute FV batch {} of {}'.format(i+1, n_batches+1))
    fv_batch = fv_gmm.predict(batch, normalized=normalized)
    fv_batches.append(fv_batch)

  fv = np.concatenate(fv_batches, axis=0)

  if fv_dump_path:
    df = pd.DataFrame(data={'labels': labels, 'features': np.vsplit(fv, features.shape[0])})
    print('Dumped feature dataframe to', fv_dump_path)
    df.to_pickle(fv_dump_path)

  assert fv.shape[0] == labels.shape[0]
  return fv, labels


def loadFisherVectors(fisher_vector_path):
  '''
  loads fisher vectors from pd dataframe and returns them as a matrix
  :param feature_df: pandas dataframe which holds features in a column 'features'
  :param feature_df_path: path to pandas dataframe that holds features
  :return: (fv, labels) - fv as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (as array) of videos
  '''
  assert fisher_vector_path is not None
  assert os.path.isfile(fisher_vector_path)

  fisher_vectors_df = pd.read_pickle(fisher_vector_path)
  assert 'labels' in fisher_vectors_df and 'features' in fisher_vectors_df

  fv = np.concatenate(fisher_vectors_df['features'], axis=0)
  labels = np.asarray(fisher_vectors_df['labels'])

  return fv, labels


def evaluateMatching(feature_vectors, labels, n_splits=5, distance_metrics=['cosine', 'euclidean', 'hamming']):
  '''
  evaluates the matching performance by computing the mean average precision and precision at k
  -> the evaluation measures are computed on n splits and then averaged over the splits
  -> the evaluation is also performed for different distance_metrics
  :param feature_vectors: features as ndarray of shape (n_videos, n_feature_dim)
  :param labels: labels as ndarray of shape (n_videos,)
  :param n_splits: number of splits for dividing the feature vectors into a memory and query set
  :param distance_metrics: list of distance metrics to use for the matching e.g. 'cosine', 'euclidean', 'mahalanobis'
  :return: result dictionary which holds the evaluation results - it has the following shape

    {'cosine': {'mAP': 0.5238396745484852, 'precision_at_1': 0.5, 'precision_at_3': 0.5, 'precision_at_5': 0.516, 'precision_at_10': 0.5039999999999999}}

  '''
  labels = np.asarray(labels)
  assert labels.ndim == 1 and feature_vectors.ndim > 1
  assert feature_vectors.shape[0] == labels.shape[0]

  kf = KFold(n_splits=n_splits, shuffle=True)

  eval_measures = ['mAP', 'precision_at_1', 'precision_at_3', 'precision_at_5', 'precision_at_10']
  inner_result_dict = dict([(eval_measure, []) for eval_measure in eval_measures])
  result_dict = dict([(distance_metric, inner_result_dict) for distance_metric in distance_metrics])

  for memory_index, query_index in kf.split(labels):
    memory_features = feature_vectors[memory_index]
    query_features = feature_vectors[query_index]
    memory_labels = labels[memory_index]
    query_labels = labels[query_index]

    for metric in distance_metrics:
      matches_df = nearestNeighborMatching(memory_features, memory_labels, query_features, query_labels, metric=metric)
      result_dict[metric]['mAP'].append(mean_average_precision(matches_df))
      result_dict[metric]['precision_at_1'].append(precision_at_k(matches_df, k=1))
      result_dict[metric]['precision_at_3'].append(precision_at_k(matches_df, k=3))
      result_dict[metric]['precision_at_5'].append(precision_at_k(matches_df, k=5))
      result_dict[metric]['precision_at_10'].append(precision_at_k(matches_df, k=10))

  # Take the mean of the results over the folds
  for metric in distance_metrics:
    for eval_measure in eval_measures:
      result_dict[metric][eval_measure] = np.mean(result_dict[metric][eval_measure])

  return result_dict

def setup_logger(logfile_name = 'piplineRuns.log'):
  log_file_path = os.path.join(LogDir, logfile_name)
  logger = logging.getLogger('PipelinRunLogger')
  logger.setLevel(logging.INFO)
  fh = logging.FileHandler(log_file_path)
  fh.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(ch)

  return logger

def runEntirePipeline(extractFeat=True, trainGMM=True, computeFV=True, dataset='20bn_val', by_bic=False):
  assert dataset in DATASETS, 'Dataset must be one of the following: ' + str(DATASETS)

  #set up logging
  logger = setup_logger(logfile_name = 'piplineRuns.log')

  logger.info('----- START PIPELINE RUN -----')
  logger.info('CONFIGS: [extractFeat: {}, trainGMM: {}, computeFV: {}, dataset: {}, by_bic: {}]'.format(extractFeat, trainGMM, computeFV, dataset, by_bic))

  overall_result_dict = {}

  for extractor in EXTRACTOR_TYPES + EXTERNALLY_EXTRACTED_FEATURES:

    ''' 1. Extract / Load Features '''
    if extractFeat:
      logger.info('Started extracting {} features from {} dataset'.format(extractor, dataset))
      features, labels = extractFeatures(extractor_type=extractor, dataset=dataset,
                                         feature_dump_path=getDumpFileName(extractor, dataset, 'features'))
      logger.info('Finished extracting {} features from {} dataset. Features have shape {}.'
                  ' Dumped features to {}'.format(extractor, dataset, features.shape, getDumpFileName(extractor, dataset, 'features')))

    else: # load features
      features, labels = loadFeatures(getDumpFileName(extractor, dataset, 'features'))
      logger.info('Loaded {} features from {}.  Features have shape {}'.format(extractor, getDumpFileName(extractor, dataset, 'features'), features.shape))


    ''' 2. Train / Load Fisher Vector GMM '''
    if trainGMM:
      logger.info('Started training FisherVector GMM')
      fv_gmm = trainFisherVectorGMM(features, by_bic=by_bic, model_dump_path=getDumpFileName(extractor, dataset, 'model'))
      logger.info('Finished training FisherVector GMM. Dumped model to {}'.format(getDumpFileName(extractor, dataset, 'model')))

    else: #load model
      fv_gmm = loadFisherVectorGMM(getDumpFileName(extractor, dataset, 'model'))
      logger.info('Loaded FisherVector GMM from {}'.format(getDumpFileName(extractor, dataset, 'model')))


    ''' 3. Compute / Load Fisher Vectors '''
    if computeFV:
      logger.info('Started Computing Fisher Vectors')
      fv, labels = computeFisherVectors(features, labels, fv_gmm, normalized=True,
                                fv_dump_path=getDumpFileName(extractor, dataset, 'fisher_vectors'))
      logger.info('Finished computing Fisher Vectors. Dumped vectors to {}'.format(getDumpFileName(extractor, dataset, 'fisher_vectors')))

    else:
      fv, labels = loadFisherVectors(getDumpFileName(extractor, dataset, 'fisher_vectors'))
      logger.info('Loaded Fisher Vectors from {}'.format(getDumpFileName(extractor, dataset, 'fisher_vectors')))

    ''' 4. Evaluate Matching '''
    n_splits = 5
    distance_metrics = ['cosine', 'euclidean', 'hamming']

    logger.info('Starting to evaluate matching results (n_splits={}, distance_metrics={})'.format(n_splits, distance_metrics))
    matching_result_dict = evaluateMatching(fv, labels, n_splits=5, distance_metrics=distance_metrics)

    logger.info('Finished matching evaluation of {} features from {} dataset - Results: \n{}'.format(extractor, dataset, matching_result_dict))

    overall_result_dict[extractor] = matching_result_dict

  # Store final results as pandas df
  result_df = overall_result_dict_to_df(overall_result_dict)
  result_df.to_pickle(getDumpFileName('', dataset, 'results'))

  logger.info('Dumped results datafram to {}'.format(getDumpFileName('', dataset, 'results')))


def getDumpFileName(extractor, dataset, type):
  assert type in ['features', 'model', 'fisher_vectors', 'results']
  if type is 'features':
    return os.path.join(FeatureDumpsDir, 'features_' + extractor + '_' + dataset + '.pickle')
  elif type is 'model':
    return os.path.join(ModelDumpsDir, 'model_' + extractor + '_' + dataset + '.pickle')
  elif type is 'fisher_vectors':
    return os.path.join(FisherVectorDumpsDir, 'fv_' + extractor + '_' + dataset + '.pickle')
  elif type is 'results':
    return os.path.join(ResultsDir, 'results_{}.pickle'.format(dataset))

def overall_result_dict_to_df(result_dict):
  df_columns = ['extractor', 'distance_measure', 'mAP', 'precision_at_1', 'precision_at_3', 'precision_at_5', 'precision_at_10']
  df_rows = []
  i = 1
  for dataset, inner_dict in result_dict.items():
    for distance_measure, results in inner_dict.items():
      row = [dataset, distance_measure]
      for result in results.values():
        row.append(result)
      df_rows.append((i,row))
      i += 1

  df = pd.DataFrame.from_dict(dict(df_rows), orient='index')
  df.columns = df_columns
  return df



  #print(df)

def main():
  # for extractor in ['resnet']:
  #   extractFeatures(extractor_type=extractor, dataset='20bn_val', batch_size=20)

  runEntirePipeline(extractFeat=False)



  # features = np.random.normal(size=(1840, 20, 1, 80))
  # labels = np.random.randint(1,10, size=(1840))
  #
  # fv_gmm = trainFisherVectorGMM(features, by_bic=False)
  # fv = computeFisherVectors(features, labels, fv_gmm)


  #features, labels = extractFeatures(extractor_type='vgg_fc1', dataset='20bn_val')
  # features, labels = loadFeatures(feature_df_path='./DataDumps/Features/vgg_fc1_20bn_val')
  # fv_gmm = trainFisherVectorGMM(features, by_bic=False)
  # fisher_vectors = computeFisherVectors(features[:100], labels[:100], fv_gmm)
  # fisher_df = loadFisherVectors("./DataDumps/Features/gmm_fisher_vectors.pickle")
  # print(fisher_df)
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