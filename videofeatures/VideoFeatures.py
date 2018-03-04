from fishervector import FisherVectorGMM
import pandas as pd
import numpy as np
import os, pickle
import logging
import gc
import datetime as dt



class Pipeline:
  def __init__(self, dataset, extractor, base_dir, dataset_name='VideoDataset'):
    """
    A pipeline object initialized with a dataset and a feature extractor allows to
     1) extract, store and load features (see repo description for currently supported features)
     2) train a GMM for a Fisher Vector encoding based on the chosen features, store and load its model parameters after training
     3) compute Fisher Vectors from the GMM, store and load them
    :param dataset: gulpIO DataLoader object which represents a dataset
    :param extractor: feature extractor object of type BaseFeatures (must implement computeFeatures function)
    :param base_dir: absolute or relative path to a directory where sub folders for logs and data dumps from the pipeline are created
    :param dataset_name: a string representing the name of the dataset used for logs and file name
    """
    self.dataset = dataset
    self.dataset_name = dataset_name
    self.extractor = extractor
    self.base_dir = base_dir

    self.model_dumps_dir = create_dir(os.path.join(base_dir, "models"))
    self.feature_dumps_dir = create_dir(os.path.join(base_dir, "features"))
    self.fisher_vector_dumps_dir = create_dir(os.path.join(base_dir, "fishervectors"))
    self.log_dir = create_dir(os.path.join(base_dir, "logs"))
    self.results_dir = create_dir(os.path.join(base_dir, "results"))

    time_stamp = str(dt.datetime.now().strftime("%m-%d-%y_%H-%M"))
    self.logger = self.setup_logger(logfile_name = time_stamp + '_video_features.log')


  def extractFeatures(self, feature_dump_path=None):
    """
    Extracts features from (gulped) dataset
    :param batch_size: number of videos per batch
    :param feature_dump_path: (optional) if set the features are stored as pandas df to the denoted location, else the df is dumped to the default path /DataDumps/Features
    :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels of videos
    """
    if not feature_dump_path:
      feature_dump_path = self.getDumpFileName(type='features')

    self.logger.info('Started extracting {} features from {} dataset'.format(self.extractor.__class__.__name__, self.dataset_name))

    features, labels = self.extractor.computeFeaturesForVideoDataset(self.dataset, pickle_path=feature_dump_path)

    self.logger.info('Finished extracting features from dataset. Features have shape {}.'
                ' Dumped features to {}'.format(np.shape(features), feature_dump_path))

    return features, labels


  def loadFeatures(self, feature_df_path=None):
    """
    loads features from pd dataframe and returns them as a matrix
    :param feature_df_path: path to pandas dataframe that holds features
    :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (list) of videos
    """

    if feature_df_path is None:
      feature_df_path = self.getDumpFileName('features')

    assert os.path.isfile(feature_df_path)
    feature_df = pd.read_pickle(feature_df_path)

    assert 'features' in feature_df and 'labels' in feature_df

    # stack video features to a 2d matrix
    features = np.concatenate(feature_df['features'], axis=0)

    labels = list(feature_df['labels'])

    if features.ndim == 3: # assume only one feature vector is given -> insert dimension
      features = features.reshape((features.shape[0], features.shape[1], 1, features.shape[2]))

    self.logger.info(
      'Loaded {} features from {}.  Features have shape {}'.format(self.extractor.__class__.__name__, feature_df_path,
                                                                   np.shape(features)))

    assert features.ndim == 4 and len(labels) == features.shape[0]
    return features, labels



  def getDumpFileName(self, type):
    assert type in ['features', 'model', 'fv_npy', 'results']
    if type is 'features':
      return os.path.join(self.feature_dumps_dir, 'features_' + self.extractor.__class__.__name__ + '_' + self.dataset_name + '.pickle')
    elif type is 'model':
      return os.path.join(self.model_dumps_dir, 'gmm_' + self.extractor.__class__.__name__ + '_' + self.dataset_name + '.pickle')
    elif type is 'fv_npy':
      return os.path.join(self.fisher_vector_dumps_dir, 'fv_' + self.extractor.__class__.__name__ + '_' + self.dataset_name)
    elif type is 'results':
      return os.path.join(self.results_dir, 'results_{}.pickle'.format(self.dataset_name))



  def trainFisherVectorGMM(self, features, by_bic=True, model_dump_path=None, n_kernels=50):
    """
    Trains a GMM for Fisher Vectors on the given features.
    :param features: features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels (list) of videos
    :param by_bic: denotes whether the gmm fit is chosen based on the lowest BIC
    :param n_kernels: provide if not fitted by bic (bic-method chooses from a fixed set of n_kernels)
    :return: the fitted object of type FisherVectorGMM
    """
    if model_dump_path is None:
      model_dump_path = self.getDumpFileName('model')

    self.logger.info('Started training FisherVector GMM')
    fv_gmm = FisherVectorGMM(n_kernels=n_kernels)
    if by_bic:
      fv_gmm.fit_by_bic(features, model_dump_path=model_dump_path)
    else:
      fv_gmm.fit(features, model_dump_path=model_dump_path)
    assert fv_gmm.fitted
    self.logger.info('Finished training FisherVector GMM. Dumped model to {}'.format(model_dump_path))

    return fv_gmm


  def loadFisherVectorGMM(self, pickle_path=None):
    if pickle_path is None:
      pickle_path = self.getDumpFileName('model')
    assert os.path.isfile(pickle_path)
    with open(pickle_path, 'rb') as f:
      fv_gmm = pickle.load(f)
    assert isinstance(fv_gmm, FisherVectorGMM)

    self.logger.info('Loaded FisherVector GMM from {}'.format(pickle_path))

    return fv_gmm


  def computeFisherVectors(self, features, labels, fv_gmm, normalized=True, mem_map_mode = False, dump_path=None, batch_size=100):
    """
    Compute the (improved) Fisher Vectors of features with a fitted FisherVectorGMM.
    :param features: features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
    :param labels: labels correspodning to features - array of shape (n_videos,)
    :param fv_gmm: fitted FisherVectorGMM instance
    :param normalized: boolean - if true: improved fisher vectors are computed
    :param mem_map_mode: set to True, if the fisher vectors should be computed in batches which are stored as numpy memory map on disk (recommended
    if the computation overflows the available RAM)
    :param dump_path: file path specifying the dump location for the computed vectors
    :param batch_size: batch size for computing the fisher vactors
    :return: (fv, labels) fisher vectors - ndarray of shape (n_videos, n_frames, 2*n_kernels, n_feature_dim)
    """
    assert isinstance(fv_gmm, FisherVectorGMM)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 4

    labels = np.asarray(labels)
    assert labels.ndim ==1
    assert features.shape[0] == labels.shape[0]

    self.logger.info('Started Computing Fisher Vectors')

    if dump_path is None:
      dump_path = self.getDumpFileName('fv_npy')

    """ compute Fisher vectors in batches and store as memory map """
    if mem_map_mode:
      n_instances = features.shape[0]
      n_batches = n_instances // batch_size
      split_tuples = list(zip([i * batch_size for i in range(0, n_batches + 1)], [i*batch_size for i in range(1,n_batches+1)] + [n_instances]))

      fv_shape = (features.shape[0], features.shape[1], fv_gmm.n_kernels*2, features.shape[3])
      fv = np.memmap(dump_path + '_memmap.npy', dtype='float32', mode='w+', shape=fv_shape)

      for i, (batch_start, batch_end) in enumerate(split_tuples):
        batch = features[batch_start:batch_end]
        if batch.shape[0] > 0:
          print('Compute FV batch {} of {}'.format(i+1, n_batches+1))
          fv[batch_start:batch_end] = fv_gmm.predict(batch, normalized=normalized)
          # clean memory
          del batch
          gc.collect()

      np.save(dump_path + '.npy', fv)
      np.save(dump_path + '_labels.npy', labels)
      del fv
      return np.load(dump_path + '.npy', mmap_mode='r'), labels

    else:
      fv = fv_gmm.predict(features, normalized=normalized)
      np.save(dump_path + '.npy', fv)
      np.save(dump_path + '_labels.npy', labels)

    assert fv.shape[0] == labels.shape[0]

    self.logger.info('Finished computing Fisher Vectors. Dumped vectors to {}'.format(dump_path))

    return fv, labels


  def loadFisherVectors(self, fisher_vector_path=None):
    """
    Loads fisher vectors from pd dataframe and returns them as a matrix
    :param feature_df: pandas dataframe which holds features in a column 'features'
    :param feature_df_path: path to pandas dataframe that holds features
    :return: (fv, labels) - fv as ndarray of shape (n_videos, n_frames, 2*n_kernels, n_dim_descriptor) and labels (as array) of videos
    """

    if fisher_vector_path is None:
      fisher_vector_path = self.getDumpFileName('fv_npy')
    assert os.path.isfile(fisher_vector_path + '_labels.npy')
    assert os.path.isfile(fisher_vector_path + '.npy')

    fv = np.load(fisher_vector_path + '.npy', mmap_mode='r')
    labels = np.load(fisher_vector_path + '_labels.npy')
    self.logger.info('Loaded Fisher Vectors from {} with shape {}'.format(fisher_vector_path + '.npy', fv.shape))

    return fv, labels


  def setup_logger(self, logfile_name = '_video_features.log'):
    log_file_path = os.path.join(self.log_dir, logfile_name)
    logger = logging.getLogger('PipelineRunLogger')
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



def create_dir(output_dir):
  assert(output_dir)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  return output_dir

