import cv2
from FeatureExtractor.BaseFeatureExtractor import BaseFeatures
from gulpio.loader import DataLoader
import numpy as np
import pandas as pd

class CVFeatures(BaseFeatures):

  def computeFeaturesForVideoDataset(self, dataloader, pickle_path=None):
    """
    Computes Feature Vectors for the video dataset provided via a dataloader object
    :param dataloader: gulpIO Dataloader object which represents a dataset
    :param pickle_path: (optional) if provided the features are pickeled to the specified location
    :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels of videos
    """
    assert isinstance(dataloader, DataLoader)

    feature_batch_list = []
    labels = []
    n_batches = len(dataloader)
    for i, (data_batch, label_batch) in enumerate(dataloader):
      assert data_batch.ndim == 5
      n_frames = data_batch.shape[1]

      frames_batch = data_batch.reshape(
        (data_batch.shape[0] * n_frames, data_batch.shape[2], data_batch.shape[3], data_batch.shape[4]))
      frames_batch = frames_batch.astype('float32')

      feature_batch = self.computeFeatures(frames_batch)
      assert feature_batch.ndim == 2
      feature_batch = feature_batch.reshape((data_batch.shape[0], data_batch.shape[1], -1, feature_batch.shape[1]))

      feature_batch_list.append(feature_batch)
      labels.extend(label_batch)
      print("batch %i of %i" % (i, n_batches))

    features = np.concatenate(feature_batch_list, axis=0)
    assert features.shape[0] == len(labels) and features.ndim == 4

    if pickle_path:
      df = pd.DataFrame(data={'labels': labels, 'features': np.vsplit(features, features.shape[0])})
      print('Dumped feature dataframe to', pickle_path)
      df.to_pickle(pickle_path)

    return features, labels

class SIFTFeatures(CVFeatures):

  def __init__(self, n_descriptors=5):
    self.n_descriptors = n_descriptors

  def computeFeatures(self, video):
    """
    todo: improve documentation
    Computes SIFT features for a single video.
    :param video: a video of shape (n_frames, width, height, channel)
    :return: the features, shape ()
    """
    descriptor_array = []
    for i in range(video.shape[0]):
      frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY).astype('uint8')
      _, descriptors = cv2.xfeatures2d.SIFT_create(nfeatures=self.n_descriptors).detectAndCompute(frame, None)

      if descriptors is not None:
        if descriptors.shape[0] < self.n_descriptors:
          descriptors = np.concatenate([descriptors, np.zeros((self.n_descriptors - descriptors.shape[0], 128))], axis=0)
        else:
          descriptors = descriptors[:self.n_descriptors]
      else:
          descriptors = np.zeros((self.n_descriptors, 128))

      assert descriptors.shape == (self.n_descriptors, 128)
      descriptor_array.append(descriptors)
    features = np.concatenate(descriptor_array, axis=0)
    return features

class SURFFeatures(CVFeatures):
  # todo: documentation
  def __init__(self, n_descriptors=5):
    self.n_descriptors = n_descriptors

  def computeFeatures(self, video):
    descriptor_array = []
    for i in range(video.shape[0]):
      frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY).astype('uint8')
      _, descriptors = cv2.xfeatures2d.SURF_create().detectAndCompute(frame, None)

      # make sure that descriptors have shape (n_descriptor, 64)
      if descriptors is not None:
        if descriptors.shape[0] < self.n_descriptors:
          descriptors = np.concatenate([descriptors, np.zeros((self.n_descriptors - descriptors.shape[0], 64))],
                                       axis=0)
        else:
          descriptors = descriptors[:self.n_descriptors]
      else:
        descriptors = np.zeros((self.n_descriptors, 64))

      assert descriptors.shape == (self.n_descriptors, 64)
      descriptor_array.append(descriptors)

    return np.concatenate(descriptor_array, axis=0)

