import sys, glob, argparse
import numpy as np
import math, cv2
from FeatureExtractor.BaseFeatureExtractor import BaseFeatures
from DatasetProvider import TwentyBNDataset
from gulpio.loader import DataLoader
import numpy as np
import pandas as pd

class CVFeatures(BaseFeatures):

  def computeFeaturesForVideoDataset(self, dataloader, pickle_path=None):
    """
    Computes Feature Vectors for the video dataset provided via a dataloader object
    :param dataloader: gulpIO Dataloader object which represents a dataset
    :param pickle_path: (optional) if provided the features are pickeled to the specified location
    :return: pandas Dataframe with two columns holding the featurevectors and the label_id of each video
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
      if i == 20:
        break

    features = np.concatenate(feature_batch_list, axis=0)
    assert features.shape[0] == len(labels)
    df = pd.DataFrame(data={'label_id': labels, 'features': np.vsplit(features, features.shape[0])})
    print(df)
    if pickle_path:
      df.to_pickle(pickle_path)

    return df

class SIFTFeatures(CVFeatures):

  def __init__(self, n_descriptors=5):
    self.n_descriptors = n_descriptors

  def computeFeatures(self, video):
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

  def __init__(self, n_descriptors=5):
    self.n_descriptors = n_descriptors

  def computeFeatures(self, video):
    descriptor_array = []
    for i in range(video.shape[0]):
      frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY).astype('uint8')
      _, descriptors = cv2.xfeatures2d.SURF_create().detectAndCompute(frame, None)

      # make sure that descriptors has shape (n_descriptor, 64)
      if descriptors is not None:
        if descriptors.shape[0] < self.n_descriptors:
          descriptors = np.concatenate([descriptors, np.zeros((self.n_descriptors - descriptors.shape[0], 128))],
                                       axis=0)
        else:
          descriptors = descriptors[:self.n_descriptors]
      else:
        descriptors = np.zeros((self.n_descriptors, 64))

      assert descriptors.shape == (self.n_descriptors, 64)
      descriptor_array.append(descriptors)

    return np.concatenate(descriptor_array, axis=0)



def main():
  model = SIFTFeatures()
  loader = TwentyBNDataset(batch_size=20).getDataLoader()
  model.computeFeaturesForVideoDataset(loader, pickle_path="/common/homes/students/rothfuss/Desktop/SIFT_fc1_20bn.pickle")

if __name__ == "__main__":
  main()