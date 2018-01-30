from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from DatasetProvider import TwentyBNDataset
from gulpio.loader import DataLoader
import numpy as np
import pandas as pd

from FeatureExtractor.BaseFeatureExtractor import BaseFeatures

VIDEO_PATH = "/home/jonasrothfuss/Downloads/Armar_Experiences_Download/video_frames/1"

class VGGFeatures(BaseFeatures):
  def __init__(self, feature = 'fc1'):
    self.base_model = VGG16()
    assert feature in ["fc1", "fc2"]
    self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(feature).output)

  def computeFeatures(self, video):
    x = preprocess_input(video)
    features = self.model.predict(x)
    return features

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
      feature_batch = feature_batch.reshape((data_batch.shape[0], data_batch.shape[1], feature_batch.shape[1]))

      feature_batch_list.append(feature_batch)
      labels.extend(label_batch)
      print("batch %i of %i" % (i, n_batches))

    features = np.concatenate(feature_batch_list, axis=0)
    assert features.shape[0] == len(labels)
    df = pd.DataFrame(data={'label_id': labels, 'features': np.vsplit(features, features.shape[0])})
    if pickle_path:
      print('Dumped feature dataframe to', pickle_path)
      df.to_pickle(pickle_path)

    return df


def main():
  model = VGGFeatures()
  loader = TwentyBNDataset(batch_size=20).getDataLoader()
  model.computeFeaturesForVideoDataset(loader, pickle_path="/common/homes/students/rothfuss/Desktop/vgg_fc1_20bn.pickle")

if __name__ == "__main__":
  main()