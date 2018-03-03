import keras.applications.vgg16 as vgg16
import keras.applications.resnet50 as resnet50
from keras.models import Model
from gulpio.loader import DataLoader
import numpy as np
import pandas as pd




class CNNFeatures:

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
      feature_batch = feature_batch.reshape((data_batch.shape[0], data_batch.shape[1], feature_batch.shape[1]))

      feature_batch_list.append(feature_batch)
      labels.extend(label_batch)
      print("batch %i of %i" % (i, n_batches))

    features = np.concatenate(feature_batch_list, axis=0)

    # reshape features to (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
    features = features.reshape((features.shape[0], features.shape[1], 1, features.shape[2]))
    assert features.shape[0] == len(labels) and features.ndim == 4

    # store as pandas dataframe
    if pickle_path:
      df = pd.DataFrame(data={'labels': labels, 'features': np.vsplit(features, features.shape[0])})
      print('Dumped feature dataframe to', pickle_path)
      df.to_pickle(pickle_path)

    return features, labels


class VGGFeatures(CNNFeatures):

  def __init__(self, feature = 'fc1'):
    self.base_model = vgg16.VGG16()
    assert feature in ["fc1", "fc2"]
    self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(feature).output)

  def computeFeatures(self, video):
    x = vgg16.preprocess_input(video)
    features = self.model.predict(x)
    return features


class ResNetFeatures(CNNFeatures):
  def __init__(self):
    self.base_model = resnet50.ResNet50()
    self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('avg_pool').output)

  def computeFeatures(self, video):
    x = resnet50.preprocess_input(video)
    features = self.model.predict(x)
    return features.reshape((-1, 2048))

