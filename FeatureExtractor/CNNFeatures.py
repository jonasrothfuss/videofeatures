from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
import os

VIDEO_PATH = "/home/jonasrothfuss/Downloads/Armar_Experiences_Download/video_frames/1"

class VGG:
  def __init__(self, feature = 'fc1'):
    self.base_model = VGG16()
    assert feature in ["fc1", "fc2"]
    self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(feature).output)

  def computeFeatures(self, video):
    x = preprocess_input(video)
    features = self.model.predict(x)
    return features


def main():
  model = VGG()


if __name__ == "__main__":
  main()