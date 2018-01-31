import numpy as np
import pandas as pd
import os

SIFT_FEAT_FILE = '/PDFData/rothfuss/data/20bn-something/SIFT/20bn_val_SIFT/20bn_val_sift_test.txt'
CSV_20BN_VALID = '/PDFData/rothfuss/data/20bn-something/something-something-v1-validation.csv'
FeatureDumpsDir = './../DataDumps/Features'


def readSIFTFeatureFile(file_path, n_descriptors=30, pickle_path=None):
  """
  Reads file containing SIFT features with the following format

  # video_id
  feature1a feature1b feature1c ...
  feature2a feature2b ...
  .....

  # video_id
  feature1a feature1b feature1c ...
  feature2a feature2b ...
  .....

  :param file_path: path to file containing the sift features
  :param n_descriptors: number of descriptors per video - videos that have less descriptors are omitted
  :param pickle_path: (optional) if set, the features and labels are stored as pandas dataframe to the specified location
  :return: (features, labels) - features as ndarray of shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor) and labels of videos
  """

  assert os.path.isfile(file_path)

  with open(file_path, 'r') as f:
    f.readline() # pops first line
    video_id = int(f.readline().split(' ')[1]) # get first video id
    feat_array = []

    video_id_list = []
    features_list = []

    for line in f:
      if '#' in line:

        if len(feat_array) > n_descriptors:
          # add features and video id to lists
          # if not enough features or no features provided -> skip this part
          features = np.stack(feat_array)[:n_descriptors,:]
          features_list.append(features)
          video_id_list.append(video_id)

        # set new video id
        video_id = int(line.split(' ')[1])
        feat_array = [] # reset feat_array
      elif len(line) > 500:
        feat = np.asarray([float(e) for e in line.split(' ')[:-1]])
        feat_array.append(feat)

    assert len(video_id_list) == len(features_list)

    features = np.stack(features_list)
    assert features.ndim == 3

    features = features.reshape((features.shape[0], features.shape[1], 1, features.shape[2]))

    id_label_dict = pd.read_csv(CSV_20BN_VALID, delimiter=';', names=['ids', 'labels'], index_col=0).to_dict()['labels']

    labels = [id_label_dict[int(id)] for id in video_id_list]

    if pickle_path:
      df = pd.DataFrame(data={'id': video_id_list, 'labels': labels, 'features': np.vsplit(features, features.shape[0])})
      print(df)
      print('Dumped feature dataframe to', pickle_path)
      df.to_pickle(pickle_path)

    return features, labels

def main():
  readSIFTFeatureFile(SIFT_FEAT_FILE, pickle_path=os.path.join(FeatureDumpsDir, '20bn_val_sift.pickle'))

if __name__ == '__main__':
  main()