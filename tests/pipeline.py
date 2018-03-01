from DatasetProvider import TwentyBNDataset
from FeatureExtractor import VGGFeatures
from Pipeline import Pipeline
import unittest
import configparser


class ArmarExperiencesTest(unittest.TestCase):
  def setUp(self):
    self.config = configparser.ConfigParser()
    self.config.read('../config.ini')
    self.train_dir = self.config['GULP']['twentybn_test_gulp']
    self.valid_dir = self.config['GULP']['twentybn_test_gulp']
    self.dataset = TwentyBNDataset(batch_size=20, train_dir=self.train_dir, valid_dir=self.valid_dir).getDataLoader(train=False)
    self.extractor = VGGFeatures(feature='fc2')
    self._base_dir = "../datadumps"

    self.pipeline = Pipeline(dataset=self.dataset, extractor=self.extractor, base_dir=self._base_dir)

  def test_feature_extraction(self):
    self.pipeline.extractFeatures()

  def test_feature_loading_and_GMM_training(self):
    features, labels = self.pipeline.loadFeatures()
    self.pipeline.trainFisherVectorGMM(features)

  def test_load_trained_gmm(self):
    self.pipeline.loadFisherVectorGMM()

  def test_compute_fv(self):
    features, labels = self.pipeline.loadFeatures()
    fv_gmm = self.pipeline.loadFisherVectorGMM()
    self.pipeline.computeFisherVectors(features=features, labels=labels, fv_gmm=fv_gmm)

  def test_load_fv(self):
    self.pipeline.loadFisherVectors()

if __name__ == '__main__':
  unittest.main()



