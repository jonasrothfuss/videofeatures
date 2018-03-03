import configparser
import unittest
import numpy as np

from videofeatures.TwentyBNDatasetProvider import TwentyBNDataset
from videofeatures.CNNFeatures import ResNetFeatures
from videofeatures.VideoFeatures import VideoFeatures


class PipelineTwentyBNTest(unittest.TestCase):
  def setUp(self):
    self.config = configparser.ConfigParser()
    self.config.read('../config.ini')
    self.train_dir = self.config['GULP']['train_data_gulp']
    self.valid_dir = self.config['GULP']['valid_data_gulp']
    self.dataset = TwentyBNDataset(batch_size=20, train_dir=self.train_dir, valid_dir=self.valid_dir).getDataLoader(
                                                                                      train=False)
    self.extractor = ResNetFeatures()
    self._base_dir = "../output"
    self.pipeline = VideoFeatures(dataset=self.dataset, extractor=self.extractor, dataset_name="twentybn",
                             base_dir=self._base_dir)


  def test_00_feature_extraction_gulp(self):
    self.pipeline.extractFeatures()

  def test_01_load_features_and_train_gmm_gulp(self):
    features, labels = self.pipeline.loadFeatures()
    self.pipeline.trainFisherVectorGMM(features)

  def test_01_load_trained_gmm_gulp(self):
    self.pipeline.loadFisherVectorGMM()

  def test_02_compute_fv_gulp(self):
    features, labels = self.pipeline.loadFeatures()
    fv_gmm = self.pipeline.loadFisherVectorGMM()
    self.pipeline.computeFisherVectors(features=features, labels=labels, fv_gmm=fv_gmm)

  def test_03_load_fv_gulp(self):
    self.pipeline.loadFisherVectors()



class PipelineTest(unittest.TestCase):
  def setUp(self):
    """ use a dummy Pipeline without setting up dataset and extractor"""
    self._base_dir = "../output"
    self.pipeline = VideoFeatures(dataset=None, extractor=None, dataset_name="nprandom",
                             base_dir=self._base_dir)


  def test_00_load_features_and_train_gmm(self):
    features = np.random.normal(size=(50, 20, 80, 1))
    self.pipeline.trainFisherVectorGMM(features)

  def test_01_compute_fv(self):
    features = np.random.normal(size=(50, 20, 80, 1))
    labels = np.random.randint(1, 10, size=50)
    fv_gmm = self.pipeline.loadFisherVectorGMM()
    self.pipeline.computeFisherVectors(features=features, labels=labels, fv_gmm=fv_gmm)

  def test_02_load_fv(self):
    self.pipeline.loadFisherVectors()


if __name__ == '__main__':
  test_classes_to_run = [PipelineTest, PipelineTwentyBNTest]

  loader = unittest.TestLoader()
  suites_list = []
  for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suites_list.append(suite)

  big_suite = unittest.TestSuite(suites_list)

  runner = unittest.TextTestRunner()
  results = runner.run(big_suite)

  unittest.main()



