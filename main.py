from Pipeline import Pipeline
from FeatureExtractor import SIFTFeatures, VGGFeatures, ResNetFeatures, SURFFeatures
from DatasetProvider import TwentyBNDataset
from DatasetProvider import ArmarDataset, ActivityNetDataset


_base_dir = "./DataDumps"

def extract_features_train_gmm_compute_fv(dataset, extractor):

  pipeline = Pipeline(dataset=dataset, extractor=extractor, base_dir = _base_dir)

  pipeline.extractFeatures()

  """ test feature loading"""
  features, labels = pipeline.loadFeatures()

  _ = pipeline.trainFisherVectorGMM(features)

  """ test gmm loading """
  fv_gmm = pipeline.loadFisherVectorGMM()

  _, _ = pipeline.computeFisherVectors(features=features, labels=labels, fv_gmm=fv_gmm)

  """ test fv loading """
  fv, labels = pipeline.loadFisherVectors()



def main():
  twentybn_train = TwentyBNDataset(batch_size=20).getDataLoader(train=True)
  vgg = VGGFeatures(feature='fc2')
  extract_features_train_gmm_compute_fv(twentybn_train, vgg)


if __name__ == '__main__':
  main()

