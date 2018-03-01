from Pipeline import Pipeline
from FeatureExtractor import SIFTFeatures, VGGFeatures, ResNetFeatures, SURFFeatures
from DatasetProvider import TwentyBNDataset
import configparser
from DatasetProvider import ArmarDataset, ActivityNetDataset





def main():
  config = configparser.ConfigParser()
  config.read('config.ini')

  #train_dir = config['GULP']['twentybn_train_gulp']
  #valid_dir = config['GULP']['twentybn_valid_gulp']



if __name__ == '__main__':
  main()

