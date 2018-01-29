from DatasetProvider import VideoDataset, Video
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from FeatureExtractor import VGG, SIFTFeatures


class ArmarDataset(VideoDataset):

  def __init__(self, root_dir):
    super(ArmarDataset, self).__init__(root_dir)










if __name__ == "__main__":
  root_dir = "/home/jonasrothfuss/Downloads/Armar_Experiences_Download/"
  #ArmarDataset()
  v = Video("/home/jonasrothfuss/Downloads/Armar_Experiences_Download/video_frames/29")
  #video_array = v.getVideoAsArray(n_frames=5)
  #vgg = VGG()
  #feat = vgg.computeFeatures(video_array)
  #print(feat.shape)
  frames = v.frames
  sift = SIFTFeatures()
  a = sift.image_descriptors(frames[1])
  print(a.shape)
  print(a)