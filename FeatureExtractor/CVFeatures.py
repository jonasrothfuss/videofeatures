import sys, glob, argparse
import numpy as np
import math, cv2


class SIFTFeatures:

  def __init__(self):
    pass

  def image_descriptors(self, file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (256, 256))
    _, descriptors = cv2.xfeatures2d.SIFT_create(nfeatures=5).detectAndCompute(img, None)
    return descriptors


if __name__ == "__main__":
  dir =  "/home/jonasrothfuss/Downloads/Armar_Experiences_Download/video_frames/1"
