from keras.preprocessing.image import load_img, img_to_array
import os, glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ImageExtensions = ["*.png", "*.jpg"]

class VideoDataset:

  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self.root_dir = root_dir


class Video:
  def __init__(self,  video_dir):
    assert os.path.isdir(video_dir)
    self.video_dir = video_dir
    self.frames = getFramePaths(video_dir)

  def getVideoAsArray(self, n_frames=None, target_size=(224,224)):
    assert len(target_size) == 2
    if n_frames is None:
      n_frames = len(self.frames)

    selectFrames = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
    print(selectFrames(n_frames, len(self.frames)))
    frame_paths = [self.frames[i] for i in selectFrames(n_frames, len(self.frames))]

    n_channels = 3

    video = np.empty(shape=(n_frames, target_size[0], target_size[1], n_channels))
    for i, frame_path in enumerate(frame_paths):
      frame = load_img(frame_path, target_size=target_size) #load_img(frame_path, target_size=target_size)
      video[i] = frame

    return video


def getFramePaths(video_dir):
  frames = []
  for ext in ImageExtensions:
    frames.extend(glob.glob(os.path.join(video_dir, ext)))
  frames = sorted(frames)
  return frames
