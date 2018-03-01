from gulpio.transforms import CenterCrop, ComposeVideo, Scale
from gulpio.dataset import GulpVideoDataset
from gulpio.loader import DataLoader


class TwentyBNDataset:

  def __init__(self, train_dir, valid_dir, batch_size=20, n_frames=20):
    transforms = ComposeVideo([CenterCrop(128), Scale((224, 224))])
    self.n_frames = n_frames

    self.train_dataset = GulpVideoDataset(train_dir, n_frames, 1, False, transform=transforms)
    self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    self.val_dataset = GulpVideoDataset(valid_dir, n_frames, 1, False, transform=transforms)
    self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

  def getDataLoader(self, train=False):
    if train:
      return self.train_loader
    else:
      return self.val_loader


