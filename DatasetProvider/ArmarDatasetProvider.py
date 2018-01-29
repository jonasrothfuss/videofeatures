from gulpio import GulpDirectory
from gulpio.transforms import CenterCrop, ComposeVideo, Scale
from gulpio.dataset import GulpVideoDataset
from gulpio.loader import DataLoader
from gulpio.adapters import AbstractDatasetAdapter, Custom20BNAdapterMixin
from numpy import random
import csv
import os

from gulpio.utils import (get_single_video_path,
                    find_images_in_folder,
                    resize_images,
                    resize_by_short_edge,
                    burst_video_into_frames,
                    temp_dir_for_bursting,
                    remove_entries_with_duplicate_ids,
                    ImageNotFound,
                    )

TRAIN_GULP_DIR = '/PDFData/ferreira/data/armar-experiences-gulp/train'
VALID_GULP_DIR = '/PDFData/ferreira/data/armar-experiences-gulp/valid'

class ArmarDataset:

  def __init__(self, batch_size=20, n_frames=20):

    transforms = ComposeVideo([CenterCrop(128), Scale((224, 224))])
    self.n_frames = n_frames

    self.train_dataset = GulpVideoDataset(TRAIN_GULP_DIR, 20, 1, False, transform=transforms)
    self.train_loader = DataLoader(self.train_dataset, batch_size=10, shuffle=False, num_workers=8, drop_last=True)

    self.val_dataset = GulpVideoDataset(VALID_GULP_DIR, n_frames, 1, False, transform=transforms)
    self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)


  def getDataLoader(self, train=False):
    if train:
      return self.train_loader
    else:
      return self.val_loader

class CustomArmarExperiencesCsvPngAdapter(AbstractDatasetAdapter, Custom20BNAdapterMixin):
    def __init__(self, csv_file, folder, output_folder,
                 shuffle=False, frame_size=-1, shm_dir_path='/dev/shm'):
        self.data = self.read_csv(csv_file)
        self.output_folder = output_folder
        self.labels2idx = self.create_label2idx_dict('label')
        self.folder = folder
        self.shuffle = shuffle
        self.frame_size = frame_size
        self.shm_dir_path = shm_dir_path
        self.all_meta = self.get_meta()
        if self.shuffle:
            random.shuffle(self.all_meta)

    def read_csv(self, csv_file):
        with open(csv_file, newline='\n') as f:
            content = csv.reader(f, delimiter=',')
            data = []
            for row in content:
                data.append({'id': row[0], 'label': row[1]})
        return data

    def get_meta(self):
        return [{'id': entry['id'],
                 'label': entry['label'],
                 'idx': self.labels2idx[entry['label']]}
                for entry in self.data]

    def __len__(self):
        return len(self.data)

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for meta in self.all_meta[slice_element]:
            video_folder = os.path.join(self.folder, str(meta['id']))
            frame_paths = find_images_in_folder(video_folder, formats=['png'])
            frames = list(resize_images(frame_paths, self.frame_size))
            result = {'meta': meta,
                      'frames': frames,
                      'id': meta['id']}
            yield result
        else:
            self.write_label2idx_dict()


