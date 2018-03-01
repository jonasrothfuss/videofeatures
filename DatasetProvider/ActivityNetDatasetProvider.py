import os, random, csv
from gulpio.adapters import AbstractDatasetAdapter, Custom20BNAdapterMixin
from gulpio.transforms import CenterCrop, ComposeVideo, Scale
from gulpio.dataset import GulpVideoDataset
from gulpio.loader import DataLoader



from gulpio.utils import (
                    resize_images,
                    burst_video_into_frames,
                    temp_dir_for_bursting,
                    remove_entries_with_duplicate_ids,
                    )



class ActivityNetDataset:

  def __init__(self, train_dir, valid_dir, batch_size=200, n_frames=20):

    transforms = ComposeVideo([CenterCrop(128), Scale((224, 224))])
    self.n_frames = n_frames

    self.train_dataset = GulpVideoDataset(train_dir, n_frames, 1, False, transform=transforms)
    self.train_loader = DataLoader(self.train_dataset, batch_size=10, shuffle=False, num_workers=8, drop_last=True)

    self.val_dataset = GulpVideoDataset(valid_dir, n_frames, 1, False, transform=transforms)
    self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

  def getDataLoader(self, train=False):
    if train:
      return self.train_loader
    else:
      return self.val_loader


class ActivityNetCsvAviAdapter(AbstractDatasetAdapter,
                               Custom20BNAdapterMixin):
  """ Adapter for ActivityNet dataset specified by CSV file and avi videos. """

  def __init__(self, csv_file, folder, output_folder,
               shuffle=False, frame_size=-1, frame_rate=12,
               shm_dir_path='/dev/shm', label_name='label',
               remove_duplicate_ids=False):
    self.data = self.read_csv(csv_file)
    self.label_name = label_name
    self.output_folder = output_folder
    self.labels2idx = self.create_label2idx_dict(self.label_name)
    self.folder = folder
    self.shuffle = bool(shuffle)
    self.frame_size = int(frame_size)
    self.frame_rate = int(frame_rate)
    self.shm_dir_path = shm_dir_path
    self.all_meta = self.get_meta()
    if remove_duplicate_ids:
      self.all_meta = remove_entries_with_duplicate_ids(
        self.output_folder, self.all_meta)
    if self.shuffle:
      random.shuffle(self.all_meta)

  def read_csv(self, csv_file):
    with open(csv_file, newline='\n') as f:
      content = csv.reader(f, delimiter=';')
      data = []
      for row in content:
        if len(row) == 1:  # For test case
          data.append({'id': row[0], 'label': "dummy"})
        else:  # For train and validation case
          data.append({'id': row[0], 'label': row[1], 'file_name': row[4]})
    return data

  def get_meta(self):
    return [{'id': entry['id'],
             'label': entry[self.label_name],
             'file_name': entry['file_name'],
             'idx': self.labels2idx[entry[self.label_name]]}
            for entry in self.data]

  def __len__(self):
    return len(self.all_meta)


  def iter_data(self, slice_element=None):
    slice_element = slice_element or slice(0, len(self))
    for meta in self.all_meta[slice_element]:
      video_path = os.path.join(self.folder, str(meta['file_name']))

      with temp_dir_for_bursting(self.shm_dir_path) as temp_burst_dir:
        frame_paths = burst_video_into_frames(
          video_path, temp_burst_dir, frame_rate=self.frame_rate)
        frames = list(resize_images(frame_paths, self.frame_size))
      result = {'meta': meta,
                'frames': frames,
                'id': meta['id']}
      yield result
    else:
      self.write_label2idx_dict()
