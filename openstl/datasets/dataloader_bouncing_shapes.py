import cv2
import gzip
import numpy as np
import os
import random

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def load_fixed_set(root, data_name='bouncing_shapes_train'):
    # Load the fixed dataset
    img_file_map = {
        'bouncing_shapes_train': 'bouncing_shapes/bouncing_shapes_train_seq.npy',
        'bouncing_shapes_test': 'bouncing_shapes/bouncing_shapes_test_seq.npy',
    }
    label_file_map = {
        'bouncing_shapes_train': 'bouncing_shapes/bouncing_shapes_train_label.npy',
        'bouncing_shapes_test': 'bouncing_shapes/bouncing_shapes_test_label.npy',
    }
    path = os.path.join(root, img_file_map[data_name])
    img_dataset = np.load(path, allow_pickle=True)
    img_dataset = img_dataset[..., np.newaxis]

    label_data = np.load(os.path.join(root, label_file_map[data_name]), allow_pickle=True)
    return img_dataset, label_data


class BouncingShapesDataset(Dataset):
    """
    Args:
        data_root (str): Path to the dataset.
        is_train (bool): Whether to use the train or test set.
        n_frames_input, n_frames_output (int): The number of input and prediction
            video frames.
        image_size (int): Input resolution of the data.
        num_objects (list): The number of moving objects in videos.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, root, is_train=True, data_name='bouncing_shapes',
                 n_frames_input=10, n_frames_output=10, image_size=64,
                 num_objects=[2], transform=None, use_augment=False):
        super(BouncingShapesDataset, self).__init__()

        self.dataset = None
        self.is_train = is_train
        self.data_name = data_name
        if self.is_train:
            self.dataset, self.labels = load_fixed_set(root, 'bouncing_shapes_train')
        else:
            self.dataset, self.labels = load_fixed_set(root, 'bouncing_shapes_test')

        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.use_augment = use_augment
        self.background = 'cifar' in data_name
        # For generating data
        self.image_size_ = image_size
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1



    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        images = self.dataset[:, idx, ...]
        labels = self.labels[:, idx, ...]

        if not self.background:
            r, w = 1, self.image_size_
            images = images.reshape((length, w, r, w, r)).transpose(
                0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        else:
            images = images.transpose(0, 3, 1, 2)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()

        # map the 0th label to be 'triangle'->0, 'circle'->1, 'rectangle'->2
        labels[:, 0] = np.array([0 if x == 'triangle' else 1 if x == 'circle' else 2 for x in labels[:, 0]])
        # print("labels: ", labels.shape)
        # print(f"labels: {labels}")

        # Ensure labels are of a supported type
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)

        if self.use_augment:
            imgs = self._augment_seq(torch.cat([input, output], dim=0), crop_scale=0.94)
            input = imgs[:self.n_frames_input, ...]
            output = imgs[self.n_frames_input:self.n_frames_input+self.n_frames_output, ...]
        

        return input, output, labels

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='bouncing_shapes',
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    image_size = in_shape[-1] if in_shape is not None else 64
    train_set = BouncingShapesDataset(root=data_root, is_train=True, data_name=data_name,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length, num_objects=[1],
                            image_size=image_size, use_augment=use_augment)
    test_set = BouncingShapesDataset(root=data_root, is_train=False, data_name=data_name,
                           n_frames_input=pre_seq_length,
                           n_frames_output=aft_seq_length, num_objects=[1],
                           image_size=image_size, use_augment=False)

    dataloader_train = create_loader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True, is_training=True,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, persistent_workers=True,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  data_name='bouncing_shapes',
                  pre_seq_length=10, aft_seq_length=10,
                  distributed=True, use_prefetcher=False)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape, item[2].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape, item[2].shape)
        break
