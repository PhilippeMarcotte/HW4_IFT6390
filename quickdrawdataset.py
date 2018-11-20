from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import os
class QuickDrawLoader(DataLoader):
    def validation(self):
        self.dataset.validation()

    def training(self):
        self.dataset.train()

class QuickDrawDataset(Dataset):
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(self, root, split='train', transform=None, target_transform=None):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train/test/unlabeled set
        self.classes = ['sink', 'pear', 'moustache', 'nose', 'skateboard', 'penguin', 'peanut', 'skull', 'panda',
                        'paintbrush', 'nail', 'apple', 'rifle', 'mug', 'sailboat', 'pineapple',
                        'spoon', 'rabbit', 'shovel', 'rollerskates', 'screwdriver', 'scorpion', 'rhinoceros', 'pool',
                        'octagon', 'pillow', 'parrot', 'squiggle', 'mouth', 'empty', 'pencil']

        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.labels = self.__loadfile("train_images.npy", "train_labels.csv")
        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])

        self.mean = np.mean(self.data)
        self.stdv = np.std(self.data)

        self.transforms_type = 'train'

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

            # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform[self.transforms_type](np.transpose(img, (1,2,0)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.loadtxt(f, delimiter=",", skiprows=1, dtype=str)[:, 1]
                labels_to_index = {k: v for v, k in enumerate(self.classes)}
                labels = [labels_to_index[label] for label in labels]

        path_to_data = os.path.join(self.root, data_file)
        # read whole file in uint8 chunks
        everything = np.load(path_to_data, encoding='latin1')
        images = np.array([t[1].reshape(1, 100, 100) for t in everything]).astype('uint8')

        return images, labels

    def validation(self):
        self.transforms_type = 'valid'

    def train(self):
        self.transforms_type = 'train'