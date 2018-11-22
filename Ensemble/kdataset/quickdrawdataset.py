from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import os
class TestQuickDrawLoader(DataLoader):
    def __iter__(self):
        self.dataset.testing()
        print(self.dataset.transforms_type, "transforms")
        return super(TestQuickDrawLoader, self).__iter__()

class ValidationQuickDrawLoader(DataLoader):
    def __iter__(self):
        self.dataset.validation()
        print(self.dataset.transforms_type, "transforms")
        return super(ValidationQuickDrawLoader, self).__iter__()

class TrainingQuickDrawLoader(DataLoader):
    def __iter__(self):
        self.dataset.training()
        print(self.dataset.transforms_type, "transforms")
        return super(TrainingQuickDrawLoader, self).__iter__()

class QuickDrawDataset(Dataset):
    splits = ('train', 'test')
    train_imgs_name = "train_images.npy"
    test_imgs_name = "test_images.npy"

    def __init__(self, root, split='train', transform=None, target_transform=None):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train/test/unlabeled set
        self.classes = np.array(['sink', 'pear', 'moustache', 'nose', 'skateboard', 'penguin', 'peanut', 'skull', 'panda',
                        'paintbrush', 'nail', 'apple', 'rifle', 'mug', 'sailboat', 'pineapple',
                        'spoon', 'rabbit', 'shovel', 'rollerskates', 'screwdriver', 'scorpion', 'rhinoceros', 'pool',
                        'octagon', 'pillow', 'parrot', 'squiggle', 'mouth', 'empty', 'pencil'])
        self.labels_to_index = {k: v for v, k in enumerate(self.classes)}


        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.labels = self.__loadfile__(self.train_imgs_name, "train_labels.csv")
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile__(self.test_imgs_name)

        self.training()

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
            img, target = self.data[index], 0

            # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform[self.transforms_type](np.transpose(img, (1,2,0)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def __loadfile__(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.loadtxt(f, delimiter=",", skiprows=1, dtype=str)[:, 1]
                labels = [self.labels_to_index[label] for label in labels]

        path_to_data = os.path.join(self.root, data_file)
        # read whole file in uint8 chunks
        everything = np.load(path_to_data, encoding='latin1')
        images = np.array([t[1].reshape(1, 100, 100) for t in everything]).astype('uint8')

        return images, labels

    def testing(self):
        self.transforms_type = 'test'

    def validation(self):
        self.transforms_type = 'valid'

    def training(self):
        self.transforms_type = 'train'

class DenoisedQuickDrawDataset(QuickDrawDataset):
    train_imgs_name = "train_images_denoised.npy"
    test_imgs_name = "test_images_denoised.npy"