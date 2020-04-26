""" Module for the data loading pipeline for the model to train """

import os
import numpy as np
from torch.utils.data import Dataset

class RawTextFace2TextDataset(Dataset):
    """ PyTorch Dataset wrapper around the Face2Text dataset
        Raw text version
    """

    def __load_data(self):
        """
        private helper for loading the annotations and file names from the annotations file
        :return: images, descs => images and descriptions
        """
        from data_processing.TextExtractor import read_annotations, basic_preprocess
        images, descs = read_annotations(self.annots_file_path)
        # preprocess the descriptions:
        descs = basic_preprocess(descs)

        return images, descs

    def __init__(self, annots_file, img_dir, img_transform=None):
        """
        constructor of the class
        :param annots_file: annotations file
        :param img_dir: path to the images directory
        :param img_transform: torch_vision transform to apply
        """

        # create state:
        self.base_path = img_dir
        self.annots_file_path = annots_file
        self.transform = img_transform

        self.images, self.descs = self.__load_data()

        # extract all the data

    def __len__(self):
        """
        obtain the length of the data-items
        :return: len => length
        """
        return len(self.images)

    def __getitem__(self, ix):
        """
        code to obtain a specific item at the given index
        :param ix: index for element query
        :return: (caption, img) => caption and the image
        """

        # read the image at the given index
        img_file_path = os.path.join(self.base_path, self.images[ix])
        img = PIL.Image.open(img_file_path)

        # transform the image if required
        if self.transform is not None:
            img = self.transform(img)

        # get the raw_text caption:
        caption = self.descs[ix]

        # return the data element
        return caption, img

def get_transform(new_size=None, flip_horizontal=False):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :param flip_horizontal: Whether to randomly mirror input images during training
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize, \
        RandomHorizontalFlip

    if not flip_horizontal:
        if new_size is not None:
            image_transform = Compose([
                Resize(new_size),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        else:
            image_transform = Compose([
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    else:
        if new_size is not None:
            image_transform = Compose([
                RandomHorizontalFlip(p=0.5),
                Resize(new_size),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        else:
            image_transform = Compose([
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    return image_transform


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dl
