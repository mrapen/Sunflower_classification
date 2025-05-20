"""This module is designed to read and process data"""

import random
import os
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class CustomDataset(Dataset):
    """This class is designed to read and process data"""
    def __init__(self, root, class_names = None, transformations = None):
        self.transformations = transformations
        data_directory = os.path.join(os.path.dirname(__file__), '..', root)
        self.images = np.load(os.path.join(data_directory, 'Sunflower_Stages.npy'))
        self.labels = np.load(os.path.join(data_directory, 'Sunflower_Stages_Labels.npy'))
        self.labels = self.labels.flatten()
        self.class_names, self.class_counts, count = {} if not class_names else class_names, {}, 0
        for _, (_, label) in enumerate(zip(self.images, self.labels)):
            if type(label) is list:
                label = label[0]
            if self.class_names is not None and label not in self.class_names:
                self.class_names[label] = count
                count += 1
            if label not in self.class_counts:
                self.class_counts[label] = 1
            else:
                self.class_counts[label] += 1

    def __len__(self):
        return len(self.images)

    def get_pos_neg_ims(self, qry_label):
        """This method returns random items from lists of matching and
        non-matching images and names for a classification name query"""
        positive_images_paths = [image for (image, label) in
                                zip(self.images, self.labels) if qry_label == label]
        negative_images_paths = [image for (image, label) in
                                zip(self.images, self.labels) if qry_label != label]
        pos_rand_int = random.randint(a = 0, b = len(positive_images_paths) - 1)
        neg_rand_int = random.randint(a = 0, b = len(negative_images_paths) - 1)
        return (positive_images_paths[pos_rand_int],
                negative_images_paths[neg_rand_int],
                [label for label in self.labels if label != qry_label][0])

    def __getitem__(self, index):
        qry_im = self.images[index]
        qry_label = self.labels[index]
        if type(qry_label) is list:
            qry_label = qry_label[0]
        pos_im, neg_im, neg_label = self.get_pos_neg_ims(qry_label = qry_label)
        qry_gt = self.class_names[qry_label]
        neg_gt = self.class_names[neg_label]
        if self.transformations is not None:
            qry_im = self.transformations(qry_im)
            pos_im = self.transformations(pos_im)
            neg_im = self.transformations(neg_im)
        data = {}
        data["qry_im"] = qry_im
        data["qry_gt"] = qry_gt
        data["pos_im"] = pos_im
        data["neg_im"] = neg_im
        data["neg_gt"] = neg_gt
        return data

def get_dls(root, transformations, batch_size, split = None, num_workers = 0):
    """This function is designed to read and
    process data explicitly for training a model"""
    if split is None:
        split = [0.9, 0.05, 0.05]
    dataset = CustomDataset(root = root, transformations = transformations)
    class_names = dataset.class_names
    class_counts = dataset.class_counts
    total_len = len(dataset)
    train_len = int(total_len * split[0])
    validation_len = int(total_len * split[1])
    test_len = total_len - train_len - validation_len
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset = dataset,
        lengths = [train_len, validation_len, test_len])
    train_dl = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size = batch_size,
                                       shuffle = False,
                                       num_workers = num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size = 1,
                                 shuffle = False,
                                 num_workers = num_workers)
    return train_dl, validation_dataloader, test_dataloader, class_names, class_counts
