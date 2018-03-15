"""Processing Functions for Multi-Label Classifier."""

import os, random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

class DatasetProcessing(Dataset):
    #def __init__(self, data_path, img_path, img_filename, label_filename, search_classes, transform=None):
    def __init__(self, data_path, img_path, search_classes, transform=None):
        self.transform = transform
        self.img_path = os.path.join(data_path, img_path)

        # # reading img file from file
        # img_filepath = os.path.join(data_path, img_filename)
        # file_p = open(img_filepath, 'r')
        # self.img_filename = [x.strip() for x in file_p]
        # file_p.close()

        # my listdir version (also shuffles)
        self.the_filename_list = os.listdir(self.img_path)
        random.shuffle(self.the_filename_list) # shuffles inplace!
        #print(self.the_filename_list)
        # my label reader: reading labels from filenames
        the_target_list = []

        # Liste "self.label" ausgegeben werden, die die One-hot labels enthält.
        # TODO: Funktion umschreiben, sodass sie selbst die Anzahl der Klassen detektiert.
        for name in self.the_filename_list:
            if search_classes[0] in name:
                the_target_list.append([1, 0, 0, 0, 0])
            elif search_classes[1] in name:
                the_target_list.append([0, 1, 0, 0, 0])
            elif search_classes[2] in name:
                the_target_list.append([0, 0, 1, 0, 0])
            elif search_classes[3] in name:
                the_target_list.append([0, 0, 0, 1, 0])
            elif search_classes[4] in name:
                the_target_list.append([0, 0, 0, 0, 1])
            else:
                print("Something went wrong with the target creation!")
        #print(the_target_list)
        self.label = np.asarray(the_target_list, dtype=np.int64)

        
        # reading labels from file
        # label_filepath = os.path.join(data_path, label_filename)
        # labels = np.loadtxt(label_filepath, dtype=np.int64)
        # self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.the_filename_list[index])) # my version
        #img = Image.open(os.path.join(self.img_path, self.img_filename[index]))

        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label

    def __len__(self):
        return len(self.the_filename_list)
