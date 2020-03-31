import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from PIL import Image
import StringIO
import h5py
import PIL
import os.path
import sys
import bisect
import lmdb
import random


## helper functions.
def get_parentdir(dir):
    parent_dir = os.path.dirname(dir)
    parent_folder = os.path.basename(parent_dir)
    return parent_folder

def classify_filelist(file_list):
    classes = map(get_parentdir, file_list)
    file_dict = {}
    for i in range(0, len(file_list)):
        if not classes[i] in file_dict.keys():
            file_dict[classes[i]] = []
        file_dict[classes[i]].append(file_list[i])
    return file_dict

def read_h5_pos(file, pos, nsamples):
    h5file = h5py.File(file)
    data = h5file['data'][pos:pos+nsamples]
    label = h5file['label'][pos:pos+nsamples]
    h5file.close()
    return data, label


def read_h5_length(file):
    # print os.path.exists(file)
    h5file = h5py.File(file)
    length = len(h5file['data'])
    h5file.close()
    return length


# network definition
class HDF5_Dataset_transpose():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        length_list = map(read_h5_length, hdf5_list)
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos(h5_file, self.curr_file_pointer, self.batch_size)
        data = np.transpose(data, (0, 1, 3, 2))
        label = np.transpose(label, (0, 1, 3, 2))
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label

