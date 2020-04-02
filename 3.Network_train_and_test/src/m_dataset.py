import numpy as np
from torch.utils.data import Dataset
import os
import h5py
import os.path
import bisect
import lmdb
from scipy.ndimage.filters import gaussian_filter


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


class Dataset_lmdb_manual_data3(Dataset):
    def __init__(self, negative_file, positive_file, max_readers, pos_portion = 0.5, signal_min = 0.03, signal_max = 0.23
                 , synthesize_portion = 0.7):
        self.negative_file = negative_file
        self.positive_file = positive_file
        self.max_readers = max_readers
        self.pos_portion = pos_portion
        self.signal_min = signal_min
        self.signal_range = signal_max - signal_min
        self.synthesize_portion = synthesize_portion

        self.positive_env = lmdb.open(self.positive_file, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.positive_ndata = self.positive_env.stat()['entries'] // 2
        with self.positive_env.begin(write=False) as txn:
            str_id = '{:08}'.format(0)
            data = txn.get('data-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        self.data_size = int(np.sqrt(data.shape[0]))

        self.negative_env = lmdb.open(self.negative_file, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.negative_ndata = self.negative_env.stat()['entries'] // 2
        self.data_num = self.positive_ndata + self.negative_ndata

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sample = {}
        rand_idx = np.random.randint(0, self.negative_ndata)
        str_id = '{:08}'.format(rand_idx)
        with self.negative_env.begin(write=False) as txn:
            data = txn.get('data-' + str_id)
            label = txn.get('label-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        data = data.reshape(1, self.data_size, self.data_size)
        label = np.fromstring(label, dtype=np.float32)
        label = label.reshape(1, self.data_size, self.data_size)

        rand_num = np.random.rand()
        if rand_num < self.pos_portion:
            rand_idx = np.random.randint(self.positive_ndata)
            str_id = '{:08}'.format(rand_idx)
            with self.positive_env.begin(write=False) as txn:
                pos_label = txn.get('label-' + str_id)
            pos_label = np.fromstring(pos_label, dtype=np.float32)

            pos_label = pos_label.reshape(1, self.data_size, self.data_size)
            label = label + pos_label

            pos_label = pos_label.reshape(self.data_size, self.data_size)
            pos_label_blur = gaussian_filter(pos_label, sigma=np.random.rand() + 0.3)
            pos_label_blur = pos_label_blur + pos_label
            pos_label_blur[pos_label_blur > 1] = 1
            pos_label_blur = pos_label_blur.reshape(1, self.data_size, self.data_size)

            rand_a = np.random.rand() * self.signal_range + self.signal_min
            data = data + pos_label_blur * rand_a
            data[data > 1] = 1
            label[label > 1] = 1

        data = np.transpose(data, (0, 2, 1))
        label = np.transpose(label, (0, 2, 1))
        sample['input'] = data
        sample['label'] = label

        return sample


class Dataset_lmdb_manual_data2_recall3(Dataset):
    def __init__(self, negative_file, positive_file, max_readers, pos_portion = 0.5, signal_min = 0.03, signal_max = 0.23
                 , synthesize_portion = 0.7):
        self.negative_file = negative_file
        self.positive_file = positive_file
        self.max_readers = max_readers
        self.pos_portion = pos_portion
        self.signal_min = signal_min
        self.signal_range = signal_max - signal_min
        self.synthesize_portion = synthesize_portion


        self.positive_env = lmdb.open(self.positive_file, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.positive_ndata = self.positive_env.stat()['entries'] // 2
        with self.positive_env.begin(write=False) as txn:
            str_id = '{:08}'.format(0)
            data = txn.get('data-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        self.data_size = int(np.sqrt(data.shape[0]))

        self.negative_env = lmdb.open(self.negative_file, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.negative_ndata = self.negative_env.stat()['entries'] // 2
        self.data_num = self.positive_ndata + self.negative_ndata

        self.recall = np.ones(self.positive_ndata) / self.positive_ndata
        self.recall_list = np.cumsum(self.recall) / np.sum(self.recall)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sample = {}

        rand_idx = np.random.randint(0, self.negative_ndata)
        str_id = '{:08}'.format(rand_idx)

        with self.negative_env.begin(write=False) as txn:
            data = txn.get('data-' + str_id)
            label = txn.get('label-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        data = data.reshape(1, self.data_size, self.data_size)
        label = np.fromstring(label, dtype=np.float32)
        label = label.reshape(1, self.data_size, self.data_size)

        rand_num = np.random.rand()
        if rand_num < self.pos_portion:
            rand_num2 = np.random.rand()
            rand_idx = bisect.bisect_left(self.recall_list, rand_num2)
            str_id = '{:08}'.format(rand_idx)
            with self.positive_env.begin(write=False) as txn:
                pos_label = txn.get('label-' + str_id)
            pos_label = np.fromstring(pos_label, dtype=np.float32)

            pos_label = pos_label.reshape(1, self.data_size, self.data_size)
            label = label + pos_label

            pos_label = pos_label.reshape(self.data_size, self.data_size)
            pos_label_blur = gaussian_filter(pos_label, sigma=np.random.rand() + 0.3)
            pos_label_blur = pos_label_blur + pos_label
            pos_label_blur[pos_label_blur > 1] = 1
            pos_label_blur = pos_label_blur.reshape(1, self.data_size, self.data_size)

            rand_a = np.random.rand() * self.signal_range + self.signal_min
            data = data + pos_label_blur * rand_a
            data[data > 1] = 1
            label[label > 1] = 1


        data = np.transpose(data, (0, 2, 1))
        label = np.transpose(label, (0, 2, 1))
        sample['input'] = data
        sample['label'] = label

        return sample