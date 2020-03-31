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



def read_lmdb(lmdb_dir):
    env = lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
    data = None
    label = None
    label_start = len(keys) / 2
    with env.begin(write=False) as txn:
        for num in range(0, label_start):
            data = txn.get(keys[num])
            label = txn.get(keys[label_start + num])
            data = np.fromstring(data, dtype=np.float32)
            data = data.reshape(8, int(np.sqrt(data.shape[0] / 8)), int(np.sqrt(data.shape[0] / 8)))
            label = np.fromstring(label, dtype=np.float32)
            label = label.reshape(int(np.sqrt(label.shape[0])), int(np.sqrt(label.shape[0])))
    env.close()
    return data, label


def read_lmdb_keys(lmdb_dir):
    env = lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
    env.close()
    return keys


def read_lmdb_random(lmdb_dir, nsamples):
    env = lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
    label_start = len(keys) / 2
    datas = []
    labels = []
    with env.begin(write=False) as txn:
        for t in range(0, nsamples):
            num = random.randint(0, label_start - 1)
            data = txn.get(keys[num])
            label = txn.get(keys[label_start + num])
            data = np.fromstring(data, dtype=np.float32)
            data = data.reshape(8, int(np.sqrt(data.shape[0] / 8)), int(np.sqrt(data.shape[0] / 8)))
            datas.append(data)
            label = np.fromstring(label, dtype=np.float32)
            label = label.reshape(int(np.sqrt(label.shape[0])), int(np.sqrt(label.shape[0])))
            labels.append(label)
    env.close()
    return datas, labels

def read_lmdb_pos(lmdb_dir, pos, nsamples):
    env = lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
    label_start = len(keys) / 2
    datas = []
    labels = []
    with env.begin(write=False) as txn:
        for t in range(pos, pos + nsamples):
            num = t % label_start
            data = txn.get(keys[num])
            label = txn.get(keys[label_start + num])
            data = np.fromstring(data, dtype=np.float32)
            data = data.reshape(8, int(np.sqrt(data.shape[0] / 8)), int(np.sqrt(data.shape[0] / 8)))
            datas.append(data)
            label = np.fromstring(label, dtype=np.float32)
            label = label.reshape(int(np.sqrt(label.shape[0])), int(np.sqrt(label.shape[0])))
            labels.append(label)
    env.close()
    return datas, labels


def read_lmdb_pos_by_keys(lmdb_dir, pos, nsamples, keys):
    env = lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    datas = []
    labels = []
    label_start = len(keys) / 2
    with env.begin(write=False) as txn:
        for t in range(pos, pos + nsamples):
            num = t % label_start
            data = txn.get(keys[num])
            label = txn.get(keys[label_start + num])
            data = np.fromstring(data, dtype=np.float32)
            data = data.reshape(8, int(np.sqrt(data.shape[0] / 8)), int(np.sqrt(data.shape[0] / 8)))
            datas.append(data)
            label = np.fromstring(label, dtype=np.float32)
            label = label.reshape(int(np.sqrt(label.shape[0])), int(np.sqrt(label.shape[0])))
            labels.append(label)
    env.close()
    return datas, labels

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


def read_h5_pos2(data0, label0, pos, nsamples):
    data = data0[pos:pos+nsamples]
    label = label0[pos:pos+nsamples]
    return data, label


def load_h5_to_RAM(file):
    h5file = h5py.File(file)
    data = h5file['data'][:]
    label = h5file['label'][:]
    h5file.close()
    return data, label

def read_h5_length(file):
    # print os.path.exists(file)
    h5file = h5py.File(file)
    length = len(h5file['data'])
    h5file.close()
    return length


class HDF5_Dataset():
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
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label


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


class HDF5_Dataset_load_in_RAM():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        # self.data = []
        # self.label = []
        self.data_map = {}
        self.label_map = {}
        for i in range(len(hdf5_list)):
            data, label = load_h5_to_RAM(hdf5_list[i])
            # self.data.append(data)
            # self.label.append(label)
            self.data_map[hdf5_list[i]] = data
            self.label_map[hdf5_list[i]] = label

        length_list = map(read_h5_length, hdf5_list)
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos2(self.data_map[h5_file], self.label_map[h5_file], self.curr_file_pointer, self.batch_size)
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label



class HDF5_Dataset_load_in_RAM_transpose():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        # self.data = []
        # self.label = []
        self.data_map = {}
        self.label_map = {}
        for i in range(len(hdf5_list)):
            data, label = load_h5_to_RAM(hdf5_list[i])
            data = np.transpose(data, (0,1,3,2))
            label = np.transpose(label, (0,1,3,2))
            # self.data.append(data)
            # self.label.append(label)
            self.data_map[hdf5_list[i]] = data
            self.label_map[hdf5_list[i]] = label

        length_list = map(read_h5_length, hdf5_list)
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def next(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos2(self.data_map[h5_file], self.label_map[h5_file], self.curr_file_pointer, self.batch_size)
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label


class DSRCNN_Dataset_HDF5(Dataset):
    def __init__(self, file_list, batchsize, transformer=None):
        self.file_list = file_list
        self.transformer = transformer
        file_pointers = {}
        nsamples = 0
        file_length_dict = {}
        for file in file_list:
            file_length_dict[file] = read_h5_length(file)
            nsamples += file_length_dict[file]
            file_pointers[file] = 0
        class_dict = classify_filelist(file_list)
        classes = class_dict.keys()
        nclass = len(classes)
        h5_pointer = [0] * nclass
        self.epoches = [0] * nclass
        self.nclass = nclass
        self.classes = classes
        self.class_dict = class_dict
        self.file_pointers = file_pointers
        self.h5_pointer = h5_pointer
        self.nsamples = nsamples
        self.batchsize = batchsize
        self.currentclass = random.randint(0, self.nclass - 1)
        self.currentfile = 0
        self.currentBatch = []
        self.file_length_dict = file_length_dict

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        # if idx % self.batchsize == 0:
            # currentclass = random.randint(0, self.nclass - 1)
            # DSRCNN_Dataset.__currentclass = currentclass
        idx = idx % self.batchsize
        return self.currentBatch[0][idx], self.currentBatch[1][idx]

    def setclass(self):
        currentclass = self.currentclass
        classname = self.classes[currentclass]
        filenum = self.h5_pointer[currentclass]
        filelist = self.class_dict[classname]
        h5_file = filelist[filenum]
        file_pointer = self.file_pointers[h5_file]
        nrecords = self.file_length_dict[h5_file]
        self.file_pointers[h5_file] = file_pointer + self.batchsize
        if (file_pointer + self.batchsize) >= nrecords:
            self.file_pointers[h5_file] -= nrecords
            nfiles = len(filelist)
            self.h5_pointer[currentclass] = self.h5_pointer[currentclass] + 1
            if self.h5_pointer[currentclass] >= nfiles:
                self.epoches[currentclass] += 1
                self.h5_pointer[currentclass] %= nfiles
        return read_h5_pos(h5_file, file_pointer, self.batchsize)


class DSRCNN_TestDataset_HDF5_cached(Dataset):
    def __init__(self, file_list, read_batch_size, transformer=None):
        self.file_list = file_list
        self.transformer = transformer
        nsamples = 0
        nsample_list = []
        for file in file_list:
            nsamples += read_h5_length(file)
            nsample_list.append(nsamples)
        self.nsamples = nsamples
        self.nsample_list = nsample_list
        self.batchsize = read_batch_size
        self.classname = get_parentdir(file_list[0])
        self.index = 0

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return None

    def updateBatch(self):
        idx = self.index
        file_num = bisect.bisect_left(self.nsample_list, idx + 1)
        h5_file = self.file_list[file_num]
        if file_num == 0:
            index = idx
        else:
            index = idx % self.nsample_list[file_num - 1]
        if index == 0:
            self.classname = get_parentdir(h5_file)
        batch_index = index % self.batchsize
        if batch_index == 0:
            self.currentBatch = read_h5_pos(h5_file, index, self.batchsize)
        self.index += self.batchsize
        self.index %= self.nsamples

    def getClassname(self):
        return self.classname



class DSRCNN_Dataset(Dataset):
    def __init__(self, file_list, batchsize, transformer=None):
        self.file_list = file_list
        self.transformer = transformer
        lmbd_keys = map(read_lmdb_keys, file_list)
        lmbd_keys_dict = {}
        lmdb_file_pointer = {}
        nsamples = 0
        for i in range(0, len(file_list)):
            lmbd_keys_dict[file_list[i]] = lmbd_keys[i]
            lmdb_file_pointer[file_list[i]] = 0
            nsamples += len(lmbd_keys[i]) / 2
        lmdb_dict = classify_filelist(file_list)
        classes = lmdb_dict.keys()
        nclass = len(classes)
        lmdb_pointer = [0] * nclass
        self.nclass = nclass
        self.classes = classes
        self.lmdb_dict =lmdb_dict
        self.lmbd_keys_dict = lmbd_keys_dict
        self.lmdb_file_pointer = lmdb_file_pointer
        self.lmdb_pointer = lmdb_pointer
        self.nsamples = nsamples
        self.batchsize = batchsize
        self.currentclass = random.randint(0, self.nclass - 1)
        self.currentfile = 0
        self.currentBatch = []

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        # if idx % self.batchsize == 0:
            # currentclass = random.randint(0, self.nclass - 1)
            # DSRCNN_Dataset.__currentclass = currentclass
        idx = idx % self.batchsize
        return self.currentBatch[0][idx], self.currentBatch[1][idx]

    def setclass(self):
        currentclass = self.currentclass
        classname = self.classes[currentclass]
        filenum = self.lmdb_pointer[currentclass]
        filelist = self.lmdb_dict[classname]
        lmdb_file = filelist[filenum]
        file_pointer = self.lmdb_file_pointer[lmdb_file]
        keys = self.lmbd_keys_dict[lmdb_file]
        # self.currentBatch =
        nrecords = len(keys) / 2
        self.lmdb_file_pointer[lmdb_file] = file_pointer + self.batchsize
        if (file_pointer + self.batchsize) >= nrecords:
            self.lmdb_file_pointer[lmdb_file] = self.lmdb_file_pointer[lmdb_file] - nrecords
            nfiles = len(filelist)
            self.lmdb_pointer[currentclass] = (self.lmdb_pointer[currentclass] + 1) % nfiles
        return read_lmdb_pos_by_keys(lmdb_file, file_pointer, self.batchsize, keys)


class DSRCNN_TestDataset_cached(Dataset):
    def __init__(self, file_list, read_batch_size, transformer=None):
        self.file_list = file_list
        self.transformer = transformer
        lmbd_keys = map(read_lmdb_keys, file_list)
        nsamples = 0
        nsample_list = []
        for i in range(0, len(file_list)):
            nsamples += len(lmbd_keys[i]) / 2
            nsample_list.append(nsamples)
        self.nsamples = nsamples
        self.nsample_list = nsample_list
        self.lmdb_keys = lmbd_keys
        self.batchsize = read_batch_size
        self.classname = get_parentdir(file_list[0])
        self.index = 0

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        file_num = bisect.bisect_left(self.nsample_list, idx + 1)
        lmdb_file = self.file_list[file_num]
        index = idx % self.nsample_list[file_num]
        if index == 0:
            self.classname = get_parentdir(lmdb_file)
        batch_index = index % self.batchsize
        if batch_index == 0:
            self.currentBatch = read_lmdb_pos_by_keys(lmdb_file, index, self.batchsize, self.lmdb_keys[file_num])
        return self.currentBatch[0][batch_index], self.currentBatch[1][batch_index]

    def updateBatch(self):
        idx = self.index
        file_num = bisect.bisect_left(self.nsample_list, idx + 1)
        lmdb_file = self.file_list[file_num]
        if file_num == 0:
            index = idx
        else:
            index = idx % self.nsample_list[file_num - 1]
        if index == 0:
            self.classname = get_parentdir(lmdb_file)
        batch_index = index % self.batchsize
        if batch_index == 0:
            self.currentBatch = read_lmdb_pos_by_keys(lmdb_file, index, self.batchsize, self.lmdb_keys[file_num])
        self.index += self.batchsize
        self.index %= self.nsamples

    def getClassname(self):
        return self.classname


class Dataset_lmdb_simple(Dataset):
    def __init__(self, lmdb_file_path, max_readers):
        self.lmdb_file_path = lmdb_file_path
        self.max_readers = max_readers

        #self.env = env
        env = lmdb.open(self.lmdb_file_path, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.file_ndata = env.stat()['entries'] // 2
        with env.begin(write=False) as txn:
            str_id = '{:08}'.format(0)
            data = txn.get('data-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        self.data_size = int(np.sqrt(data.shape[0]))
        self.data_num = self.file_ndata
        env.close()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sample = {}
        str_id = '{:08}'.format(idx)
        env = lmdb.open(self.lmdb_file_path , max_readers=self.max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            data = txn.get('data-' + str_id)
            label = txn.get('label-' + str_id)
        env.close()
        data = np.fromstring(data, dtype=np.float32)
        data = data.reshape(1, self.data_size, self.data_size)
        label = np.fromstring(label, dtype=np.float32)
        label = label.reshape(1, self.data_size, self.data_size)

        sample['input'] = data
        sample['label'] = label

        return sample


class Dataset_lmdb_simple2(Dataset):
    def __init__(self, lmdb_file_path, max_readers):
        self.lmdb_file_path = lmdb_file_path
        self.max_readers = max_readers

        self.env = lmdb.open(self.lmdb_file_path, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.file_ndata = self.env.stat()['entries'] // 2
        with self.env.begin(write=False) as txn:
            str_id = '{:08}'.format(0)
            data = txn.get('data-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        self.data_size = int(np.sqrt(data.shape[0]))
        self.data_num = self.file_ndata

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sample = {}
        str_id = '{:08}'.format(idx)
        with self.env.begin(write=False) as txn:
            data = txn.get('data-' + str_id)
            label = txn.get('label-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        data = data.reshape(1, self.data_size, self.data_size)
        label = np.fromstring(label, dtype=np.float32)
        label = label.reshape(1, self.data_size, self.data_size)

        sample['input'] = data
        sample['label'] = label

        return sample


class Dataset_lmdb_manual_data(Dataset):
    def __init__(self, negative_file, positive_file, max_readers, pos_portion = 0.5, signal_min = 0.01, signal_max = 0.4):
        self.negative_file = negative_file
        self.positive_file = positive_file
        self.max_readers = max_readers
        self.pos_portion = pos_portion
        self.signal_min = signal_min
        self.signal_range = signal_max - signal_min

        env = lmdb.open(self.positive_file, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.positive_ndata = env.stat()['entries'] // 2
        with env.begin(write=False) as txn:
            str_id = '{:08}'.format(0)
            data = txn.get('data-' + str_id)
        data = np.fromstring(data, dtype=np.float32)
        env.close()

        env = lmdb.open(self.negative_file, max_readers=max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        self.negative_ndata = env.stat()['entries'] // 2
        env.close()

        self.data_size = int(np.sqrt(data.shape[0]))
        self.data_num = self.positive_ndata + self.negative_ndata

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sample = {}

        rand_idx = np.random.randint(0, self.negative_ndata)
        str_id = '{:08}'.format(rand_idx)
        env = lmdb.open(self.negative_file, max_readers=self.max_readers, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            data = txn.get('data-' + str_id)
            label = txn.get('label-' + str_id)
        env.close()
        data = np.fromstring(data, dtype=np.float32)
        data = data.reshape(1, self.data_size, self.data_size)
        label = np.fromstring(label, dtype=np.float32)
        label = label.reshape(1, self.data_size, self.data_size)

        rand_num = np.random.rand()
        if rand_num < self.pos_portion:
            rand_idx = np.random.randint(0, self.positive_ndata)
            str_id = '{:08}'.format(rand_idx)
            env = lmdb.open(self.positive_file, max_readers=self.max_readers, readonly=True, lock=False,
                            readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                pos_label = txn.get('label-' + str_id)
            env.close()
            pos_label = np.fromstring(pos_label, dtype=np.float32)
            pos_label = pos_label.reshape(1, self.data_size, self.data_size)
            label = label + pos_label
            rand_a = np.random.rand() * self.signal_range + self.signal_min
            data = data + pos_label * rand_a

        data = np.transpose(data, (0, 2, 1))
        label = np.transpose(label, (0, 2, 1))
        sample['input'] = data
        sample['label'] = label

        return sample


class Dataset_lmdb_manual_data2(Dataset):
    def __init__(self, negative_file, positive_file, max_readers, pos_portion = 0.5, signal_min = 0.01, signal_max = 0.4):
        self.negative_file = negative_file
        self.positive_file = positive_file
        self.max_readers = max_readers
        self.pos_portion = pos_portion
        self.signal_min = signal_min
        self.signal_range = signal_max - signal_min


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
            rand_idx = np.random.randint(0, self.positive_ndata)
            str_id = '{:08}'.format(rand_idx)
            with self.positive_env.begin(write=False) as txn:
                pos_label = txn.get('label-' + str_id)
            pos_label = np.fromstring(pos_label, dtype=np.float32)
            pos_label = pos_label.reshape(1, self.data_size, self.data_size)
            label = label + pos_label
            rand_a = np.random.rand() * self.signal_range + self.signal_min
            data = data + pos_label * rand_a

        data = np.transpose(data, (0, 2, 1))
        label = np.transpose(label, (0, 2, 1))
        sample['input'] = data
        sample['label'] = label

        return sample


class Dataset_lmdb_manual_data2_recall(Dataset):
    def __init__(self, negative_file, positive_file, max_readers, pos_portion = 0.5, signal_min = 0.1, signal_max = 0.4
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
            rand_num3 = np.random.rand()
            if rand_num3 < self.synthesize_portion:
                rand_num2 = np.random.rand()
                rand_idx = bisect.bisect_left(self.recall_list, rand_num2)
                str_id = '{:08}'.format(rand_idx)
                with self.positive_env.begin(write=False) as txn:
                    pos_label = txn.get('label-' + str_id)
                pos_label = np.fromstring(pos_label, dtype=np.float32)
                pos_label = pos_label.reshape(1, self.data_size, self.data_size)
                label = label + pos_label
                rand_a = np.random.rand() * self.signal_range + self.signal_min
                data = data + pos_label * rand_a
                data[data > 1] = 1
                label[label > 1] = 1
            else:
                rand_idx = np.random.randint(0, self.positive_ndata)
                str_id = '{:08}'.format(rand_idx)
                with self.positive_env.begin(write=False) as txn:
                    data = txn.get('data-' + str_id)
                    label = txn.get('label-' + str_id)
                data = np.fromstring(data, dtype=np.float32)
                data = data.reshape(1, self.data_size, self.data_size)
                label = np.fromstring(label, dtype=np.float32)
                label = label.reshape(1, self.data_size, self.data_size)

        data = np.transpose(data, (0, 2, 1))
        label = np.transpose(label, (0, 2, 1))
        sample['input'] = data
        sample['label'] = label

        return sample



from scipy.ndimage.filters import gaussian_filter
class Dataset_lmdb_manual_data2_recall2(Dataset):
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
            rand_num3 = np.random.rand()
            if rand_num3 < self.synthesize_portion:
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
                pos_label_blur = pos_label_blur.reshape(1, self.data_size, self.data_size)
                rand_a = np.random.rand() * self.signal_range + self.signal_min
                data = data + pos_label_blur * rand_a
                data[data > 1] = 1
                label[label > 1] = 1
            else:
                rand_idx = np.random.randint(0, self.positive_ndata)
                str_id = '{:08}'.format(rand_idx)
                with self.positive_env.begin(write=False) as txn:
                    data = txn.get('data-' + str_id)
                    label = txn.get('label-' + str_id)
                data = np.fromstring(data, dtype=np.float32)
                data = data.reshape(1, self.data_size, self.data_size)
                label = np.fromstring(label, dtype=np.float32)
                label = label.reshape(1, self.data_size, self.data_size)

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
            rand_num3 = np.random.rand()
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

        # self.recall = np.ones(self.positive_ndata) / self.positive_ndata
        # self.recall_list = np.cumsum(self.recall) / np.sum(self.recall)

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


class Dataset_lmdb_manual_data4(Dataset):
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

        # self.recall = np.ones(self.positive_ndata) / self.positive_ndata
        # self.recall_list = np.cumsum(self.recall) / np.sum(self.recall)

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
