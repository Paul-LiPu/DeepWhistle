# from utils.logger import Logger
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import logging
import torch.optim.lr_scheduler

from utils.m_global import dtype

from utils.m_dataset import HDF5_Dataset
import utils.m_dataset as m_dataset
import src.models as models
import src.losses as losses

import utils.m_func as m_func
import sys
import time
import cv2
import h5py

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def removeLineFeed(line):
    return line[:-1]

def read_lmdb_list(file):
    with open(file) as f:
        data = f.readlines()
    data = map(removeLineFeed, data)
    return data

def parse_class(classname):
    split_result = classname.split('_')
    noise = split_result[0]
    scale = split_result[1]
    noise = noise[4:]
    scale = scale[1:]
    return noise, int(scale)

def sort_list(images):
    list.sort(images, key=str.lower)


class Config():
    def __init__(self):
        pass

def patch2image(X, inter=0, out_shape=None, color=None):
    """
    Stitch a batch of patches together into one image.
    :param X: input patches, in range[0,1](float) or [0,255](uint8)
    :shape: BxCxHxW or BxHxW or BxHW(if H=W)
    :return: stitched image.
    """

    # [0, 1] -> [0,255]

    n_samples = X.shape[0]
    if out_shape is None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1
        nh, nw = rows, int(n_samples/rows)
    else:
        nh = out_shape[0]
        nw = out_shape[1]

    # if the patch is flattened, unflat it
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    # construct the stitching image
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh + inter * (nh + 1), w * nw + inter * (nh + 1), 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh + inter * (nh + 1), w * nw + inter * (nh + 1))).astype(X.dtype)

    if not color is None:
        img[:, :, 0] = color[0]
        img[:, :, 1] = color[1]
        img[:, :, 2] = color[2]

    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        img[j * h + (j + 1) * inter : j * h + (j + 1) * inter + h, i * w + (i + 1) * inter: i * w + (i + 1) * inter + w] = x
    return img


def readh5(file):
    h5file = h5py.File(file)
    data = h5file['data']
    label = h5file['label']
    data = data[...]
    label = label[...]
    h5file.close()
    return data, label


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def im_unit8(images, lb=0, ub=1):
    """
    convert numpy img from any range to [0, 255] and unit8
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    if images.dtype == np.uint8:
        return images
    images = np.clip((images - lb) * 1.0 / (ub - lb) * 255, 0, 255).astype('uint8')
    return images


test_file = 'test.txt'
test_filelist = read_lmdb_list(test_file)

data_base_dir = '/home/lipu/Documents/whale_recognition/Train_data/LMDB'
# data_base_dir = '/home/vision/lipu/projects/whale_recognition/train_data_lmdb'
negative_data_file = data_base_dir + '/' + 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_patch_mixed_1-1_negative_patches'
# positive_data_file = data_base_dir + '/' + 'BSD_500_train_GT_width2_positive_patch'
positive_data_file = data_base_dir + '/' + 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_patch_mixed_1-1_reduced0.0625_1_positive_patches'
test_data_file = data_base_dir + '/' + 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_TEST'



# Configure Training Parameters.
current_directory = os.getcwd()
config = Config()
config.name = 'DSRCNN'
config.beta1 = 0.9
config.beta2 = 0.999
config.learning_rate = 0.001
config.weight_lr = 1
config.bias_lr = 0.1
config.iterations = 600000
config.stepsize = 250000
config.gamma = 0.1
config.weight_decay = 0.00001
config.batch_size = 64
config.test_batchsize = 64
config.n_train_log = 100
config.n_test_log = 1000
config.timestamp = config.name + '_' + time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
config.test_folder = current_directory + '/test_result/' + config.timestamp + '/'
make_dir(config.test_folder)
# config.test_folder += config.timestamp
config.n_save_model = 10000
config.save_folder = current_directory + '/models/' + config.timestamp
make_dir(config.save_folder)
config.save_folder += '/' + config.name
config.log_folder = current_directory + '/logs/' + config.timestamp
config.workers = 16
# config.saved_model =  current_directory + '/models/' + model_file

# Configure logger
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.basicConfig(filename=config.log_folder+'.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
# tf_board = Logger(current_directory + '/tf_logs/' + config.timestamp + '/')

logging.info("Start program")

def worker_init_fn(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


# Register dataset
train_samples = 0
test_samples = 0
train_dataset = m_dataset.Dataset_lmdb_manual_data3(negative_file=negative_data_file, positive_file=positive_data_file, max_readers=config.workers, pos_portion=1)
train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=False, num_workers=config.workers, worker_init_fn=worker_init_fn)


n_disp = 1
gray = 0.5

output_dir = '/home/lipu/Documents/whale_recognition/images/' + os.path.basename(positive_data_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i_batch, sample_batched in enumerate(train_dataloader):
    input = Variable(sample_batched['input'])
    label = Variable(sample_batched['label'])

    print('%s' % (str(i_batch)))
    disp_data = input.data.numpy()
    disp_label = label.data.numpy()

    img = patch2image(disp_data, 2, color=[gray, gray, gray])
    img = im_unit8(img)

    output_file = output_dir + '/' + 'data_' + str(i_batch) + '.png'
    cv2.imwrite(output_file, img)


