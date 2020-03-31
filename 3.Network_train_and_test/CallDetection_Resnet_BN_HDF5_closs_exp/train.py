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


# helper functions.
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


# Read data locations from configuration files.
train_file = 'train.txt'
train_filelist = read_lmdb_list(train_file)

test_file = 'test.txt'
test_filelist = read_lmdb_list(test_file)


# Configure Training Parameters.
current_directory = os.getcwd()
config = Config()
config.name = 'DSRCNN'
config.beta1 = 0.9
config.beta2 = 0.999
config.learning_rate = 0.001
config.weight_lr = 1
config.bias_lr = 0.1
config.iterations = 1000000
config.stepsize = 400000
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


# Configure logger
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.basicConfig(filename=config.log_folder+'.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)

logging.info("Start program")


# Register dataset
train_samples = 0
test_samples = 0
train_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=train_filelist, batchsize=config.batch_size)
test_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=test_filelist, batchsize=config.batch_size)
train_samples += len(train_dataset)
test_samples += len(test_dataset)


# Register network
net = models.Detection_ResNet_BN2(width=32).type(dtype)
net.apply(m_func.weights_init_He_normal)
all_net = net


# Register optimizer and scheduler
config.epoch_size = train_samples / config.batch_size
optimizer = optim.Adam(all_net.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=1e-8, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.stepsize), gamma=config.gamma)


# Test before training
iterations = 0
epoch = 0
all_net.eval()
psnr_list = m_func.evaluate_detection_network(net, test_dataset, config, iterations)
mean_psnr = np.mean(psnr_list)
logging.info('Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
all_net.train()

# Start Training.
logging.info("Start Training ...")
loss_list = []
while True:
    all_net.train()
    current_lr = optimizer.param_groups[0]['lr']
    mse_total_loss = Variable(torch.zeros(1)).type(dtype)
    iterations += 1
    sample_batched = next(train_dataset)
    input = Variable(torch.from_numpy(np.asarray(sample_batched[0]))).type(dtype)
    output = net(input)
    label = Variable(torch.from_numpy(np.asarray(sample_batched[1]))).type(dtype)
    weight = 1
    loss = losses.C_Loss(output, label)

    loss_list.append(loss.cpu().data.numpy()[0])
    if iterations % config.n_train_log == 0:
        mean_loss = np.mean(loss_list)
        logging.info(
            "[epoch " + str(epoch) + " iter " + str(iterations) + "]:" + ' lr: ' + str(current_lr) +
            ' ' + "loss: " + str(mean_loss))


    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()
    all_net.zero_grad()


    if iterations % config.n_save_model == 0:
        logging.info("Saving model ...")
        torch.save(all_net.state_dict(), config.save_folder+'-iter_'+str(iterations))


    if iterations % config.n_test_log == 0:
        logging.info("Testing ...")
        all_net.eval()
        psnr_list = m_func.evaluate_detection_network(net, test_dataset, config, iterations)
        mean_psnr = np.mean(psnr_list)
        logging.info(
            'Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
        all_net.train()

    if iterations % config.epoch_size == 0:
        epoch += 1

    if iterations > config.iterations:
        break


logging.info('Train epoch' + ' : ' + str(epoch))
