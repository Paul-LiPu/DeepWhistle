import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import os

import logging
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from src.m_global import dtype
import src.m_dataset as m_dataset
import src.models as models
import src.losses as losses

import src.m_func as m_func
import sys
import time


class Config():
    def __init__(self):
        pass


import argparse

timestamp = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, required=True, help='h5 or lmdb')
parser.add_argument('--train_data', type=str, required=True, help='path to a text file, each line of which contains one training data file')
parser.add_argument('--test_data', type=str, required=True, help='path to a text file, each line of which contains one testing data file')
parser.add_argument('--recall_val_data', type=str, required=True, help='path to a text file, each line of which contains one validation file for recall calculation')
parser.add_argument('--exp_name', type=str, default=timestamp, help='experiment name')
parser.add_argument('--model_name', type=str, default='DWC', help='model name')
parser.add_argument('--learning_rate', type=float, default=0.001, help='base learning rate')
parser.add_argument('--iterations', type=int, default=600000, help='total training iterations')
parser.add_argument('--stepsize', type=int, default=250000, help='number of iterations to do decay of learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batchsize', type=int, default=64, help='test batch size')
parser.add_argument('--n_train_log', type=int, default=100, help='number of iterations to print out training log')
parser.add_argument('--n_test_log', type=int, default=1000, help='number of iterations to print out testing log')
parser.add_argument('--n_save_model', type=int, default=10000, help='number of iterations to save the model')
parser.add_argument('--recall_guided', type=bool, default=False, help='if use lmdb training data, whether to use recall guided training')
parser.add_argument('--pretrained_model', type=str, default='', help='pretrained model to initialize this model')

config = parser.parse_args()
# Read data locations from configuration files.
train_file = config.train_data
train_filelist = m_func.read_lmdb_list(config.train_data)

test_file = config.test_data
test_filelist = m_func.read_lmdb_list(config.test_data)

val_filelist = None
if config.recall_guided:
    val_filelist = m_func.read_lmdb_list(config.recall_val_data)

# Configure Training Parameters.
exp_directory = os.path.dirname(os.path.realpath(__file__))
config.name = config.model_name
config.beta1 = 0.9
config.beta2 = 0.999
config.weight_lr = 1
config.bias_lr = 0.1
config.workers = 4

config.test_folder = exp_directory + '/test_result/' + config.exp_name + '/'
m_func.make_dir(config.test_folder)
config.save_folder = exp_directory + '/models/' + config.exp_name
m_func.make_dir(config.save_folder)
config.save_folder += '/' + config.name
config.log_folder = exp_directory + '/logs/' + config.exp_name

# Configure logger
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.basicConfig(filename=config.log_folder+'.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)

logging.info("Start program")

# Register training dataset
val_dataset = None
if config.data_type == 'h5':
    train_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=train_filelist, batchsize=config.batch_size)
elif config.data_type == 'lmdb':
    train_dataset = m_dataset.Dataset_lmdb_manual_data3(negative_file=train_filelist[1],
                                                        positive_file=train_filelist[0], max_readers=config.workers)
    if config.recall_guided:
        train_dataset = m_dataset.Dataset_lmdb_manual_data2_recall3(negative_file=train_filelist[1],
                                                                    positive_file=train_filelist[0],
                                                                    max_readers=config.workers)
        val_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=val_filelist, batchsize=config.batch_size)

else:
    raise NotImplementedError


# Register testing dataset
test_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=test_filelist, batchsize=config.test_batchsize)

train_samples = len(train_dataset)
test_samples = len(test_dataset)

if config.data_type == 'lmdb':
    train_dataset = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                  num_workers=config.workers, worker_init_fn=m_func.worker_init_fn)

test_batch_num, test_pic_num = m_func.find_test_image_h5(test_dataset, config)
logging.info('Test Picture batch: ' + str(test_batch_num) + '\tpicture num: ' + str(test_pic_num))

# Register network
net = models.Detection_ResNet_BN2(width=32).type(dtype)
net.apply(m_func.weights_init_He_normal)
if config.pretrained_model != '':
    logging.info('Load weights from: %s' % (config.pretrained_model))
    net.load_state_dict(torch.load(config.pretrained_model))
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


def train_one_iteration(net, input, label, config, iterations, epoch, loss_list):
    net.train()
    current_lr = optimizer.param_groups[0]['lr']
    iterations += 1

    input = Variable(input).type(dtype)
    label = Variable(label).type(dtype)
    output = net(input)
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
        torch.save(all_net.state_dict(), config.save_folder + '-iter_' + str(iterations))

    if iterations % config.n_test_log == 0:
        logging.info("Testing ...")
        net.eval()
        psnr_list = m_func.evaluate_detection_network(net, test_dataset, config, iterations)
        mean_psnr = np.mean(psnr_list)
        logging.info(
            'Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
        net.train()

    if iterations % config.epoch_size == 0:
        epoch += 1

    return iterations, epoch, loss_list


# Start Training.
logging.info("Start Training ...")
loss_list = []
while True:
    if config.data_type == 'h5':
        sample_batched = next(train_dataset)
        input = torch.from_numpy(sample_batched[0])
        label = torch.from_numpy(sample_batched[1])
        iterations, epoch, loss_list = train_one_iteration(all_net, input, label, config, iterations, epoch, loss_list)
    else:
        if config.recall_guided:
            all_net.eval()
            recall = m_func.evaluate_detection_network_hdf5_PR(net, val_dataset, config)
            recall_mean = np.mean(recall)
            recall = 1 - recall
            recall_list = np.cumsum(recall) / np.sum(recall)
            train_dataset.dataset.recall_list = recall_list
            logging.info(
                "[epoch " + str(epoch) + "] mean recall: " + str(recall_mean))
            all_net.train()
        for i_batch, sample_batched in enumerate(train_dataset):
            input = sample_batched['input']
            label = sample_batched['label']
            iterations, epoch, loss_list = train_one_iteration(all_net, input, label, config, iterations, epoch,
                                                               loss_list)
            if iterations > config.iterations:
                break
    if iterations > config.iterations:
        break

logging.info('Train epoch' + ' : ' + str(epoch))
