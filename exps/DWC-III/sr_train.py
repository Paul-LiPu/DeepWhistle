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


# Read lmdb locations from configuration files.
# train_file = 'train.txt'
# train_filelist = read_lmdb_list(train_file)
#

val_file = 'val.txt'
val_filelist = read_lmdb_list(val_file)

test_file = 'test.txt'
test_filelist = read_lmdb_list(test_file)

data_base_dir = '/home/lipu/Documents/whale_recognition/Train_data/LMDB'
negative_data_file = data_base_dir + '/' + 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_patch_mixed_1-1_negative_patches'
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
train_dataset = m_dataset.Dataset_lmdb_manual_data3(negative_file=negative_data_file, positive_file=positive_data_file, max_readers=config.workers)
train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=False, num_workers=config.workers, worker_init_fn=worker_init_fn)
val_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=val_filelist, batchsize=config.batch_size)
# test_dataset = m_dataset.Dataset_lmdb_simple(lmdb_file_path=test_data_file, max_readers=1)
# test_dataloader = DataLoader(test_dataset, batch_size = config.test_batchsize, shuffle=False, num_workers=1)
test_dataset = m_dataset.HDF5_Dataset_transpose(hdf5_list=test_filelist, batchsize=config.batch_size)
train_samples += len(train_dataset)
test_samples += len(test_dataset)

test_batch_num, test_pic_num = m_func.find_test_image_h5(test_dataset, config)
logging.info('Test Picture batch: ' + str(test_batch_num) + '\tpicture num: ' + str(test_pic_num))


# Register network
net = models.Detection_ResNet_BN2(width=32).type(dtype)
# net = models.Detection_ResNet_BN(width=16).type(dtype)
net.apply(m_func.weights_init_He_normal)
all_net = net
# net.load_state_dict(torch.load(config.saved_model))


# Register optimizer and scheduler
config.epoch_size = train_samples / config.batch_size
optimizer = optim.Adam(all_net.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=1e-8, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.stepsize), gamma=config.gamma)


# Test before training
iterations = 0
epoch = 0
all_net.eval()
psnr_list = m_func.evaluate_detection_network2(net, test_dataset, config, iterations, test_batch_num, test_pic_num)
mean_psnr = np.mean(psnr_list)
logging.info('Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
# tf_board.scalar_summary('test/PSNR', mean_psnr, iterations)
all_net.train()




# Start Training.
logging.info("Start Training ...")
loss_list = []
s_time = time.time()
while True:
    for i_batch, sample_batched in enumerate(train_dataloader):
        all_net.train()
        current_lr = optimizer.param_groups[0]['lr']
        mse_total_loss = Variable(torch.zeros(1)).type(dtype)
        iterations += 1
        # temp = sample_batched['input']
        # input = Variable(torch.from_numpy(np.asarray(sample_batched['input']))).type(dtype)
        input = Variable(sample_batched['input']).type(dtype)
        output = net(input)
        # label = Variable(torch.from_numpy(np.asarray(sample_batched['label']))).type(dtype)
        label = Variable(sample_batched['label']).type(dtype)
        weight = 1
        # loss = losses.C_Loss(output, label) + weight * losses.continuity_loss(output, 1)
        loss = losses.C_Loss(output, label)

        loss_list.append(loss.cpu().data.numpy()[0])
        if iterations % config.n_train_log == 0:
            e_time = time.time()
            mean_loss = np.mean(loss_list)
            logging.info(
                "[epoch " + str(epoch) + " iter " + str(iterations) + "]:" + ' lr: ' + str(current_lr) +
                ' ' + "loss: " + str(mean_loss) + '  time: ' + str(e_time - s_time))
            # tf_board.scalar_summary('train/loss', mean_loss, iterations)
            loss_list = []
            # tf_board.scalar_summary('train/lr', current_lr, iterations)
            s_time = time.time()

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
            psnr_list = m_func.evaluate_detection_network2(net, test_dataset, config, iterations,test_batch_num, test_pic_num)
            mean_psnr = np.mean(psnr_list)
            logging.info(
                'Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
            # tf_board.scalar_summary('test/PSNR', mean_psnr, iterations)
            all_net.train()

        if iterations % config.epoch_size == 0:
            epoch += 1

        if iterations > config.iterations:
            break

    if iterations > config.iterations:
        break

# val_dataset.env.close()
train_dataset.negative_env.close()
train_dataset.positive_env.close()
logging.info('Train epoch' + ' : ' + str(epoch))
