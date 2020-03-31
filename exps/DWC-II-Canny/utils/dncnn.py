import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.models.vgg as vgg
import torch.optim as optim
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import logging
import h5py
import torch.optim.lr_scheduler
from utils.m_global import dtype
from utils.logger import Logger

from utils.m_dataset import DnCNNDataset#DejpegDataset
from utils.m_func import mod4crop, add_noise_cpu, weights_init_xavier as weights_init, evaluate
from src.models import DnCNN as Generator, DnCNNBlock as ResBlock
from src.losses import euclideanLoss as mse_loss
import sys    
file_name =  os.path.basename(sys.argv[0])
file_name = file_name.split('.')[0]+'_'
# In[3]:

train_base_path = '/mnt/lustre/DATAshare2/linjunfan/denoise/Train400_40_226816_1772.lmdb'
test_base_path = '/mnt/lustre/DATAshare2/linjunfan/denoise/Set12.lmdb'
test_base_path2 = '/mnt/lustre/DATAshare2/linjunfan/denoise/BSD68.lmdb'

test_sizes = None
with open('/mnt/lustre/DATAshare2/linjunfan/denoise/Set12_sizes', 'r') as fhand:
    test_filelist = fhand.read()
    test_sizes = eval(test_filelist)
    fhand.close()

test_sizes2 = None
with open('/mnt/lustre/DATAshare2/linjunfan/denoise/BSD68_sizes', 'r') as fhand:
    test_filelist = fhand.read()
    test_sizes2 = eval(test_filelist)
    fhand.close()

class Config():
    def __init__(self):
        pass

config = Config()
config.beta1 = 0.9
config.beta2 = 0.999
config.learning_rate = 0.001
config.iterations = 1000000
config.batch_size = 128
config.input_size = 40
config.n_train_log = 100
config.n_test_log = 2500
config.shuffle = True
config.workers = 16
config.timestamp = file_name+str(int(np.ceil(os.times()[-1])))
config.test_folder = '/mnt/lustre/linjunfan/Data_t1/deblur/python/dncnn/test_result/'+config.timestamp+'/'
os.system("mkdir -p "+str(config.test_folder))
config.test_folder+=config.timestamp
config.n_save_model = 10000
config.save_folder = '/mnt/lustre/linjunfan/Data_t1/deblur/python/dncnn/models/'+config.timestamp
config.log_folder = '/mnt/lustre/linjunfan/Data_t1/deblur/python/dncnn/logs/'+config.timestamp
logging.basicConfig(filename=config.log_folder+'_log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

tf_board = Logger('/mnt/lustre/linjunfan/Data_t1/deblur/python/dncnn/tf_logs/'+config.timestamp+'/')

logging.info("Start program")

netG = Generator(ResBlock, nn.BatchNorm2d)
netG = netG.eval()
netG.apply(weights_init)
netG = netG.type(dtype)
logging.info("Done loading netG")

# In[16]:
train_denoise_dataset = DnCNNDataset(db_path=train_base_path, input_transformer=(lambda img: add_noise_cpu(img, 25.)),  transformer=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(train_denoise_dataset, batch_size = config.batch_size, shuffle=config.shuffle, num_workers=config.workers)

config.epoch_size = len(train_denoise_dataset)/config.batch_size
logging.info("Epoch size is"+str(config.epoch_size))

test_denoise_dataset = DnCNNDataset(db_path=test_base_path, input_transformer=(lambda img: add_noise_cpu(img, 25.)), transformer=transforms.Compose([transforms.ToTensor()]), test=test_sizes)
test_dataloader = DataLoader(test_denoise_dataset, batch_size=1, shuffle=False, num_workers=1)

test_denoise_dataset2 = DnCNNDataset(db_path=test_base_path2, input_transformer=(lambda img: add_noise_cpu(img, 25.)), transformer=transforms.Compose([transforms.ToTensor()]), test=test_sizes2)
test_dataloader2 = DataLoader(test_denoise_dataset2, batch_size=1, shuffle=False, num_workers=1)

iterations = 0
epoch = 0

netG.eval()
psnr_list = evaluate(netG, test_dataloader, config, iterations=0, name='Set12')
logging.info("Valiation###[epoch "+str(epoch)+" iter "+str(iterations)+"]:  mean psnr "+str(np.mean(psnr_list)))

tf_board.scalar_summary('Set12/test_psnr', np.mean(psnr_list), iterations)

psnr_list = evaluate(netG, test_dataloader2, config, iterations=0, name='BSD68')
logging.info("Testing###[epoch "+str(epoch)+" iter "+str(iterations)+"]:  mean psnr "+str(np.mean(psnr_list)))

tf_board.scalar_summary('BSD68/test_psnr', np.mean(psnr_list), iterations)

optimizer = torch.optim.Adam(netG.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=1e-8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


logging.info("Start Triaining ...")
input_data = Variable(torch.zeros((config.batch_size, 1, config.input_size, config.input_size)), requires_grad=False).type(dtype)
label_data = Variable(torch.zeros((config.batch_size, 1, config.input_size, config.input_size)), requires_grad=False).type(dtype)


while True:
    netG.train()
    scheduler.step()
    for i_batch, sample_batched in enumerate(train_dataloader):
        input_data.data.copy_(sample_batched['input'])
        label_data.data.copy_(sample_batched['label'])

        gen_output = netG(input_data)

        vgg_loss = mse_loss(gen_output, label_data, config.input_size**2)

        gen_total_loss = vgg_loss
        gen_total_loss.backward()

        optimizer.step()
        netG.zero_grad()

        iterations += 1
        if iterations % config.n_save_model == 0:
            logging.info("Saving model ...")
            torch.save(netG.state_dict(), config.save_folder+'-Giter_'+str(iterations))
            netG.eval()
            psnr_list = evaluate(netG, test_dataloader2, config, iterations = iterations, name='BSD68')
            logging.info("Testing###[epoch "+str(epoch)+" iter "+str(iterations)+"]:  mean psnr "+str(np.mean(psnr_list)))
            tf_board.scalar_summary('BSD68/test_psnr', np.mean(psnr_list), iterations)
            netG.train()

        if iterations % config.n_train_log == 0:
            logging.info("[epoch "+str(epoch)+" iter "+str(iterations)+"]:"+" total: "+str(gen_total_loss.cpu().data.numpy()[0]))
            tf_board.scalar_summary('train/mse', gen_total_loss.cpu().data.numpy()[0], iterations)

        if iterations % config.n_test_log == 0:
            netG.eval()
            psnr_list = evaluate(netG, test_dataloader, config, iterations = iterations, name='Set12')
            logging.info("Valiation###[epoch "+str(epoch)+" iter "+str(iterations)+"]:  mean psnr "+str(np.mean(psnr_list)))
            tf_board.scalar_summary('Set12/test_psnr', np.mean(psnr_list), iterations)
            netG.train()

        if iterations > config.iterations:
            break
    epoch+=1


