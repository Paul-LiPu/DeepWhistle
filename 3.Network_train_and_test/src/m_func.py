import numpy as np
import torch
import torch.nn.init as init
import cv2
from torch.autograd import Variable
from .m_global import dtype
import os
import fnmatch

def weights_init_constant(m, std):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean = 0.0, std = std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)#zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, std)
        m.bias.data.zero_()


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #std = np.sqrt(2./(m.kernel_size[0]*m.kernel_size[1]*m.out_channels))
        #m.weight.data.normal_(0.0, std)
        #m.bias.data.zero_()

        init.xavier_normal(m.weight.data)
        if m.bias is not None:
            init.constant(m.bias.data, 0.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.zero_()


def weights_init_msra(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Conv') != -1:
        std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels))
        # init.kaiming_uniform(m.weight.data, mode='fan_in')
        m.weight.data.normal_(mean=0.0, std=std)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        #print m.weight.data.numpy()
        m.weight.data.fill_(1.)
        #print m.weight.data.numpy()
        m.bias.data.fill_(0.)#zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.zero_()

def weights_init_He_normal(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Transpose') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        # std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        # m.weight.data.normal_(0.0, std)
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()



def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / (mse + 1e-5))
    return psnr

def cal_mse(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    return mse



def evaluate_detection_network(net, test_dataset, config, iterations):
    psnr = []
    for i_test_batch in range(0, len(test_dataset) // config.test_batchsize ):
        test_batched = next(test_dataset)
        input = Variable(torch.from_numpy(np.asarray(test_batched[0]))).type(dtype)
        output = net(input)
        output = np.clip((output.cpu().data.numpy()) * 255., 0, 255).astype(np.uint8)
        label = np.clip(np.asarray(test_batched[1]) * 255., 0, 255).astype(np.uint8)

        if i_test_batch == 0:
            output_patch = 10
            output_image = np.clip(input.data.cpu().numpy()[output_patch, 0, :, :] * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(config.test_folder + '/test_image_iter_' + str(iterations) + '_input.png', output_image)
            output_image = output[output_patch, 0, :, :]
            cv2.imwrite(config.test_folder + '/test_image_iter_'+str(iterations)+'_output.png', output_image)
            output_image = label[output_patch, 0, :, :]
            cv2.imwrite(config.test_folder + '/test_image_iter_' + str(iterations) + '_GT.png', output_image)
        for i in range(0, len(label)):
            # test_psnr = cal_psnr(output[i,], label[i,])
            # psnr.append(test_psnr)
            test_mse = cal_mse(output[i,], label[i,])
            psnr.append(test_mse)
    return psnr


def evaluate_detection_network_hdf5_PR(net, test_dataset, config):
    recall = np.asarray([])
    for i_test_batch in range(0, len(test_dataset) // config.batch_size):
        sample_batched = next(test_dataset)
        input = Variable(torch.from_numpy(sample_batched[0])).type(dtype)
        binary_criteria = 0.5
        output = net(input)
        output = output.cpu().data.numpy()
        output[output >= binary_criteria] = 1
        output[output < binary_criteria] = 0
        label = sample_batched[1]
        output = np.reshape(output, (output.shape[0], output.shape[1] * output.shape[2] * output.shape[3]))
        label = np.reshape(label, (label.shape[0], label.shape[1] * label.shape[2] * label.shape[3]))
        true_positive = np.sum(output * label, axis=1)
        # precision = true_positive / (np.sum(output, axis=1) + 1e-6)
        recall_batch = (true_positive + 1e-6) / (np.sum(label, axis=1) + 1e-6)
        recall = np.append(recall, recall_batch)
    # recall[recall > 1] = 1
    return recall


# helper functions.
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def removeLineFeed(line):
    temp = line.split('\n')
    return temp[0]

def read_lmdb_list(file):
    with open(file) as f:
        data = f.readlines()
    data = list(map(removeLineFeed, data))
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

def worker_init_fn(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def find_test_image_h5(test_dataset, config):
    for i_test_batch in range(0, len(test_dataset) // config.test_batchsize):
        test_batched = next(test_dataset)
        label = np.asarray(test_batched[1])
        label = np.reshape(label, (label.shape[0] * label.shape[1], label.shape[2] * label.shape[3]))
        label_sum = np.sum(label, axis=1)
        for i in range(label.shape[0]):
            if label_sum[i] > 0:
                return i_test_batch, i



#Helper functions.
def list_all_dir(path):
    result = []
    files = os.listdir(path)
    for file in files:
        m = os.path.join(path, file)
        if os.path.isdir(m):
            result.append(m)
    return result

def list_all(path):
    files = os.listdir(path)
    return list(map(join_path(path), files))

def findfiles(path, fnmatchex='*.*'):
    result = []
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnmatchex):
            fullname = os.path.join(root, filename)
            result.append(fullname)
    return result

def list_png_files(path):
    return findfiles(path, '*.png')

def read_images(files):
    result = []
    for file in files:
        if os.path.isdir(file):
            result.append(file)
    return result

def join_path(base_dir):
    def sub_func(file):
        return os.path.join(base_dir, file)
    return sub_func


def load_image(image_file):
    image = cv2.imread(image_file)
    return image

def load_image_list(image_file_list):
    images = list(map(cv2.imread, image_file_list))
    return images

def RGB_TO_YCRCB(image_rgb):
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YCR_CB)
    return image_yuv

def YCRCB_TO_RGB(image_yuv):
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YCR_CB2BGR)
    return image_rgb

def RGB_TO_YCRCB_BATCH(images):
    return list(map(RGB_TO_YCRCB, images))

def extractChannel(channel):
    def sub_func(image):
        return image[:, :, channel]
    return sub_func

def image_to_file(image, file):
    cv2.imwrite(file, image)

def extractChannel_batch(channel):
    def subfunc(image_list):
        return list(map(extractChannel(channel), image_list))
    return subfunc