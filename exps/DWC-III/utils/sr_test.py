import os
from utils.m_global import dtype
import src.models as models
import time
import cv2
import fnmatch
import numpy as np
from torch.autograd import Variable
import torch

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
    return map(join_path(path), files)

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
    images = map(cv2.imread, image_file_list)
    return images

def RGB_TO_YCRCB(image_rgb):
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YCR_CB)
    return image_yuv

def YCRCB_TO_RGB(image_yuv):
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YCR_CB2BGR)
    return image_rgb

def RGB_TO_YCRCB_BATCH(images):
    return map(RGB_TO_YCRCB, images)

def extractChannel(channel):
    def sub_func(image):
        return image[:, :, channel]
    return sub_func

def image_to_file(image, file):
    cv2.imwrite(file, image)

def extractChannel_batch(channel):
    def subfunc(image_list):
        return map(extractChannel(channel), image_list)
    return subfunc



class Config():
    def __init__(self):
        pass



iterations = '100'
exp_name = 'DSRCNN_20181003_13-07-33_old_data'
exp_folder = '/home/lipu/Documents/whale_recognition/experiments/PYTORCH/CallDetection_Resnet_BN_HDF5_closs_only'
model_file = exp_folder + '/models/' + exp_name + '/DSRCNN-iter_' + iterations + '0000'


test_data = 'bottlenose_common_silbido_TEST_smoothInput1_0.025'
# test_data = 'bottlenose_common_silbido_TEST_smoothInput2_0.025'
# test_data = 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_TEST_smoothInput1_0.025'
# test_data = 'common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_TEST_smoothInput2_0.025'
test_base_folder = '/home/lipu/Documents/whale_recognition/Test_data'
test_folder = test_base_folder + '/' + test_data


# Read lmdb locations from configuration files.

# Register network
# net = models.Detection_ResNet_BN2(width=32).type(dtype)
net = models.Detection_ResNet_BN_PCEN(width=32).type(dtype)
all_net = net
all_net.load_state_dict(torch.load(model_file))


# Test

# output_dir = '/home/lipu/Documents/whale_recognition/MyCode/matlab_code/exp'
output_dir = './exp'
output_folder = output_dir + '/' + 'test_' + exp_name + '_' + iterations + 'w' + '_' + test_data
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_folders = list_all_dir(test_folder)
images = map(list_png_files, image_folders)
images = [image for line in images for image in line]

print("Test Start")
all_net.eval()
for test_image in images:
    image_name = os.path.basename(test_image)
    filename = image_name.split('.')[0]
    image = load_image(test_image)

    if 'GT' in image_name:
        image_to_file(image, os.path.join(output_folder, image_name))
        continue

    image = image.astype(np.float32) / 255
    output_y = np.clip(image * 255, 0, 255).astype(np.uint8)
    image_to_file(output_y, os.path.join(output_folder, filename + '_input.png'))

    image = image[:, :, 0]
    image = image.transpose()
    imgs_y = image[np.newaxis, np.newaxis, :]

    start_time = time.time()
    input = Variable(torch.from_numpy(imgs_y)).type(dtype)
    output = net(input)
    output_y = output.cpu().data.numpy()
    output_y = np.squeeze(output_y)
    output_y = output_y.transpose()
    output_y = np.clip(output_y * 255, 0, 255).astype(np.uint8)
    image_to_file(output_y, os.path.join(output_folder, image_name))
    end_time = time.time()
    print("Test for %s : %f s" % (image_name, end_time - start_time))
