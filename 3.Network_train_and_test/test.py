import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.m_global import dtype
import src.models as models
import src.m_func as m_func
import time
import cv2
import fnmatch
import numpy as np
from torch.autograd import Variable
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, required=True, help='path to model file')
parser.add_argument('--test_img_dir', type=str, required=True, help='path to test image directory')
parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')
config = parser.parse_args()
model_file = config.model_file
test_folder = config.test_img_dir
# Test
output_folder = config.output_dir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Register network, load weights.
net = models.Detection_ResNet_BN2(width=32).type(dtype)
all_net = net
all_net.load_state_dict(torch.load(model_file))

image_folders = m_func.list_all_dir(test_folder)
images = map(m_func.list_png_files, image_folders)
images = [image for line in images for image in line]

# Test
print("Test Start")
all_net.eval()
for test_image in images:
    test_dir = os.path.dirname(test_image)
    test_group = os.path.basename(test_dir)
    output_subdir = output_folder + '/' + test_group
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    image_name = os.path.basename(test_image)
    filename = image_name.split('.')[0]
    image = m_func.load_image(test_image)

    if 'GT' in image_name:
        m_func.image_to_file(image, os.path.join(output_subdir, image_name))
        continue

    image = image.astype(np.float32) / 255
    output_y = np.clip(image * 255, 0, 255).astype(np.uint8)

    image = image[:, :, 0]
    imgs_y = image[np.newaxis, np.newaxis, :]

    start_time = time.time()
    input = Variable(torch.from_numpy(imgs_y)).type(dtype)
    output = net(input)
    output_y = output.cpu().data.numpy()
    output_y = np.squeeze(output_y)
    output_y = np.clip(output_y * 255, 0, 255).astype(np.uint8)
    m_func.image_to_file(output_y, os.path.join(output_subdir, filename + '_predict.png'))
    end_time = time.time()
    print("Test for %s : %f s" % (image_name, end_time - start_time))
