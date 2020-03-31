import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

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

# h5_file = '/home/lipu/Documents/whale_recognition/Train_data/HDF5/DCL/common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_positive_patch/' +\
#           'train_common_bottlenose_framel-8_step-2_log_magspec_wavio_24bit_block_lineGT_patch50_stride25_1.h5'
h5_file = '/home/lipu/Documents/whale_recognition/Train_data/HDF5/DCL/common_bottlenose_logmag_plotGT_positive_patch/train_common_bottlenose_logmag_plotGT_patch50_stride25_1.h5'

# h5_file = '/home/lipu/Documents/whale_recognition/Train_data/HDF5/DCL/BSD_500_train_GT_width2_positive_patch/' + \
#     'train_BSD_500_train_GT_width2_patch50_stride25_1.h5'

h5_name = os.path.basename(os.path.dirname(h5_file))
output_dir = '/home/lipu/Documents/paper_submission/Images/patch_image/' + h5_name
checkDir(output_dir)


data, label = readh5(h5_file)
data = np.transpose(data, (0, 1, 3, 2))
label = np.transpose(label, (0, 1, 3, 2))
temp = 0
n_data = data.shape[0]
n_disp = 1
gray = 0.5

for i in range(0, n_data - n_disp, n_disp):
    print('%s / %s' % (str(i+n_disp), str(n_data)))
    disp_data = data[i:i+n_disp, ...]
    disp_label = label[i:i+n_disp, ...]
    # img = patch2image(disp_data, 2, color=[gray, gray, gray])
    # img = im_unit8(img)
    # output_file = output_dir + '/' + 'data_' + str(i) + '_' + str(i+n_disp) + '.png'
    # cv2.imwrite(output_file, img)

    img = patch2image(disp_label, 2, color=[gray, gray, gray])
    img = im_unit8(img)

    output_file = output_dir + '/' + 'label_' + str(i) + '_' + str(i + n_disp) + '.png'
    cv2.imwrite(output_file, img)



