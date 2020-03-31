import numpy as np
import torch
import torch.nn.init as init
import cv2
from torch.autograd import Variable
from utils.m_global import dtype

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
    if mse == 0:
        return -1
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def evaluate_detection_network(net, test_dataset, config, iterations):
    psnr = []
    for i_test_batch in xrange(0, len(test_dataset) / config.test_batchsize ):
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
        for i in xrange(0, len(label)):
            test_psnr = cal_psnr(output[i,], label[i,])
            psnr.append(test_psnr)
    return psnr


def evaluate_detection_network2(net, test_dataset, config, iterations, test_batch_num, test_pic_num):
    psnr = []
    for i_test_batch in xrange(0, len(test_dataset) / config.test_batchsize ):
        test_batched = next(test_dataset)
        input = Variable(torch.from_numpy(np.asarray(test_batched[0]))).type(dtype)
        output = net(input)
        output = np.clip((output.cpu().data.numpy()) * 255., 0, 255).astype(np.uint8)
        label = np.clip(np.asarray(test_batched[1]) * 255., 0, 255).astype(np.uint8)

        if i_test_batch == test_batch_num:
            output_patch = test_pic_num
            output_image = np.clip(input.data.cpu().numpy()[output_patch, 0, :, :] * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(config.test_folder + '/test_image_iter_' + str(iterations) + '_input.png', output_image)
            output_image = output[output_patch, 0, :, :]
            cv2.imwrite(config.test_folder + '/test_image_iter_'+str(iterations)+'_output.png', output_image)
            output_image = label[output_patch, 0, :, :]
            cv2.imwrite(config.test_folder + '/test_image_iter_' + str(iterations) + '_GT.png', output_image)
        for i in xrange(0, len(label)):
            test_psnr = cal_psnr(output[i,], label[i,])
            if test_psnr == -1:
                continue
            psnr.append(test_psnr)
    return psnr


def find_test_image_h5(test_dataset, config):
    for i_test_batch in xrange(0, len(test_dataset) / config.test_batchsize):
        test_batched = next(test_dataset)
        label = np.asarray(test_batched[1])
        label = np.reshape(label, (label.shape[0] * label.shape[1], label.shape[2] * label.shape[3]))
        label_sum = np.sum(label, axis=1)
        for i in range(label.shape[0]):
            if label_sum[i] > 0:
                return i_test_batch, i


def find_test_image(test_dataset):
    for i_batch, sample_batched in enumerate(test_dataset):
        label = sample_batched['label'].numpy()
        label = np.reshape(label, (label.shape[0] * label.shape[1], label.shape[2] * label.shape[3]))
        label_sum = np.sum(label, axis=1)
        for i in range(label.shape[0]):
            if label_sum[i] > 0:
                return i_batch, i


def evaluate_detection_network_hdf5_PR(net, test_dataset, config):
    recall = np.asarray([])
    for i_test_batch in xrange(0, len(test_dataset) / config.batch_size):
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
