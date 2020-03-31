import numpy as np
import torch
import torch.nn.init as init
import cv2
from torch.autograd import Variable
from utils.m_global import dtype


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
