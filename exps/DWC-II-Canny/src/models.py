import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.m_global import dtype


class Detection_ResNet(nn.Module):
    def __init__(self, width=64):
        super(Detection_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.relu1 = nn.PReLU(num_parameters=width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x_res = self.conv3(self.relu2(self.conv2(x)))
        x = x + x_res
        x_res = self.conv5(self.relu4(self.conv4(x)))
        x = x + x_res
        x_res = self.conv7(self.relu6(self.conv6(x)))
        x = x + x_res
        x_res = self.conv9(self.relu8(self.conv8(x)))
        x = x + x_res
        x = self.conv10(x)
        return x


class Detection_ResNet_BN(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.relu1 = nn.PReLU(num_parameters=width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        x = self.conv10(x)
        return x



class Detection_ResNet_BN2(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN2, self).__init__()
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        x = self.conv10(x)
        return x



class Detection_ResNet_BN_PCEN(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN_PCEN, self).__init__()
        self.epsilon = 1e-6
        # self.alpha = Variable(torch.ones(1) * 0.98, requires_grad=True).type(dtype)
        # self.delta = Variable(torch.ones(1) * 2, requires_grad=True).type(dtype)
        # self.gamma = Variable(torch.ones(1) * 0.5, requires_grad=True).type(dtype)
        self.alpha = nn.Parameter(torch.ones(1) * 0.98, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1) * 2, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)
        self.norminput = []

    def forward(self, x):
        # x = (x[:, 0:1, :, :] / ((self.epsilon + x[:, 1:2, :, :]) ^ self.alpha) + self.delta) ^ self.gamma - self.delta ^ self.gamma
        x = torch.pow(x[:, 0:1, :, :] / torch.pow(self.epsilon + x[:, 1:2, :, :], self.alpha) + self.delta, self.gamma)\
            - torch.pow(self.delta, self.gamma)
        self.norminput = x
        x = self.conv1(x)
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        x = self.conv10(x)
        return x



class Detection_ResNet_BN2_pyramid(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN2_pyramid, self).__init__()
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.down1 = nn.Conv2d(width, width * 2, 3, padding=1, stride=2)
        self.down1_conv1 = nn.Conv2d(width * 2, width * 2, 3, padding=1, stride=1)
        self.down1_conv1_bn = nn.BatchNorm2d(num_features=width * 2, eps=1e-05, momentum=0.1, affine=True)
        self.down1_relu1 = nn.PReLU(num_parameters=width * 2)
        self.down1_conv2 = nn.Conv2d(width * 2, width * 2, 3, padding=1, stride=1)
        self.down1_conv2_bn = nn.BatchNorm2d(num_features=width * 2, eps=1e-05, momentum=0.1, affine=True)
        self.down1_conv3 = nn.Conv2d(width * 2, width * 2, 3, padding=1, stride=1)
        self.down1_conv3_bn = nn.BatchNorm2d(num_features=width * 2, eps=1e-05, momentum=0.1, affine=True)
        self.down1_relu3 = nn.PReLU(num_parameters=width * 2)
        self.down1_conv4 = nn.Conv2d(width * 2, width * 2, 3, padding=1, stride=1)
        self.down1_conv4_bn = nn.BatchNorm2d(num_features=width * 2, eps=1e-05, momentum=0.1, affine=True)
        self.down1_conv5 = nn.Conv2d(width * 2, 1, 3, padding=1, stride=1)
        self.down2 = nn.Conv2d(width * 2, width * 4, 3, padding=1, stride=2)
        self.down2_conv1 = nn.Conv2d(width * 4, width * 4, 3, padding=1, stride=1)
        self.down2_conv1_bn = nn.BatchNorm2d(num_features=width * 4, eps=1e-05, momentum=0.1, affine=True)
        self.down2_relu1 = nn.PReLU(num_parameters=width * 4)
        self.down2_conv2 = nn.Conv2d(width * 4, width * 4, 3, padding=1, stride=1)
        self.down2_conv2_bn = nn.BatchNorm2d(num_features=width * 4, eps=1e-05, momentum=0.1, affine=True)
        self.down2_conv3 = nn.Conv2d(width * 4, width * 4, 3, padding=1, stride=1)
        self.down2_conv3_bn = nn.BatchNorm2d(num_features=width * 4, eps=1e-05, momentum=0.1, affine=True)
        self.down2_relu3 = nn.PReLU(num_parameters=width * 4)
        self.down2_conv4 = nn.Conv2d(width * 4, width * 4, 3, padding=1, stride=1)
        self.down2_conv4_bn = nn.BatchNorm2d(num_features=width * 4, eps=1e-05, momentum=0.1, affine=True)
        self.down2_conv5 = nn.Conv2d(width * 4, 1, 3, padding=1, stride=1)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)
        self.upsample = nn.functional.upsample
        self.prediction1 = []
        self.prediction2 = []
        self.prediction3 = []

    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        self.prediction1 = self.conv10(x)

        x = self.down1(x)
        x_res = self.down1_conv2_bn(self.down1_conv2(self.down1_relu1(self.down1_conv1_bn(self.down1_conv1(x)))))
        x = x + x_res
        x_res = self.down1_conv4_bn(self.down1_conv4(self.down1_relu3(self.down1_conv3_bn(self.down1_conv3(x)))))
        x = x + x_res
        self.prediction2 = self.down1_conv5(x)
        self.prediction2 = self.upsample(self.prediction2, scale_factor=2, mode='bilinear')

        x = self.down2(x)
        x_res = self.down2_conv2_bn(self.down2_conv2(self.down2_relu1(self.down2_conv1_bn(self.down2_conv1(x)))))
        x = x + x_res
        x_res = self.down2_conv4_bn(self.down2_conv4(self.down2_relu3(self.down2_conv3_bn(self.down2_conv3(x)))))
        x = x + x_res
        self.prediction3 = self.down2_conv5(x)
        self.prediction3 = self.upsample(self.prediction3, scale_factor=4, mode='bilinear')

        return self.prediction1



class Detection_ResNet_BN_PCEN_fixed_alpha(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN_PCEN_fixed_alpha, self).__init__()
        self.epsilon = 1e-6
        # self.alpha = Variable(torch.ones(1) * 0.98, requires_grad=True).type(dtype)
        # self.delta = Variable(torch.ones(1) * 2, requires_grad=True).type(dtype)
        # self.gamma = Variable(torch.ones(1) * 0.5, requires_grad=True).type(dtype)
        self.alpha = Variable(torch.ones(1) * 0.98, requires_grad=True).type(dtype)
        self.delta = nn.Parameter(torch.ones(1) * 2, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)
        self.norminput = []

    def forward(self, x):
        # x = (x[:, 0:1, :, :] / ((self.epsilon + x[:, 1:2, :, :]) ^ self.alpha) + self.delta) ^ self.gamma - self.delta ^ self.gamma
        x = torch.pow(x[:, 0:1, :, :] / torch.pow(self.epsilon + x[:, 1:2, :, :], self.alpha) + self.delta, self.gamma)\
            - torch.pow(self.delta, self.gamma)
        self.norminput = x
        x = self.conv1(x)
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        x = self.conv10(x)
        return x