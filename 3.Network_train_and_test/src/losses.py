import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

from utils.m_global import dtype

bce_loss = nn.BCELoss().type(dtype)
mse_loss = nn.MSELoss().type(dtype)
l1_loss = nn.L1Loss()

class Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, epsilon=1e-3):
        super(Charbonnier_loss, self).__init__()
        self.eps = epsilon ** 2
        # self.eps = Variable(torch.from_numpy(np.asarray([epsilon ** 2])))
        # self.eps = Variable(torch.ones())


    def forward(self, X, Y):
        batchsize = X.data.shape[0]
        diff = X - Y
        square_err = diff ** 2
        square_err_sum_list = torch.sum(square_err, dim=1)
        square_err_sum_list = torch.sum(square_err_sum_list, dim=1)
        square_err_sum_list = torch.sum(square_err_sum_list, dim=1)
        square_err_sum_list = square_err_sum_list + self.eps
        error = torch.sqrt(square_err_sum_list)
        loss = torch.sum(error) / batchsize
        return loss


class MSEloss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, epsilon=1e-3):
        super(MSEloss, self).__init__()
        self.eps = epsilon ** 2

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        sum_square_err = torch.sum(diff * diff)
        loss = sum_square_err / X.data.shape[0] / 2.
        # loss = torch.sum(error)
        return loss

class weighted_MSEloss(nn.Module):
    def __init__(self):
        super(weighted_MSEloss, self).__init__()

    def forward(self, X, Y, weight):
        diff = torch.add(X, -Y)
        diff_square = diff * diff
        positive_label_diff = diff_square * Y
        sum_square_err = torch.sum(diff_square + weight * positive_label_diff)
        loss = sum_square_err / X.data.shape[0] / 2.
        return loss


class Continuity_loss(nn.Module):
    def __init__(self):
        super(Continuity_loss, self).__init__()

    def forward(self, X, neighbor):
        loss = Variable(torch.zeros(1)).type(dtype)
        for i in range(neighbor):
            for j in range(neighbor):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j == 0:
                    corr = X[:-i, :] * X[i:, :]
                if i == 0 and j > 0:
                    corr = X[:-i, :] * X[i:, :]
                if i > 0 and j > 0:
                    corr = X[:-i, :-j] * X[i:, j:]
                corr = torch.abs(corr)
                loss -= torch.sum(corr * torch.log(corr))
        loss = loss / X.data.shape[0]
        return loss


def continuity_loss(output, neighbor):
    return Continuity_loss()(output, neighbor)


def weightedEuclideanLoss(output, label, weight):
    return weighted_MSEloss()(output, label, weight)


def euclideanLoss(output, label, input_size):
    mse = mse_loss(output, label)
    mse = mse*((input_size)/2.)
    return mse

def euclideanLoss2(output, label):
    mse = MSEloss()(output, label)
    return mse

def L1NormLoss(output, label, input_size):
    l1norm = l1_loss(output, label)
    l1norm = l1norm*((input_size)/2.)
    return l1norm

def C_Loss(output, label):
    c_loss_func = Charbonnier_loss(epsilon=1e-3)
    return c_loss_func(output, label)