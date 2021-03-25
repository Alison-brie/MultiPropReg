import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from layers import SpatialTransformer, shape, ResizeTransform
import ext.pynd as pynd
from MIND import MIND


class LossFunction_mpr_MIND(nn.Module):
    def __init__(self):
        super(LossFunction_mpr_MIND, self).__init__()

        self.ncc_loss = MIND()
        self.gradient_loss = gradient_loss()
        self.flow_jacdet_loss = flow_jacdet_loss()
        self.multi_loss = multi_loss_MIND()

    def forward(self, y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                hyper_1, hyper_2, hyper_3, hyper_4):
        ncc = self.ncc_loss(tgt, y)
        grad = self.gradient_loss(flow)
        multi = self.multi_loss(src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4)
        jac = self.flow_jacdet_loss(flow)
        loss = multi + 10 * grad + 15 * ncc + 0.1 * jac
        return loss, ncc, grad


class LossFunction_mpr_ncc(nn.Module):
    def __init__(self):
        super(LossFunction_mpr_ncc, self).__init__()
        self.ncc_loss = ncc_loss()
        self.gradient_loss = gradient_loss()
        self.flow_jacdet_loss = flow_jacdet_loss()
        self.multi_loss = multi_loss_ncc()

    def forward(self, y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                hyper_1, hyper_2, hyper_3, hyper_4):
        ncc = self.ncc_loss(tgt, y)
        grad = self.gradient_loss(flow)
        multi = self.multi_loss(src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4)
        loss = multi + hyper_1 * ncc + hyper_2 * grad
        return loss, ncc, grad


class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0


class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J, win=None):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        conv_fn = getattr(F, 'conv%dd' % ndims)
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -1 * torch.mean(cc)


class flow_jacdet_loss(nn.Module):
    def __init__(self):
        super(flow_jacdet_loss, self).__init__()

    def Get_Grad(self, y):
        ndims = 3

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            # r = [d, *range(d), *range(d + 1, ndims + 2)]
            # y = K.permute_dimensions(y, r)
            y = y.permute(d, *range(d), *range(d + 1, ndims + 2))

            dfi = y[1:, ...] - y[:-1, ...]
            # [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            dfi = F.pad(dfi, pad=(0,0, 0,0, 0,0, 0,0, 1, 0), mode="constant", value=0)

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            # r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            # df[i] = K.permute_dimensions(dfi, r)
            df[i] = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        # df[2] = K.permute_dimensions(df[2], (1, 0, 2, 3, 4))
        df[2] = df[2].permute(1, 0, 2, 3, 4)
        return df

    def forward(self, x):
        flow = x[0, :, :, :, :]
        vol_size = flow.shape[:-1]
        grid = np.stack(pynd.ndutils.volsize2ndgrid(vol_size), len(vol_size))
        grid = np.reshape(grid, (1,) + grid.shape)
        grid = torch.from_numpy(grid)
        J = self.Get_Grad(x + grid)
        # J = np.gradient(flow + grid)

        dx = J[0][0, :, :, :, :]
        dy = J[1][:, 0, :, :, :]
        dz = J[2][:, :, 0, :, :]

        Jdet0 = dx[:, :, :, 0] * (dy[:, :, :, 1] * dz[:, :, :, 2] - dy[:, :, :, 2] * dz[:, :, :, 1])
        Jdet1 = dx[:, :, :, 1] * (dy[:, :, :, 0] * dz[:, :, :, 2] - dy[:, :, :, 2] * dz[:, :, :, 0])
        Jdet2 = dx[:, :, :, 2] * (dy[:, :, :, 0] * dz[:, :, :, 1] - dy[:, :, :, 1] * dz[:, :, :, 0])

        Jdet = Jdet0 - Jdet1 + Jdet2

        loss = np.sum(np.maximum(0.0, -Jdet))

        return loss


class multi_loss_ncc(nn.Module):
  def __init__(self):
    super(multi_loss_ncc, self).__init__()

    inshape = shape
    down_shape2 = [int(d / 4) for d in inshape]
    down_shape1 = [int(d / 2) for d in inshape]
    self.ncc_loss = ncc_loss()
    self.gradient_loss = gradient_loss()
    self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
    self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
    self.resize_1 = ResizeTransform(2, len(inshape))
    self.resize_2 = ResizeTransform(4, len(inshape))

  def forward(self, src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4):
    loss = 0.
    zoomed_x1 = self.resize_1(tgt)
    zoomed_x2 = self.resize_1(src)
    warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
    loss += hyper_3 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[7])

    zoomed_x1 = self.resize_2(tgt)
    zoomed_x2 = self.resize_2(src)
    warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
    loss += hyper_4 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[5])

    return loss

class multi_loss_MIND(nn.Module):
  def __init__(self):
    super(multi_loss_MIND, self).__init__()

    inshape = shape
    down_shape2 = [int(d / 4) for d in inshape]
    down_shape1 = [int(d / 2) for d in inshape]
    self.sim_loss = MIND()
    self.gradient_loss = gradient_loss()
    self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
    self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
    self.resize_1 = ResizeTransform(2, len(inshape))
    self.resize_2 = ResizeTransform(4, len(inshape))

  def forward(self, src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4):
    zoomed_x1 = self.resize_1(tgt)
    zoomed_x2 = self.resize_1(src)
    warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
    loss_1 = hyper_3 * self.sim_loss(warped_zoomed_x2, zoomed_x1)

    zoomed_x1 = self.resize_2(tgt)
    zoomed_x2 = self.resize_2(src)
    warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
    loss_2 = hyper_4 * self.sim_loss(warped_zoomed_x2, zoomed_x1)
    loss = loss_1 + loss_2

    return loss


def flow_jacdet(flow):

    vol_size = flow.shape[:-1]
    grid = np.stack(pynd.ndutils.volsize2ndgrid(vol_size), len(vol_size))
    J = np.gradient(flow + grid)

    dx = J[0]
    dy = J[1]
    dz = J[2]

    Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
    Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
    Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def ncc(I, J):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)
