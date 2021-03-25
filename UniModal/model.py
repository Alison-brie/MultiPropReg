import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import ext.pynd as pynd
import layers

shape = (160, 192, 224)

class FeatureLearning(nn.Module):

    def __init__(self):

        super(FeatureLearning, self).__init__()

        # FeatureLearning/Encoder functions
        dim = 3
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 16, 16, 1)) # 1
        self.enc.append(conv_block(dim, 16, 16, 1)) # 2
        self.enc.append(conv_block(dim, 16, 32, 2)) # 3
        self.enc.append(conv_block(dim, 32, 32, 1)) # 4
        self.enc.append(conv_block(dim, 32, 32, 1)) # 5

    def forward(self, src, tgt):

        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))

        return c11, c21, c12, c22



class MPR_net_Te(nn.Module):

    def __init__(self, criterion):
        super(MPR_net_Te, self).__init__()
        dim = 3
        int_steps = 7
        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.criterion = criterion
        self.FeatureLearning = FeatureLearning()
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform = SpatialTransformer()

        od = 32 + 1
        self.conv2_0 = conv_block(dim, od, 48, 1)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.predict_flow2a = predict_flow(16)

        self.dc_conv2_0 = conv(3, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2_1 = conv(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv2_2 = conv(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.predict_flow2b = predict_flow(32)

        od = 1 + 16 + 16 + 3
        self.conv1_0 = conv_block(dim, od, 48, 1)
        self.conv1_1 = conv_block(dim, 48, 32, 1)
        self.conv1_2 = conv_block(dim, 32, 16, 1)
        self.predict_flow1a = predict_flow(16)

        self.dc_conv1_0 = conv(3, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv1_1 = conv(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv1_2 = conv(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.predict_flow1b = predict_flow(32)


        self.resize = layers.ResizeTransform(1 / 2, dim)

        self.integrate2 = layers.VecInt(down_shape2, int_steps)
        self.integrate1 = layers.VecInt(down_shape1, int_steps)


    def forward(self, src, tgt):
        c11, c21, c12, c22 = self.FeatureLearning(src, tgt)

        corr2 = MatchCost(c22, c12)
        x = torch.cat((corr2, c22), 1)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        flow2 = self.predict_flow2a(x)
        upfeat2 = self.resize(x)

        x = self.dc_conv2_0(flow2)
        x = self.dc_conv2_1(x)
        x = self.dc_conv2_2(x)
        refine_flow2 = self.predict_flow2b(x) + flow2
        int_flow2 = self.integrate2(refine_flow2)
        up_int_flow2 = self.resize(int_flow2)
        features_s_warped = self.spatial_transform_f(c11, up_int_flow2)


        corr1 = MatchCost(c21, features_s_warped)
        x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        flow1 = self.predict_flow1a(x) + up_int_flow2

        x = self.dc_conv1_0(flow1)
        x = self.dc_conv1_1(x)
        x = self.dc_conv1_2(x)
        refine_flow1 = self.predict_flow1b(x) + flow1
        int_flow1 = self.integrate1(refine_flow1)
        flow = self.resize(int_flow1)
        y = self.spatial_transform(src, flow)
        return y, flow, int_flow1, refine_flow1, int_flow2, refine_flow2

    def hyper_parameters(self):
        return self.FeatureLearning.parameters()

    def y_parameters(self):
        return self.parameters()

    def _loss(self, src, tgt):
        y, flow, flow1, refine_flow1, flow2, refine_flow2 = self.forward(src, tgt)
        return self.criterion(y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                              self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)

    def new(self):
        model_new = MPR_net_Te(self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

class MPR_net_HO(nn.Module):

    def __init__(self, criterion):
        super(MPR_net_HO, self).__init__()
        dim = 3
        int_steps = 7
        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.criterion = criterion
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1) # (160, 192, 224)
        self.spatial_transform = SpatialTransformer()

        # FeatureLearning/Encoder functions
        dim = 3
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 16, 16, 1))  # 1
        self.enc.append(conv_block(dim, 16, 16, 1))  # 2
        self.enc.append(conv_block(dim, 16, 32, 2))  # 3
        self.enc.append(conv_block(dim, 32, 32, 1))  # 4
        self.enc.append(conv_block(dim, 32, 32, 1))  # 5

        od = 32 + 1
        self.conv2_0 = conv_block(dim, od, 48, 1) # [48, 32, 16]
        self.enc.append(self.conv2_0)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv2_1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv2_2)
        self.predict_flow2a = predict_flow(16)
        self.enc.append(self.predict_flow2a)

        self.dc_conv2_0 = conv(3, 48, kernel_size=3, stride=1, padding=1, dilation=1) # [48, 48, 32]
        self.enc.append(self.dc_conv2_0)
        self.dc_conv2_1 = conv(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv2_1)
        self.dc_conv2_2 = conv(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv2_2)
        self.predict_flow2b = predict_flow(32)
        self.enc.append(self.predict_flow2b)

        od = 1 + 16 + 16 + 3
        self.conv1_0 = conv_block(dim, od, 48, 1)
        self.enc.append(self.conv1_0)
        self.conv1_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv1_1)
        self.conv1_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv1_2)
        self.predict_flow1a = predict_flow(16)
        self.enc.append(self.predict_flow1a)

        self.dc_conv1_0 = conv(3, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.enc.append(self.dc_conv1_0)
        self.dc_conv1_1 = conv(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv1_1)
        self.dc_conv1_2 = conv(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv1_2)
        self.predict_flow1b = predict_flow(32)
        self.enc.append(self.predict_flow1b)


        self.resize = layers.ResizeTransform(1 / 2, dim)

        self.integrate2 = layers.VecInt(down_shape2, int_steps)
        self.integrate1 = layers.VecInt(down_shape1, int_steps)
        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # 初始化
        self.hyper_1.data.fill_(10)
        self.hyper_2.data.fill_(15)
        self.hyper_3.data.fill_(3.2)
        self.hyper_4.data.fill_(0.8)

    def forward(self, src, tgt):

        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))

        corr2 = MatchCost(c22, c12)
        x = torch.cat((corr2, c22), 1)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        flow2 = self.predict_flow2a(x)
        upfeat2 = self.resize(x)

        x = self.dc_conv2_0(flow2)
        x = self.dc_conv2_1(x)
        x = self.dc_conv2_2(x)
        refine_flow2 = self.predict_flow2b(x) + flow2
        int_flow2 = self.integrate2(refine_flow2)
        up_int_flow2 = self.resize(int_flow2)
        features_s_warped = self.spatial_transform_f(c11, up_int_flow2)


        corr1 = MatchCost(c21, features_s_warped)
        x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        flow1 = self.predict_flow1a(x) + up_int_flow2

        x = self.dc_conv1_0(flow1)
        x = self.dc_conv1_1(x)
        x = self.dc_conv1_2(x)
        refine_flow1 = self.predict_flow1b(x) + flow1
        int_flow1 = self.integrate1(refine_flow1)
        flow = self.resize(int_flow1)
        y = self.spatial_transform(src, flow)
        return y, flow, int_flow1, refine_flow1, int_flow2, refine_flow2

    def hyper_parameters(self):
        return [self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4]

    def y_parameters(self):
        return self.parameters()

    def _loss(self, src, tgt):
        y, flow, flow1, refine_flow1, flow2, refine_flow2 = self.forward(src, tgt)
        return self.criterion(y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                              self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)

    def new(self):
        model_new = MPR_net_HO(self.criterion).cuda()
        for x, y in zip(model_new.hyper_parameters(), self.hyper_parameters()):
            x.data.copy_(y.data)
        return model_new

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize= shape , mode='bilinear'): # (160, 192, 224)
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

def predict_flow(in_planes):
    dim = 3
    conv_fn = getattr(nn, 'Conv%dd' % dim)
    return conv_fn(in_planes, dim, kernel_size=3, padding=1)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))

def MatchCost(features_t, features_s):
    mc = torch.norm(features_t - features_s, p=1, dim=1)
    mc = mc[ ..., np.newaxis]
    return mc.permute(0, 4, 1, 2, 3)

class LossFunction_mpr(nn.Module):
    def __init__(self):
        super(LossFunction_mpr, self).__init__()

        self.ncc_loss = ncc_loss()
        self.gradient_loss = gradient_loss()
        self.flow_jacdet_loss = flow_jacdet_loss()
        self.multi_loss = multi_loss()

    def forward(self, y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                hyper_1, hyper_2, hyper_3, hyper_4):
        ncc = self.ncc_loss(tgt, y)
        grad = self.gradient_loss(flow)
        # jac = self.flow_jacdet_loss(flow)
        multi = self.multi_loss(src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4)
        # loss = multi + 10 * grad + 15 * ncc + 0.1 * jac
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

class multi_loss(nn.Module):
    def __init__(self):
        super(multi_loss, self).__init__()

        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.ncc_loss = ncc_loss()
        self.gradient_loss = gradient_loss()
        self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
        self.resize_1 = layers.ResizeTransform(2, len(inshape))
        self.resize_2 = layers.ResizeTransform(4, len(inshape))

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

        # loss += weights[1] * self.gradient_loss(refine_flow1) * 0.1
        # loss += weights[0] * self.gradient_loss(refine_flow2) * 0.1

        return loss




