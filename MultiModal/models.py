import torch.nn as nn
from layers import *

class MPR_net_Tr(nn.Module):

    def __init__(self, criterion):
        super(MPR_net_Tr, self).__init__()
        int_steps = 7
        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.criterion = criterion
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform = SpatialTransformer()

        # FeatureLearning/Encoder functions
        dim = 3
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 8, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 8, 8, 1))  # 1
        self.enc.append(conv_block(dim, 8, 8, 1))  # 2
        self.enc.append(conv_block(dim, 8, 16, 2))  # 3
        self.enc.append(conv_block(dim, 16, 16, 1))  # 4
        self.enc.append(conv_block(dim, 16, 16, 1))  # 5

        od = 16 + 1
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

        od = 1 + 8 + 16 + 3
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


        self.resize = ResizeTransform(1 / 2, dim)

        self.integrate2 = VecInt(down_shape2, int_steps)
        self.integrate1 = VecInt(down_shape1, int_steps)
        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)

        # 初始化
        self.hyper_1.data.fill_(10)
        self.hyper_2.data.fill_(1) #
        self.hyper_3.data.fill_(3.2)
        self.hyper_4.data.fill_(0.8)

    def forward(self, src, tgt):
        ##################### Feature extraction #######################
        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))

        ##################### Estimation at scale-2 #######################
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

        ##################### Estimation at scale-1 #######################
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

        ##################### Upsampling at scale-0 #######################
        flow = self.resize(int_flow1)
        y = self.spatial_transform(src, flow)
        return y, flow, int_flow1, refine_flow1, int_flow2, refine_flow2

    def _loss(self, src, tgt):
        y, flow, flow1, refine_flow1, flow2, refine_flow2 = self.forward(src, tgt)
        return self.criterion(y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                              self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)




class MPR_net_Uni(nn.Module):

    def __init__(self, criterion):
        super(MPR_net_Uni, self).__init__()
        int_steps = 7
        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.criterion = criterion
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform = SpatialTransformer()

        # FeatureLearning/Encoder functions
        dim = 3
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 8, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 8, 8, 1))  # 1
        self.enc.append(conv_block(dim, 8, 8, 1))  # 2
        self.enc.append(conv_block(dim, 8, 16, 2))  # 3
        self.enc.append(conv_block(dim, 16, 16, 1))  # 4
        self.enc.append(conv_block(dim, 16, 16, 1))  # 5

        od = 16 + 1
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

        od = 1 + 8 + 16 + 3
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


        self.resize = ResizeTransform(1 / 2, dim)

        self.integrate2 = VecInt(down_shape2, int_steps)
        self.integrate1 = VecInt(down_shape1, int_steps)
        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)

        # 初始化
        self.hyper_1.data.fill_(10)
        self.hyper_2.data.fill_(15)
        self.hyper_3.data.fill_(3.2)
        self.hyper_4.data.fill_(0.8)

    def forward(self, src, tgt):
        ##################### Feature extraction #######################
        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))

        ##################### Estimation at scale-2 #######################
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

        ##################### Estimation at scale-1 #######################
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

        ##################### Upsampling at scale-0 #######################
        flow = self.resize(int_flow1)
        y = self.spatial_transform(src, flow)
        return y, flow, int_flow1, refine_flow1, int_flow2, refine_flow2

    def _loss(self, src, tgt):
        y, flow, flow1, refine_flow1, flow2, refine_flow2 = self.forward(src, tgt)
        return self.criterion(y, tgt, src, flow, flow1, refine_flow1, flow2, refine_flow2,
                              self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)






