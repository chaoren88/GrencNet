import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DyC_attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(DyC_attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        # if init_weight:
        #     self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = DyC_attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


## Channel Attention (CA) Layer
class CA_attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_attention, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            Dynamic_conv2d(channel, channel //16, 1, stride=1, padding=(1 - 1) // 2),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(channel // 16, channel, 1, stride=1, padding=(1 - 1) // 2),

        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool1(x)
        y = self.avg_pool2(self.conv_du(y))
        return x * self.sig(y)
##########################################################################


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Dynamic_conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
##########################################################################


class Dynamic_Joint_Attention(nn.Module):
    def __init__(self, n_feat, kernel_size=3):

        super(Dynamic_Joint_Attention, self).__init__()
        modules_body = []
        for i in range(3):
            modules_body.append(
                BasicConv(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True),
            )
        modules_body.append(BasicConv(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False))
        self.SA = spatial_attn_layer()  # Spatial Attention
        self.CA = CA_attention(n_feat)  # Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = BasicConv(n_feat * 2, n_feat, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res + x


class Corrector(nn.Module):
    def __init__(self, nf = 64, nf_2=64, input_para=1):
        super(Corrector, self).__init__()
        self.head_noisy = BasicConv(input_para, nf_2, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        self.ConvNet = nn.Sequential(*[
            BasicConv(nf*2, nf, 1, stride=1, padding=(1 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
        ])

        self.att = simam_module()
        self.conv1 = BasicConv(nf, input_para, 3, stride=1, padding=(3 - 1) // 2, relu=False)


    def forward(self, feature_maps, noisy_map):
        para_maps = self.head_noisy(noisy_map)
        cat_input = self.ConvNet(torch.cat((feature_maps, para_maps), dim=1))
        return self.conv1(self.att(cat_input))


class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            BasicConv(in_nc, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        ])
        # self.att = CALayer(nf)
        self.att = simam_module()
        self.conv1 = BasicConv(nf, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, input):
        return self.conv1(self.att(self.ConvNet(input)))

class GFDDR(nn.Module):
    def __init__(self):
        super(GFDDR, self).__init__()
        num_crb = 10
        n_feats = 64
        reduction = 16
        inp_chans = 1

        modules_head = [
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True)]

        self.head1 = BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True)

        modules_body = [
            Dynamic_Joint_Attention(n_feats) \
            for _ in range(num_crb)]

        modules_tail = [
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True)]
        self.conv1 = BasicConv(inp_chans, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.conv2 = BasicConv(3*n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True)
        self.conv3 = BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        self.NFDR = nn.Sequential(*[
            BasicConv(inp_chans, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True, bias=True),
            BasicConv(n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True, bias=True),
            BasicConv(n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True, bias=True),
            BasicConv(n_feats, n_feats // 16, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True),
            nn.ReLU(inplace=True),
            BasicConv(n_feats // 16, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        ])
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, last_maps,feature_maps, noisy_map):
        para_maps = self.conv1(noisy_map)
        feature_maps1 = self.head1(feature_maps)
        conv_feature_maps = self.head(last_maps)
        cat_input = self.conv2(torch.cat((conv_feature_maps, feature_maps1, para_maps), dim=1))
        x = self.conv3(self.body(cat_input))
        para_maps1 = self.NFDR(noisy_map)
        f = torch.mul(para_maps1,x) + conv_feature_maps
        return self.tail(f)


class DN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, para=1):
        super(DN, self).__init__()
        self.head = nn.Sequential(
            BasicConv(in_nc, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        )
        self.C = Corrector()
        self.P = Predictor(in_nc=3, nf=nf)
        self.F = GFDDR()

        self.tail = nn.Sequential(
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, out_nc, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        )

    def forward(self, noisyImage):
        M0 = self.head(noisyImage)
        n0 = self.P(noisyImage)
        M1 = self.F(M0, M0, n0) + M0
        noise_map = []

        for i in range(4):
            n0 = n0 + self.C(M1, n0)
            M1 = self.F(M1, M0, n0) + M0
            noise_map.append(n0)
        return noise_map,self.tail(M1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Net = DN().cuda()
# input = torch.rand(1,3,256,256).cuda()
# print(Net(input))
# print_network(Net)

