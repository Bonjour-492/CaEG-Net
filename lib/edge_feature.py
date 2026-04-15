import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import EMCAD
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.UncertaintyAttention import UncertaintyAwareAttention

class edge(nn.Module):
    def __init__(self):
        super(edge, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.nf = 32  # channal
        self.nc = 2  # num_class
        act_fn = nn.ReLU(inplace=True)
        self.edge_conv0 = nn.Sequential(nn.Conv2d(64, self.nf, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv1 = nn.Sequential(nn.Conv2d(256, self.nf, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv3 = nn.Sequential(nn.Conv2d(self.nf, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1), act_fn)
        self.up_4_BGM = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.UncertaintyAwareAttention1 = UncertaintyAwareAttention(in_dim=64)
        self.UncertaintyAwareAttention2 = UncertaintyAwareAttention(in_dim=256)
        self.UncertaintyAwareAttention3 = UncertaintyAwareAttention(in_dim=512)
        self.UncertaintyAwareAttention4 = UncertaintyAwareAttention(in_dim=1024)


    def data_normal(self, x):
        min = x.min()
        if min < 0:
            x = x + torch.abs(min)
            min = x.min()
        max = x.max()
        dst = max - min
        normal_x = (x - min).true_divide(dst)
        return normal_x

    def edge_features(self, x1, x2):  # def BGM
        x21 = self.edge_conv1(x2)
        #[1, 32, 88, 88]
        edge_guidance = self.edge_conv2(self.edge_conv0(x1) + x21)
        #[1, 32, 88, 88]
        edge_out = self.up_4_BGM(self.edge_conv3(edge_guidance))
        #[1, 1, 352, 352]
        edge_out = edge_out.view(-1, 1, 224, 224)
        #[1, 352, 352]
        edge_out = self.data_normal(edge_out)
        #[1, 352, 352]

        return edge_out, edge_guidance

    def forward(self, x):
        # if grayscale input, convert to 3 channels
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)
        x1 = self.UncertaintyAwareAttention1(x1)
        #torch.Size([6, 64, 56, 56])
        x2 = self.resnet.layer1(x1)
        x2 = self.UncertaintyAwareAttention2(x2)
        #torch.Size([6, 256, 56, 56])
        x3 = self.resnet.layer2(x2)
        x3 = self.UncertaintyAwareAttention3(x3)
        #torch.Size([6, 512, 28, 28])
        x4 = self.resnet.layer3(x3)
        x4 = self.UncertaintyAwareAttention4(x4)
        #torch.Size([6, 1024, 14, 14])
        edge_out, edge_guidance = self.edge_features(x1, x2)


        return [x2,x3,x4], edge_out, edge_guidance
        #[x2,x3,x4]:[6, 256, 56, 56][6, 512, 28, 28][6, 1024, 14, 14] edge_out,edge_guidance:[6,224,224] [6,32,56,56]

if __name__ == '__main__':
    model = edge().cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()
    P = model(input_tensor)