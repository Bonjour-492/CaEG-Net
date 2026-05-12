import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import EMCAD
from lib.edge_feature import edge
import torch.nn.functional as F




class CaEGNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='E:\pytorch\ours\pretrained_model\pvt_v2_b2.pth'):
        super(CaEGNet, self).__init__()
        # conv block to convert single channel to 3 channels
        self.hg = 512
        self.nf = 32  # channal
        self.nc = 2  # num_class
        act_fn = nn.ReLU(inplace=True)
        self.edge_feature = edge()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        # backbone network initialization with pretrained weight
        if  encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = pretrained_dir
            channels=[512, 320, 128, 64]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = pretrained_dir
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
        print('Model %s created, param count: %d' %
                     ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        self.fu_4 = nn.Sequential(nn.Conv2d(channels[0] + self.nf, channels[0], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[0]), act_fn)
        self.fu_3 = nn.Sequential(nn.Conv2d(channels[1] + self.nf, channels[1], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[1]), act_fn)
        self.fu_2 = nn.Sequential(nn.Conv2d(channels[2] + self.nf , channels[2], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[2]), act_fn)
        self.fu_1 = nn.Sequential(nn.Conv2d(channels[3] + self.nf , channels[3], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[3]), act_fn)

        self.fu_8 = nn.Sequential(nn.Conv2d(channels[0] + self.hg, channels[0], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[0]), act_fn)
        self.fu_7 = nn.Sequential(nn.Conv2d(channels[1] + self.hg, channels[1], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[1]), act_fn)
        self.fu_6 = nn.Sequential(nn.Conv2d(channels[2] + self.hg, channels[2], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[2]), act_fn)
        self.fu_5 = nn.Sequential(nn.Conv2d(channels[3] + self.hg, channels[3], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(channels[3]), act_fn)

        self.concat1 = nn.Sequential(nn.Conv2d(256, channels[3], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(channels[3]), act_fn)
        self.concat2 = nn.Sequential(nn.Conv2d(512, channels[2], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(channels[2]), act_fn)
        self.concat3 = nn.Sequential(nn.Conv2d(1024, channels[1], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(channels[1]), act_fn)

        self.cat1 = nn.Sequential(nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(512), act_fn)
        self.cat2 = nn.Sequential(nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(512), act_fn)
        self.cat3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(512), act_fn)
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        self.num_classes = num_classes
        self.activate = nn.Sigmoid()


    def forward(self, img, args, p_fs, p_w1, epoch_num, i_batch, mode):   #EMCAD-BG-Casual
        # if grayscale input, convert to 3 channels
        if img.size()[1] == 1:
            img = self.conv(img)


        feature_list, edge_out, edge_guidance = self.edge_feature(img)
        # [x2,x3,x4]:[6, 256, 56, 56][6, 512, 28, 28][6, 1024, 14, 14] edge_out,edge_guidance:[6,224,224] [2,32,56,56]
        feature_list[0] = self.concat1(feature_list[0])#[6, 64, 56, 56]
        feature_list[1] = self.concat2(feature_list[1])#[6, 128, 28, 28]
        feature_list[2] = self.concat3(feature_list[2])#[6, 512, 14, 14]
        if mode == 'train':
            x1, x2, x3, x4, weight1, pre_features1, pre_weight1 = self.backbone(img,feature_list,args, p_fs, p_w1, epoch_num, i_batch, mode = 'train')
        else:
            x1, x2, x3, x4 = self.backbone(img,feature_list,args, p_fs, p_w1, epoch_num, i_batch, mode = 'test')
        #X1[2, 64, 56, 56] x2([2, 128, 28, 28]) x3([2, 320, 14, 14]) x4([2, 512, 7, 7])

        dec_outs = self.decoder(x4, [x3, x2, x1])
        #dec_outse([6, 512, 7, 7]) ([6, 320, 14, 14])([6, 128, 28, 28])([6, 64, 56, 56])

        dec_outs[0] = self.fu_4(torch.cat((dec_outs[0], F.interpolate(edge_guidance, scale_factor=1 / 8, mode='bilinear')), dim=1))#edge_guidence[6,32,56,56]
        dec_outs[1] = self.fu_3(torch.cat((dec_outs[1], F.interpolate(edge_guidance, scale_factor=1 / 4, mode='bilinear')), dim=1))
        dec_outs[2] = self.fu_2(torch.cat((dec_outs[2], F.interpolate(edge_guidance, scale_factor=1 / 2, mode='bilinear')), dim=1))
        dec_outs[3] = self.fu_1(torch.cat((dec_outs[3], edge_guidance), dim=1))
        # ([6, 512, 7, 7]) ([6, 320, 14, 14]) ([6, 128, 28, 28]) ([6, 64, 56, 56])

        # prediction heads
        p4 = self.out_head4(dec_outs[0])#[2, 1, 7, 7]
        p3 = self.out_head3(dec_outs[1])#[2, 1, 14, 14]
        p2 = self.out_head2(dec_outs[2])#[2, 1, 28, 28]
        p1 = self.out_head1(dec_outs[3])#[2, 1, 56, 56]

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')#[6, 9, 224, 224]
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')#[6, 9, 224, 224]
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')#[6, 9, 224, 224]
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')#[6, 9, 224, 224]


        # p4 = self.activate(p4)
        # p3 = self.activate(p3)
        # p2 = self.activate(p2)
        p1 = self.activate(p1)
        edge_out = self.activate(edge_out)

        if mode == 'test':


             return [p4, p3, p2, p1]

        # return [p4, p3, p2, p1]
        return [p4, p3, p2, p1], weight1, pre_features1, pre_weight1,edge_out
        # return [p4, p3, p2, p1], weight1, pre_features1, pre_weight1, weight2, pre_features2, pre_weight2, edge_out



if __name__ == '__main__':
    model = CaEGNet().cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())

