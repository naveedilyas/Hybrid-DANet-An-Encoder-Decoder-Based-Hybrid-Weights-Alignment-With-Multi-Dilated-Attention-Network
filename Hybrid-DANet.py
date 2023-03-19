#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 15:22
# @Author  : Eric Ching
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class VAEBranch(nn.Module):

    def __init__(self, input_shape, init_channels, out_channels, squeeze_channels=None):
        super(VAEBranch, self).__init__()
        self.input_shape = input_shape

        if squeeze_channels:
            self.squeeze_channels = squeeze_channels
        else:
            self.squeeze_channels = init_channels * 4

        self.hidden_conv = nn.Sequential(nn.GroupNorm(8, init_channels * 8),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(init_channels * 8, self.squeeze_channels, (3, 3, 3),
                                                   padding=(1, 1, 1)),
                                         nn.AdaptiveAvgPool3d(1))

        self.mu_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)
        self.logvar_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)

        recon_shape = np.prod(self.input_shape) // (16 ** 3)

        self.reconstraction = nn.Sequential(nn.Linear(self.squeeze_channels // 2, init_channels * 8 * recon_shape),
                                            nn.ReLU(inplace=True))

        self.vconv4 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 8, (1, 1, 1)),
                                    nn.Upsample(scale_factor=2))

        self.vconv3 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 4, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 4, init_channels * 4))

        self.vconv2 = nn.Sequential(nn.Conv3d(init_channels * 4, init_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 2, init_channels * 2))

        self.vconv1 = nn.Sequential(nn.Conv3d(init_channels * 2, init_channels, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels, init_channels))

        self.vconv0 = nn.Conv3d(init_channels, out_channels, (1, 1, 1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.hidden_conv(x)
        batch_size = x.size()[0]
        x = x.view((batch_size, -1))
        mu = x[:, :self.squeeze_channels // 2]
        mu = self.mu_fc(mu)
        logvar = x[:, self.squeeze_channels // 2:]
        logvar = self.logvar_fc(logvar)
        z = self.reparameterize(mu, logvar)
        re_x = self.reconstraction(z)
        recon_shape = [batch_size,
                       self.squeeze_channels // 2,
                       self.input_shape[0] // 16,
                       self.input_shape[1] // 16,
                       self.input_shape[2] // 16]
        print(len(recon_shape))
        re_x = re_x.view(recon_shape)
        #x = x.view((batch_size, -1))
        x = self.vconv4(re_x)
        x = self.vconv3(x)
        x = self.vconv2(x)
        x = self.vconv1(x)
        vout = self.vconv0(x)

        return vout, mu, logvar


class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self,in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        #self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)

        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)

        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        uout = F.sigmoid(uout)

        return uout


class UnetVAE3D(nn.Module):

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UnetVAE3D, self).__init__()
        self.unet = UNet3D(input_shape, in_channels, out_channels, init_channels, p)
        self.vae_branch = VAEBranch(input_shape, init_channels, out_channels=in_channels)

    def forward(self, x):
        uout, c4d = self.unet(x)
        vout, mu, logvar = self.vae_branch(c4d)

        return uout, vout, mu, logvar









#################################################################
################################################################
import torch
import torch.nn as nn
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class AttU_Net(nn.Module):
    def __init__(self, img_ch=4, output_ch=3):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16)
        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.Conv4 = conv_block(ch_in=64, ch_out=128)
        self.Conv5 = conv_block(ch_in=128, ch_out=256)
        self.droput = nn.Dropout(p=0.2)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Att5 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Att2 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv3d(16, output_ch, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.droput(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        #print("Upconv 5 output", d5.shape)
        x4 = self.Att5(g=d5, x=x4)
        #print("Attention 5 output",x4.shape)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)


        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        uout = F.sigmoid(d1)
        return uout

# if __name__ == "__main__":
#     input = torch.randn(1,4,160,192,128)
#     model = AttU_Net()
#     output = model(input)
#     print(output.shape)



#################################################################
#################################################################

import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x

# class Multi_Scale(nn.Module):
#     def __init__(self, in_ch, sq_ch, exp_ch1, exp_ch2, exp_ch3):
#         super(Multi_Scale, self).__init__()
#         self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

#         self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

#         self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x1 = self.squeeze_layer(x)
#         x2 = self.expnad_layer1_1x1(x1)
#         x3 = self.expnad_layer2_3x3(x1)

#         x4 = self.expnad_layer3_1x1(x2)
#         x5 = self.expnad_layer4_3x3(x2)

#         x6 = self.expnad_layer5_1x1(x3)
#         #print("x6",x6.shape)
#         x7 = self.expnad_layer6_3x3(x3)
#         #print("x7", x7.shape)

#         xc1 = self.relu(torch.cat((x4, x5), dim=1))
#         xc2 = self.relu(torch.cat((x6, x7), dim=1))
#         xc3 = self.relu(torch.cat((xc1, xc2), dim=1))
#         return xc3
#########################################################

class Multi_Scale(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #self. att1 = Attention_Block_Modified(exp_ch,exp_ch,4)
        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)
        #print("x3",x3.shape)
        #x4 = self.att1(x3)
        #print("x444",x4.shape)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        #xc3 = self.relu(torch.cat((x2, x4), dim=1))
        return xc3

#########################################################

class Attention_Block(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))
        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x * psi

class Atten_UN_Multi_Scale(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(Atten_UN_Multi_Scale, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        self.fire1 = Multi_Scale(in_ch=16,sq_ch=4,exp_ch=8)
        #self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)
        #self.fire1 = Multi_Scale(16, 8, 8, 8, 4)


        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.fire2 = Multi_Scale(in_ch=32,sq_ch=8,exp_ch=16)
        #self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.fire3 = Multi_Scale(in_ch=64,sq_ch=16,exp_ch=32)
        #self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.fire4 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        #self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.fire5 = Multi_Scale(in_ch=256,sq_ch=64,exp_ch=128)
        #self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        self.upfire6 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        #self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        self.upfire7 = Multi_Scale(in_ch=64,sq_ch=16, exp_ch=32)
        #self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        self.upfire8 = Multi_Scale(in_ch=32, sq_ch=8, exp_ch=16)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        self.upfire9 = Multi_Scale(in_ch=16, sq_ch=4, exp_ch=8)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        #out = torch.sigmoid(out)
        return out



################################################################# Atten_UN_Multi_Scale_Modified     ################################################################
#################################################################                                   ################################################################

import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x

# class Multi_Scale(nn.Module):
#     def __init__(self, in_ch, sq_ch, exp_ch1, exp_ch2, exp_ch3):
#         super(Multi_Scale, self).__init__()
#         self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#
#         self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))
#
#         self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x1 = self.squeeze_layer(x)
#         x2 = self.expnad_layer1_1x1(x1)
#         x3 = self.expnad_layer2_3x3(x1)
#
#         x4 = self.expnad_layer3_1x1(x2)
#         x5 = self.expnad_layer4_3x3(x2)
#
#         x6 = self.expnad_layer5_1x1(x3)
#         #print("x6",x6.shape)
#         x7 = self.expnad_layer6_3x3(x3)
#         #print("x7", x7.shape)
#
#         xc1 = self.relu(torch.cat((x4, x5), dim=1))
#         xc2 = self.relu(torch.cat((x6, x7), dim=1))
#         xc3 = self.relu(torch.cat((xc1, xc2), dim=1))
#         return xc3


class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi

class Atten_UN_Multi_Scale_Modified(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(Atten_UN_Multi_Scale_Modified, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        #self.fire1 = Multi_Scale(in_ch=16,sq_ch=4,exp_ch=8)
        self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        #self.fire2 = Multi_Scale(in_ch=32,sq_ch=8,exp_ch=16)
        self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        #self.fire3 = Multi_Scale(in_ch=64,sq_ch=16,exp_ch=32)
        self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        #self.fire4 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        #self.fire5 = Multi_Scale(in_ch=256,sq_ch=64,exp_ch=128)
        self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        #self.upfire6 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        #self.upfire7 = Multi_Scale(in_ch=64,sq_ch=16, exp_ch=32)
        self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=8, exp_ch=16)
        self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=4, exp_ch=8)
        self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        #out = torch.sigmoid(out)
        return out

################################################################# Atten_UN_Multi_Scale_Modified     ################################################################
#################################################################                                   ################################################################


import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x

class Multi_Scale_Fire(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale_Fire, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        return xc3





class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        #print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        #print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi

class Atten_UN_Multi_Scale_Modified_Fire(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(Atten_UN_Multi_Scale_Modified_Fire, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        self.fire1 = Multi_Scale_Fire(in_ch=16,sq_ch=4,exp_ch=8)
        #self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.fire2 = Multi_Scale_Fire(in_ch=32,sq_ch=8,exp_ch=16)
        #self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.fire3 = Multi_Scale_Fire(in_ch=64,sq_ch=16,exp_ch=32)
        #self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.fire4 = Multi_Scale_Fire(in_ch=128,sq_ch=32,exp_ch=64)
        #self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.fire5 = Multi_Scale_Fire(in_ch=256,sq_ch=64,exp_ch=128)
        #self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        self.upfire6 = Multi_Scale_Fire(in_ch=128,sq_ch=32,exp_ch=64)
        #self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        self.upfire7 = Multi_Scale_Fire(in_ch=64,sq_ch=16, exp_ch=32)
        #self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        self.upfire8 = Multi_Scale_Fire(in_ch=32, sq_ch=8, exp_ch=16)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        self.upfire9 = Multi_Scale_Fire(in_ch=16, sq_ch=4, exp_ch=8)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        #out = torch.sigmoid(out)
        return out

# if __name__ == "__main__":
#     #input  = torch.randn(1,4,160,192,128)
#     input = torch.randn(1, 4, 128, 128, 128)
#     #input = torch.randn(1, 4, 164, 164, 164)
#     #input2 = torch.randn((1,8,100,100,60))
#     model = Atten_UN_Multi_Scale_Modified_Fire()
#     output = model(input)
#     print(output.shape)


#################################################################   Atten_UN_Dense_Encoder           #######################################################
#################################################################                                    #######################################################


import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x


'''

class Multi_Scale(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #self. att1 = Attention_Block_Modified(exp_ch,exp_ch,4)
        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)
        #print("x3",x3.shape)
        #x4 = self.att1(x3)
        #print("x444",x4.shape)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        #xc3 = self.relu(torch.cat((x2, x4), dim=1))
        return xc3


'''


class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        #print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        #print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi
class DDCB(nn.Module):
    def __init__(self, in_planes):
        super(DDCB, self).__init__()
        self.seen = 0
        # self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=1),nn.Conv3d(16, 8, 3, padding=1), nn.ReLU(True))
        # self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 8, 32, 1), nn.ReLU(True),nn.Conv3d(32, 16, 3, padding=2, dilation=2),nn.Conv3d(16, 8, 3, padding=2, dilation=2), nn.ReLU(True))
        # self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 16, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=3, dilation=3),nn.Conv3d(16, 8, 3, padding=3, dilation=3), nn.ReLU(True))
        # self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 16, in_planes, 3, padding=1, dilation=1), nn.ReLU(True))

        self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 16, 1), nn.ReLU(True), nn.Conv3d(16, 8, 3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 8, 16, 1), nn.ReLU(True),nn.Conv3d(16, 8, 3, padding=2, dilation=2))
        self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 16, 32, 1), nn.ReLU(True), nn.Conv3d(32, 8, 3, padding=3, dilation=3))
        self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 24, in_planes, 3, padding=1, dilation=1), nn.ReLU(True))

    def forward(self, x):
        #print (x.shape)
        x1_raw = self.conv1(x)
        #print (x1_raw.shape)
        x1 = torch.cat([x, x1_raw], 1)
        #print (x1.shape)
        x2_raw = self.conv2(x1)
        #print (x2_raw.shape)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        #print (x2.shape)
        x3_raw = self.conv3(x2)
        #print (x3_raw.shape)
        x3 = torch.cat([x, x2_raw, x3_raw], 1)
        #print (x3.shape)
        output = self.conv4(x3)
        #print (output.shape)
        return output
class Atten_UN_Dense_Encoder(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(Atten_UN_Dense_Encoder, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        self.dense1 = DDCB(16)
        #self.fire1 = Multi_Scale(in_ch=16,sq_ch=4,exp_ch=8)
        #self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.dense2 = DDCB(32)
        #self.fire2 = Multi_Scale(in_ch=32,sq_ch=8,exp_ch=16)
        #self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.dense3 = DDCB(64)
        #self.fire3 = Multi_Scale(in_ch=64,sq_ch=16,exp_ch=32)
        #self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.dense4 = DDCB(128)
        #self.fire4 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        #self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.dense5 = DDCB(256)
        #self.fire5 = Multi_Scale(in_ch=256,sq_ch=64,exp_ch=128)
        #self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        #self.upfire6 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        #self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        #self.upfire7 = Multi_Scale(in_ch=64,sq_ch=16, exp_ch=32)
        #self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=8, exp_ch=16)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=4, exp_ch=8)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        #x1 = self.dense1(x1)
        #x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        #x2 = self.dense2(x2)
        #x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        #x3 = self.dense3(x3)
        #x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        #x4 = self.dense4(x4)
        #x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        #x5 = self.dense5(x5)
        #x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        #d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        #d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        #d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        #d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        #out = torch.sigmoid(out)
        return out

# class DDCB(nn.Module):
#     def __init__(self, in_planes):
#         super(DDCB, self).__init__()
#         self.seen = 0
#         self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 256, 1), nn.ReLU(True), nn.Conv3d(256, 128, 3, padding=1),nn.Conv3d(128, 64, 3, padding=1), nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 64, 256, 1), nn.ReLU(True),nn.Conv3d(256, 128, 3, padding=2, dilation=2),nn.Conv3d(128, 64, 3, padding=2, dilation=2), nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 128, 256, 1), nn.ReLU(True), nn.Conv3d(256, 128, 3, padding=3, dilation=3),nn.Conv3d(128, 64, 3, padding=3, dilation=3), nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 128, in_planes, 3, padding=1, dilation=1), nn.ReLU(True))
#
#     def forward(self, x):
#         #print (x.shape)
#         x1_raw = self.conv1(x)
#         #print (x1_raw.shape)
#         x1 = torch.cat([x, x1_raw], 1)
#         #print (x1.shape)
#         x2_raw = self.conv2(x1)
#         #print (x2_raw.shape)
#         x2 = torch.cat([x, x1_raw, x2_raw], 1)
#         #print (x2.shape)
#         x3_raw = self.conv3(x2)
#         #print (x3_raw.shape)
#         x3 = torch.cat([x, x2_raw, x3_raw], 1)
#         #print (x3.shape)
#         output = self.conv4(x3)
#         #print (output.shape)
#         return output
# if __name__ == "__main__":
#     #input  = torch.randn(1,4,160,192,128)
#     input = torch.randn(1, 4, 128, 128, 128)
#     #input = torch.randn(1, 4, 240, 240, 155)
#     #input = torch.randn(1, 4, 164, 164, 164)
#     #input2 = torch.randn((1,8,100,100,60))
#     model = Atten_UN_Dense_Encoder()
#     output = model(input)
#     print(output.shape)











#unet3d=UNet3D((12,12,20))
#print(unet3d)
# '''
# if __name__ == '__main__':
#     import time
#     import torch
#     from torch.autograd import Variable
#     from torchsummaryX import summary
#
#     torch.cuda.set_device(0)
#     net =UnetVAE3D(128,128,32).cuda().eval()
#
#     data = Variable(torch.randn(4, 128, 128, 32, 8)).cuda()
#
#     out = net(data)
#
#     summary(net,data)
    # print("out size: {}".format(out.size()))


###################################################################################    ###########################################################



#################################################################


import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x


'''
class Multi_Scale(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #self. att1 = Attention_Block_Modified(exp_ch,exp_ch,4)
        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)
        #print("x3",x3.shape)
        #x4 = self.att1(x3)
        #print("x444",x4.shape)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        #xc3 = self.relu(torch.cat((x2, x4), dim=1))
        return xc3


'''


class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        #print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        #print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi
class DDCB(nn.Module):
    def __init__(self, in_planes):
        super(DDCB, self).__init__()
        self.seen = 0
        self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=1),nn.Conv3d(16, 8, 3, padding=1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 8, 32, 1), nn.ReLU(True),nn.Conv3d(32, 16, 3, padding=2, dilation=2),nn.Conv3d(16, 8, 3, padding=2, dilation=2), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 16, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=3, dilation=3),nn.Conv3d(16, 8, 3, padding=3, dilation=3), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 16, in_planes, 3, padding=1, dilation=1), nn.ReLU(True))

    def forward(self, x):
        #print (x.shape)
        x1_raw = self.conv1(x)
        #print (x1_raw.shape)
        x1 = torch.cat([x, x1_raw], 1)
        #print (x1.shape)
        x2_raw = self.conv2(x1)
        #print (x2_raw.shape)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        #print (x2.shape)
        x3_raw = self.conv3(x2)
        #print (x3_raw.shape)
        x3 = torch.cat([x, x2_raw, x3_raw], 1)
        #print (x3.shape)
        output = self.conv4(x3)
        #print (output.shape)
        return output
class Atten_UN_Dense_Encoder_Decoder(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(Atten_UN_Dense_Encoder_Decoder, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        self.dense1 = DDCB(16)


        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.dense2 = DDCB(32)

        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.dense3 = DDCB(64)

        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.dense4 = DDCB(128)

        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.dense5 = DDCB(256)

        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        self.dense6 = DDCB(128)


        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        self.dense7 = DDCB(64)


        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        self.dense8 = DDCB(32)


        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        self.dense9 = DDCB(16)


        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.dense1(x1)


        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        x2 = self.dense2(x2)



        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        x3 = self.dense3(x3)



        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        x4 = self.dense4(x4)




        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        x5 = self.dense5(x5)


        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        d5 = self.dense6(d5)
        de5 = (x4+d5)


        d4 = self.up4(de5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        d4 = self.dense7(d4)
        de4 = (d4+x3)

        d3 = self.up3(de4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        d3 = self.dense8(d3)
        de3 = (d3+x2)


        d2 = self.up2(de3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        d2 = self.dense9(d2)
        de2 = (d2+x1)


        out = self.conv_1x1(de2)
        #out = torch.sigmoid(out)
        return out

#########################################################################################    UNet_Double_Atten          ##################################################################

import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x

class Multi_Scale(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #self. att1 = Attention_Block_Modified(exp_ch,exp_ch,4)
        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)
        #print("x3",x3.shape)
        #x4 = self.att1(x3)
        #print("x444",x4.shape)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        #xc3 = self.relu(torch.cat((x2, x4), dim=1))
        return xc3

class Attention_Block(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))
        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x * psi

class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi
# class DDCB(nn.Module):
#     def __init__(self, in_planes):
#         super(DDCB, self).__init__()
#         self.seen = 0
#         self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=1),nn.Conv3d(16, 8, 3, padding=1), nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 8, 32, 1), nn.ReLU(True),nn.Conv3d(32, 16, 3, padding=2, dilation=2),nn.Conv3d(16, 8, 3, padding=2, dilation=2), nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 16, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=3, dilation=3),nn.Conv3d(16, 8, 3, padding=3, dilation=3), nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 16, in_planes, 3, padding=1, dilation=1), nn.ReLU(True))
#
#     def forward(self, x):
#         #print (x.shape)
#         x1_raw = self.conv1(x)
#         #print (x1_raw.shape)
#         x1 = torch.cat([x, x1_raw], 1)
#         #print (x1.shape)
#         x2_raw = self.conv2(x1)
#         #print (x2_raw.shape)
#         x2 = torch.cat([x, x1_raw, x2_raw], 1)
#         #print (x2.shape)
#         x3_raw = self.conv3(x2)
#         #print (x3_raw.shape)
#         x3 = torch.cat([x, x2_raw, x3_raw], 1)
#         #print (x3.shape)
#         output = self.conv4(x3)
#         #print (output.shape)
#         return output
class UNet_Double_Atten(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(UNet_Double_Atten, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        #self.dense1 = DDCB(16)
        self.Att_Modified1 = Attention_Block_Modified(F_g=16, F_l=16, F_int=8)
        #self.fire1 = Multi_Scale(in_ch=16,sq_ch=4,exp_ch=8)
        #self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.Att_Modified2 = Attention_Block_Modified(F_g=32, F_l=32, F_int=16)
        #self.dense2 = DDCB(32)
        #self.fire2 = Multi_Scale(in_ch=32,sq_ch=8,exp_ch=16)
        #self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.Att_Modified3 = Attention_Block_Modified(F_g=64, F_l=64, F_int=32)
        #self.dense3 = DDCB(64)
        #self.fire3 = Multi_Scale(in_ch=64,sq_ch=16,exp_ch=32)
        #self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.Att_Modified4 = Attention_Block_Modified(F_g=128, F_l=128, F_int=64)
        #self.dense4 = DDCB(128)
        #self.fire4 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        #self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.Att_Modified5 = Attention_Block_Modified(F_g=256, F_l=256, F_int=128)
        #self.dense5 = DDCB(256)
        #self.fire5 = Multi_Scale(in_ch=256,sq_ch=64,exp_ch=128)
        #self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        #self.upfire6 = Multi_Scale(in_ch=128,sq_ch=32,exp_ch=64)
        #self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        #self.upfire7 = Multi_Scale(in_ch=64,sq_ch=16, exp_ch=32)
        #self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=8, exp_ch=16)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=4, exp_ch=8)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        #x1 = self.dense1(x1)
        x1 = self.Att_Modified1(x1,x1)
        #x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        #x2 = self.dense2(x2)
        x2 = self.Att_Modified2(x2,x2)
        #x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        #x3 = self.dense3(x3)
        x3 = self.Att_Modified3(x3,x3)
        #x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        #x4 = self.dense4(x4)
        x4 = self.Att_Modified4(x4,x4)
        #x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        #x5 = self.dense5(x5)
        x5 = self.Att_Modified5(x5,x5)
        #x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        #d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        #d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        #d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        #d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        #out = torch.sigmoid(out)
        return out


################################################################# UNet_Att_Modified_Fire_Residual#############
######################################################################################################################


import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x

class Multi_Scale_Fire(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale_Fire, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        return xc3





class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        #print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        #print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi

#######################################################################################
class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm3d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip
#######################################################################################
class UNet_Att_Modified_Fire_Residual(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(UNet_Att_Modified_Fire_Residual, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        self.resid1 = residual_block(16, 16)
        self.fire1 = Multi_Scale_Fire(in_ch=16,sq_ch=4,exp_ch=8)
        #self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.resid2 = residual_block(32,32)
        self.fire2 = Multi_Scale_Fire(in_ch=32,sq_ch=8,exp_ch=16)
        #self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.resid3 = residual_block(64, 64)
        self.fire3 = Multi_Scale_Fire(in_ch=64,sq_ch=16,exp_ch=32)
        #self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.resid4 = residual_block(128, 128)
        self.fire4 = Multi_Scale_Fire(in_ch=128,sq_ch=32,exp_ch=64)
        #self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.resid5 = residual_block(256, 256)
        self.fire5 = Multi_Scale_Fire(in_ch=256,sq_ch=64,exp_ch=128)
        #self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        self.resid6 = residual_block(128, 128)
        self.upfire6 = Multi_Scale_Fire(in_ch=128,sq_ch=32,exp_ch=64)
        #self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        self.resid7 = residual_block(64, 64)
        self.upfire7 = Multi_Scale_Fire(in_ch=64,sq_ch=16, exp_ch=32)
        #self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        self.resid8 = residual_block(32, 32)
        self.upfire8 = Multi_Scale_Fire(in_ch=32, sq_ch=8, exp_ch=16)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        self.resid9 = residual_block(16, 16)
        self.upfire9 = Multi_Scale_Fire(in_ch=16, sq_ch=4, exp_ch=8)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.resid1(x1)
        x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        x2 = self.resid2(x2)
        x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        x3 = self.resid3(x3)
        x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        x4 = self.resid4(x4)
        x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        x5 = self.resid5(x5)
        x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        d5 = self.resid6(d5)
        d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        d4 = self.resid7(d4)
        d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        d3 = self.resid8(d3)
        d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        d2 = self.resid9(d2)
        d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        out = torch.sigmoid(out)
        return out

    
    
################################################

################################################################# UNet_Att_Modified_Fire_Dense           #############
######################################################################################################################


import torch.nn as nn
import torch
class Double_conv_block(nn.Module):
    def __init__(self,in_ch,out_ch,n_groups=4):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(
                    nn.GroupNorm(n_groups, in_ch),
                    nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                    #nn.BatchNorm3d(out_ch),
                    nn.GroupNorm(n_groups, out_ch),
                    nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x

class upconv(nn.Module):
    def __init__(self,in_ch, out_ch,n_groups=4):
        super(upconv, self).__init__()
        self.up = nn.Sequential(
                      nn.Upsample(scale_factor=2),
                      nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                      #nn.BatchNorm3d(out_ch),
                      nn.GroupNorm(n_groups, out_ch),
                      nn.ReLU(inplace=True))

    def forward(self,x):
        x= self.up(x)
        return x

class Multi_Scale_Fire(nn.Module):
    def __init__(self, in_ch, sq_ch, exp_ch):
        super(Multi_Scale_Fire, self).__init__()
        self.squeeze_layer = nn.Conv3d(in_ch, sq_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer1_1x1 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.expnad_layer2_3x3 = nn.Conv3d(sq_ch, exp_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        #self.expnad_layer3_1x1 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer4_3x3 = nn.Conv3d(exp_ch1, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        #self.expnad_layer5_1x1 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #self.expnad_layer6_3x3 = nn.Conv3d(exp_ch2, exp_ch3, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.squeeze_layer(x)
        x2 = self.expnad_layer1_1x1(x1)
        x3 = self.expnad_layer2_3x3(x1)

        #x4 = self.expnad_layer3_1x1(x2)
        #x5 = self.expnad_layer4_3x3(x2)

        #x6 = self.expnad_layer5_1x1(x3)
        #print("x6",x6.shape)
        #x7 = self.expnad_layer6_3x3(x3)
        #print("x7", x7.shape)

        #xc1 = self.relu(torch.cat((x4, x5), dim=1))
        #xc2 = self.relu(torch.cat((x6, x7), dim=1))
        xc3 = self.relu(torch.cat((x2, x3), dim=1))
        return xc3





class Attention_Block_Modified(nn.Module):
    def __init__(self,F_g, F_l, F_int,n_groups=4):
        super(Attention_Block_Modified, self).__init__()
        self.W_g_01 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=1,padding=1),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_01 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=2, padding=2),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
                   nn.Conv3d(F_g,F_int,kernel_size=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=5,padding=5),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups,F_int))
        self.W_x_02 = nn.Sequential(
                   nn.Conv3d(F_l,F_int,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   nn.Conv3d(F_int,F_int,kernel_size=(3,3,3),dilation=7, padding=7),
                   #nn.BatchNorm3d(F_int))
                   nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
                   nn.Conv3d(F_int,1,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=True),
                   #nn.GroupNorm(n_groups, 1),
                   nn.BatchNorm3d(1),
                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1+x1)
        #print("c1 shape",c1.shape)

        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2+x2)
        #print("c2 shape",c2.shape)

        c_comb = self.relu(c1 +c2)
        psi_01 = self.psi(c_comb)
        return x + psi_01
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # return x * psi

#######################################################################################
# class batchnorm_relu(nn.Module):
#     def __init__(self, in_c):
#         super().__init__()
#
#         self.bn = nn.BatchNorm3d(in_c)
#         self.relu = nn.ReLU()
#
#     def forward(self, inputs):
#         x = self.bn(inputs)
#         x = self.relu(x)
#         return x
#
# class residual_block(nn.Module):
#     def __init__(self, in_c, out_c, stride=1):
#         super().__init__()
#
#         """ Convolutional layer """
#         self.b1 = batchnorm_relu(in_c)
#         self.c1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
#         self.b2 = batchnorm_relu(out_c)
#         self.c2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, stride=1)
#
#         """ Shortcut Connection (Identity Mapping) """
#         self.s = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, stride=stride)
#
#     def forward(self, inputs):
#         x = self.b1(inputs)
#         x = self.c1(x)
#         x = self.b2(x)
#         x = self.c2(x)
#         s = self.s(inputs)
#
#         skip = x + s
#         return skip
#######################################################################################
class DDCB(nn.Module):
    def __init__(self, in_planes):
        super(DDCB, self).__init__()
        self.seen = 0
        self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=1),nn.Conv3d(16, 8, 3, padding=1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 8, 32, 1), nn.ReLU(True),nn.Conv3d(32, 16, 3, padding=2, dilation=2),nn.Conv3d(16, 8, 3, padding=2, dilation=2), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 16, 32, 1), nn.ReLU(True), nn.Conv3d(32, 16, 3, padding=3, dilation=3),nn.Conv3d(16, 8, 3, padding=3, dilation=3), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 16, in_planes, 3, padding=1, dilation=1), nn.ReLU(True))

        # self.conv1 = nn.Sequential(nn.Conv3d(in_planes, 16, 1), nn.ReLU(True), nn.Conv3d(16, 8, 3, padding=1))
        # self.conv2 = nn.Sequential(nn.Conv3d(in_planes + 8, 16, 1), nn.ReLU(True),nn.Conv3d(16, 8, 3, padding=2, dilation=2))
        # self.conv3 = nn.Sequential(nn.Conv3d(in_planes + 16, 32, 1), nn.ReLU(True), nn.Conv3d(32, 8, 3, padding=3, dilation=3))
        # self.conv4 = nn.Sequential(nn.Conv3d(in_planes + 24, in_planes, 3, padding=5, dilation=5), nn.ReLU(True))

    def forward(self, x):
        #print (x.shape)
        x1_raw = self.conv1(x)
        #print (x1_raw.shape)
        x1 = torch.cat([x, x1_raw], 1)
        #print (x1.shape)
        x2_raw = self.conv2(x1)
        #print (x2_raw.shape)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        #print (x2.shape)
        x3_raw = self.conv3(x2)
        #print (x3_raw.shape)
        x3 = torch.cat([x, x2_raw, x3_raw], 1)
        #print (x3.shape)
        output = self.conv4(x3)
        #print (output.shape)
        return output


#######################################################################################
class UNet_Att_Modified_Fire_Dense(nn.Module):
    def __init__(self, in_ch=4, output_final_ch=3):
        super(UNet_Att_Modified_Fire_Dense, self).__init__()
        #self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))


        self.conv1 = Double_conv_block(in_ch,out_ch=16)
        self.dense1 = DDCB(16)
        self.fire1 = Multi_Scale_Fire(in_ch=16,sq_ch=4,exp_ch=8)
        #self.fire1 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.ds1 = nn.Conv3d(16,16,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv2 = Double_conv_block(in_ch=16,out_ch=32)
        self.dense2 = DDCB(32)
        self.fire2 = Multi_Scale_Fire(in_ch=32,sq_ch=8,exp_ch=16)
        #self.fire2 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)
        self.ds2 = nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv3 = Double_conv_block(in_ch=32,out_ch=64)
        self.dense3 = DDCB(64)
        self.fire3 = Multi_Scale_Fire(in_ch=64,sq_ch=16,exp_ch=32)
        #self.fire3 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)
        self.ds3 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv4 = Double_conv_block(in_ch=64,out_ch=128)
        self.dense4 = DDCB(128)
        self.fire4 = Multi_Scale_Fire(in_ch=128,sq_ch=32,exp_ch=64)
        #self.fire4 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)
        self.ds4 = nn.Conv3d(128,128,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.conv5 = Double_conv_block(in_ch=128,out_ch=256)
        self.dense5 = DDCB(256)
        self.fire5 = Multi_Scale_Fire(in_ch=256,sq_ch=64,exp_ch=128)
        #self.fire5 = Multi_Scale(in_ch=256, sq_ch=128, exp_ch1=128, exp_ch2=128, exp_ch3=64)
        self.dropout = nn.Dropout(p=0.2)

        self.up5 = upconv(in_ch=256,out_ch=128)
        self.Att5 = Attention_Block_Modified(F_g=128,F_l=128,F_int=64)
        self.upconv5 = Double_conv_block(in_ch=256,out_ch=128)
        self.dense6 = DDCB(128)
        self.upfire6 = Multi_Scale_Fire(in_ch=128,sq_ch=32,exp_ch=64)
        #self.upfire6 = Multi_Scale(in_ch=128, sq_ch=64, exp_ch1=64, exp_ch2=64, exp_ch3=32)

        self.up4 = upconv(in_ch=128,out_ch=64)
        self.Att4 = Attention_Block_Modified(F_g=64, F_l=64,F_int=32)
        self.upconv4 = Double_conv_block(in_ch=128,out_ch=64)
        self.dense7 = DDCB(64)
        self.upfire7 = Multi_Scale_Fire(in_ch=64,sq_ch=16, exp_ch=32)
        #self.upfire7 = Multi_Scale(in_ch=64, sq_ch=32, exp_ch1=32, exp_ch2=32, exp_ch3=16)

        self.up3 = upconv(in_ch=64,out_ch=32)
        self.Att3 = Attention_Block_Modified(F_g=32,F_l=32,F_int=16)
        self.upconv3 = Double_conv_block(in_ch=64,out_ch=32)
        self.dense8 = DDCB(32)
        self.upfire8 = Multi_Scale_Fire(in_ch=32, sq_ch=8, exp_ch=16)
        #self.upfire8 = Multi_Scale(in_ch=32, sq_ch=16, exp_ch1=16, exp_ch2=16, exp_ch3=8)

        self.up2 = upconv(in_ch=32,out_ch=16)
        self.Att2 = Attention_Block_Modified(F_g=16,F_l=16,F_int=8)
        self.upconv2 = Double_conv_block(in_ch=32,out_ch=16)
        self.dense9 = DDCB(16)
        self.upfire9 = Multi_Scale_Fire(in_ch=16, sq_ch=4, exp_ch=8)
        #self.upfire9 = Multi_Scale(in_ch=16, sq_ch=8, exp_ch1=8, exp_ch2=8, exp_ch3=4)

        self.conv_1x1 = nn.Conv3d(16,output_final_ch,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))


    def forward(self,x):
        x1 = self.conv1(x)
        #print(x1.shape)
        x1 = self.dense1(x1)
        #print(x1.shape)
        x1 = self.fire1(x1)
        #x1 = self.dropout(x1)
        #x2 = self.Maxpool(x1)

        x2 = self.ds1(x1)
        x2 = self.conv2(x2)
        x2 = self.dense2(x2)
        x2 = self.fire2(x2)
        #x2 = self.dropout(x2)

        #x3 = self.Maxpool(x2)
        x3 = self.ds2(x2)
        x3 = self.conv3(x3)
        x3 = self.dense3(x3)
        x3 = self.fire3(x3)
        #x3 = self.dropout(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.ds3(x3)
        x4 = self.conv4(x4)
        x4 = self.dense4(x4)
        x4 = self.fire4(x4)
        #x4 = self.dropout(x4)


        #x5 = self.Maxpool(x4)
        x5 = self.ds4(x4)
        x5 = self.conv5(x5)
        x5 = self.dense5(x5)
        x5 = self.fire5(x5)

        x5 = self.dropout(x5)
        d5 = self.up5(x5)
        at1  = self.Att5(d5,x4)
        d5 = torch.cat((at1,x4),dim=1)
        d5 = self.upconv5(d5)
        d5 = self.dense6(d5)
        d5 = self.upfire6(d5)

        d4 = self.up4(d5)
        at2 = self.Att4(d4,x3)
        d4 = torch.cat((at2,x3),dim=1)
        d4 = self.upconv4(d4)
        d4 = self.dense7(d4)
        d4 = self.upfire7(d4)

        d3 = self.up3(d4)
        at3 = self.Att3(d3,x2)
        d3 = torch.cat((at3,x2),dim=1)
        d3 = self.upconv3(d3)
        d3 = self.dense8(d3)
        d3 = self.upfire8(d3)

        d2 = self.up2(d3)
        at2 = self.Att2(d2,x1)
        d2 = torch.cat((at2,x1),dim=1)
        d2 = self.upconv2(d2)
        d2 = self.dense9(d2)
        d2 = self.upfire9(d2)

        out = self.conv_1x1(d2)
        out = torch.sigmoid(out)
        return out

# if __name__ == "__main__":
#     #input  = torch.randn(1,4,160,192,128)
#     input = torch.randn(1, 4, 128, 128, 128)
#     #input = torch.randn(1, 4, 164, 164, 164)
#     #input2 = torch.randn((1,8,100,100,60))
#     model = UNet_Att_Modified_Fire_Residual()
#     output = model(input)
#     print(output.shape)


###############################################
    
# if __name__ == "__main__":
#     #input  = torch.randn(1,4,160,192,128)
#     input = torch.randn(1, 4, 128, 128, 128)
#     #input = torch.randn(1, 4, 164, 164, 164)
#     #input2 = torch.randn((1,8,100,100,60))
#     model = UNet_Att_Modified_Fire_Residual()
#     output = model(input)
#     print(output.shape)

















# if __name__ == "__main__":
#     #input  = torch.randn(1,4,160,192,128)
#     input = torch.randn(1, 4, 128, 128, 128)
#     #input = torch.randn(1, 4, 240, 240, 155)
#     #input = torch.randn(1, 4, 164, 164, 164)
#     #input2 = torch.randn((1,8,100,100,60))
#     model = Atten_UN_Dense_Encoder()
#     output = model(input)
#     print(output.shape)
