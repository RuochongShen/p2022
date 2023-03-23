import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, Transpose=False):
        super().__init__()
        if Transpose:
            self.up_block = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            self.up_block = nn.Sequential(
                nn.Upsample(2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up_block(x)


class UpBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, Transpose=False):
        super().__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        else:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = DoubleConv(in_ch, out_ch)
        self.up.apply(self.init_weights)

    def forward(self, x1, x2):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


class BasicUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.downblock1 = DownBlock(64, 128)
        self.downblock2 = DownBlock(128, 256)
        self.downblock3 = DownBlock(256, 512)
        self.downblock4 = DownBlock(512, 1024)

        self.upblock4 = UpBlock2(1024, 512)
        self.upblock3 = UpBlock2(512, 256)
        self.upblock2 = UpBlock2(256, 128)
        self.upblock1 = UpBlock2(128, 64)

        self.outc = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.downblock1(enc1)
        enc3 = self.downblock2(enc2)
        enc4 = self.downblock3(enc3)
        enc5 = self.downblock4(enc4)
        dec4 = self.upblock4(enc5, enc4)
        dec3 = self.upblock3(dec4, enc3)
        dec2 = self.upblock2(dec3, enc2)
        dec1 = self.upblock1(dec2, enc1)
        out = self.outc(dec1)

        return out


class BasicUNet1(nn.Module):
    def __init__(self, in_ch, out_ch, depth=5):
        super().__init__()
        self.depth = depth
        ch_lst = [64 * (2 ** i) for i in range(depth)]
        self.conv1 = DoubleConv(in_ch, ch_lst[0])
        self.downblocks = [DownBlock(ch_lst[i], ch_lst[i + 1]) for i in range(depth - 1)]
        self.upblocks = [UpBlock(ch_lst[i + 1], ch_lst[i]) for i in range(depth - 1)]
        self.upconvs = [DoubleConv(ch_lst[i + 1], ch_lst[i]) for i in range(depth - 1)]
        self.output = nn.Conv2d(ch_lst[0], out_ch, 1)

    def forward(self, x):
        enc_os = [self.conv1(x)] + [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            enc_os[i + 1] = self.downblocks[i](enc_os[i])

        dec_os = [enc_os[3]] + [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            h = self.upblocks[self.depth - 1 - i](dec_os[i])
            diffY = enc_os[i].size()[2] - h.size()[2]
            diffX = enc_os[i].size()[3] - h.size()[3]
            h = F.pad(h, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

            h = torch.cat((enc_os[i], h), dim=1)
            dec_os[i] = self.upconvs[self.depth - 1 - i](h)

        return self.output(dec_os[-1])


class BasicUNet2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(in_ch, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.Up5 = UpBlock(1024, 512)
        self.Up_conv5 = DoubleConv(1024, 512)

        self.Up4 = UpBlock(512, 256)
        self.Up_conv4 = DoubleConv(512, 256)

        self.Up3 = UpBlock(256, 128)
        self.Up_conv3 = DoubleConv(256, 128)

        self.Up2 = UpBlock(128, 64)
        self.Up_conv2 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, out_ch, 1)

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

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


def init_weights_v1_f(m):
    if type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight.data, 0, 0.01)
        init.constant(m.bias, 0)
    else:
        init.normal_(m.weight.data, 0, 0.01)


def init_weights_v1(net):
    net.apply(init_weights_v1_f)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class discriminatorBlock(nn.Module):
    def __init__(self, in_filters, out_filters, bn=True):
        super().__init__()
        self.bn = bn
        self.model = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bnlayer = nn.BatchNorm2d(out_filters, 0.8)

    def forward(self, x):
        x = self.model(x)
        if self.bn:
            x = self.bnlayer(x)

        return x


class BasicDiscriminator(nn.Module):
    def __init__(self, pic_size):
        super().__init__()
        self.model = nn.Sequential(
            discriminatorBlock(1, 16, False),
            discriminatorBlock(16, 32),
            discriminatorBlock(32, 64),
            discriminatorBlock(64, 128),
        )
        self.adv_layer = nn.Sequential(nn.Linear(128 * pic_size//(2**8), 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)  # batch_size * (ch*pic_size)
        validity = self.adv_layer(x)

        return validity


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc, pic_size, ndf=64, n_layers=3):
        """
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 64, kernel_size=kw, stride=1, padding=padw)]
        sequence += [nn.Linear(64 * pic_size, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=64, img_size=64):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=64, num_heads=8, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=64,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=64, n_classes=1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class GenerativeHead(nn.Module):
    def __init__(self, patch_size=16, emb_size=64):
        super().__init__()
        self.projL = nn.Sequential(
            Rearrange('b s1 s2 -> b (s1 s2)', s1=patch_size, s2=emb_size),
            nn.Linear(patch_size * emb_size, emb_size * emb_size),
            Rearrange('b (c s1 s2) -> b c s1 s2', c=1, s1=emb_size, s2=emb_size)
        )

    def forward(self, x):
        x = x[:, 1:, :]
        x = self.projL(x)
        return x


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_size=64,
                 img_size=64,
                 depth=6,
                 n_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            # ClassificationHead(emb_size, n_classes)
            GenerativeHead(patch_size, emb_size)
        )
