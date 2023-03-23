import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sinusoidal_embedding
from blocks import *


class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, cond_ch=1, MyBlock=MyBlock1):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 256, 256), 1, 32),
            MyBlock((32, 256, 256), 32, 32),
            MyBlock((32, 256, 256), 32, 32)
        )
        self.down1 = nn.Conv2d(32, 32, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 32)
        self.b2 = nn.Sequential(
            MyBlock((32, 128, 128), 32, 64),
            MyBlock((64, 128, 128), 64, 64),
            MyBlock((64, 128, 128), 64, 64)
        )
        self.down2 = nn.Conv2d(64, 64, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 64)
        self.b3 = nn.Sequential(
            MyBlock((64, 64, 64), 64, 128),
            MyBlock((128, 64, 64), 128, 128),
            MyBlock((128, 64, 64), 128, 128)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1)
        )

        # conditional
        self.te1c = self._make_te(time_emb_dim, cond_ch)
        self.b1c = nn.Sequential(
            MyBlock((1, 256, 256), cond_ch, 32),
            MyBlock((32, 256, 256), 32, 32),
            MyBlock((32, 256, 256), 32, 32)
        )
        self.down1c = nn.Conv2d(32, 32, 4, 2, 1)

        self.te2c = self._make_te(time_emb_dim, 32)
        self.b2c = nn.Sequential(
            MyBlock((32, 128, 128), 32, 64),
            MyBlock((64, 128, 128), 64, 64),
            MyBlock((64, 128, 128), 64, 64)
        )
        self.down2c = nn.Conv2d(64, 64, 4, 2, 1)

        self.te3c = self._make_te(time_emb_dim, 64)
        self.b3c = nn.Sequential(
            MyBlock((64, 64, 64), 64, 128),
            MyBlock((128, 64, 64), 128, 128),
            MyBlock((128, 64, 64), 128, 128)
        )
        self.down3c = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 128)
        self.b_mid = nn.Sequential(
            MyBlock((128, 32, 32), 128, 64),
            MyBlock((64, 32, 32), 64, 64),
            MyBlock((64, 32, 32), 64, 128)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 256)
        self.b4 = nn.Sequential(
            MyBlock((256, 64, 64), 256, 128),
            MyBlock((128, 64, 64), 128, 64),
            MyBlock((64, 64, 64), 64, 64)
        )

        self.up2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 128)
        self.b5 = nn.Sequential(
            MyBlock((128, 128, 128), 128, 64),
            MyBlock((64, 128, 128), 64, 32),
            MyBlock((32, 128, 128), 32, 32)
        )

        self.up3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 64)
        self.b_out = nn.Sequential(
            MyBlock((64, 256, 256), 64, 32),
            MyBlock((32, 256, 256), 32, 32),
            MyBlock((32, 256, 256), 32, 32, normalize=False)
        )

        self.conv_out = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x, t, condx):
        # (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 32, 256, 256)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 64, 128, 128)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 128, 64, 64)

        out1c = self.b1c(condx + self.te1c(t).reshape(n, -1, 1, 1))  # (N, 32, 256, 256)
        out2c = self.b2c(self.down1c(out1c) + self.te2c(t).reshape(n, -1, 1, 1))  # (N, 64, 128, 128)
        out3c = self.b3c(self.down2c(out2c) + self.te3c(t).reshape(n, -1, 1, 1))  # (N, 128, 64, 64)

        out_mid = self.b_mid(self.down3(out3) + self.down3c(out3c) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


