import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
from tqdm import tqdm

from ctbasic import *
from models import *
from metrics import *


miuWater = 0.19


def tif2arr(tif):
    return tif/1000 * miuWater + miuWater


# arr/miuWater * 1000 - 1000
def arr2tif(arr):
    return (np.maximum(arr/miuWater * 1000, 0)).astype(np.uint16)


def myshowpics(pics, figsize=(64, 36)):
    """
      pics = [(pic1, title1), (pic2, title2), ...]
      """
    n = len(pics)
    if n == 1:
        pic, tit = pics[0]
        plt.figure(figsize=figsize)
        plt.imshow(pic, cmap=plt.cm.Greys_r)
        plt.title(tit)
        plt.show()
        return
    ax = (None for _ in range(n))
    fig, ax = plt.subplots(n, 1, figsize=figsize, sharex=False, sharey=False)
    for i in range(n):
        pic, tit = pics[i]
        ax[i].set_title(tit)
        ax[i].imshow(pic, cmap=plt.cm.Greys_r)

    plt.show()


if __name__ == '__main__':
    """from dataset_v1 import *
    net_unet_3ch = BasicUNet(3, 1)
    train_3ch(net_unet_3ch)
    testout_v1_base_unet_3ch = test_3ch(net_unet_3ch, imRaw_3ch.transpose(3,2,1,0), np.reshape(imRaw, (512,512,1,1)).transpose(3,2,1,0))

    myshowpics([(np.transpose(arr2tif(testout_v1_base_unet_3ch[0, 0, :, :].cpu().numpy()), (1, 0)), "basic UNet small training v1 with 3 channels")])
    """
    testv1 = loadmat('C:/Users/ScottShen/bins/UniMelb/phd/project2022/mywork/sample_2.mat')
    metalBW = testv1['imData'][0][0][0]
    imRaw = testv1['imData'][0][0][1]
    imBHC = testv1['imData'][0][0][2]
    imLI = testv1['imData'][0][0][3]
    projRef = testv1['imData'][0][0][5]
    proj = testv1['imData'][0][0][6]

    myM, myN = imRaw.shape
    imRaw_3ch = np.zeros([myM, myN, 3])
    imRaw_3ch[:, :, 0] = imRaw
    imRaw_3ch[:, :, 1] = imBHC
    imRaw_3ch[:, :, 2] = imLI
    imRaw_3ch = imRaw_3ch[:, :, :, np.newaxis]

    from dataset_v2_deeplesion import *
    net_unet_v2 = BasicUNet(1, 1)
    train_v2(net_unet_v2)
    test_v2_unet = test_v2(net_unet_v2, imRaw, miuWater=miuWater)
    myshowpics([(np.transpose(arr2tif(test_v2_unet[0, 0, :, :].cpu().numpy()), (1, 0)),
                 "basic UNet small training v2")])

    calc_metrics(test_v2_unet, imRaw)
    Image.fromarray(np.transpose(arr2tif(test_v2_unet[0, 0, :, :].cpu().numpy()), (1, 0))).save(
        "testv2.tif")
