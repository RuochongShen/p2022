import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
from tqdm import tqdm

from medical.ctbasic import *
from models import *
from metrics import *


def save_chosen_set(n, op_upperbd, file_path):
    """
    Using the pandas.to_csv()/numpy.tofile()/numpy.savetxt()/file-handling/writerows() function to write array to CSV

    arr = np.asarray([ [7,8,9], [5,8,9]])
    rows = ["{},{},{}".format(i, j, k) for i, j, k in arr]
    text = "\n".join(rows)
    with open('sample.csv', 'w') as f:
        f.write(text)

    csv.writer
    """
    chosen_lst = np.random.randint(op_upperbd, size=n)
    np.savetxt(file_path, chosen_lst, fmt='%d', delimiter=' ')


def load_chosen_set(file_path):
    """
    readrow()
    """
    chosen_lst = np.loadtxt(file_path, dtype=int)
    return chosen_lst


def normalization_medimage(image, miuWater=0.19, norm_para1=32768, opt=None):
    if opt is None:
        image = (image - norm_para1) / 1000 * miuWater + miuWater
    else:
        pass
    return image


class testData_v2(data.Dataset):
    def __init__(self, path, istest=False, n_chosen=1, cond_ch=1, miuWater=0.19, to_be_pred=None, loadfile='textset.txt', totaln=None, sinogram=False):
        self.folder = path
        self.istest = istest
        self.n_chosen = n_chosen
        self.cond_ch = cond_ch
        self.miuWater = miuWater
        self.to_be_pred = to_be_pred
        self.datalst = []
        self.chosen_arr = []
        self.loadfile = loadfile
        if loadfile:
            self.chosen_arr = load_chosen_set(loadfile)
        self.totaln = totaln
        self.sinogram = sinogram

        """
        save pattern:
        '/metaldata/datasets/adn_data/deep_lesion/train(test)/(000001_01_01)/(103)/imData(1-90)'
        """

        counter, itest = 0, 0
        for scan_folder in os.listdir(self.folder):
            for scan_no in os.listdir(os.path.join(self.folder, scan_folder)):
                img_folder = os.path.join(self.folder, scan_folder, scan_no)
                upperb = 10 if istest else 90
                if self.istest:
                    chosen_arr = int(self.chosen_arr[itest])
                    itest += 1
                else:
                    chosen_arr = np.random.randint(upperb, size=n_chosen) if n_chosen < upperb else range(upperb)
                # self.chosen_arr.append(chosen_arr)
                tmp_pairs = [os.path.join(img_folder, 'gt.mat')]

                if self.istest:
                    tmp_pairs.append(os.path.join(img_folder, str(chosen_arr+1) + '.mat'))
                    tmp_pairs.append(os.path.join(img_folder, 'imData', str(chosen_arr+1)+'.mat'))
                else:
                    for n in chosen_arr:
                        tmp_pairs.append(os.path.join(img_folder, str(n+1)+'.mat'))
                        tmp_pairs.append(os.path.join(img_folder, 'imData', str(n+1) + '.mat'))

                self.datalst.append(tmp_pairs)
                counter += 1
                if totaln and counter == totaln:
                    return

        # print(len(self.datalst), len(self.datalst[0]), self.datalst[0][0], self.datalst[0][1])

    def __len__(self):
        if self.istest and self.to_be_pred:
            return 1
        return self.n_chosen * len(self.datalst)

    def __getitem__(self, idx):
        if self.istest and self.to_be_pred:
            image = self.to_be_pred.transpose()
            return torch.Tensor(image[np.newaxis, :, :]), torch.Tensor(image[np.newaxis, :, :])
        elif self.istest and not self.sinogram:
            miuWater = self.miuWater
            image = (loadmat(self.datalst[idx][1])['image']).astype(float)
            imData = loadmat(self.datalst[idx][2])
            label = normalization_medimage((loadmat(self.datalst[idx][0])['image']).astype(float))
            label = label.transpose()

            # differ with condition channel
            if self.cond_ch == 1:
                return torch.Tensor(image[np.newaxis, :, :]), torch.Tensor(label[np.newaxis, :, :])
            else:
                condx = np.append(image[np.newaxis, :, :], imData['imLI'].astype(float)[np.newaxis, :, :], axis=0)
                if self.cond_ch == 2:
                    return torch.Tensor(condx), torch.Tensor(label[np.newaxis, :, :])
                else:
                    condx = np.append(condx, imData['imBHC'].astype(float)[np.newaxis, :, :], axis=0)
                    if self.cond_ch == 3:
                        pass
                    return torch.Tensor(condx), torch.Tensor(label[np.newaxis, :, :])
        elif not self.sinogram:
            i = idx // self.n_chosen
            d = (idx - self.n_chosen * i) * 2 + 1   # 0 for gt; raw/additional (LI, BHC, ...) start from 1/2 ...
            miuWater = self.miuWater
            image = (loadmat(self.datalst[i][d])['image']).astype(float)
            imData = loadmat(self.datalst[i][d+1])
            label = normalization_medimage((loadmat(self.datalst[i][0])['image']).astype(float))
            label = label.transpose()
            # label = label - np.min(label)

            # differ with condition channel
            if self.cond_ch == 1:
                return torch.Tensor(image[np.newaxis, :, :]), torch.Tensor(label[np.newaxis, :, :])
            else:
                condx = np.append(image[np.newaxis, :, :], imData['imLI'].astype(float)[np.newaxis, :, :], axis=0)
                if self.cond_ch == 2:
                    return torch.Tensor(condx), torch.Tensor(label[np.newaxis, :, :])
                else:
                    condx = np.append(condx, imData['imBHC'].astype(float)[np.newaxis, :, :], axis=0)
                    if self.cond_ch == 3:
                        pass
                    return torch.Tensor(condx), torch.Tensor(label[np.newaxis, :, :])
        else:
            sinogram = (loadmat(self.datalst[idx][2])['proj']).astype(float)
            label = normalization_medimage((loadmat(self.datalst[idx][0])['image']).astype(float))
            label = label.transpose()
            #label_proj = astra_create_projector('line_fanflat', proj_geom, vol_geom);
            return torch.Tensor(sinogram[np.newaxis, :, :]), torch.Tensor(label_proj[np.newaxis, :, :])



