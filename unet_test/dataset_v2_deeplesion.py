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

from ctbasic import *
from models import *
from metrics import *


class testData_v2(data.Dataset):
    def __init__(self, istest=False, n_chosen=3, miuWater=0.19, to_be_pred=None):
        self.folder = 'C:\\Users\\ScottShen\\PycharmProjects\\adn\\data\\deep_lesion\\train\\'
        self.istest = istest
        self.n_chosen = n_chosen
        self.miuWater = miuWater
        self.to_be_pred = to_be_pred
        self.datalst = []

        for scan_folder in os.listdir(self.folder):
            for scan_no in os.listdir(os.path.join(self.folder, scan_folder)):
                img_folder = os.path.join(self.folder, scan_folder, scan_no)
                chosen_arr = np.random.randint(90, size=n_chosen) if n_chosen < 90 else range(90)
                tmp_pairs = [os.path.join(img_folder, 'gt.mat')]
                for n in chosen_arr:
                    tmp_pairs.append(os.path.join(img_folder, str(n+1)+'.mat'))
                self.datalst.append(tmp_pairs)

    def __len__(self):
        if self.istest:
            return 1
        return self.n_chosen * len(self.datalst)

    def __getitem__(self, idx):
        if self.istest:
            image = self.to_be_pred.transpose()
            return torch.Tensor(image[np.newaxis, :, :]), torch.Tensor(image[np.newaxis, :, :])
        else:
            i = idx // 3
            d = idx - 3 * i + 1
            miuWater = self.miuWater
            image = loadmat(self.datalst[i][d])['image']
            label = loadmat(self.datalst[i][0])['image'] / 1000 * miuWater
            label = label.transpose()
            label = label - np.min(label)
            return torch.Tensor(image[np.newaxis, :, :]), torch.Tensor(label[np.newaxis, :, :])


def train_v2(net, miuWater=0.19, Cuda=True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))
    batch_size = 4
    num_workers = 0

    train_data = testData_v2(miuWater=miuWater)
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    mse_loss = torch.nn.MSELoss()
    for epoch in range(1):
        running_loss = 0.0
        for i, (mt, gt) in tqdm(enumerate(train_dataloader)):
            #print((mt.shape,gt.shape))
            tinput = mt.to(device)
            target = gt.to(device)
            optimizer.zero_grad()
            score = net(tinput)
            loss = mse_loss(score, target)
            loss.backward()
            optimizer.step()
            running_loss = loss.item() + running_loss
            if i % 100 == 0:
                print('[%d, %5d] loss: %.9f' % (epoch, i, running_loss))
                running_loss = 0.0
        torch.save(net.state_dict(),"mynet.params")


def test_v2(net, to_be_pred, miuWater=0.19, Cuda=True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))

    test_data = testData_v2(miuWater=miuWater, istest=True, to_be_pred=to_be_pred)
    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
    with torch.no_grad():
        for i, (mt, gt) in tqdm(enumerate(test_dataloader)):
            #print((mt.shape,gt.shape))
            tinput = mt.to(device)
            marTest = net(tinput)
    return marTest