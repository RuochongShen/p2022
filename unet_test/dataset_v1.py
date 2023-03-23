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

dataset_read = loadmat('C:/Users/ScottShen/bins/UniMelb/phd/project2022/mywork/imdb-500.mat')

training_data = dataset_read['imdb'][0][0][0][0][0][1]
sets_id = dataset_read['imdb'][0][0][0][0][0][2][0]
labels = dataset_read['imdb'][0][0][0][0][0][3]

training_input = np.array([training_data[:, :, :, i] for i in range(500) if sets_id[i]==1])
training_target = np.array([labels[:, :, :, i] for i in range(500) if sets_id[i]==1])
valid_input = np.array([training_data[:, :, :, i] for i in range(500) if sets_id[i]==2])
valid_target = np.array([labels[:, :, :, i] for i in range(500) if sets_id[i]==2])

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
imRaw_3ch = imRaw_3ch[ :, :, :, np.newaxis]

miuWater = 0.19

#testv1_raw = (np.maximum((imRaw - miuWater)/miuWater * 1000,0)).astype(np.uint16)
testv1_raw = (np.maximum(imRaw/miuWater * 1000, 0)).astype(np.uint16)


def tif2arr(tif):
    return tif/1000 * miuWater + miuWater


# arr/miuWater * 1000 - 1000
def arr2tif(arr):
    return (np.maximum(arr/miuWater * 1000, 0)).astype(np.uint16)


class testData_1ch(data.Dataset):
    def __init__(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

    def __len__(self):
        return self.training_data.shape[0]

    def __getitem__(self, idx):
        image = self.training_data[idx, 0:1, :, :]
        label = self.labels[idx, :, :, :]
        return torch.Tensor(image), torch.Tensor(label)


def train_1ch(net, Cuda = True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))
    batch_size = 4
    num_workers = 0

    train_data = testData_1ch(training_data.transpose((3,2,1,0)), labels.transpose((3,2,1,0)))
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    mse_loss = torch.nn.MSELoss()
    for epoch in range(20):
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


class testData_3ch(data.Dataset):
    def __init__(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

    def __len__(self):
        return self.training_data.shape[0]

    def __getitem__(self, idx):
        image = self.training_data[idx, :, :, :]
        label = self.labels[idx, :, :, :]
        return torch.Tensor(image), torch.Tensor(label)


def train_3ch(net, Cuda = True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))
    batch_size = 4
    num_workers = 0

    train_data = testData_3ch(training_data.transpose((3,2,1,0)), labels.transpose((3,2,1,0)))
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    mse_loss = torch.nn.MSELoss()
    for epoch in range(20):
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


class testData_gan_3ch(data.Dataset):
    # data.shape = batch_num, channel_num, height, weight
    def __init__(self, metal_data, tr_data, isTrain):
        self.metal_data = metal_data
        self.tr_data = tr_data
        self.isTrain = isTrain

    def __len__(self):
        return self.metal_data.shape[0]

    def __getitem__(self, idx):
        metal_img = self.metal_data[idx, :, :, :]
        tr_img = self.tr_data[idx, :, :, :]
        return torch.Tensor(metal_img), torch.Tensor(tr_img)


def train_gan_3ch(net_G, net_D, Cuda = True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net_G.to(device)
    net_D.to(device)
    # net.load_state_dict(torch.load("mynet.params"))
    batch_size = 4
    num_workers = 0

    train_data = testData_gan_3ch(training_data.transpose((3,2,1,0)), labels.transpose((3,2,1,0)), True)
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=5e-5)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=5e-4)
    adversarial_loss = torch.nn.BCELoss()
    for epoch in range(20):
        running_loss_g = 0.0
        running_loss_d = 0.0
        for i, (mt, tr) in tqdm(enumerate(train_dataloader)):
            #print((mt.shape,gt.shape))
            tr_tag = torch.full((tr.shape[0], 1), 1.0, requires_grad=False).to(device)
            mt_tag = torch.full((mt.shape[0], 1), 0.0, requires_grad=False).to(device)
            mt = mt.to(device)
            tr = tr.to(device)

            optimizer_G.zero_grad()
            metal_mar = net_G(mt)
            #print(metal_mar.shape)
            g_loss = adversarial_loss(net_D(metal_mar), tr_tag)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            tr_loss = adversarial_loss(net_D(tr), tr_tag)
            mt_loss = adversarial_loss(net_D(metal_mar.detach()), mt_tag)
            d_loss = (tr_loss + mt_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            running_loss_g = g_loss.item() + running_loss_g
            running_loss_d = d_loss.item() + running_loss_d

            if i % 100 == 0:
                print('[%d, %5d] loss_g: %.9f, loss_d: %.9f' % (epoch, i, running_loss_g, running_loss_d))
                running_loss_g = 0.0
                running_loss_d = 0.0

        torch.save(net_G.state_dict(),"mynet_G.params")
        torch.save(net_D.state_dict(),"mynet_D.params")


class testData_trans_3ch(data.Dataset):
    def __init__(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

    def __len__(self):
        return self.training_data.shape[0]

    def __getitem__(self, idx):
        image = self.training_data[idx, :, :, :]
        label = self.labels[idx, :, :, :]
        return torch.Tensor(image), torch.Tensor(label)


def train_trans_3ch(net, Cuda = True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))
    batch_size = 4
    num_workers = 0

    train_data = testData_trans_3ch(training_data.transpose((3,2,1,0)), labels.transpose((3,2,1,0)))
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    mse_loss = torch.nn.MSELoss()
    for epoch in range(20):
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


def test_1ch(net, Cuda=True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))

    test_data = testData_1ch(np.reshape(imRaw, (512,512,1,1)).transpose(3,2,1,0), np.reshape(imRaw, (512,512,1,1)).transpose(3,2,1,0))
    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)
    with torch.no_grad():
        for i, (mt, gt) in tqdm(enumerate(test_dataloader)):
            #print((mt.shape,gt.shape))
            tinput = mt.to(device)
            marTest = net(tinput)
    return marTest


def test_3ch(net, to_be_pred, label, Cuda=True):
    if Cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)
    # net.load_state_dict(torch.load("mynet.params"))

    test_data = testData_3ch(to_be_pred, label)
    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)
    with torch.no_grad():
        for i, (mt, gt) in tqdm(enumerate(test_dataloader)):
            #print((mt.shape,gt.shape))
            tinput = mt.to(device,dtype=torch.float)
            marTest = net(tinput)
    return marTest


if __name__ == '__main__':
    net_base_unet = BasicUNet(1, 1)
    train_1ch(net_base_unet, False)
    testout_v1_base_unet = test_1ch(net_base_unet, False)
