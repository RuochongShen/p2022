import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# tif
import PIL
from PIL import Image

import skimage.transform
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans

import scipy
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interp1d, griddata

import pickle
import pandas as pd
# import glob


class CTpara:
    def __init__(self, angsize, AngNum, SOD, imPixNum, imPixScale=1):
        """
        This code is to set parameters of the equi-angular CT scanning
        """
        self.angsize = angsize  # angle between two neighbor rays
        # self.DetNum         # number of detector bins
        self.AngNum = AngNum  # number of projection views
        self.SOD = SOD  # source-to-origin distance, [cm]
        self.imPixNum = imPixNum  # image pixels along x or y direction
        self.imPixScale = imPixScale  # the real size of each pixel, [cm]

        # normalize
        self.SOD = self.SOD / self.imPixScale


def parallelbeam(image, pspacing=1, rangedeg=360):
  """
  forward transform for parallel beam
  """
  CTparam = CTpara(pspacing, rangedeg, 0, 1, 1)
  image = np.array(image)
  theta = np.linspace(0., CTparam.angsize*CTparam.AngNum, num=CTparam.AngNum, endpoint=False)
  sinogram = skimage.transform.radon(image, theta=theta, circle=False)
  return sinogram


def iparallelbeam(proj, pspacing=1, rangedeg=360):
  """
  reconstruction with the filtered back projection for parallel beam
  """
  CTparam = CTpara(pspacing, rangedeg, 0, 1, 1)
  proj = np.array(proj)
  theta = np.linspace(0., CTparam.angsize*CTparam.AngNum, num=CTparam.AngNum, endpoint=False)
  image = skimage.transform.iradon(proj, theta=theta, filter_name='ramp', circle=False)
  return image


def shiftAndInterpRotationAngles(gamma, theta, pthetapad, Ppad):
    m, n = gamma.size, theta.size
    F = np.zeros((m, n))
    for i in range(m):
        F[i, :] = interp1d(pthetapad + gamma[i], Ppad[i, :], fill_value="extrapolate")(theta)
    return F


def fanbeam(image, CTparam, pspacing=1, prangedeg=360):
    """
    forward transform for fan beam
    range of rotation: circle
    linear interpolation
    """
    P = parallelbeam(image, pspacing=pspacing, rangedeg=prangedeg)
    d = CTparam.SOD
    dthetaDeg = prangedeg / CTparam.AngNum  # fan rotation angle in degrees
    m, n = P.shape

    # padding to have an odd m
    P = np.concatenate((np.zeros((2 - m % 2, n)), P, np.zeros((1, n))), axis=0)
    m, n = P.shape

    dploc = pspacing  # parallel sensor spacing in degrees
    ploc = np.array([dploc * i for i in range(-(m // 2), m // 2 + 1)])  # parallel sensors
    fanspacing = CTparam.angsize
    fanspacingRad = fanspacing * np.pi / 180

    # equal-angle: fansensor spacing = arcsin(para spacing/SOD)
    nGammaRad = np.floor(np.arcsin(m // 2 * dploc / d) / fanspacingRad)
    gammaRad = fanspacingRad * np.arange(-nGammaRad, nGammaRad + 1)
    gammaDeg = gammaRad * 180 / np.pi  # transformed sensors
    gammaMax, gammaMin = np.max(gammaDeg), np.min(gammaDeg)
    gammaRange = gammaMax - gammaMin

    # form a vector of fan rotation angles in degrees
    dpthetaDeg = prangedeg / n
    pthetaDeg = dpthetaDeg * np.arange(n)
    thetaDeg = dthetaDeg * np.arange(CTparam.AngNum * (360 / prangedeg))

    # interpolate to get desired sample locations
    Pint = np.zeros((gammaDeg.size, n))
    t = d * np.sin(gammaRad)
    for i in range(n):
        Pint[:, i] = interp1d(ploc, P[:, i].ravel(), fill_value="extrapolate")(t)

    # if prangedeg==360
    Ppad, pthetapad = Pint, pthetaDeg

    Ppad = np.concatenate((Ppad[:, pthetapad >= 360 - gammaRange],
                           Ppad,
                           Ppad[:, pthetapad <= gammaRange]),
                          axis=1)
    pthetapad = np.concatenate((pthetapad[pthetapad >= 360 - gammaRange] - 360,
                                pthetapad,
                                pthetapad[pthetapad <= gammaRange] + 360))
    F = shiftAndInterpRotationAngles(gammaDeg, thetaDeg, pthetapad, Ppad)

    # if prangedeg==360
    Ppad2, pthetapad2 = Ppad[::-1, :], pthetapad - 180
    theta2 = (thetaDeg + 180) % 360 - 180
    F2 = shiftAndInterpRotationAngles(gammaDeg, theta2, pthetapad2, Ppad2)
    F = (F + F2) / 2

    return np.nan_to_num(F)


def ifanbeam(proj, CTparam, pspacing=1, prangedeg=360):
    """
    reconstruction with the filtered back projection for fan beam
    """
    F = proj
    d = CTparam.SOD
    m, n = F.shape

    # padding to have an odd m
    F = np.concatenate((np.zeros((2 - m % 2, n)), F, np.zeros((1, n))), axis=0)
    m, n = F.shape

    fanspacing = CTparam.angsize
    gammaDeg = fanspacing * np.arange(-(m // 2), m // 2 + 1)

    # if 360
    thetaDeg = np.arange(n) * 360 / n

    nploc = np.floor(d * np.sin(fanspacing * (m // 2) * np.pi / 180) / pspacing)
    ploc = pspacing * np.arange(-nploc, nploc + 1)

    dpthetaDeg = pspacing
    pthetaDeg = dpthetaDeg * np.arange(prangedeg)

    n4 = int(np.ceil(n / 4))
    Fpad = np.concatenate((F[:, -n4:], F, F[:, :n4]), axis=1)
    thetapad = np.concatenate((thetaDeg[-n4:] - 360, thetaDeg, thetaDeg[:n4] + 360))

    mpad, npad = Fpad.shape
    Fsh = np.zeros((mpad, pthetaDeg.size))
    for i in range(mpad):
        Fsh[i, :] = interp1d(thetapad - gammaDeg[i], Fpad[i, :].ravel(),
                             fill_value="extrapolate")(pthetaDeg)

    P = np.zeros((ploc.size, pthetaDeg.size))
    t = d * np.sin(gammaDeg * np.pi / 180)
    for i in range(pthetaDeg.size):
        P[:, i] = interp1d(t, Fsh[:, i].ravel(), fill_value="extrapolate")(ploc)

    P = np.nan_to_num(P)
    image = iparallelbeam(P, pspacing=pspacing, rangedeg=prangedeg)
    im = image.shape[0]
    if im > CTparam.imPixNum:
        if im - CTparam.imPixNum <= 3:
            image = image[1:-1, 1:-1]
        if im - CTparam.imPixNum == 3:
            image = image[1:, 1:]

    return image


def projInterp(proj, metalTrace):
  """
  projection linear interpolation
  Input:
    proj:         uncorrected projection
    metalTrace:   metal trace in projection domain (binary image)
  Output:
    Pinterp:      linear interpolation corrected projection
  """
  nbin, nview = proj.shape
  Pinterp = np.zeros(proj.shape)

  # use 1-D linear interpolation
  for i in range(0, nview):
    metal_slice = metalTrace[:, i]
    proj_slice = proj[:, i]

    x = np.arange(nbin)[metal_slice==0]  # non-metal positions
    y = proj_slice[metal_slice==0]  # projection at non-metal positions
    f = interp1d(x, y)  # kind='linear' as default. interp1d(x, y, kind='')

    Pinterp[:, i] = f(np.arange(nbin))

  return Pinterp


def marLI(proj, metalTrace, CTparam, isfanbeam=True):
  """
  This code is to reduce metal artifacts using linear interpolation
  Input:
    proj:         uncorrected projection
    metalTrace:   metal trace in projection domain (binary image)
  Output:
    imLI:         linear interpolation corrected image (1/cm)
  """
  Pinterp = projInterp(proj, metalTrace)
  if isfanbeam:
    imLI = ifanbeam(Pinterp, CTparam)
  else:
    imLI = iparallelbeam(Pinterp)
  imLI = imLI / CTparam.imPixScale
  return imLI


def marBHC(proj, metalBW, CTparam, isfanbeam=True):
    bins, views = proj.shape
    if isfanbeam:
        projMetal = fanbeam(metalBW.astype(float), CTparam)
    else:
        projMetal = parallelbeam(metalBW.astype(float))
    projMetal = projMetal * CTparam.imPixScale
    Pinterp = projInterp(proj, projMetal > 0)
    projDiff = proj - Pinterp
    projDiff1 = np.reshape(projDiff, (bins * views,))
    projMetal1 = np.reshape(projMetal, (bins * views,))
    projMetalbw1 = np.reshape(projMetal > 0, (bins * views,))

    # first order beam hardening correction
    A = np.zeros((bins * views, 3))
    A[:, 0] = projMetalbw1 * projMetal1
    A[:, 1] = projMetalbw1 * projMetal1 ** 2
    A[:, 2] = projMetalbw1 * projMetal1 ** 3
    X0 = np.linalg.lstsq(A, projMetalbw1 * projDiff1)[0].ravel()  # coefficients of the polynomial

    projFit1 = X0[0] * A[:, 0] + X0[1] * A[:, 1] + X0[2] * A[:, 2]

    projDelta1 = projMetalbw1 * (X0[0] * projMetal1 - projFit1)
    projBHC = proj + np.reshape(projDelta1, (bins, views))

    if isfanbeam:
        imBHC = ifanbeam(projBHC, CTparam)
    else:
        imBHC = iparallelbeam(projBHC)
    imBHC = imBHC / CTparam.imPixScale
    return imBHC


def myshowpics(pics, figsize=(64, 36)):
  """
  Util function for showing results
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


def im2HU(im, miuWater):
  res = (im - miuWater) / miuWater * 1000
  res[res < 0] = 0
  res = res.astype(np.uint16)
  return res


def HU2im(imHU, miuWater):
  return imHU.astype(float) / 1000 * miuWater + miuWater


def demo():
    """
    can be executed on google colab;
    take care to change the paths and functions (like display -> plt.imshow())
    """
    path_read = "/content/drive/MyDrive/FurtherStudy/p2022/test/25-48.tif"
    path_png_save = "/content/drive/MyDrive/FurtherStudy/p2022/test/test1.png"
    path_tif_save_LI = "imLI_test.tif"
    path_tif_save_BHC = "imBHC_test.tif"

    # read image and show
    im = Image.open(path_read)
    im_arr = np.array(im)
    # display(Image.fromarray(im_arr / 16.0).convert("L"))

    # save as png
    new_png = Image.fromarray(im_arr / 16.0).convert("L")
    new_png.save(path_png_save)

    # parameters and inputs
    CTpara1 = CTpara(0.001 * 180, 984, 59.5, 512, 0.08)
    miuWater = 0.192
    imRawHU = np.array(im)
    imRawCT = HU2im(imRawHU, miuWater)

    # prepare mid parameters
    para_proj = parallelbeam(imRawCT)
    para_proj = para_proj * CTpara1.imPixScale
    proj = fanbeam(imRawCT, CTpara1)
    proj = proj * CTpara1.imPixScale
    myshowpics([(para_proj, "parallel projection"), (proj, "fanbeam projection")])

    threshold = 6000
    metalBW = (imRawHU > threshold).astype(float)
    myshowpics([(metalBW, "metal")])

    metalTrace = fanbeam(metalBW, CTpara1)
    metalTrace = metalTrace * CTpara1.imPixScale > 0
    para_metalTrace = parallelbeam(metalBW)
    para_metalTrace = para_metalTrace * CTpara1.imPixScale > 0
    myshowpics([(para_metalTrace, "parallel trace"), (metalTrace, "fanbeam trace")])

    # check LI and BHC based on either parallel beam or fanbeam and show
    imLI = marLI(proj, metalTrace, CTpara1)
    imBHC = marBHC(proj, metalBW, CTpara1)

    para_imLI = marLI(para_proj, para_metalTrace, CTpara1, isfanbeam=False)
    para_imBHC = marBHC(para_proj, metalBW, CTpara1, isfanbeam=False)

    myshowpics([(imLI, "fanbeam LI"), (imBHC, "fanbeam BHC"), (para_imLI, "parallel LI"), (para_imBHC, "parallel BHC")])

    imLI_HU = im2HU(imLI, miuWater)
    imBHC_HU = im2HU(imBHC, miuWater)

    print(np.max(imLI_HU), np.max(im), np.min(imLI_HU), np.min(im))
    print(np.max(imBHC_HU), np.max(im), np.min(imBHC_HU), np.min(im))
    print(np.max(im2HU(para_imLI, miuWater)), np.max(im), np.min(im2HU(para_imLI, miuWater)), np.min(im))

    # set the path
    cv2.imwrite(path_tif_save_LI, imLI_HU)
    cv2.imwrite(path_tif_save_BHC, imBHC_HU)
