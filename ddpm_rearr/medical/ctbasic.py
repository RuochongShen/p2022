import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from PIL import Image
import skimage.transform
from skimage.filters import threshold_multiotsu
import scipy
from scipy.interpolate import interp1d, griddata
import matlab.engine

# TODO

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
    theta = np.linspace(0., CTparam.angsize * CTparam.AngNum, num=CTparam.AngNum, endpoint=False)
    sinogram = skimage.transform.radon(image, theta=theta, circle=False)
    return sinogram


def iparallelbeam(proj, pspacing=1, rangedeg=360):
    """
    reconstruction with the filtered back projection for parallel beam
    """
    CTparam = CTpara(pspacing, rangedeg, 0, 1, 1)
    proj = np.array(proj)
    theta = np.linspace(0., CTparam.angsize * CTparam.AngNum, num=CTparam.AngNum, endpoint=False)
    image = skimage.transform.iradon(proj, theta=theta, filter_name='ramp', circle=False)
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

        x = np.arange(nbin)[metal_slice == 0]  # non-metal positions
        y = proj_slice[metal_slice == 0]  # projection at non-metal positions
        f = interp1d(x, y)  # kind='linear' as default. interp1d(x, y, kind='')

        Pinterp[:, i] = f(np.arange(nbin))

    return Pinterp


def marLI(proj, metalTrace):
    """
    This code is to reduce metal artifacts using linear interpolation
    Input:
      proj:         uncorrected projection
      metalTrace:   metal trace in projection domain (binary image)
    Output:
      imLI:         linear interpolation corrected image (1/cm)
    """
    Pinterp = projInterp(proj, metalTrace)
    imLI = iparallelbeam(Pinterp)
    return imLI


def marBHC(proj, metalBW):
    bins, views = proj.shape
    projMetal = parallelbeam(metalBW.astype(float))
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

    imBHC = iparallelbeam(projBHC)
    return imBHC


def myLI(image, metalBW):
    proj = parallelbeam(image)
    metalTrace = parallelbeam(metalBW.astype(float))
    return marLI(proj, metalTrace)


def myBHC(image, metalBW):
    proj = parallelbeam(image)
    return marBHC(proj, metalBW)


def fanbeam(image):
    m = matlab.engine()
