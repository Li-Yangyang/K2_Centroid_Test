# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:20:41 2017

@author: yangyang
"""
import numpy as np
from math import modf, radians

def PRF2DET(flux, OBJx, OBJy, DATx, DATy, wx, wy, a, splineInterpolation):
    """
    PRF interpolation function
    """

    # trigonometry
    cosa = np.cos(radians(a))
    sina = np.sin(radians(a))

    # where in the pixel is the source position?
    PRFfit = np.zeros((np.size(DATy), np.size(DATx)))
    for i in range(len(flux)):
        FRCx, INTx = modf(OBJx[i])
        FRCy, INTy = modf(OBJy[i])
        if FRCx > 0.5:
            FRCx -= 1.0
            INTx += 1.0
        if FRCy > 0.5:
            FRCy -= 1.0
            INTy += 1.0
        FRCx = -FRCx
        FRCy = -FRCy

        # constuct model PRF in detector coordinates
        for (j, y) in enumerate(DATy):
            for (k, x) in enumerate(DATx):
                xx = x - INTx + FRCx
                yy = y - INTy + FRCy
                dx = xx * cosa - yy * sina
                dy = xx * sina + yy * cosa
                PRFfit[j, k] = PRFfit[j, k] + splineInterpolation(dy * wy, dx * wx) * flux[i]

    return PRFfit
    
def PRF(params, *args):
    """
    PRF model
    """
    # arguments
    DATx = args[0]
    DATy = args[1]
    DATimg = args[2]
    DATerr = args[3]
    nsrc = args[4]
    splineInterpolation = args[5]
    col = args[6]
    row = args[7]

    # parameters
    f = np.empty((nsrc))
    x = np.empty((nsrc))
    y = np.empty((nsrc))
    for i in range(nsrc):
        f[i] = params[i]
        x[i] = params[nsrc + i]
        y[i] = params[nsrc * 2 + i]

    # calculate PRF model binned to the detector pixel size
    PRFfit = PRF2DET(f,x,y,DATx,DATy,1.0,1.0,0.0,splineInterpolation)
    # calculate the sum squared difference between data and model
    PRFres = np.nansum(np.square(DATimg - PRFfit))
    # keep the fit centered
    if max(abs(col - x[0]), abs(row - y[0])) > 10.0:
        PRFres = 1.0e300
    return PRFres

