# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:08:01 2017

@author: yangyang
"""

from __future__ import print_function, division

from astropy.io import fits
import numpy as np
import batman

import sys, os, re, time, os.path, glob
import argparse
import logging
import model
from astropy.wcs import WCS

import matplotlib.pylab as plt 

def add_img(fixel_file, time_range):
    t = fits.open(fixel_file)
    mask = np.where(t[1].data['QUALITY']==0)
    f_pix = t[2].data['FPIX'][mask]
    mean_img = np.zeros(34)
    for i in range(len(time_range)):
        idx_i = np.where((t[1].data['TIME'][mask]>time_range[i][0])&(t[1].data['TIME'][mask]<time_range[i][1]))
        mean_img = np.vstack((mean_img,f_pix[idx_i]))
    mean_img = np.average(mean_img[1:],axis=0)
    return mean_img

def img_in_apr(ap_file, img):
    k = 0
    ap = np.array([([0] *18) for p in range(16)]) 
    for i in range(ap_file.shape[0]):
        for j in range(ap_file.shape[1]):
            if ap_file[i][j] != 0:
                ap[i][j] = img[k]
                k = k+1
            else:
                ap[i][j] = 0
    return ap
    
def unc_img(err_file, maskimg, t, time_range):
    unc = np.zeros((maskimg.shape[0],maskimg.shape[1]))
    no = 0
    for i in range(len(time_range)):
        idx_i = np.where((t>time_range[i][0])&(t<time_range[i][1]))
        unc = unc + np.sum(np.square(err_file[idx_i]*maskimg),axis=0)
        no = no + len(idx_i[0])
    unc = np.sqrt(unc/no)
    return unc
    
filedir = "/home/yangyang/Documents/Code/astronomy/centroidtest/test/"
i_o = model.In_out_transit()
ini = i_o.from_ini(os.path.join(filedir,'model.ini'))
LC = fits.open(os.path.join(filedir,"hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits"))
mask = np.where(LC[1].data['QUALITY'] == 0)
t = LC[1].data['TIME'][mask]
timemodel = np.linspace(t.min(),t.max(),t.shape[0])
LC_Model = i_o.model(ini, timemodel)
plt.scatter(timemodel,LC_Model,marker='.')
it = i_o.in_transit_range(timemodel,LC_Model,ini)
ot = i_o.out_transit_range(timemodel,it,ini)
img_i = add_img("hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits",it)
img_o = add_img("hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits",ot)
wcs = WCS(LC[3].header, key='P')
diff_img = img_in_apr(LC[3].data,img_o-img_i)
ax = plt.subplot(111, projection=wcs)
ax.imshow(diff_img)
ax.scatter(9.870969162523352-1,16-7.849627832013198-1,marker='x',color='white')
