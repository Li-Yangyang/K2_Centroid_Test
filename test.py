# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:51:39 2017

@author: yangyang
"""

import numpy as np
import matplotlib.pylab as plt
import os
import model
from astropy.io import fits
filedir = "/home/yangyang/Documents/Code/astronomy/centroidtest/test/"
i_o = model.in_out_transit()
ini = i_o.from_ini(os.path.join(filedir,'model.ini'))
LC = fits.open(os.path.join(filedir,"hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits"))
mask = np.where(LC[1].data['QUALITY'] == 0)
t = LC[1].data['TIME'][mask]
timemodel = np.linspace(t.min(),t.max(),t.shape[0])
LC_Model = i_o.model(ini, timemodel)
plt.scatter(timemodel,LC_Model,marker='.')
it = i_o.in_transit_range(timemodel,LC_Model,ini)
# plot in_transit
for i in range(len(it)):
     idx_i = np.where((timemodel<it[i][1]) & (timemodel>it[i][0]))
     plt.scatter(timemodel[idx_i],LC_Model[idx_i],marker='.',color='red')
     #plot put_transit
ot = i_o.out_transit_range(timemodel,it,ini)
for i in range(len(ot)):
     idx_o = np.where((timemodel<ot[i][1]) & (timemodel>ot[i][0]))
     plt.scatter(timemodel[idx_o],LC_Model[idx_o],marker='.',color='green')
     #plt.vlines(ini[1]-duration(ini)/2.0,0.99,1)
     #plt.vlines(ini[1]+duration(ini)/2.0,0.99,1)
plt.xlim(1999+ini[0]*1,2001+ini[0]*1)