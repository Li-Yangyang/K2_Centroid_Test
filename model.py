# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:54:45 2017

@author: yangyang
"""
from __future__ import print_function, division

from astropy.io import fits
import numpy as np
import batman

import sys, os, re, time, os.path, glob
import argparse
import logging
from configobj import ConfigObj


class In_out_transit(object):
    
    def __int__(self, filedir, LC):
        self.filedir = filedir
        self.LC = LC

    def model(self,modelpara,time_points):
        '''
        read parameters from fitting results and create light curve model with
        same time points
    
        '''
        period, t0, aoR, rprs, inc, ecc, omeg, kepu1, kepu2 = modelpara
        tparams = batman.TransitParams()
        tparams.t0 = t0                     #time of inferior conjunction
        tparams.per = period                #orbital period 
        tparams.rp = rprs                   #planet radius (in units of stellar radii)
        tparams.a = aoR                     #semi-major axis (in units of stellar radii)
        tparams.inc = inc                   #orbital inclination (in degrees)
        tparams.ecc = ecc                   #eccentricity
        tparams.w = omeg                    #longitude of periastron (in degrees)
        tparams.limb_dark = "quadratic"         #limb darkening model
        tparams.u = [kepu1, kepu2]            #limb darkening coefficients [u1, u2, u3, u4]
    
        Model = batman.TransitModel(tparams, time_points, transittype='primary')
        LCModel=Model.light_curve(tparams)
        return LCModel
        
    def from_ini(self, ini_file):
        config = ConfigObj(ini_file)
        para = []
        for i in range(9):
            para.append(float(config.items()[i][1]))
        return np.transpose(para)
                
    def time_cov(self,t):
        new_t=t+2400000.5-2454833
        return new_t
                    
    def duration(self,ini):
        duration = ini[0]/np.pi*np.arcsin(1/ini[2]*np.sqrt((1+ini[3])**2-(ini[2]*np.cos(ini[4]*np.pi/180)**2)**2)/np.sin(ini[4]*np.pi/180))
        return duration

    def in_transit_range(self,timemodel, LC_Model, ini):
        min_index = np.where(np.r_[True, LC_Model[1:] < LC_Model[:-1]] & np.r_[LC_Model[:-1] < LC_Model[1:], True]==True)
        r = []    
        for i in range(len(min_index[0])):
            depth = 1- LC_Model[min_index[0][i]]
            idx1 = np.where((timemodel<(timemodel[min_index[0][i]]+self.duration(ini)/2.0)) & (timemodel>(timemodel[min_index[0][i]]-self.duration(ini)/2.0)))
            idx2 = np.where(LC_Model[idx1]<(1-0.75*depth))
            rg = [timemodel[idx1][idx2].min(), timemodel[idx1][idx2].max()]
            r.append(rg)
        r =np.array(r)
        return r    

    def out_transit_range(self,timemodel, it, ini):
        index = []
        for i in range(len(it)):
            idx2 = np.where(timemodel==it[i][0])[0][0]-3
            idx3 = np.where(timemodel==it[i][1])[0][0]+3
            index.append([idx2,idx3])
        index = np.array(index)
        tmp = timemodel[index]
        t1 = [tmp[:, 0]-self.duration(ini),tmp[:,0]]
        t2 = [tmp[:, 1],tmp[:,1]+self.duration(ini)]
        return np.vstack((np.transpose(t1),np.transpose(t2)))
    

