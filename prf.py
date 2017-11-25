# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:04:24 2017

@author: yangyang
"""

from __future__ import print_function, division

from astropy.io import fits
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import interpolation
from scipy.optimize import fmin_powell
from scipy.stats import multivariate_normal
import batman

import sys, os, re, time, os.path, glob
import math
import time
import argparse
import logging
import model
import diffimg
from PRFfunc import PRF, PRF2DET
from astropy.wcs import WCS

import matplotlib.pylab as plt 

def pixel_info(pixel_file):
    pf = fits.open(pixel_file)
    kepid = pf[0].header['KEPLERID']
    channel = pf[0].header['CHANNEL']
    try:
        skygroup = pf[0].header['SKYGROUP']
        skygroup = str(skygroup)
    except:
        skygroup = '0'
    module = pf[0].header['MODULE']
    output = pf[0].header['OUTPUT']
    campaign = pf[0].header['CAMPAIGN']
    data_rel = pf[0].header['DATA_REL']
    try:
        season = pf[0].header['SEASON']
        season = str(season)
    except:
        season = '0'
    ra = pf[0].header['RA_OBJ']
    dec = pf[0].header['DEC_OBJ']
    kepmag = pf[0].header['KEPMAG']
    tdim5 = pf['TARGETTABLES'].header['TDIM5']
    xdim = int(tdim5.strip().strip('(').strip(')').split(',')[0])
    ydim = int(tdim5.strip().strip('(').strip(')').split(',')[1])
    crv5p1 = pf['TARGETTABLES'].header['1CRV5P']
    column = crv5p1
    crv5p2 = pf['TARGETTABLES'].header['2CRV5P']
    row = crv5p2
    return (kepid, channel, skygroup, module, output, campaign, data_rel, season,\
            ra, dec, column, row, kepmag, xdim, ydim)
            
def prf_info(prf_file, hdu):
    """read pixel response file"""

    prf = fits.open(prf_file)

    # read bitmap image
    img = prf[hdu].data
    naxis1 = prf[hdu].header['NAXIS1']
    naxis2 = prf[hdu].header['NAXIS2']
    # read WCS keywords
    crpix1p = prf[hdu].header['CRPIX1P']
    crpix2p = prf[hdu].header['CRPIX2P']
    crval1p = prf[hdu].header['CRVAL1p']
    crval2p = prf[hdu].header['CRVAL2p']
    cdelt1p = prf[hdu].header['CDELT1P']
    cdelt2p = prf[hdu].header['CDELT2P']

    prf.close()
    return img, crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p

def bitInBitmap(bitmap, bit):
    """bit map decoding"""
    flag = False
    for i in range(10, -1,- 1):
        if bitmap - 2**i >= 0:
            bitmap = bitmap - 2**i
            if 2**i == bit:
                flag = True
        else:
            continue
    return flag
    
def intScale2D(image, imscale):
    """intensity scale limits of 2d array"""
    nstat = 2
    work1 = np.array([], dtype=np.float32)
    (ysiz, xsiz) = np.shape(image)
    for i in range(ysiz):
        for j in range(xsiz):
            if np.isfinite(image[i, j]) and image[i, j] > 0.0:
                work1 = np.append(work1, image[i, j])
    work2 = np.array(np.sort(work1))
    if int(float(len(work2)) / 1000 + 0.5) > nstat:
        nstat = int(float(len(work2)) / 1000 + 0.5)
    zmin = np.median(work2[:nstat])
    zmax = np.median(work2[-nstat:])
    if imscale == 'logarithmic':
        image = np.log10(image)
        zmin = math.log10(zmin)
        zmax = math.log10(zmax)
    if imscale == 'squareroot':
        image = np.sqrt(image)
        zmin = math.sqrt(zmin)
        zmax = math.sqrt(zmax)

    return image, zmin, zmax
    

def prf(DATimg, DATimg_eff, or_pix_file, prfdir, maskimg, columns, rows, fluxes, projection, background=False,\
           border=1, focus=False, xtol=1e-4, ftol=0.0001, outfile=None, plot=False,\
           imscale='linear', cmap='YlOrBr', apercol='#ffffff', verbose=True):
               #construct inital guess vector for fit
               f =fluxes
               x = columns
               y = rows
               nsrc = 1
               guess = np.array([f, x, y])
               
               if background:
                   if border == 0:
                       guess.append(0.0)
                   else:
                       for i in range((border + 1) * 2):
                           guess.append(0.0)
    
               if focus:
                   guess = guess + [1.0, 1.0, 0.0]
               
               kepid, channel, skygroup, module, output, campaign, data_rel, season, ra, \
               dec, column, row, kepmag, xdim, ydim = \
               pixel_info(or_pix_file)
               npix = np.size(np.nonzero(maskimg)[0])
               
               # print target data
               if verbose:
                   print('')
                   print('      KepID: {}'.format(kepid))
                   print(' RA (J2000): {}'.format(ra))
                   print('Dec (J2000): {}'.format(dec))
                   print('     KepMag: {}'.format(kepmag))
                   print('   SkyGroup: {}'.format(skygroup))
                   print('     Season: {}'.format(str(season)))
                   print('     Campaign: {}'.format(campaign))
                   print('     Data Realse: {}'.format(data_rel))
                   print('    Channel: {}'.format(channel))
                   print('     Module: {}'.format(module))
                   print('     Output: {}'.format(output))
                   print('')
                   
               #construct pixel image
               DATx = np.arange(column,column + xdim)
               DATy = np.arange(row, row + ydim)
               
               #determine PRF calibration file
               if int(module) < 10:
                   prefix = 'kplr0'
               else:
                   prefix = 'kplr'
               prfglob = prfdir + '/' + prefix + str(module) + '.' + str(output) + '*' + '_prf.fits'
               prffile = glob.glob(prfglob)[0]
               
               # read PRF images
               prfn = [0,0,0,0,0]
               crpix1p = np.zeros(5, dtype='float32')
               crpix2p = np.zeros(5, dtype='float32')
               crval1p = np.zeros(5, dtype='float32')
               crval2p = np.zeros(5, dtype='float32')
               cdelt1p = np.zeros(5, dtype='float32')
               cdelt2p = np.zeros(5, dtype='float32')
               for i in range(5):
                   prfn[i], crpix1p[i], crpix2p[i], crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i] = \
                       prf_info(prffile, i+1)
               prfn = np.array(prfn)
               PRFx = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
               PRFy = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
               PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p[0]
               PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p[0]
               
               # interpolate the calibrated PRF shape to the target position
               prf = np.zeros(np.shape(prfn[0]), dtype='float32')
               prfWeight = np.zeros(5, dtype='float32')
               for i in range(5):
                   prfWeight[i] = math.sqrt((column - crval1p[i])**2 + (row - crval2p[i])**2)
                   if prfWeight[i] == 0.0:
                       prfWeight[i] = 1.0e-6
                   prf = prf + prfn[i] / prfWeight[i]
               prf = prf / np.nansum(prf) / cdelt1p[0] / cdelt2p[0]

               # location of the data image centered on the PRF image (in PRF pixel units)
               prfDimY = int(ydim / cdelt1p[0])
               prfDimX = int(xdim / cdelt2p[0])
               PRFy0 = int(np.round((np.shape(prf)[0] - prfDimY) / 2))
               PRFx0 = int(np.round((np.shape(prf)[1] - prfDimX) / 2))
               
               # interpolation function over the PRF
               splineInterpolation = RectBivariateSpline(PRFx,PRFy,prf)
               
               # fit PRF model to pixel data
               start = time.time()
               args = (DATx, DATy, DATimg, DATimg_eff, nsrc, splineInterpolation, float(x), float(y))
               ans = fmin_powell(PRF, guess, args=args, xtol=xtol, ftol=ftol, disp=True)
               print("Convergence time = {}s\n".format(time.time() - start))
               
               # pad the PRF data if the PRF array is smaller than the data array
               flux = []
               OBJx = []
               OBJy = []
               PRFmod = np.zeros((prfDimY, prfDimX))
               if PRFy0 < 0 or PRFx0 < 0.0:
                   PRFmod = np.zeros((prfDimY, prfDimX))
                   superPRF = np.zeros((prfDimY + 1, prfDimX + 1))
                   superPRF[abs(PRFy0):abs(PRFy0) + np.shape(prf)[0],\
                            abs(PRFx0):abs(PRFx0) + np.shape(prf)[1]] = prf
                   prf = superPRF * 1.0
                   PRFy0 = 0
                   PRFx0 = 0
               # rotate the PRF model around its center
               if focus:
                   angle = ans[-1]
                   prf = interpolation.rotate(prf, -angle, reshape=False,\
                                              mode='nearest')
               for i in range(nsrc):
                   flux.append(ans[i])
                   OBJx.append(ans[nsrc + i])
                   OBJy.append(ans[nsrc * 2 + i])
                   # calculate best-fit model
                   y = (OBJy[i] - np.mean(DATy)) / cdelt1p[0]
                   x = (OBJx[i] - np.mean(DATx)) / cdelt2p[0]
                   prfTmp = interpolation.shift(prf, [y, x], order=3, mode='constant')
                   prfTmp = prfTmp[PRFy0:PRFy0 + prfDimY, PRFx0:PRFx0 + prfDimX]
                   PRFmod = PRFmod + prfTmp * flux[i]
                   wx = 1.0
                   wy = 1.0
                   angle = 0
                   b = 0.0
                   # write out best fit parameters
                   if verbose:
                       txt = ("Flux = {0} e-/s X = {1} pix Y = {2} pix".format(flux[i], OBJx[i], OBJy[i]))
                       print(txt)
                       
                   # measure flux fraction and contamination
                   PRFall = PRF2DET(flux, OBJx, OBJy, DATx, DATy, wx, wy, angle,\
                             splineInterpolation)
                   PRFone = PRF2DET([flux[0]], [OBJx[0]], [OBJy[0]], DATx, DATy,\
                             wx, wy, angle, splineInterpolation)
                   FluxInMaskAll = np.nansum(PRFall)
                   FluxInMaskOne = np.nansum(PRFone)
                   FluxInAperAll = 0.0
                   FluxInAperOne = 0.0
                   for i in range(1, ydim):
                       for j in range(1, xdim):
                           if (maskimg[i, j]==1):
                               FluxInAperAll += PRFall[i, j]
                               FluxInAperOne += PRFone[i, j]
                   FluxFraction = FluxInAperOne / flux[0]
                   
                   try:
                       Contamination = (FluxInAperAll - FluxInAperOne) / FluxInAperAll
                   except:
                       Contamination = 0.0
                   print("\nTotal flux in mask = {0} e-/s".format(FluxInMaskAll))
                   print("\nTarget flux in mask = {0} e-/s".format(FluxInMaskOne))
                   print("\nTotal flux in aperture = {0} e-/s".format(FluxInAperAll))
                   print("\nTarget flux in aperture = {0} e-/s".format(FluxInAperOne))
                   print("\nTarget flux fraction in aperture = {0} %".format(FluxFraction * 100.0))
                   print("\nContamination fraction in aperture = {0} %".format(Contamination * 100.0))
                   
                   # construct model PRF in detector coordinates
                   PRFfit = PRFall + 0.0
                   # calculate residual of DATA - FIT
                   PRFres = DATimg - PRFfit
                   FLUXres = np.nansum(PRFres) / npix
                   
                   # calculate the sum squared difference between data and model
                   Pearson = abs(np.nansum(np.square(DATimg - PRFfit) / PRFfit))
                   Chi2 = np.nansum(np.square(DATimg - PRFfit) / np.square(DATimg_eff))
                   DegOfFreedom = npix - len(guess) - 1
                   
                   print("\n Residual flux = {0} e-/s".format(FLUXres))
                   print("\n Pearson\'s chi^2 test = {0} for {1} dof".format(Pearson, DegOfFreedom))
                   print("\n Chi^2 test = {0} for {1} dof".format(Chi2, DegOfFreedom))
                   # image scale and intensity limits for plotting images
                   imgdat_pl, zminfl, zmaxfl = intScale2D(DATimg, imscale)
                   imgprf_pl, zminpr, zmaxpr = intScale2D(PRFmod, imscale)
                   imgfit_pl, zminfi, zmaxfi = intScale2D(PRFfit, imscale)
                   imgres_pl, zminre, zmaxre = intScale2D(PRFres, 'linear')
                   if imscale == 'linear':
                       zmaxpr *= 0.9
                   elif imscale == 'logarithmic':
                       zmaxpr = np.max(zmaxpr)
                       zminpr = zmaxpr / 2

                   plt.figure(figsize=(18, 14))
                   plt.clf()
                   ax1 = plt.subplot(221,projection=projection)
                   ax1.set_xlabel("CCD Column")
                   ax1.set_ylabel("CCD Row")
                   ax2 = plt.subplot(222)
                   ax3 = plt.subplot(223,projection=projection)
                   ax3.set_xlabel("CCD Column")
                   ax3.set_ylabel("CCD Row")
                   ax4 = plt.subplot(224,projection=projection)
                   ax4.set_xlabel("CCD Column")
                   ax4.set_ylabel("CCD Row")
                   ax1.imshow(imgdat_pl, aspect='auto',interpolation='nearest',\
                              vmin=zminfl,vmax=zmaxfl, cmap=cmap)
                   ax1.text(0.05, 0.9,'observation',horizontalalignment='left',verticalalignment='center',\
                                     fontsize=26,fontweight=500)
                   ax2.imshow(imgprf_pl, aspect='auto',interpolation='nearest',\
                              vmin=zminfl,vmax=zmaxfl, cmap=cmap)
                   ax2.text(47, 60,'model',horizontalalignment='left',verticalalignment='center',\
                                     fontsize=26,fontweight=500)
                   ax3.imshow(imgfit_pl, aspect='auto',interpolation='nearest',\
                              vmin=zminfl,vmax=zmaxfl, cmap=cmap)
                   ax3.scatter(projection.all_world2pix(OBJx,OBJy,1)[0]-1, 15-projection.all_world2pix(OBJx,OBJy,1)[1],marker='X',color='white',s=100)
                   ax3.text(0.05, 0.9,'fit',horizontalalignment='left',verticalalignment='center',\
                                     fontsize=26,fontweight=500)
                   ax4.imshow(imgres_pl, aspect='auto',interpolation='nearest',\
                              vmin=zminfl,vmax=zmaxfl, cmap=cmap)
                   ax4.text(0.05, 0.9,'residual',horizontalalignment='left',verticalalignment='center',\
                                     fontsize=26,fontweight=500)
                   
                   plt.savefig(outfile)
                   
                   return flux, OBJx, OBJy

def cent_RMS(fixel_file, orient, time_range):
    t = fits.open(fixel_file)
    mask = np.where(t[1].data['SAP_QUALITY']==0)
    if orient=='column':
        cent = t[1].data['PSF_CENTR1'][mask]
        cent_err = t[1].data['PSF_CENTR1_ERR'][mask]
    elif orient =='row':
        cent = t[1].data['PSF_CENTR2'][mask]
        cent_err = t[1].data['PSF_CENTR2_ERR'][mask]
    cent_tmp = []
    cent_weight = []
    for i in range(len(time_range)):
        idx_i = np.where((t[1].data['TIME'][mask]>time_range[i][0])&(t[1].data['TIME'][mask]<time_range[i][1]))
        cent_tmp = np.append(cent_tmp,cent[idx_i])
        cent_weight = np.append(cent_weight,cent_err[idx_i])
    return cent_tmp, cent_weight                   
                   
filedir = "/home/yangyang/Documents/Code/astronomy/centroidtest/test/"
i_o = model.In_out_transit()
ini = i_o.from_ini(os.path.join(filedir,'model.ini'))
LC = fits.open(os.path.join(filedir,"hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits"))
pf = fits.open("ktwo201920032-c01_lpd-targ.fits")
mask = np.where(LC[1].data['QUALITY'] == 0)
t = LC[1].data['TIME'][mask]
timemodel = np.linspace(t.min(),t.max(),t.shape[0])
LC_Model = i_o.model(ini, timemodel)
plt.scatter(timemodel,LC_Model,marker='.')
it = i_o.in_transit_range(timemodel,LC_Model,ini)
ot = i_o.out_transit_range(timemodel,it,ini)
img_i = diffimg.add_img("hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits",it)
img_o = diffimg.add_img("hlsp_everest_k2_llc_201920032-c01_kepler_v2.0_lc.fits",ot)
wcs = WCS(LC[3].header, key='P')
diff_img = diffimg.img_in_apr(LC[3].data,img_o-img_i)
diff_img_i = diffimg.img_in_apr(LC[3].data,img_i)
diff_img_o = diffimg.img_in_apr(LC[3].data,img_o)
unc = diffimg.unc_img(pf[1].data['FLUX_ERR'][mask],LC[3].data, pf[1].data['TIME'][mask],np.vstack((it,ot)))
unc_i = diffimg.unc_img(pf[1].data['FLUX_ERR'][mask],LC[3].data, pf[1].data['TIME'][mask],it)
unc_o = diffimg.unc_img(pf[1].data['FLUX_ERR'][mask],LC[3].data, pf[1].data['TIME'][mask],ot)
#ax = plt.subplot(111, projection=wcs)
#ax.imshow(diff_img)
#ax.scatter(9.870969162523352-1,16-7.849627832013198-1,marker='x',color='white')

#centroid calculation from out-transit & diff imgage
wcs2 = WCS(LC[3].header)
column = 329.87096916252335
row = 571.8496278320132
out_of_transit_cent = prf(diff_img_o, unc_o, "ktwo201920032-c01_lpd-targ.fits","kplr2011265_prf", LC[3].data, column, row, np.nansum(img_o), projection=wcs, outfile='out-of-transit.png')
diff_cent = prf(diff_img, unc, "ktwo201920032-c01_lpd-targ.fits","kplr2011265_prf", LC[3].data, column, row, np.nansum(diff_img), projection=wcs, outfile='diff.png')
out_of_transit_x = out_of_transit_cent[1][0]
out_of_transit_y = out_of_transit_cent[2][0]
x2, y2 = wcs.all_world2pix(out_of_transit_x,out_of_transit_y,1)
x2_, y2_ = wcs2.all_pix2world(x2,y2,1)
diff_x = diff_cent[1][0]
diff_y = diff_cent[2][0]
x1, y1 = wcs.all_world2pix(diff_x,diff_y,1)
x1_, y1_ = wcs2.all_pix2world(x1,y1,1)
x0, y0 = wcs.all_world2pix(column,row,1)
x0_,y0_ = wcs2.all_pix2world(x0,y0,1)
#centroid uncertainties estimation(from centroid time series)------------------
c = cent_RMS('ktwo201920032-c01_llc.fits', 'column', ot)
w = 1.0/c[1]**2/np.sum(1.0/c[1]**2)
c2 = cent_RMS('ktwo201920032-c01_llc.fits', 'row', ot)
w2 = 1.0/c2[1]**2/np.sum(1.0/c2[1]**2)
w_total = np.transpose([w,w2])
center = wcs2.all_pix2world(wcs.all_world2pix(np.transpose([c[0],c2[0]]),1),1)-np.tile([x1_,y1_],(len(w),1))
m=np.sum(w_total*center,axis=0)
std=np.sum(w_total*(center - np.tile(m,(len(center),1)))**2,axis=0)**0.5*3600
#erro propagation (standard deviation)
total_ang = (((x1_ - x2_)*np.cos(y2_*np.pi/180.0)*3600)**2+((y1_ - y2_)*3600)**2)**0.5
std_t = np.sqrt((((x1_ - x2_)*np.cos(y2_*np.pi/180.0)**2))**2*std[0]**2+((x1_ - x2_)**2*0.5*np.sin(2*y2_*np.pi/180.0)*np.pi/180+(y1_-y2_))**2*std[1]**2)*3600/total_ang
sigma3 = 3*std_t
#plot--------------------------------------------------------------------------
centroid_off_x = (x1_ - x2_)*np.cos(y2_*np.pi/180.0)*3600
centroid_off_y = (y1_ - y2_)*3600
fig1 = plt.figure(figsize=(14,10))
ax1 = plt.subplot()
circle1=plt.Circle((centroid_off_x,centroid_off_y),sigma3,color='blue',fill=False)
plt.gcf().gca().add_artist(circle1)
X, Y = np.mgrid[-5:5:.01, -5:5:.01]
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([centroid_off_x,centroid_off_y], [[std[0], 0], [0, std[1]]])
ax1.contourf(X, Y, rv.pdf(pos),offset=0.15, level=[0.069182369442786328,0.069182369442786329])
ax1.axes.errorbar(centroid_off_x, centroid_off_y, xerr=std[0],yerr=std[1],marker='+',color='green')
ax1.scatter(centroid_off_x, centroid_off_y,marker='X',color='white',s=150)
ax1.scatter(0,0, marker='*', color = 'red',s=150)
ax1.set_xlim(xmin=-5,xmax=5)
ax1.set_ylim(ymin=-5,ymax=5)
ax1.set_title('Offset Relative to Out of Transit Centroid',fontweight=1000)
ax1.set_xlabel('RA Offset (arcsec)', fontweight=1000)
ax1.set_ylabel('Dec Offset (arcsec)', fontweight=1000)
fig1.savefig("Centroid_Offset1.eps",format='eps')
fig2 = plt.figure(figsize=(14,10))
plt.scatter((x1_ - x0_)*np.cos(y0_*np.pi/180.0)*3600, (y1_ - y0_)*3600, marker='X', color = 'purple',s=150)
plt.scatter(0,0, marker='*', color = 'red',s=150)
plt.set_xlim(xmin=-1,xmax=1)
plt.set_ylim(ymin=-1,ymax=1)
plt.set_title('Offset Relative to K2 Position',fontweight=1000)
plt.set_xlabel('RA Offset (arcsec)', fontweight=1000)
plt.set_ylabel('Dec Offset (arcsec)', fontweight=1000)
fig2.savefig("Centroid_Offset2.eps",format='eps')