# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:21:08 2022

@author: logslab
"""

import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

from pytracker.video_utils import MultiWormTracker
import pytracker.video_utils as vutils



# ... Filenames ... 
DIR_NAME          = 'videos/_cut'
BLOB_FILENAME     = os.path.join( DIR_NAME, 'video_data_blobs.pkl')
BLOB_REF_FILENAME = os.path.join( DIR_NAME, 'video_reference_contour.pkl')
TRAJ_FILENAME     = os.path.join( DIR_NAME, 'trajectories.pkl')
IMG_FILENAME      = os.path.join( DIR_NAME, 'trajectories.png')
BKGD_FILENAME     = os.path.join( DIR_NAME, 'video_fondo.png')


#... Parameters
AVG_WDW = 1
MIN_LENGTH    = 100

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



# Import data
with open(TRAJ_FILENAME,'rb') as f:
    data = pickle.load(f)

WORMS = [ worm for worm in data if len(worm.x)>MIN_LENGTH ]

_nfalive = [ worm.t0+len(worm.x) for worm in WORMS ]
n_frames = np.max(_nfalive)
n_worms  = len(WORMS)

data_xy = np.nan*np.zeros( (n_worms, n_frames-AVG_WDW+1, 2) )
for i_worm, worm in enumerate( WORMS):
    length = len( worm.x)-AVG_WDW+1
    ini = worm.t0
    fin = ini + length
    data_xy[i_worm, ini:fin, 0] = moving_average( worm.x, AVG_WDW)
    data_xy[i_worm, ini:fin, 1] = moving_average( worm.y, AVG_WDW)
    


data_vw = np.nan*np.zeros( (n_worms, n_frames-AVG_WDW, 2) )
data_vw[:,:,0] = np.sqrt( np.diff( data_xy[:,:,0], axis=1)**2 + np.diff( data_xy[:,:,1], axis=1)**2 )
data_vw[:,:,1] = np.arctan2( np.diff( data_xy[:,:,1], axis=1), np.diff( data_xy[:,:,0], axis=1) )



idx = 3
LW= 0.5
BINS_V = np.linspace(0,10,20)
BINS_W = np.linspace(-np.pi, np.pi,20)

fondo = vutils.load_background( BKGD_FILENAME )*0.5

plt.figure( figsize=(6,6), dpi=600 )

plt.subplot(2,2,1)
plt.imshow( fondo, cmap='gray', vmax=255, vmin=0)
plt.plot( data_xy[idx,:,0], data_xy[idx,:,1] )
# plt.xticks( ticks=plt.xticks()[0], labels='')
# plt.yticks( ticks=plt.yticks()[0], labels='')

plt.subplot(2,2,2)
plt.plot( data_vw[idx,:,0] + 4 , lw=LW)
plt.hlines(4, 0, n_frames, 'k')
plt.plot( data_vw[idx,:,1] , lw=LW)
plt.yticks( ticks=plt.yticks()[0], labels='')
plt.ylabel('Angular speed      Speed')

plt.subplot(2,2,3)
plt.hist( data_vw[idx,:,0], bins=BINS_V, density=True)
plt.yscale('log')
plt.xlabel('Speed (pxl/2s)')
plt.ylabel('Probability density')

plt.subplot(2,2,4)
plt.hist( data_vw[idx,:,1], bins=BINS_W, density=True)
plt.xlabel('Angular speed (rad/2s)')
plt.ylabel('Probability density')

plt.tight_layout()





