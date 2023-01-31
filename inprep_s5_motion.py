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
AVG_WDW = 4

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



# Import trajectories data
with open(TRAJ_FILENAME,'rb') as f:
    WORMS = pickle.load(f)

# Load blob data from BLOB_FILENAME
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f) 
    
    
idx = 10
n_frames = WORMS[idx].x.shape[0]
n_hist = 20
    
for jj, contour in enumerate( WORMS[idx].c):
    frame = np.zeros((80,80), dtype='uint8')

    if jj>n_hist:
        for _ in range(n_hist,0,-1):
            color = int(255 - 255*(_)/n_hist)
            frame = cv2.drawContours( frame, [WORMS[idx].c[jj-_] +[40,40] ], -1, ( color ), -1, cv2.LINE_AA )
    frame = cv2.drawContours( frame, [contour +[40,40] ], -1, (255) , -1, cv2.LINE_AA )

    
    frame = cv2.resize( frame, (800,600))
    cv2.imshow( 'wdw', frame)
    if cv2.waitKey(20) == ord('q'):
        break
cv2.destroyAllWindows()
    
# tstart = WORMS[idx].t0    
# worm_contours = []
# for jj_frame in range( n_frames):
#     worm_coords = WORMS[idx].x[jj_frame], WORMS[idx].y[jj_frame]
    
#     for kk_contour, c in enumerate(CONTOURS[tstart + jj_frame ]):
#         centroid = vutils.centroid(c)
#         if centroid[0]==worm_coords[0] and centroid[1]==worm_coords[1]:
#             worm_contours.append(c) 
#             print(jj_frame, 'found!', kk_contour)
    
#     frame = np.zeros((50,50), dtype='uint8')
#     frame = cv2.drawContours( frame, [worm_contours[-1]- vutils.centroid(worm_contours[-1]) +[20,20] ], -1, (255) , -1, cv2.LINE_AA )
#     frame = cv2.resize( frame, (800,600))
#     cv2.imshow( 'wdw', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()