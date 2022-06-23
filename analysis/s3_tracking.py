# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:59:54 2022

@author: Pablo
"""

import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture
import video_utils as vutils


# ... Filenames ... 
BLOB_FILENAME     = 'video_data_blobs.pkl'
BLOB_REF_FILENAME = 'video_reference_contour.pkl'

# ... General parameters ...
HU_THRESH         = 1e-10
METRIC_THRESH = 5 #3.85
AREA_MIN      = 40
AREA_MAX      = 115
MAX_SPEED     = 5

# ... Output ...
WORMS = []



print('Loading data from %s...' % BLOB_FILENAME)
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f) 
    n_frames = len(CONTOURS)

print('Loading reference blob from %s...' % BLOB_REF_FILENAME)
with open( BLOB_REF_FILENAME, 'rb') as f:
    CNT_REF = pickle.load(f)

HU_REF = vutils.logHu( CNT_REF, HU_THRESH ) 

drawing_points=[]
drawing_colors=[]
def new_worm(contour):
    _center = vutils.centroid(contour)
    x = [_center[0]]*3
    y = [_center[1]]*3
    return [x,y]

for t in tqdm(range(n_frames)):
    
    contours = CONTOURS[t]
    blobs_t = np.array( [ vutils.centroid(c) for c in contours] )
    blobs_used = []
    
    worms_t_1=[]
    for worm in WORMS:
        vx = np.clip( worm[0][-1]-worm[0][-2], -MAX_SPEED,MAX_SPEED)
        vy = np.clip( worm[1][-1]-worm[1][-2], -MAX_SPEED,MAX_SPEED)
        
        x_exp = worm[0][-1] + vx
        y_exp = worm[1][-1] + vy
        worms_t_1.append( [x_exp, y_exp ])
    
    
    # jj es el worm index de los que FIJO existen t-1 y por tanto en t
    # idx es el worm index de los worms putativos en t
    # ...
    worms_t_1 = np.array( worms_t_1 )
    for jj, worm in enumerate( worms_t_1 ):
        dist = np.sqrt( np.sum( ( worm - blobs_t )**2, axis=1) )
        if np.any( dist<50) :
            idx = np.argmin(dist)
            # Linking of jj worm in t-1 to idx blob in t
            WORMS[jj][0].append( blobs_t[idx][0] )
            WORMS[jj][1].append( blobs_t[idx][1] )
            blobs_used.append(idx)
            print('[%d] Linked %d worm to blob %d' % (t, jj, idx) )
        else:
            # Linking of jj worm in t-1 to its potential place in t
            # ... the worm is probably being occluded
            WORMS[jj][0].append( worms_t_1[jj][0] )
            WORMS[jj][1].append( worms_t_1[jj][1] )
            
            
          
    
    # Prepare show image
    frame = np.zeros((1944, 2592,3), dtype='uint8')
    cv2.drawContours(frame, contours, -1, (255,255,255), -1, cv2.LINE_AA)
    # draw circle around current worms in yellow
    _ = [cv2.circle(frame, ( w[0][-1], w[1][-1]), 15, drawing_colors[jj].tolist(), 2) for jj, w in enumerate(WORMS)]
    _ = [cv2.line(frame, (w[0][-1],w[1][-1]), worms_t_1[jj], (0,0,255), 2, cv2.LINE_AA) for jj, w in enumerate(WORMS)]
    
    frame = vutils.zoom_in(frame, (1200,1000), 2)
    frame = cv2.resize(frame, (800,600))
    cv2.putText(frame,  "frame : %d" % t,
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    
    
    cv2.imshow('wdw', frame)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
    # These are the blobs that are potentially new, solitary, worms
    blobs_left = [jj for jj in range(len(blobs_t)) if jj not in blobs_used ]
    
    for jj in blobs_left:
        cnt = contours[jj]
        if vutils.is_worm(cnt, HU_REF, METRIC_THRESH, AREA_MIN, AREA_MAX, hu_thold=HU_THRESH) :
            WORMS.append( new_worm(cnt) )
            drawing_colors.append( np.random.randint(255, size=(3,)) )

    
    
cv2.destroyAllWindows()

plt.figure(dpi=600)
for w in WORMS:
    plt.plot( w[0], w[1],'.-' , ms=3)
plt.xlim(500,1500)
plt.ylim(500,1500)

