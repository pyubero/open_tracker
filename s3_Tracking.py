# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:59:54 2022

@author: Pablo
With ~600 worms in the video, ~4 fps
With ~60 worms in the video,  ~60fps
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
# DIR_NAME          = './videos/_cut'
# DIR_NAME          = './videos/Carla_EC/Carla_N2_EC_2211101415_002'
DIR_NAME          = './videos/Carla_EC/Analysis'
BLOB_FILENAME     = os.path.join( DIR_NAME, 'video_data_blobs.pkl')
BLOB_REF_FILENAME = os.path.join( DIR_NAME, 'video_reference_contour.pkl')
TRAJ_FILENAME     = os.path.join( DIR_NAME, 'trajectories.pkl')
IMG_FILENAME      = os.path.join( DIR_NAME, 'trajectories.png')
BKGD_FILENAME     = os.path.join( DIR_NAME, 'video_fondo.png')
ROIS_FILENAME     = os.path.join( DIR_NAME, 'rois.pkl')


# ... Reference calibration ...
HU_THRESH     = 1e-08
METRIC_THRESH = 3.50
AREA_MIN      = 130
AREA_MAX      = 280

# ... General parameters ...
SPEED_DAMP    = 0.5 
INERTIA       = 5
MAX_STEP      = 35
WAIT_TIME     = 10
VERBOSE  = False

# ... Output ...
WORMS = []
COLORS= []
FRAME_WIDTH = 2592 #2592
FRAME_HEIGHT= 1944 #1944
ZOOM        = 1.0
TRUE_VIDEO  = False
EXPORT_TRAJ = True

# Load ROIS
with open( ROIS_FILENAME, 'rb') as f:
    data = pickle.load(f) 
plate = data[0]


# Step 1. Load blob data from BLOB_FILENAME
print('Loading data from %s...' % BLOB_FILENAME)
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f) 
    n_frames = len(CONTOURS)



# Step 2. Prepare reference contour..
#... load blob
print('Loading reference blob from %s...' % BLOB_REF_FILENAME)
with open( BLOB_REF_FILENAME, 'rb') as f:
    CNT_REF = pickle.load(f)

#... compute HU_REF vector
HU_REF = vutils.logHu( CNT_REF, HU_THRESH ) 

#... prepare check_worm function, returns True if the contour is sufficiently
#... similar to that of the reference worm.
def check_worm(contour):
    return vutils.is_worm(contour, HU_REF, METRIC_THRESH, AREA_MIN, AREA_MAX, HU_THRESH)


  
# Step 3. Create Tracker Object and list of COLORS, one for each worm
COLORS  = []
TRACKER = MultiWormTracker( max_step=MAX_STEP,
                            inertia = INERTIA,
                            n_step=20,
                            speed_damp=SPEED_DAMP,
                            is_worm = check_worm,
                            keep_alive = 2,
                            verbose = VERBOSE)

#... if true_video == True, then load the video
# cap = cv2.VideoCapture( DIR_NAME+'.avi')


tStart = datetime.now()
for t in tqdm(range(n_frames)):    
    contours = CONTOURS[t]
  
    TRACKER.update( contours )
    WORMS = TRACKER.WORMS.copy()
    
    
        
    # Assign a unique color to each worm
    while len(WORMS) > len(COLORS):
        COLORS.append( np.random.randint(255, size=(3,)) )        
    

    # Reconstruct image from contours
    if TRUE_VIDEO:
        ret, frame = cap.read()
    else:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH,3), dtype='uint8')
        cv2.drawContours(frame, contours, -1, (255,255,255), -1, cv2.LINE_AA)
        frame = 255-frame
    
    #... draw circle around current worms 
    _ = [cv2.circle(frame, ( int(w.x[-1]), int(w.y[-1])), MAX_STEP, COLORS[jj].tolist(), 2) for jj, w in enumerate(WORMS) if w.x[-1]>0 ]
    # _ = [cv2.putText(frame, "%d" % jj, ( int(w.x[-1]+10), int(w.y[-1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[jj].tolist(), 2, cv2.LINE_AA) for jj, w in enumerate(WORMS) if w.x[-1]>0 ]
    
    #... draw estelas of worms   
    for worm_jj, worm in enumerate(WORMS):
        [cv2.circle(frame, ( int(x), int(y)), 2, COLORS[worm_jj].tolist(), -1) for x,y in zip(worm.x[-200:], worm.y[-200:]) if x>0 ]

    #... draw plate 
    cv2.circle( frame, (plate[0], plate[1]), plate[2],(0,255,0),5)

    #... zoom in and resize output
    frame = vutils.zoom_in(frame, [0.5,0.5] , ZOOM)
    frame = cv2.resize(frame, (800,600))
    
    #... add labels
    cv2.putText(frame,  "frame : %d" % t,
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(frame,  "#worms : %d" % np.sum([ w.x[-1]>0 for w in WORMS]),
                (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    
    #... display image
    cv2.imshow('wdw', frame)    
    key = cv2.waitKey( WAIT_TIME )
    if key==ord('q'):
        break
    if key==ord('p'):
        WAIT_TIME = 1 - WAIT_TIME
    
    
   
    
cv2.destroyAllWindows()







print('---------------------------------')
print('Speed: %1.3f fps'% (t/(datetime.now()-tStart).total_seconds() )  )
print('Number of worms at the end: %d' % len(WORMS) )

if EXPORT_TRAJ:
    # Easily export all worm objects
    with open(TRAJ_FILENAME,'wb') as f:
        pickle.dump( WORMS, f)


    # Or export their coordinates
    n_worms  = len(WORMS)
    T0 = WORMS[0].t0
    
    data_xy = np.nan*np.zeros( (n_worms, n_frames+1, 2) )
    for i_worm, worm in enumerate( WORMS):
        length = len( worm.x)
        ini = worm.t0 - T0
        fin = ini + length
        print(ini, fin)
        data_xy[i_worm, ini:fin, 0] = worm.x
        data_xy[i_worm, ini:fin, 1] = worm.y
    
    np.savez( TRAJ_FILENAME+'.npz', data=data_xy)




fondo = vutils.load_background( BKGD_FILENAME )*0.5

color = plt.cm.viridis(np.linspace(0, 1, len(WORMS)))
color = color[np.random.permutation(len(WORMS)),:]

plt.figure( figsize=(6,6), dpi=600 )
plt.imshow( fondo, cmap='gray', vmax=255, vmin=0)
for jj, worm in enumerate( WORMS) :
    plt.plot( worm.x, worm.y, lw=0.5, c=color[jj,:] )
    
plt.xticks( ticks=plt.xticks()[0], labels='')
plt.yticks( ticks=plt.yticks()[0], labels='')
plt.xlim(0, fondo.shape[1])
plt.ylim(0, fondo.shape[0])
plt.gca().invert_yaxis()
plt.title(DIR_NAME)
plt.savefig( IMG_FILENAME , bbox_inches='tight', )