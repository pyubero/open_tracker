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
from datetime import datetime
from tqdm import tqdm
from video_utils import MultiWormTracker
import video_utils as vutils



# ... Filenames ... 
DIR_NAME          = 'video_grad_5mM_sinCond_50ulOP50_2206091243_000'
BLOB_FILENAME     = os.path.join( DIR_NAME, 'video_data_blobs.pkl')
BLOB_REF_FILENAME = os.path.join( DIR_NAME, 'video_reference_contour.pkl')
TRAJ_FILENAME     = os.path.join( DIR_NAME, 'trajectories.pkl')

# ... Reference calibration ...
HU_THRESH     = 1e-10
METRIC_THRESH = 3.0
AREA_MIN      = 60
AREA_MAX      = 150

# ... General parameters ...
SPEED_DAMP    = 0.8     
MAX_STEP      = 25
WAIT_TIME     = 1
VERBOSE  = False

# ... Output ...
WORMS = []
COLORS= []
FRAME_WIDTH = 2592
FRAME_HEIGHT= 1944
ZOOM        = 1.5
TRUE_VIDEO  = False
EXPORT_TRAJ = False



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
                            n_step=20,
                            speed_damp=SPEED_DAMP,
                            is_worm = check_worm,
                            verbose = VERBOSE)

#... if true_video == True, then load the video
cap = cv2.VideoCapture( DIR_NAME+'.avi')


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
    
    #... draw circle around current worms 
    _ = [cv2.circle(frame, ( int(w.x[-1]), int(w.y[-1])), MAX_STEP, COLORS[jj].tolist(), 2) for jj, w in enumerate(WORMS)]
    _ = [cv2.putText(frame, "%d" % jj, ( int(w.x[-1]+10), int(w.y[-1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[jj].tolist(), 2, cv2.LINE_AA) for jj, w in enumerate(WORMS)]
    
    #... zoom in and resize output
    frame = vutils.zoom_in(frame, [0.5,0.5] , ZOOM)
    frame = cv2.resize(frame, (1280,960))
    
    #... add labels
    cv2.putText(frame,  "frame : %d" % t,
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(frame,  "#worms : %d" % len(WORMS),
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
    with open(TRAJ_FILENAME,'wb') as f:
        pickle.dump( WORMS, f)

