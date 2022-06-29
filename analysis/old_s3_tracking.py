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
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp
from video_utils import Worm
import video_utils as vutils



# ... Filenames ... 
DIR_NAME          = 'video_grad_5mM_sinCond_50ulOP50_2206091243_000'
# DIR_NAME          = 'SampleVideo'
BLOB_FILENAME     = os.path.join( DIR_NAME, 'video_data_blobs.pkl')
BLOB_REF_FILENAME = os.path.join( DIR_NAME, 'video_reference_contour.pkl')
TRAJ_FILENAME     = os.path.join( DIR_NAME, 'trajectories.pkl')

# ... General parameters ...
HU_THRESH     = 1e-10
METRIC_THRESH = 3.5
AREA_MIN      = 50
AREA_MAX      = 200
SPEED_DUMP    = 0.5     #... dumping parameter of speed in the Kalman filter
WAIT_TIME     = 1
VERBOSE  = False
# ... Output ...
WORMS = []
COLORS= []


## Now, the class MultiWormTracker should have an api similar to opencv's trackers.






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


tStart = datetime.now()
for t in tqdm(range(n_frames)):    
    contours = CONTOURS[t]
    
    # I need the centroid of blobs available in time t 
    #        the  area    of blobs available in time t
    #        the worms present at time t-1
    #        the speed of worms at time t-1
    #        the expected position of worms at time t
    
    blobs_t  = np.array( [ vutils.centroid(c) for c in contours] )
    blobs_t_area= np.array( [cv2.contourArea(c) for c in contours ] )


    # jj es el worm index de los que FIJO existen t-1 y por tanto en t
    # idx es el worm index de los worms putativos en t
    blobs_t_used = []
    blobs_t_disp = blobs_t.copy()
    worms_t_1_disp= np.array(range(len(WORMS)))
    
    radii = np.linspace(0,20,10)

    # By looping across radii, I do a progressive scan to identify proximal worms
    for _radius in radii:
        
        # Then for each worm that has not been tracked...        
        for worm_jj in worms_t_1_disp.copy():
            
            #... I compute its distance to all blobs
            _worm_expected = WORMS[worm_jj].expected_position( alpha= SPEED_DUMP)
            dist = np.sqrt( np.sum( ( _worm_expected - blobs_t_disp )**2, axis=1) )
            
            #... if any blob is closer than _radius
            if np.any( dist < _radius ):
                
                #... identify the most proximal blob
                idx = np.argmin(dist)
                
                #... link worm_jj to blob idx
                WORMS[ worm_jj ].update( blobs_t[idx] )
                
                #... and then make blob_t[idx] unavailable and mark idx as used
                blobs_t_used.append(idx)
                blobs_t_disp[idx][0] = 999999
                blobs_t_disp[idx][1] = 999999
                
                #... and delete worm_jj from the "to-track" list
                worms_t_1_disp = np.delete( worms_t_1_disp, np.where(worms_t_1_disp==worm_jj) ) 
                
                # Print some output
                if VERBOSE:
                    print('[%d] Linked worm %d to blob %d at distance %1.1f' % (t, worm_jj, idx, dist[idx]) )
    
    # Then, there will still be some un-tracked worms, waiting... in the dark...
    if VERBOSE:
        print('Untracked worms:', worms_t_1_disp)
    
    # BUT, untracked worms that are suspiciously close to unused blobs (perhaps large ones) should be linked together            
    for worm_jj in worms_t_1_disp:
        _worm_expected = WORMS[worm_jj].expected_position( alpha= SPEED_DUMP)
        dist = np.sqrt( np.sum( ( _worm_expected - blobs_t )**2, axis=1) )
        
        if np.any( dist < 40 ):
            #... link worm_jj to blob idx
            idx = np.argmin(dist)
            WORMS[ worm_jj ].update( blobs_t[idx] )
            if VERBOSE:
                print('--- Linked worm %d to blob %d at distance %1.1f' % (worm_jj, idx, dist[idx]) )
          
    
    # Reconstruct image from contours
    frame = np.zeros((1944, 2592,3), dtype='uint8')
    # frame = np.zeros((960, 1280,3), dtype='uint8')
    cv2.drawContours(frame, contours, -1, (255,255,255), -1, cv2.LINE_AA)
    
    #... draw circle around current worms in yellow
    _ = [cv2.circle(frame, ( int(w.x[-1]), int(w.y[-1])), 15, COLORS[jj].tolist(), 2) for jj, w in enumerate(WORMS)]
    _ = [cv2.putText(frame, "%d" % jj, ( int(w.x[-1]+10), int(w.y[-1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[jj].tolist(), 2, cv2.LINE_AA) for jj, w in enumerate(WORMS)]
    _ = [cv2.putText(frame, "%d" % jj, vutils.centroid(c), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA) for jj, c in enumerate(contours)]
    
    #... zoom in and resize output
    # frame = vutils.zoom_in(frame, [0.5,0.5] , 1.5)
    frame = cv2.resize(frame, (800,600))
    
    #... add frame number label
    cv2.putText(frame,  "frame : %d" % t,
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    
    #... display image
    cv2.imshow('wdw', frame)    
    key = cv2.waitKey( WAIT_TIME )
    if key==ord('q'):
        break
    if key==ord('p'):
        WAIT_TIME = 1 - WAIT_TIME
    
    
    # These are the blobs that are potentially new, solitary, worms
    blobs_left = [jj for jj in range(len(blobs_t)) if jj not in blobs_t_used ]
    
    for jj in blobs_left:
        cnt = contours[jj]
        if vutils.is_worm(cnt, HU_REF, METRIC_THRESH, AREA_MIN, AREA_MAX, hu_thold=HU_THRESH) :
            _position = vutils.centroid(cnt)
            
            WORMS.append( Worm(_position, t0=t) )
            COLORS.append( np.random.randint(255, size=(3,)) )

    
    
cv2.destroyAllWindows()

print('---------------------------------')
print('Analysis speed: %1.3f fps'% (t/(datetime.now()-tStart).total_seconds() )  )
print('Number of worms at the end: %d' % len(WORMS) )

plt.figure(dpi=600, figsize=(12,4))
plt.subplot(1,2,1)
for w in WORMS:
    plt.plot( w.x, w.y,'.-' , ms=3)
# plt.xlim(500,1500)
# plt.ylim(500,1500)

plt.subplot(1,2,2)
plt.hist( WORMS[0].speed_module(), alpha=0.5, density=True)
plt.hist( WORMS[2].speed_module(), alpha=0.5, density=True)
plt.yscale('log')


ks_2samp(WORMS[0].speed_module(), WORMS[2].speed_module())


with open(TRAJ_FILENAME,'wb') as f:
    pickle.dump( WORMS, f)