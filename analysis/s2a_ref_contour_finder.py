# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:49:13 2022

@author: Pablo
"""

import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
import video_utils as vutils

msg1 = "Welcome to ref_contour_finder.\n"+\
       "This script will help you identify a contour\n"+ \
       "located in a frame to use it as a reference of your object. Try to choose one\n"+\
       "as ideal as possible, and then press S to export it. The same reference\n"+\
       "can then be used throughout different videos.\n"\
       "Press P to skip a frame\n"+\
       "Press S to export the current contour\n"+\
       "Press any other key to skip contour\n"+\
       "Press Q to quit."
       

    
PROJECT_NAME    = 'video_grad_5mM_sinCond_50ulOP50_2206091243_000'# 'SampleVideo'       
BLOB_FILENAME   = os.path.join(PROJECT_NAME, 'video_data_blobs.pkl' ) 
OUTPUT_FILENAME = os.path.join(PROJECT_NAME, 'video_reference_contour')
#536 0

print(msg1)
print('')
print('Loading data from %s...' % BLOB_FILENAME )
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f) 
    n_frames = len(CONTOURS)



_frame = 0
_running = True

while _running:
    _frame += 1
    if _frame > n_frames:
        break
    
    contours = CONTOURS[_frame]

    # Reconstruct frame
    frame = np.zeros((1944,2594), dtype='uint8')
    frame = cv2.drawContours(frame, contours ,-1, (255) , -1, cv2.LINE_AA )
    frame = cv2.cvtColor( frame, cv2.COLOR_GRAY2BGR )

    
    for kk, cnt in enumerate(contours):
        _corner = np.min(cnt, axis=0)
        cnt = cnt - _corner
        _size = np.max( (50, np.max(cnt)))
        
        # Image "img" contains a zoom of only the specific contour
        img = np.zeros( (_size,_size,3), dtype='uint8')
        img = cv2.drawContours(img, [ cnt ], -1, (255,255,255) , -1, cv2.LINE_AA )
    
    
        # Prepare left side
        out_l = frame.copy()
        cv2.drawContours(out_l, [cnt+_corner] ,-1, (0,255,0) , -1, cv2.LINE_AA )    
        out_l = cv2.resize( out_l, (800,600) )
        cv2.putText( out_l, "frame : %d" % _frame, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
        
        
        # Prepare right side
        out_r = cv2.resize(img, (600,600))
        cv2.putText( out_r, "cnt : %d" % kk, (10,550), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
        
        
        # Prepare display frame
        output = cv2.hconcat( [out_l, out_r] )
        cv2.imshow('wdw', output)
        key = cv2.waitKey(0)
        
        if key == ord('p'):
            break
        if key == ord('s'):
            cnt_ref = cnt
            vutils.export(OUTPUT_FILENAME+'.pkl', cnt)
            print('Contour %d - %d exported!' % (_frame, kk) )
            
            plt.figure(figsize=(5,5), dpi=300)
            plt.imshow(out_r)
            plt.xlim(0,599)
            plt.ylim(0,599)
            plt.xticks( plt.xticks()[0], labels='')
            plt.yticks( plt.yticks()[0], labels='')
            plt.savefig( OUTPUT_FILENAME+'.png', dpi=300)
            
            
        if key == ord('q'):
            _running = False
            break
        
cv2.destroyAllWindows()



