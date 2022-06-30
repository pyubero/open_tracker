# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 18:48:01 2022

@author: Pablo
"""


import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp


BLOB_FILENAME = 'data_analysis.pkl'

print('Loading data...')
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f) 


n_frames = len(CONTOURS)




def logHu(contour):
    return cv2.HuMoments( cv2.moments(contour))[:,0]

def metric(hu, hu_ref):
    # return np.corrcoef( hu, hu_ref)[0,1]**2
    return np.sqrt(np.sum((hu-hu_ref)**2))

def is_worm(contour, cnt_ref, threshold=0.05, size=None):
    area = cv2.contourArea(contour)
    
    # First check size
    if size is not None:
        if  area>np.min(size) or area>np.max(size):
            return False
        
    # Then check Hu moments
    r2 = metric( logHu(contour), logHu(cnt_ref) )
    if r2<threshold:
        return True
    else:
        return False
    


ref_cnt = CONTOURS[263][0]
logHu_ref = logHu(ref_cnt)


for jj in range(n_frames):
    contours = CONTOURS[jj]

    # Reconstruct frame
    frame = np.zeros((1944,2594), dtype='uint8')
    frame = cv2.drawContours(frame, contours ,-1, (255) , -1, cv2.LINE_AA )
    frame = cv2.cvtColor( frame, cv2.COLOR_GRAY2BGR )
    #... add frame number flag
    cv2.putText( frame, "frame : %d" % jj, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    print('Frame %d' % jj)
    # Compute Humoments correlations
    for cnt in contours:
        #... compute centroid
        M = cv2.moments(cnt)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        
        r2 = metric( logHu(cnt), logHu_ref)
        print(r2)
        if r2<0.1:
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText( frame, "%1.2f" % r2 , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    
    frame = cv2.resize(frame, (800,600) )
    cv2.imshow('wdw', frame)
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()
        break



# cnt_all = []
# lbl_all = []


# for jj in range(n_frames):
#     contours = CONTOURS[jj]
#     labels_frame = []    
#     # Reconstruct frame
#     frame = np.zeros((1944,2594), dtype='uint8')
#     frame = cv2.drawContours(frame, contours ,-1, (255) , -1, cv2.LINE_AA )
#     frame = cv2.cvtColor( frame, cv2.COLOR_GRAY2BGR )
#     cv2.putText( frame, "frame : %d" % jj, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

#     # cv2.imshow('wdw', frame)
#     # if cv2.waitKey(0)==ord('q'):
#     #     cv2.destroyAllWindows()
#     #     break
    
#     for cnt in contours:
#         _corner = np.min(cnt, axis=0)
#         cnt = cnt - _corner
#         _size = np.max( (50, np.max(cnt)))
        
#         img = np.zeros( (_size,_size,3), dtype='uint8')
#         img = cv2.drawContours(img, [ cnt ], -1, (255,255,255) , -1, cv2.LINE_AA )
    
#         # Prepare left side
#         out_l = frame.copy()
#         cv2.drawContours(out_l, [cnt+_corner] ,-1, (0,255,0) , -1, cv2.LINE_AA )    
#         out_l = cv2.resize( out_l, (800,600) )
        
        
#         out_r = cv2.resize(img, (600,600))
#         output = cv2.hconcat( [out_l, out_r] )
#         cv2.imshow('wdw', output)
#         key = cv2.waitKey(0)
        
#         if key==ord('q'):
#             break
#         elif key==ord('y'):
#             labels_frame.append(1)
#             lbl_all.append(1)
#             cnt_all.append(cnt)
    
#         elif key==ord('n'):
#             labels_frame.append(0)
#             lbl_all.append(0)
#             cnt_all.append(cnt)
        
#         elif key == ord('i'):
#             labels_frame.append(2)
#             lbl_all.append(2)
#             cnt_all.append(cnt)
            
            



# lbl_all = np.array( lbl_all)
# cnt_all = np.array( cnt_all)
# mom_all = np.zeros((24, len(lbl_all)))
# hu_all  = np.zeros((7, len(lbl_all)))

# for jj, cnt in enumerate(cnt_all):
#     M = cv2.moments(cnt)
#     mom_all[:,jj] = [value for key, value in M.items() ]
#     hu_all[:,jj]  = cv2.HuMoments(M)[:,0]

# idx0 = np.argwhere( lbl_all==0)[:,0]
# idx1 = np.argwhere( lbl_all==1)[:,0]
# idx2 = np.argwhere( lbl_all==2)[:,0]


# indices= [18,17]
# plt.plot(mom_all[ indices[0],idx0], mom_all[ indices[1], idx0],'o')
# plt.plot(mom_all[ indices[0],idx1], mom_all[ indices[1], idx1],'o')
# plt.plot(mom_all[ indices[0],idx2], mom_all[ indices[1], idx2],'o')
# plt.legend()


# for jj in range(7):
#     xobs = np.mean(hu_all[jj, idx1])
#     xexp = np.mean(hu_all[jj, idx0])
#     xstd = np.std(hu_all[jj, idx0])
    
#     Z_score = (xobs-xexp)/xstd
#     ks2 = ks_2samp(hu_all[jj,idx0], hu_all[jj, idx1]).pvalue
#     print(jj, "%1.4f\t" % Z_score,  ks2)

# for jj in range(24):
#     xobs = np.mean(mom_all[jj, idx1])
#     xexp = np.mean(mom_all[jj, idx0])
#     xstd = np.std(mom_all[jj, idx0])

#     Z_score = (xobs-xexp)/xstd
#     ks2 = ks_2samp(mom_all[jj,idx0], mom_all[jj, idx1]).pvalue
#     if abs(Z_score)>1:
#         print(jj, "%1.4f\t" % Z_score, ks2 )



