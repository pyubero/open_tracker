# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:42:23 2022

@author: logslab
"""


import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

BLOB_FILENAME = 'data_analysis.pkl'


print('Loading data...')
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f)   
    

# Extract a broad estimation of worm number with time
nworms = [ len(c) for c in CONTOURS ]
plt.plot(nworms)


# Print coordinates
plt.figure()
# for cnt in CONTOURS:
for jj in tqdm( range(len(CONTOURS))):
    cnt = CONTOURS[jj]
    if len(cnt)>0:
        for blob in cnt:
            M = cv2.moments( blob )
            plt.plot( M['m10']/M['m00'], M['m01']/M['m00'] , 'o', ms=1, alpha=0.1)




# Obtain Hu moments of all contours in frame 800
blobs800 = CONTOURS[100]
num_blobs= len(blobs800)
HU = np.zeros(  (num_blobs, 7 ) )
MOM = np.zeros( (num_blobs, 24) )
for jj, blob in enumerate( blobs800 ):
    M = cv2.moments(blob)
    
    MOM[jj,:] = [ value for key, value in M.items() ]
    HU[jj,:] = cv2.HuMoments(M)[:,0]





x = MOM[:,0]
y = MOM[:,1]

plt.figure( dpi=300)
plt.plot( x, y,'.')
for i in range(num_blobs):
    plt.gca().annotate( i, (x[i], y[i]) )


blob_idx = 200

img = np.zeros( (100,100), dtype='uint8')

_centroid= np.min(blobs800[blob_idx], axis=0)

img = cv2.drawContours(img, [ blobs800[blob_idx]-_centroid,], -1, (255,255,255) , -1, cv2.LINE_AA )

plt.figure()
plt.imshow(img)

moms_img = cv2.HuMoments( cv2.moments( img ))[:,0]
moms_cnt = cv2.HuMoments( cv2.moments( blobs800[blob_idx] ))[:,0]

print(moms_img[-1], moms_cnt[-1])