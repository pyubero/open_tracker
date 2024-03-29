# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:59:54 2022

@author: Pablo
"""
import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import pytracker.video_utils as vutils

# ... Filenames ...
DIR_NAME          = './videos/N2_espontaneo/Analysis'
BLOB_FILENAME     = os.path.join( DIR_NAME, 'video_data_blobs.pkl')
BLOB_REF_FILENAME = os.path.join( DIR_NAME, 'video_reference_contour.pkl')
OUTPUT            = os.path.join( DIR_NAME, './video_likely_worms')

# ... General parameters ...
NMAX              = 100
HU_THRESH         = 1e-8


print('Loading data from %s...' % BLOB_FILENAME)
with open( BLOB_FILENAME, 'rb') as f:
    CONTOURS = pickle.load( f)
    n_frames = len(CONTOURS)

print('Loading reference blob...')
with open( BLOB_REF_FILENAME, 'rb') as f:
    CNT_REF = pickle.load(f)


# Compute the metric for all contours
hu_ref = vutils.logHu( CNT_REF, HU_THRESH )
metric_all, area_all = [], []

for contours in tqdm(CONTOURS):
    for c in contours:
        metric_all.append( vutils.metric( vutils.logHu(c, HU_THRESH), hu_ref)  )
        area_all.append( cv2.contourArea(c) )

metric_all = np.array(metric_all)
area_all = np.array(area_all)


# Fit histogram to a gaussian mixture, to
# ... extract the first component
gmm = GaussianMixture(6,
                     covariance_type='full',
                     random_state=0).fit(  np.expand_dims(metric_all, -1) )
idx = np.argmin( gmm.means_ )
ref_metric= gmm.means_[idx,0] -0.5*np.sqrt(gmm.covariances_[idx,0])


# Then fit the resulting histogram of areas to another gaussian
idc = np.argwhere(metric_all<ref_metric)[:,0]
gmm2 = GaussianMixture(2,
                     covariance_type='full',
                     random_state=0).fit(  np.expand_dims(area_all[idc], -1) )


# Find gaussian that is closest to reference contour area
idx = np.argmin( np.abs(gmm2.means_ - cv2.contourArea(CNT_REF)))
ref_area_min = gmm2.means_[idx,0] - np.sqrt(gmm2.covariances_[idx,0])
ref_area_max = gmm2.means_[idx,0] + np.sqrt(gmm2.covariances_[idx,0])


# >>> FIGURE 1 <<<
# Print the histogram of metrics
plt.figure( figsize=(12,4), dpi=100)
plt.subplot(1,2,1)
plt.hist(metric_all, bins=np.linspace(metric_all.min(), metric_all.max(),1000))
plt.yscale('log')

for mean in gmm.means_[:,0]:
    plt.vlines(mean, 1, 1000,'r')
plt.vlines( ref_metric, 1, 500, 'g')
plt.xlabel('Distance from reference')
plt.ylabel('Probability density')

plt.subplot(1,2,2)
plt.hist(area_all[idc], bins=100)
plt.yscale('log')

for mean in gmm2.means_[:,0]:
    plt.vlines(mean, 1, 1000,'r')
plt.vlines( ref_area_min, 1, 500, 'g')
plt.vlines( ref_area_max, 1, 500, 'g')
plt.xlabel('Area')
plt.show()


# >>> FIGURE 2 <<<
# Plot many examples of nicely matching contours.
plt.figure( figsize=(20,20), dpi=300 )
total_found = 0
for contours in CONTOURS:
    for c in contours:
        if vutils.is_worm(c, hu_ref, ref_metric, ref_area_min, ref_area_max, hu_thold=HU_THRESH) and total_found<100:
            total_found += 1

            _img = np.zeros((50,50), dtype='uint8')
            cv2.drawContours( _img, [c-np.min(c, axis=0) ], -1, (255), -1, cv2.LINE_AA)

            plt.subplot(10,10, total_found)
            plt.imshow(_img)
            plt.xticks( plt.xticks()[0],  labels='')
            plt.yticks( plt.yticks()[0],  labels='')
            plt.xlabel( "%1.4f - %d" % (vutils.metric( vutils.logHu(c, HU_THRESH), hu_ref) , cv2.contourArea(c) ) )
            plt.xlim(0,50)
            plt.ylim(0,50)

plt.tight_layout()
plt.savefig( OUTPUT+'.png', dpi=300 )
print('Figure saved!')
print('---------------------')
print('Your parameters are:')
print('metric < %1.4f' % ref_metric )
print('Area  >= %d' % ref_area_min)
print('Area  <= %d' % ref_area_max)
