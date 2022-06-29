# -*- coding: utf-8 -*-
"""
@author: Pablo Yubero

To detect moving objects we typically use background subtraction. 

There are two ways to do background subtraction. 
    1) The traditional one, that is, obtaining a "ground truth" without any,
or as little objects as possible, and then literally subtracting the background
to every frame.
    2) The modern and fancy one. In which the ground truth created dynamically,
thus in the first frames accuracy decreases, but it later improves. The main
benefit of it is that slow "ambiental" changes, like lighting changes, are 
"included" frame by frame in the "dynamic" ground truth.

For more info, please visit 


"""
import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import video_utils as vutils



# Variables que no van a cambiar
VIDEO_PATH  = 'video_grad_5mM_sinCond_50ulOP50_2206091243_000.avi'
# VIDEO_PATH      = 'test_video.avi'
# VIDEO_PATH      = '2206211503_000.avi'
# VIDEO_PATH = 'SampleVideo.avi'
OUTPUT_FILENAME = 'video_data_blobs'
BKGD_FILENAME   = 'video_fondo'
BG_FRAMES   = 199
BG_SKIP     = 10
MIN_AREA    = 10
MAX_AREA    = 40000
BLUR_SIZE   = 5
FORMFACTOR  = 1
MAX_FRAMES  = 999999
USE_MOG = False
GENERATE_BACKGROUND = False

# Output variables
CONTOURS  = [] 

# Style of annotations
font     = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 1
color    = (0, 255, 0)
thickness = 1

wait_time = 1 #... wait time of each frame during the preview



# Create a separate folder
vutils.clear_dir( VIDEO_PATH.split('.')[0] )



# Cargamos el video
video = cv2.VideoCapture( VIDEO_PATH )
ret, frame = video.read()
_h , _w , _ = frame.shape


# ... y cargamos el fondo
if GENERATE_BACKGROUND:
    fondo = vutils.generate_background(video, n_imgs = BG_FRAMES, skip = BG_SKIP )
    cv2.imwrite( os.path.join( VIDEO_PATH.split('.')[0], BKGD_FILENAME+'.png'), fondo )
    
fondo = vutils.load_background( os.path.join( VIDEO_PATH.split('.')[0], BKGD_FILENAME+'.png') )
fondo = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )

#... y detectamos la placa y creamos su máscara
plate = vutils.detect_plate( fondo , size_ratio=1.3)
print('Scale is: %1.2f px/mm' % (plate[2]/55) )

# ... 
plate[2] = int(0.95*plate[2])
mask_plate = np.zeros_like(fondo)
mask_plate = cv2.circle(mask_plate, (plate[0],plate[1]), plate[2], (255,), -1)


# Uncomment to show masked ROI on the background
# output = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
# cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),2)
# cv2.imshow('w', cv2.bitwise_and(output, mask_plate) )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if USE_MOG:
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    
video = cv2.VideoCapture( VIDEO_PATH )

tStart = datetime.now()
while ret:
    ret, frame = video.read()

    if not ret:
        break
    
    
    
    if FORMFACTOR != 1:
        frame = cv2.resize( frame, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
        
    curr_frame = video.get( cv2.CAP_PROP_POS_FRAMES )
    if curr_frame > MAX_FRAMES:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )   

    if USE_MOG:
        gray = backSub.apply(gray)
    else:    
        gray = cv2.subtract(fondo, gray)
    
    gray = cv2.bitwise_and(gray, mask_plate)
    gray = vutils.adjust_gamma( gray, gamma=0.8)
    gray = cv2.medianBlur(gray, BLUR_SIZE)
    gray = vutils.auto_BC(gray)
    gray = vutils.adjust_gamma( gray, gamma=2.0)

    _, thresh = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
    

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8) )
    thresh = cv2.medianBlur(thresh, BLUR_SIZE)

    
    cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntArea = [cv2.contourArea(c) for c in cnt ]
    
    
    filtered_contours = [ c for c in cnt if MIN_AREA < cv2.contourArea(c) < MAX_AREA ]
    CONTOURS.append(  filtered_contours) 
    print( curr_frame, len(cnt), len(filtered_contours) )
    
    
    
    
    output = thresh.copy()
    if len( output.shape ) ==2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR )  
    
    
    _= [cv2.circle( output, vutils.centroid(c),20,(0,255,0),2) for c in filtered_contours]
    
    cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),2)

    # output = vutils.zoom_in( output, [0.5, 0.5], 2)
    output = cv2.resize( output, (800, 600) )
    output = cv2.putText( output, "%d" % curr_frame, (20,40), font, fontsize, color, thickness, cv2.LINE_AA)

    cv2.imshow( 'window', output)
    
    
    key = cv2.waitKey(wait_time)
    
    if key==ord('q'):
        break
    elif key==ord('p'):
        wait_time = (1-wait_time)


cv2.destroyAllWindows()

speed = video.get( cv2.CAP_PROP_POS_FRAMES)/(datetime.now()-tStart).total_seconds()
print('Analysis speed: %1.2f fps' % (speed) )


print('Data exported to %s.' % (OUTPUT_FILENAME+'.pkl') )
with open( os.path.join( VIDEO_PATH.split('.')[0], OUTPUT_FILENAME+'.pkl'), 'wb') as f:
    pickle.dump(CONTOURS, f)   
   
    
   
if USE_MOG :
    print('Frames in background history: %d' % backSub.getHistory())
    print('Number of gaussians in mix:   %d' % backSub.getNMixtures())

# # 5. Vaya, parece que funciona bien, pero los gusanos son muy oscuros!
# # Mas info en: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# # Otsu's thresholding after Gaussian filtering, and some thinning
# blur = cv2.GaussianBlur(final,(BLUR_SIZE,BLUR_SIZE),0)
# ret,thresh = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # Si los gusanos quedan muy thicc, se puede probar a erosionar la imagen binaria
# # Mas info en https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
# #kernel = np.ones((3,3),np.uint8)
# #eroded = cv2.erode( thresh , kernel , iterations = 1)

# cv2.imshow( 'wdw', thresh )
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 6. Intenta contar cuantos gusanos hay:
# # Mas info en https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
# # ... obtenemos los blobs/contornos
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # ... contamos el area total y filtramos los contornos por area
# total_contoured_area = np.sum([ cv2.contourArea(c) for c in contours])
# filtered_contours = [c for c in contours if MIN_AREA<cv2.contourArea(c)<MAX_AREA ]
# worm_areas = [ cv2.contourArea(c) for c in filtered_contours]

# count_naive = len(filtered_contours)
# count_better= total_contoured_area/np.mean(worm_areas)
# error_better= count_better*np.std(worm_areas)/np.mean(worm_areas)
# print('We find approximately %d worms.' % count_naive )
# print('But computing by size we estimate n_worms = %1.2f +/- %1.2f' % (count_better, error_better) )





# # 7. Reproducir video desde el principio restando el fondo
# video = cv2.VideoCapture( VIDEO_PATH )
# ret, frame = video.read()
# for _ in tqdm(range( nframes )):
#     # Convert to gray and subtract
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     final = cv2.subtract(fondo, gray)
    
#     # Blur and threshold
#     blur = cv2.GaussianBlur(final,(BLUR_SIZE,BLUR_SIZE),0)
#     ret,thresh = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
#     # Find contours
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
#     # Count worms after filtering by area
#     total_contoured_area = np.sum([ cv2.contourArea(c) for c in contours])
#     filtered_contours = [c for c in contours if MIN_AREA<cv2.contourArea(c) ]
#     worm_areas = [ cv2.contourArea(c) for c in filtered_contours]
    
#     count_naive = len(filtered_contours)
#     count_better= total_contoured_area/np.mean(worm_areas)
#     error_better= count_better*np.std(worm_areas)/np.mean(worm_areas)
    
#     NWORMS.append( count_better )
    
    
#     newdim = (1280,960)
#     output = frame
#     cv2.drawContours( output, filtered_contours, -1, (0,255,0), cv2.LINE_4, 1 )
#     cv2.imshow( 'wdw', cv2.resize( output, newdim) )
#     if cv2.waitKey(1)==ord('q'):
#         break
    
#     #... read next frame
#     ret, frame = video.read()


# cv2.destroyAllWindows()

# plt.plot(NWORMS)
# plt.xlabel('Time (frame)')
# plt.ylabel('Number of moving worms')

