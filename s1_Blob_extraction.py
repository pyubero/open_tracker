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
import pickle
import numpy as np
from datetime import datetime
from pytracker import video_utils as vutils



# ... Filenames ... 
DIR_PATH  = './videos/video_grad_100mM_sinCond_100ulOP50_000'
VIDEO_FILENAME  = DIR_PATH+'.avi'
OUTPUT_FILENAME = os.path.join( DIR_PATH, 'video_data_blobs.pkl') 
BKGD_FILENAME   = os.path.join( DIR_PATH, 'video_fondo.png')

# ... General parameters ...
BG_FRAMES   = 199      #... number of frames to model the background
BG_SKIP     =  5       #... number of discarded frames during background creation
MIN_AREA    = 10       #... minimum contour area, helps ignore salt and pepper noise
MAX_AREA    = 40000    #... maximum contour area
BLUR_SIZE   = 5        #... Kernel size for blurring and opening/closing operations
FORMFACTOR  = 1        #... form factor of output, typically 1
MAX_FRAMES  = 999999   #... maximum number of frames to process (in case the video is super long)
WAIT_TIME   = 1        #... wait time of each frame during the preview
PLATE_SIZE  = 1.5      #... expected plate size relative to frame size
ZOOM        = 1        #... zoom factor during the preview
USE_MOG       = True  #... activate automatic background subtraction using MOG       
GENERATE_BKGD = False  #... or generate "manual" and static background model

# ... Output ...
CONTOURS  = [] 

#... Style of annotations ...
font     = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 1
color    = (0, 255, 0)
thickness = 1



# Step 0. Create a separate folder with the project data
vutils.clear_dir( DIR_PATH )


# Step 1. Load video
video = cv2.VideoCapture( VIDEO_FILENAME )
ret, frame = video.read()
_h , _w , _ = frame.shape


# Step 2. Generate or load the background
if GENERATE_BKGD:
    fondo = vutils.generate_background(video, n_imgs = BG_FRAMES, skip = BG_SKIP )
    cv2.imwrite( BKGD_FILENAME, fondo )

if USE_MOG:
    backSub = cv2.createBackgroundSubtractorMOG2()


fondo = vutils.load_background( BKGD_FILENAME )
fondo = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )


# Step 3. Detect plate...
plate = vutils.detect_plate( fondo , size_ratio=PLATE_SIZE, blur_kernel=10)
print('Scale is: %1.2f px/mm' % (plate[2]/55) )

# ... and create a mask.
plate[2] = int(0.95*plate[2])
mask_plate = np.zeros_like(fondo)
mask_plate = cv2.circle(mask_plate, (plate[0],plate[1]), plate[2], (255,), -1)


# Uncomment to show masked ROI on the background
# output = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
# cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),2)
# cv2.imshow('w', cv2.bitwise_and(output, mask_plate) )
# cv2.waitKey(0)
# cv2.destroyAllWindows()


    
# Step 4. Start blob extraction
video = cv2.VideoCapture( VIDEO_FILENAME )

tStart = datetime.now()
while True:
    
    #... load frame and exit if the video finished
    ret, frame = video.read()

    if not ret:
        break
    
    #... resize frame if FORMFACTOR is different than 1
    if FORMFACTOR != 1:
        frame = cv2.resize( frame, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
        
    #... compute current frame, and exit if MAX_FRAMES reached        
    curr_frame = video.get( cv2.CAP_PROP_POS_FRAMES )
    if curr_frame > MAX_FRAMES:
        break
    
    #... convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )   

    #... subtract background
    if USE_MOG:
        gray = backSub.apply(gray)
    else:    
        gray = cv2.subtract(fondo, gray)
    
    # >>> PREPROCESSING I <<<
    gray = cv2.bitwise_and(gray, mask_plate)
    gray = vutils.adjust_gamma( gray, gamma=0.8)
    gray = vutils.auto_BC(gray)
    gray = cv2.medianBlur(gray, BLUR_SIZE)
    gray = vutils.adjust_gamma( gray, gamma=2.0)

    # >>> PREPROCESSING II <<<
    _, thresh = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((BLUR_SIZE,BLUR_SIZE), np.uint8) )
    # thresh = cv2.medianBlur(thresh, BLUR_SIZE)

    #... find and export contours
    cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntArea = [cv2.contourArea(c) for c in cnt ]
    
    #... filter contours by area, and append them to the output variable
    filtered_contours = [ c for c in cnt if MIN_AREA < cv2.contourArea(c) < MAX_AREA ]
    CONTOURS.append(  filtered_contours) 
    
    
    
    # >>> PREVIEW <<<
    # output = frame.copy()
    output = gray.copy()
    # output = thresh.copy()
    
    #... convert to color if output is in grayscale
    if len( output.shape ) ==2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR )  
    
    #... draw a circle around every good contour
    _= [cv2.circle( output, vutils.centroid(c),20,(0,255,0),2) for c in filtered_contours]
    
    #... draw a circle showing the estimated position of the plate
    cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),2)

    
    output = vutils.zoom_in( output, [0.5, 0.5], ZOOM)
    output = cv2.resize( output, (1280, 960) )
    output = cv2.putText( output, "%d" % curr_frame, (20,40), font, fontsize, color, thickness, cv2.LINE_AA)

    cv2.imshow( 'window', output)
    key = cv2.waitKey(WAIT_TIME)
    if key==ord('q'):
        break
    elif key==ord('p'):
        WAIT_TIME = (1-WAIT_TIME)





cv2.destroyAllWindows()


speed = video.get( cv2.CAP_PROP_POS_FRAMES)/(datetime.now()-tStart).total_seconds()
print('Analysis speed: %1.2f fps' % (speed) )


print('Data exported to %s.' % (OUTPUT_FILENAME+'.pkl') )
with open( OUTPUT_FILENAME, 'wb') as f:
    pickle.dump(CONTOURS, f)   
   
    