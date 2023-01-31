# -*- coding: utf-8 -*-
"""
@author: Pablo Yubero

To detect moving objects we typically use background subtraction. 

There are two ways to do background subtraction. 
    1) The traditional one, that is, obtaining a "ground truth" without any,
or as little objects as possible, and then literally subtracting the background
to every frame.
    2) The modern and fancy one. In which the ground truth is created dynamically,
thus in the first frames accuracy decreases, but it later improves. The main
benefit of it is that slow "ambiental" changes, like lighting changes, are 
"included" frame by frame in the "dynamic" ground truth.


"""
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pytracker import video_utils as vutils
from matplotlib import pyplot as plt


# ... Filenames ... 
DIR_PATH  = './videos/Carla_EC/Carla_N2_EC_2211101415_002'
VIDEO_FILENAME  = DIR_PATH+'.avi'
OUTPUT_FILENAME = os.path.join( DIR_PATH, 'video_data_blobs.pkl') 
BKGD_FILENAME   = os.path.join( DIR_PATH, 'video_fondo.png')
ROIS_FILENAME   = os.path.join( DIR_PATH, 'rois.pkl')

# ... General parameters ...
SKIP_FRAMES = 0        #... skip a number of initial frames from the video
BG_FRAMES   = 100      #... number of frames to model the background
BG_SKIP     =  20      #... number of discarded frames during background creation
MIN_AREA    = 50       #... minimum contour area, helps ignore salt and pepper noise
MAX_AREA    = 40000    #... maximum contour area
BLUR_SIZE   = 3        #... Kernel size for blurring and opening/closing operations
FORMFACTOR  = 1        #... form factor of output, typically 1
MAX_FRAMES  = 999999   #... maximum number of frames to process (in case the video is super long)
WAIT_TIME   = 1        #... wait time of each frame during the preview
PLATE_SIZE  = 0.65      #... expected plate size relative to frame size
R0_PLATE    = [0.5,0.5]
CHUNK_SIZE  = 0.08
ZOOM        = 1.2        #... zoom factor during the preview
USE_MOG       = False  #... activate automatic background subtraction using MOG       
GENERATE_BKGD = False  #... or generate "manual" and static background model
EXPORT_DATA   = True

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
    print('Generating background...')
    fondo = vutils.generate_background(video, n_imgs = BG_FRAMES, skip = BG_SKIP, mad=3 )
    cv2.imwrite( BKGD_FILENAME, fondo )
    print('Background saved.')
    
if USE_MOG:
    backSub = cv2.createBackgroundSubtractorMOG2()

#... load background from file
fondo = vutils.load_background( BKGD_FILENAME )
fondo = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )


# Step 3. Detect plate...
plate = vutils.detect_plate( fondo , size_ratio=PLATE_SIZE, blur_kernel=3, r0=R0_PLATE)
plate[2] = plate[2]*1.02
print('Scale is: %1.2f px/mm' % (plate[2]/55) )

# ... and create a mask.
plate[2] = int(0.99*plate[2] )
mask_plate = np.zeros_like(fondo)
mask_plate = cv2.circle(mask_plate, (plate[0],plate[1]), plate[2], (255,), -1)

if EXPORT_DATA:
    with open( ROIS_FILENAME, 'wb') as f:
        pickle.dump( [plate,], f) 

# Uncomment to show masked ROI on the background
output = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
output = cv2.cvtColor( output, cv2.COLOR_GRAY2BGR) 
cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),5)
cv2.imshow('w', cv2.resize(output, (800,600) ) )
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 4. Start blob extraction
video = cv2.VideoCapture( VIDEO_FILENAME )
n_frames= video.get( cv2.CAP_PROP_FRAME_COUNT )

tStart = datetime.now()
for _ in tqdm(range(int(n_frames))):
    
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
    elif curr_frame<SKIP_FRAMES:
        continue
    
    #... convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )   

    #... subtract background
    if USE_MOG:
        gray = backSub.apply(gray)
    else:    
        gray = cv2.subtract(fondo, gray)
    
    # >>> PREPROCESSING I <<<
    gray = cv2.bitwise_and(gray, mask_plate)
    gray = vutils.adjust_gamma( gray, gamma=0.8 )
    gray = cv2.medianBlur(gray, BLUR_SIZE)
    gray = vutils.adjust_gamma( gray, gamma=3.0)
    gray = vutils.auto_BC(gray)

    # >>> PREPROCESSING II <<<
    _, thresh = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((BLUR_SIZE,BLUR_SIZE), np.uint8) )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, np.ones((3,3), np.uint8) )
    thresh = cv2.medianBlur(thresh, BLUR_SIZE)

    #... find and export contours
    cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntArea = [cv2.contourArea(c) for c in cnt ]
    
    #... filter contours by area, and append them to the output variable
    filtered_contours = [ c for c in cnt if MIN_AREA < cv2.contourArea(c) < MAX_AREA ]
    CONTOURS.append(  filtered_contours) 
    
    
    # >>> PREVIEW <<<
    # output = frame.copy()
    # output = 255-gray.copy()
    output = thresh.copy()
    
    #... convert to color if output is in grayscale
    if len( output.shape ) ==2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR )  
    
    #... draw a circle around every good contour
    _= [cv2.circle( output, vutils.centroid(c),20,(0,255,0),2) for c in filtered_contours]
    
    #... draw a circle showing the estimated position of the plate
    cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),5)
    cv2.circle( output,(chunk[0], chunk[1]), chunk[2],(0,0,255),5)

    #... prepare output frame
    output = vutils.zoom_in( output, [0.5, 0.5], ZOOM)
    output = cv2.resize( output, (800, 600) )
    output = cv2.putText( output, "%d" % curr_frame, (20,40), font, fontsize, color, thickness, cv2.LINE_AA)

    cv2.imshow( 'window', output)
    
    #... keep loop running until Q or P are pressed
    key = cv2.waitKey(WAIT_TIME)
    if key==ord('q'):
        break
    elif key==ord('p'):
        WAIT_TIME = (1-WAIT_TIME)


# The loop finished
cv2.destroyAllWindows()

# Print some speed statistics
speed = video.get( cv2.CAP_PROP_POS_FRAMES)/(datetime.now()-tStart).total_seconds()
print('Analysis speed: %1.2f fps' % (speed) )


if EXPORT_DATA:
    print('Data exported to %s.' % OUTPUT_FILENAME )
    with open( OUTPUT_FILENAME, 'wb') as f:
        pickle.dump(CONTOURS, f)   
else:
    print('Data not exported.')

# Uncomment to display an initial plot w/ the number of
# detected worms in each frame.    
# nworms = np.array([len(c) for c in CONTOURS] )
# plt.figure( figsize=(6,4), dpi=300)
# plt.plot( nworms)
# plt.xlabel('Number of frames')
# plt.ylabel('Number of worms')   
# plt.yscale('log')
# print( np.mean( nworms[-20:]), np.std(nworms[-20:])) 