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


def fast_detect_plate(frame, plate, relMargin=0.02):
''' Recall that HoughCircles returns (x, y, radius) triplets.'''

    circles = cv2.HoughCircles(
                cv2.pyrDown(frame),
                cv2.HOUGH_GRADIENT,
                dp=8,
                minDist=1000,
                param1=50,
                param2=50,
                minRadius=int((1 - relMargin) * plate[2] / 2),
                maxRadius=int((1 + relMargin) * plate[2] / 2)
              )

    if circles.shape[1] > 1:
        dist = (circles[0, :, 0] * 2 - plate[0])**2 + \
               (circles[0, :, 1] * 2 - plate[1])**2

        idx = np.argmin(dist)
    else:
        idx = 0

    return (2 * circles[0, idx, :]).astype('int')


# ... Filenames ...
DIR_PATH = './videos/N2_espontaneo_2303031132_000'
VIDEO_FILENAME = DIR_PATH+'.avi'
OUTPUT_FILENAME = os.path.join(DIR_PATH, 'video_data_blobs.pkl')
BKGD_FILENAME = os.path.join(DIR_PATH, 'video_fondo.png')
ROIS_FILENAME = os.path.join(DIR_PATH, 'rois.pkl')

# ... General parameters ...
SKIP_FRAMES = 1650        # skip a number of initial frames from the video
BG_FRAMES = 100        # number of frames to model the background
BG_SKIP = 20           # number of discarded frames during background creation
MIN_AREA = 10           # min contour area, avoids salt and pepper noise
MAX_AREA = 600          # maximum contour area
BLUR_SIZE = 5          # Kernel size of morphological operations
FORMFACTOR = 1         # form factor of output, typically 1
MAX_FRAMES = 1800    # max frames to process (if long video)
WAIT_TIME = 1          # display speed / wait time
PLATE_SIZE_REL = 0.65  # expected plate size relative to frame size
PLATE_SIZE_TRU = 40    # in mm
R0_PLATE = [0.5, 0.5]  # location estimate of the center of the plate
NORM_CTS = [125, 50]   # frame normalization constants
ZOOM = 1.2             # zoom factor during the preview
USE_MOG = False        # automatic background subtraction using MOG
GENERATE_BKGD = False  # or generate "manual" and static background model
EXPORT_DATA = False

# Other important definitions
KERNEL = np.ones((BLUR_SIZE, BLUR_SIZE)).astype("uint8")


def is_contour_valid(contour):
    '''Checks for valid contour area'''
    return MIN_AREA < cv2.contourArea(contour) < MAX_AREA


def filter_contours(contours):
    '''Filters a list of contours wrt area limits.'''
    return [c for c in contours if is_contour_valid(c)]


# ... Output ...
CONTOURS = []

# ... Style of annotations ...
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSIZE = 1
COLOR = (0, 255, 0)
THICKNESS = 1

# Step 0. Create a separate folder with the project data
vutils.clear_dir(DIR_PATH)

# Step 1. Load video
video = cv2.VideoCapture(VIDEO_FILENAME)
ret, frame = video.read()
assert ret

_h, _w, _ = frame.shape

# Step 2. Generate or load the background
if GENERATE_BKGD:
    print('Generating background...')

    fondo = vutils.generate_background(
        video,
        n_imgs=BG_FRAMES,
        skip=BG_SKIP,
        mad=3,
        processing=lambda frame: vutils.normalize(frame, NORM_CTS)
    )

    cv2.imwrite(BKGD_FILENAME, fondo)
    print('Background saved.')

elif USE_MOG:
    backSub = cv2.createBackgroundSubtractorMOG2()

# ... load background from file
fondo = vutils.load_background(BKGD_FILENAME)
fondo = cv2.resize(fondo, (int(_w*FORMFACTOR), int(_h*FORMFACTOR)))

# Step 3. Detect plate...
# plate = vutils.detect_plate(
#             fondo,
#             size_ratio=PLATE_SIZE_REL,
#             blur_kernel=3,
#             r0=R0_PLATE
#         )
plate = fast_detect_plate(fondo, [1279, 935, 745], relMargin=0.01)
print(f'Scale is: {plate[2]/PLATE_SIZE_TRU:0.2f} px/mm')

# ... and create a mask.
mask_plate = np.zeros_like(fondo)
mask_plate = cv2.circle(mask_plate, (plate[0], plate[1]), int(0.97*plate[2]), (255,), -1)

if EXPORT_DATA:
    with open(ROIS_FILENAME, 'wb') as f:
        pickle.dump([plate,], f)

# Uncomment to show masked ROI on the background
output = cv2.resize(fondo, (int(_w*FORMFACTOR), int(_h*FORMFACTOR)))
output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
cv2.circle(output, (plate[0], plate[1]), plate[2], (0, 255, 0), 5)
cv2.imshow('w', cv2.resize(output, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 4. Start blob extraction
video = cv2.VideoCapture(VIDEO_FILENAME)
n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
assert video.set(1, SKIP_FRAMES)  # Where frame_no is the frame you want


tStart = datetime.now()
#for _ in tqdm(range(int(n_frames))):
for _ in range(int(n_frames)):

    # ... load frame and exit if the video finished
    ret, frame = video.read()

    if not ret:
        break

    # ... resize frame if FORMFACTOR is different than 1
    if FORMFACTOR != 1:
        frame = cv2.resize(frame, (int(_w*FORMFACTOR), int(_h*FORMFACTOR)))

    # ... exit if MAX_FRAMES reached
    curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    if curr_frame > MAX_FRAMES:
        break

    # ... convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = vutils.normalize(gray, NORM_CTS)


    # ... subtract background
    if USE_MOG:
        gray = backSub.apply(gray)
    else:
        gray = cv2.subtract(fondo, gray)

    # >>> PREPROCESSING I <<<
    gray = cv2.bitwise_and(gray, mask_plate)
    # gray = vutils.adjust_gamma(gray, gamma=0.8)
    gray = cv2.medianBlur(gray, BLUR_SIZE)
    # gray = vutils.adjust_gamma(gray, gamma=3.0)
    gray = vutils.auto_BC(gray)

    # >>> PREPROCESSING II <<<
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, KERNEL)

    # ... find and export contours
    cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntArea = [cv2.contourArea(c) for c in cnt]

    # ... filter contours by area, and append them to the output variable
    filtered_contours = filter_contours(cnt)
    CONTOURS.append(filtered_contours)

    # >>> PREVIEW <<<
    # output = frame.copy()
    # output = gray # cv2.subtract(255, gray.copy())
    output = thresh.copy()

    # ... convert to color if output is in grayscale
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # ... draw a circle around every good contour
    _ = [cv2.circle(
            output,
            vutils.centroid(c),
            20,
            (0, 255, 0),
            2
        ) for c in filtered_contours
        ]

    # ... draw a circle showing the estimated position of the plate
    cv2.circle(
        output,
        (plate[0], plate[1]),
        plate[2],
        (0, 255, 0),
        5
    )

    # ... prepare output frame
    output = vutils.zoom_in(output, [0.5, 0.5], ZOOM)
    output = cv2.resize(output, (800, 600))
    output = cv2.putText(output,
                         f"{curr_frame}",
                         (20, 40),
                         FONT,
                         FONTSIZE,
                         COLOR,
                         THICKNESS,
                         cv2.LINE_AA
                         )

    cv2.imshow('window', output)

    # ... keep loop running until Q or P are pressed
    key = cv2.waitKey(WAIT_TIME)
    if key == ord('q'):
        break
    elif key == ord('p'):
        WAIT_TIME = (1-WAIT_TIME)

# The loop finished
cv2.destroyAllWindows()

# Print some speed statistics
total_frames = video.get(cv2.CAP_PROP_POS_FRAMES) - SKIP_FRAMES
total_time = (datetime.now() - tStart).total_seconds()
speed = total_frames / total_time
print(f'\nAnalysis speed: {speed:0.2f} fps')


if EXPORT_DATA:
    print(f'Data exported to {OUTPUT_FILENAME}.')
    with open(OUTPUT_FILENAME, 'wb') as f:
        pickle.dump(CONTOURS, f)
else:
    print('Data not exported.')

# Assess worm sizes
areas = [cv2.contourArea(cnt) for C in CONTOURS for cnt in C]
plt.hist(areas, bins=30)
plt.show()
print(f"Worm size  5th quantile: {np.quantile(areas, q=0.05)}")
print(f"Worm size 50th quantile: {np.quantile(areas, q=0.50)}")
print(f"Worm size 95th quantile: {np.quantile(areas, q=0.98)}")

# Uncomment to display an initial plot w/ the number of
# detected worms in each frame.
# nworms = np.array([len(c) for c in CONTOURS] )
# plt.figure( figsize=(6,4), dpi=300)
# plt.plot( nworms)
# plt.xlabel('Number of frames')
# plt.ylabel('Number of worms')
# plt.yscale('log')
# print( np.mean( nworms[-20:]), np.std(nworms[-20:]))
