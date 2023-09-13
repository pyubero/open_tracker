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
from tqdm import tqdm
from datetime import datetime
from pytracker import video_utils as vutils
from matplotlib import pyplot as plt


# ... Filenames ...
DIR_PATH = './videos/N2_espontaneo/'
# VIDEO_FILENAME  = DIR_PATH+'.avi'
OUTPUT_DIR = os.path.join(DIR_PATH, 'Analysis/')
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'video_data_blobs.pkl')
BKGD_FILENAME = os.path.join(OUTPUT_DIR, 'video_fondo_#.png')
ROIS_FILENAME = os.path.join(OUTPUT_DIR, 'rois.pkl')

# ... General parameters ...
SKIP_INI_FRAMES = 0     # ... skip a number of initial frames from the video
BG_FRAMES = 100         # ... # of frames to model the background
BG_SKIP = 34            # ... # of discarded frames during background creation
MIN_AREA = 40           # ... min contour area, avoids salt and pepper noise
MAX_AREA = 400          # ... maximum contour area (typical worm area ~100px)
BLUR_SIZE = 5           # ... filtering Kernel size
FORMFACTOR = 1          # ... form factor of output, typically 1
MAX_FRAMES = 999999     # ... max # of frames (if long video)
WAIT_TIME = 1           # ... wait time between preview frames
PLATE_SIZE = 0.65       # ... expected plate size relative to frame size
R0_PLATE = [0.5, 0.5]
CHUNK_SIZE = 0.08
NORM_CTS = [125, 50]    # frame normalization constants
ZOOM = 1.2              # ... zoom factor during the preview
TRUE_PLATE_SIZE = 20    # ... true arena radius in mm
EXPORT_DATA = True

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
STYLE_ARGS = [
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    1,
    cv2.LINE_AA
]

# Step -1. Analyze DIR_PATH to find multiple videos with the same tag.
if os.path.isdir(DIR_PATH):
    VIDEO_LIST = [_ for _ in os.listdir(DIR_PATH) if _.split('.')[-1] == 'avi']
    VIDEO_LIST.sort()
    print(f'Found {len(VIDEO_LIST)} videos in {DIR_PATH}.')

# Step 0. Create a separate folder with the project data
vutils.clear_dir(OUTPUT_DIR)

# Step 1. Load first video
video = cv2.VideoCapture(os.path.join(DIR_PATH, VIDEO_LIST[1]))
ret, frame = video.read()
_h, _w, _ = frame.shape

# Step 2. Load background
fondo = vutils.load_background(BKGD_FILENAME.replace("_#",""))
fondo = cv2.resize(fondo, (int(_w*FORMFACTOR), int(_h*FORMFACTOR)))

# Step 3. Detect plate...
plate = vutils.detect_plate(
            fondo,
            size_ratio=PLATE_SIZE,
            blur_kernel=3,
            r0=R0_PLATE
        )
print(f'Scale is: {plate[2]/TRUE_PLATE_SIZE:0.2f} px/mm')

# Step 3b. ... detect chunk...
chunk = np.zeros(3,).astype('int')
# chunk = vutils.detect_plate(
#           fondo,
#           size_ratio=CHUNK_SIZE,
#           blur_kernel=3,
#           r0=[0.3, 0.5]
#         )
# print('Scale is: %1.2f px/mm' % (chunk[2]/5) )

# ... and create a mask.
mask_plate = np.zeros_like(fondo)
mask_plate = cv2.circle(
                mask_plate,
                (plate[0], plate[1]),
                int(plate[2]),
                (255,),
                -1
            )
# mask_plate = cv2.circle(
#               mask_plate,
#               (chunk[0], chunk[1]),
#               chunk[2],
#               (255,),
#               -1
#              )

if EXPORT_DATA:
    with open(ROIS_FILENAME, 'wb') as f:
        pickle.dump([plate, chunk], f)

# Uncomment to show masked ROI on the background
output = cv2.resize(fondo, (int(_w*FORMFACTOR), int(_h*FORMFACTOR)))
output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
cv2.circle(output, (plate[0], plate[1]), plate[2], (0, 255, 0), 5)
cv2.circle(output, (chunk[0], chunk[1]), chunk[2], (0, 0, 255), 5)
cv2.imshow('w', cv2.resize(output, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 4. Start blob extraction
#VIDEO_FILENAME = VIDEO_LIST[0]
for ivideo, VIDEO_FILENAME in enumerate(VIDEO_LIST):
    video = cv2.VideoCapture(os.path.join(DIR_PATH, VIDEO_FILENAME))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Generate background
    fondo = vutils.generate_background(
        video,
        n_imgs=BG_FRAMES,
        skip=BG_SKIP,
        mad=3,
        processing=lambda frame: vutils.normalize(frame, NORM_CTS)
    )
    cv2.imwrite(BKGD_FILENAME.replace("#", f"{ivideo}"), fondo)

    assert video.set(1, 0)

    tStart = datetime.now()
    for _ in tqdm(range(int(n_frames))):

        # ... load frame and exit if the video finished
        ret, frame = video.read()

        if not ret:
            break

        # ... resize frame if FORMFACTOR is different than 1
        if FORMFACTOR != 1:
            frame = cv2.resize(frame, (int(_w*FORMFACTOR), int(_h*FORMFACTOR)))

        # ... compute current frame, and exit if MAX_FRAMES reached
        curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        if curr_frame > MAX_FRAMES:
            break

        # ... convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = vutils.normalize(gray, NORM_CTS)

        # ... subtract background
        gray = cv2.subtract(fondo, gray)

        # >>> PREPROCESSING I <<<
        gray = cv2.bitwise_and(gray, mask_plate)
        gray = cv2.medianBlur(gray, BLUR_SIZE)
        gray = vutils.auto_BC(gray)

        # >>> PREPROCESSING II <<<
        _, thresh = cv2.threshold(
                        gray,
                        0,
                        255,
                        cv2.THRESH_BINARY+cv2.THRESH_OTSU
                    )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL)

        # ... find and export contours
        cnt, _ = cv2.findContours(
                    thresh,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )
        cntArea = [cv2.contourArea(c) for c in cnt]

        # ... filter contours by area, and store them
        filtered_contours = filter_contours(cnt)
        CONTOURS.append(filtered_contours)

        # >>> PREVIEW <<<
        # output = frame.copy()
        output = gray.copy()
        # output = thresh.copy()

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
            )
            for c in filtered_contours]

        # ... draw a circle showing the estimated position of the plate
        cv2.circle(output, (plate[0], plate[1]), plate[2], (0, 255, 0), 5)
        cv2.circle(output, (chunk[0], chunk[1]), chunk[2], (0, 0, 255), 5)

        output = vutils.zoom_in(output, [0.5, 0.5], ZOOM)
        output = cv2.resize(output, (800, 600))
        output = cv2.putText(output, f"{curr_frame}", *STYLE_ARGS)

        cv2.imshow('window', output)
        key = cv2.waitKey(WAIT_TIME)
        if key == ord('q'):
            break

        elif key == ord('p'):
            WAIT_TIME = (1 - WAIT_TIME)

    cv2.destroyAllWindows()

    total_frames = video.get(cv2.CAP_PROP_POS_FRAMES)
    total_time = (datetime.now() - tStart).total_seconds()
    speed = total_frames / total_time
    print(f'\nAnalysis speed: {speed:0.2f} fps')

    if EXPORT_DATA:
        total_contours = [cv2.contourArea(cnt) for C in CONTOURS for cnt in C]

        print(f'Data of {total_contours} contours exported to {OUTPUT_FILENAME}.')

        with open(OUTPUT_FILENAME, 'wb') as f:
            pickle.dump(CONTOURS, f)

    else:
        print('Data not exported.')

nworms = np.array([len(c) for c in CONTOURS])
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(nworms)
plt.xlabel('Number of frames')
plt.ylabel('Number of worms')
plt.yscale('log')
print(np.mean(nworms[-20:]), np.std(nworms[-20:]))
