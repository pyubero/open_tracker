
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

from pytracker.video_utils import MultiWormTracker
import pytracker.video_utils as vutils



# ... Filenames ...
DIR_VIDEOS        = './videos/TDC1_espontaneo'
DIR_NAME          = './videos/TDC1_espontaneo/Analysis'
# BLOB_FILENAME     = os.path.join(DIR_NAME, 'video_data_blobs.pkl')
# BLOB_REF_FILENAME = os.path.join(DIR_NAME, 'video_reference_contour.pkl')
TRAJ_FILENAME     = os.path.join(DIR_NAME, 'trajectories.pkl')
NPZ_FILENAME      = os.path.join(DIR_NAME, 'trajectories.pkl.npz')
# IMG_FILENAME      = os.path.join(DIR_NAME, 'trajectories.png')
# BKGD_FILENAME     = os.path.join(DIR_NAME, 'video_fondo.png')
ROIS_FILENAME     = os.path.join(DIR_NAME, 'rois.pkl')




OUTPUT_HEIGHT = 1400
OUTPUT_WIDTH = 1400 #1280
COLORS = np.random.randint(150, size=(3, 1000))
RADIUS = 25
FONTSIZE = 1.5
TRAIL_LEN = 50
iframe = -1

# Load list of videos
if os.path.isdir(DIR_VIDEOS):
    VIDEO_LIST = [_ for _ in os.listdir(DIR_VIDEOS) if _.split('.')[-1] == 'avi']
    VIDEO_LIST.sort()
    print(f'Found {len(VIDEO_LIST)} videos in {DIR_VIDEOS}.')

# Load trajectories file
# ... remove first time entry because it is always duplicate
data = np.load(NPZ_FILENAME)['data'][:, 1:, :]  # 275 * 14313 * 2

# Load ROIS
with open( ROIS_FILENAME, 'rb') as f:
    plate = pickle.load(f)[0]

# Find t0
with open(TRAJ_FILENAME, 'rb') as file:
    T0 = pickle.load(file)[0].t0
    print(f"Found t0 = {T0}.")

# Create output
OUTPUT_CAP = cv2.VideoWriter(
    os.path.join(DIR_NAME, 'analysed.avi'),
    cv2.VideoWriter_fourcc(*"H264"),
    30,
    (int(OUTPUT_WIDTH), int(OUTPUT_HEIGHT)),
    True
)



for ivideo, VIDEO_FILENAME in enumerate(VIDEO_LIST):
# ivideo, VIDEO_FILENAME = 0, VIDEO_LIST[0]

    video = cv2.VideoCapture(os.path.join(DIR_VIDEOS, VIDEO_FILENAME))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    tStart = datetime.now()
    for _ in tqdm(range(int(n_frames))):
        iframe += 1

        ret, frame = video.read()
        _h, _w, _ = frame.shape

        # Find which worms are in the current frame iframe
        valid_worms = np.nonzero(np.sum(data[:, iframe, :], axis=-1) > 0)[0]

        # Draw a small circle around each
        _ = [cv2.circle(frame, (int(data[jj, iframe, 0]), int(data[jj, iframe, 1])), int(RADIUS), COLORS[:, jj].tolist(), 2) for jj in valid_worms ]

        # Write the id next to each worm
        _ = [cv2.putText(frame, "%d" % jj, (int(data[jj, iframe, 0]), int(data[jj, iframe, 1])), cv2.FONT_HERSHEY_SIMPLEX, FONTSIZE, COLORS[:, jj].tolist(), 4, cv2.LINE_AA) for jj in valid_worms]

        # Write the frame number in each frame
        _ = cv2.putText(frame, "%d" % iframe, (int(590), int(300)), cv2.FONT_HERSHEY_SIMPLEX, FONTSIZE, (0,0,255), 3, cv2.LINE_AA)

        # Write filename
        _ = cv2.putText(frame, DIR_VIDEOS, (int(590), int(240)), cv2.FONT_HERSHEY_SIMPLEX, FONTSIZE, (0,0,255), 2, cv2.LINE_AA)

        # Draw a small trail
        tini = np.max((iframe - TRAIL_LEN, 0))
        for jj in valid_worms:
            _ = [cv2.circle(frame, (int(coords[0]), int(coords[1])), 1, COLORS[:, jj].tolist(), -1) for coords in data[jj, tini:iframe, :] if np.all(coords > 0)]


        # Draw square
        y_ini, x_ini = plate[0]-plate[2], plate[1]-plate[2]
        y_fin, x_fin = (plate[0]+plate[2], plate[1]+plate[2])
        # _ = cv2.rectangle(frame, start_point, end_point, (255,0,0), 4)
        cropped = frame[x_ini:x_fin, y_ini:y_fin]


        # Prepare and export output
        output = cv2.resize(cropped, (int(OUTPUT_WIDTH), int(OUTPUT_HEIGHT)))
        OUTPUT_CAP.write(output)

        # Display output
        # cv2.imshow("window", output)
        # if cv2.waitKey(1) == ord("q"):
            # break


# cv2.destroyAllWindows()
OUTPUT_CAP.release()



