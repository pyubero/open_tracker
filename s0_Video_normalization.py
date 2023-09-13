# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 08:32:03 2023

@author: Pablo
"""
import os
import pickle
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from pytracker import video_utils as vutils


def preprocess_frame(frame):
    formfactor = 0.25
    _h, _w, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (int(_w*formfactor), int(_h*formfactor)))
    return frame


def normalize(frame, ref):
    muy, sy = ref
    mux = np.mean(frame)
    sx = np.std(frame)

    frame = muy + (sy/sx)*(frame - mux)
    frame[frame < 0] = 0
    frame[frame > 255] = 255
    return frame.astype("uint8")

# ... Filenames ...
DIR_PATH = './videos/N2_espontaneo_2303031132_000'
VIDEO_FILENAME = DIR_PATH+'.avi'

video = cv2.VideoCapture(VIDEO_FILENAME)
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Big changes in frame jj =  1746
assert video.set(1, 1700)  # Where frame_no is the frame you want

ref = [125, 70]

brightness = []
brightness_ = []
for jj in tqdm(range(100)):
    suc, frame = video.read()
    frame = preprocess_frame(frame)
    frame_norm = normalize(frame, ref)

    brightness.append(np.std(frame))
    brightness_.append(np.std(frame_norm))

    cv2.imshow("window", frame_norm)
    if cv2.waitKey(10) == ord("q"):
        break
cv2.destroyAllWindows()

plt.plot(brightness)
plt.plot(brightness_)
plt.show()


