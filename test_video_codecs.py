# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:13:10 2022

@author: logslab
"""

import os
import cv2
import numpy
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

N = 100
FORMATS = ['MJPG', 'DIVX', 'DIV4', 'MP42'] # , 

# faltan: h264/5, hevc
# malos: el resto, incluyendo hfyu, 



FRAME = cv2.imread('./test_video/snapshot.png')
FRAME = cv2.cvtColor( FRAME, cv2.COLOR_BGR2GRAY)
FILENAME = './test_video/video.mkv'

SIZES=[]
ERRORS=[]
TIMES = []


color      = int(0)
resolution = (FRAME.shape[1], FRAME.shape[0] )
speed      = 30



for fmt in FORMATS:
    print('----- %4s -----' % fmt )
    fourcc     = cv2.VideoWriter_fourcc( *fmt )

    
    # File output size
    tStart = datetime.now()
    videoWriter = cv2.VideoWriter(FILENAME, fourcc, speed,  resolution, color )
    for jj in tqdm(range(N)):
        videoWriter.write( FRAME )
    videoWriter.release()
    TIMES.append( (datetime.now() - tStart).total_seconds() )
    
    size = os.path.getsize(FILENAME)/1024/1024/N
    SIZES.append( size )

    
    # File quality
    vc = cv2.VideoCapture(FILENAME, 0)
    for jj in tqdm(range(10)):
        ret, frame = vc.read()
        if not ret:
            print('Video too short, minimum of 10 frames.')
            break
    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)    
    
    error = np.mean( (1.0*gray-FRAME)**2 )
    ERRORS.append(error)
    
    
    print('')
    print('Size per frame: %1.3f' % size )
    print('MSE: %1.2f' % error)
    print('time: %ds' % TIMES[-1] )
    
    
    
    
    
    
    











