# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:04:51 2022

@author: logslab
"""


import cv2


FILENAME = 'video_000.avi'
FORMFACTOR = 0.25

# Open video
cv  = cv2.VideoCapture( FILENAME )

#... get first frame
suc, frame = cv.read()

#... and resize it to find its final size
ff_frame = cv2.resize( frame, None , fx=FORMFACTOR, fy=FORMFACTOR )


height , width , c = ff_frame.shape



# Create Video Writer
out_filename = '.'.join( [ FILENAME.split('.')[0]+'_ff%1.2f' % FORMFACTOR, FILENAME.split('.')[-1]] )
out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))


#... write first frame
out.write(ff_frame)


# Then loop...
while True:
    suc, frame = cv.read()
    if not suc: break

    ff_frame = cv2.resize( frame, None , fx=FORMFACTOR, fy=FORMFACTOR )    
    out.write(ff_frame)
    
cv.release()
out.release()
















