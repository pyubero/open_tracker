# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:33:00 2022

@author: Pablo
"""

import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm




def generate_background( videoCapture, n_imgs = 0, processing = None ):
    nframes = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)/5)
    if n_imgs <= 0:
        n_imgs = nframes
    elif n_imgs > nframes:
        n_imgs = nframes

    # Cargamos el primer frame
    ret, frame = videoCapture.read()

    #... y lo pasamos a grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if processing is not None:
        gray = processing(gray)


    #... y obtenemos el tamaño del video
    width, height = gray.shape

    # Creamos el modelo del fondo
    fondo_ = np.zeros((width, height, n_imgs))
    fondo_[:,:,0] = gray
    
    for ii in tqdm( range( n_imgs-1) ):
        ret, frame = videoCapture.read()                      # leer siguiente frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasar a grises
        
        if processing is not None:
            gray = processing(gray)
            
        fondo_[:,:,1+ii] = gray
    
    # 2. Calcular el fondo como la mediana de fondo_ sobre el eje temporal (axis=2)
    fondo = np.median( fondo_, axis=2).astype('uint8')
    
    return fondo

def load_background( filename  ):
    fondo = cv2.imread(filename)
    fondo_gray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY )
    return fondo_gray
    
def auto_BC(frame):
    alpha = 255 / (frame.max()- frame.min())
    beta = - frame.min()*alpha
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def adjust_gamma(image, gamma=1.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table) 

def detect_plate(frame, size_ratio=1.0, mindist=100):
    height, width = frame.shape
    exp_radius = height/2*size_ratio

    circles = cv2.HoughCircles(frame,cv2.HOUGH_GRADIENT,
                               dp=2,
                               minDist=mindist,
                               param1=50,
                               param2=30,
                               minRadius= int(0.8*exp_radius) ,
                               maxRadius= int(1.2*exp_radius) )
    
    true_circles=[]
    for c in circles[0,:]:
        if (0.8*width/2)<c[0]<(1.2*width/2):
            if (0.8*height/2)<c[1]<(1.2*height/2):
                true_circles.append(c)
    true_circles= np.array(true_circles)
    print('Found %d good candidates' % len(true_circles))
    
        
    idx = np.argmin( true_circles[:,2]-exp_radius*size_ratio)
    return true_circles[idx].astype('int')
    
def zoom_in(frame, center, formfactor):
    height, width = frame.shape[:2]    
    
    DeltaX = width/2/formfactor
    DeltaY = height/2/formfactor
    
    # If center is provided in relative units, compute absolute values
    if (0<center[0]<1) and (0<center[1]<1):
        center[0] = center[0]*width
        center[1] = center[1]*height
    
    xini, xfin = int(center[0]-DeltaX), int(center[0]+DeltaX)
    yini, yfin = int(center[1]-DeltaY), int(center[1]+DeltaY)

    if xini <= 0:
        xini, xfin = int(0), int(2*DeltaX+1)
    if xfin >= width:
        xini, xfin = int( width-2*DeltaX-1), int(width)
        
    if yini <= 0:
        yini, yfin = int(0), int(2*DeltaY+1)
    if yfin >=height:
        yini, yfin = int( height-2*DeltaY-1), int(height)        
        
        
    output = cv2.resize( frame[yini:yfin, xini:xfin][:] , (width, height) )
    return output

def contours_to_list(contours):
    return [  [ [ int(pts[0][0]), int(pts[0][1])] for pts in c ] for c in contours ]

def list_to_contours(contours_list):
    return [  np.array(c) for c in contours_list ]


def export(filename, contour):
    with open(filename, 'wb+') as f:
        pickle.dump(contour, f)

def logHu(contour, thold=0):
    value = cv2.HuMoments( cv2.moments(contour))[:,0]
    sign = np.sign(value)
    return sign*np.log( thold + np.abs(value) )

def metric(hu, hu_ref):
    #The metric always has to satisfy that the closer to 0 the better
    #...
    # return 1 - np.corrcoef( hu, hu_ref)[0,1]
    return np.sqrt(np.sum((hu-hu_ref)**2))

def is_worm(contour, hu_ref, ref_metric, area_min, area_max, hu_thold=0):
    A = metric(logHu(contour, thold=hu_thold), hu_ref) < ref_metric
    B = area_min<cv2.contourArea(contour)<area_max
    if A and B:
        return True
    else:
        return False

def load_contour_from_image(filename, idx=0):
    _ref = cv2.imread( filename, 0)
    _, thresh = cv2.threshold(_ref, 5, 255, cv2.THRESH_BINARY )
    cnt, _    = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnt[idx]

def clear_dir(dirpath):
    if os.path.isdir( dirpath):
        print('Folder %s found...' % OUTPUT)
        print('... deleting files.')
        files = os.listdir(dirpath)
        _ = [os.remove( os.path.join( dirpath, file) ) for file in files ]    
    else:    
        os.mkdir(dirpath)
        print("New directory %s created." % dirpath)

def centroid(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY= int(M["m01"] / M["m00"])
    return [cX,cY]
    
