# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:33:00 2022

@author: Pablo
"""
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm


#.............................................................................#
class Worm:
    def __init__(self, starting_position, t0=0, dtype='uint16'):
        self.t0= t0
        self.x = starting_position[0]*np.ones((4,), dtype=dtype)
        self.y = starting_position[1]*np.ones((4,), dtype=dtype)
        self.dtype = dtype
        
    def coordinates(self, mode='cartesian'):
        if mode =='cartesian':
            return np.array( [ self.x, self.y ] ).T
        if mode =='polar':
            r = np.sqrt(np.sum(self.coordinates()**2, axis=1))    
            z = np.arctan2( self.y, self.x)
            return np.array( [r, z]).T
    
    def speed(self):
        return np.array( [np.diff(self.x), np.diff(self.y)] ).T
    
    def speed_module(self):
        return np.sqrt(np.sum(self.speed()**2, axis=1))
    
    def speed_angle(self):
        return np.arctan2( np.diff(self.y), np.diff(self.x) )
    
    def update(self, new_position):
        self.x = np.append( self.x, new_position[0] )
        self.y = np.append( self.y, new_position[1] )
        
    def expected_position(self, alpha=1):
        return self.coordinates()[-1] + alpha*self.speed()[-1]
        




#.............................................................................#
class MultiWormTracker:
    def __init__(self, max_step=100, n_step=10, speed_damp=0.5, is_worm = None, verbose=False):
        self.WORMS = []
        
        self.MAX_STEP = max_step
        self.N_STEP   = n_step
        self.SPEED_DAMP= speed_damp
        self.IS_WORM   = is_worm
        self.VERBOSE   = verbose
        self.N_CALLS   = -1
    
    def __centroid(self, contour):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY= int(M["m01"] / M["m00"])
        return [cX,cY]
        
    
    def update(self, contours, VERBOSE = True):
        self.N_CALLS += 1
        
        # I need the centroid of blobs available in time t 
        #        the  area    of blobs available in time t
        #        the worms present at time t-1
        #        the speed of worms at time t-1
        radii = np.linspace( 0 , self.MAX_STEP , self.N_STEP )
        blobs_t  = np.array( [ self.__centroid(c) for c in contours] )
        
    
        # jj es el worm index de los que FIJO existen t-1 y por tanto en t
        # idx es el worm index de los worms putativos en t
        blobs_t_used = []
        blobs_t_disp = blobs_t.copy()
        worms_t_1_disp= np.array(range(len(self.WORMS)))
        
    
        ###### 1. Progressive scanning ######
        # By looping across radii, I do a progressive scan to identify proximal worms
        for _radius in radii:
            
            # Then for each worm that has not been tracked...        
            for worm_jj in np.random.permutation( worms_t_1_disp.copy() ):
                
                #... I compute its distance to all blobs
                _worm_expected = self.WORMS[worm_jj].expected_position( alpha = self.SPEED_DAMP)
                dist = np.sqrt( np.sum( ( _worm_expected - blobs_t_disp )**2, axis=1) )
                
                #... if any blob is closer than _radius
                if np.any( dist < _radius ):
                    
                    #... identify the most proximal blob
                    idx = np.argmin(dist)
                    
                    #... link worm_jj to blob idx
                    self.WORMS[ worm_jj ].update( blobs_t[idx] )
                    
                    #... and then make blob_t[idx] unavailable and mark idx as used
                    blobs_t_used.append(idx)
                    blobs_t_disp[idx][0] = 999999
                    blobs_t_disp[idx][1] = 999999
                    
                    #... and delete worm_jj from the "to-track" list
                    worms_t_1_disp = np.delete( worms_t_1_disp, np.where(worms_t_1_disp==worm_jj) ) 
                    
                    # Print some output
                    if self.VERBOSE:
                        print('--- Linked worm %d to blob %d at distance %1.1f' % ( worm_jj, idx, dist[idx]) )
        
        # Then, there will still be some un-tracked worms, waiting... in the dark...
        if self.VERBOSE:
            print('Untracked worms:', worms_t_1_disp)
        
        ###### 2. Naive tracking ######
        # BUT, untracked worms that are suspiciously close to unused blobs (perhaps large ones) should be linked together            
        for worm_jj in np.random.permutation( worms_t_1_disp):
            _worm_expected = self.WORMS[worm_jj].expected_position( alpha = self.SPEED_DAMP)
            dist = np.sqrt( np.sum( ( _worm_expected - blobs_t )**2, axis=1) )
            
            if np.any( dist < self.MAX_STEP ):
                #... link worm_jj to blob idx
                idx = np.argmin(dist)
                self.WORMS[ worm_jj ].update( blobs_t[idx] )
                if self.VERBOSE:
                    print('--- Linked worm %d to blob %d at distance %1.1f' % (worm_jj, idx, dist[idx]) )
    
        

        ###### 3. New worm identification ######
        # These are the blobs that are potentially new, solitary, worms
        blobs_left = [jj for jj in range(len(blobs_t)) if jj not in blobs_t_used ]
        if self.IS_WORM is None:
            # Then consider all blobs news worms
            for blob_jj in blobs_left:
                cnt = contours[blob_jj]
                _position = self.__centroid(cnt)
                self.WORMS.append( Worm(_position, t0 = self.N_CALLS) )
            
            
        else:
            # Otherwise, check with is_worm to filter likely worms
            for blob_jj in blobs_left:
                cnt = contours[blob_jj]
                if self.IS_WORM(cnt):
                    _position = self.__centroid(cnt)
                    self.WORMS.append( Worm(_position, t0 = self.N_CALLS) )
            
         
        










def traj2matrix(trajectories):
    nworms = len(trajectories)
    ntimes = len(trajectories[0].x)
    
    output = np.zeros( (2, nworms, ntimes) )
    for worm_jj, worm in enumerate(trajectories):
        l = len(worm.x)
        output[0, worm_jj, -l:] = worm.x
        output[1, worm_jj, -l:] = worm.y
    return output






def generate_background( videoCapture, n_imgs = 0, skip=0, processing = None ):
    nframes = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))-2
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
        
        # skip some frame
        for _ in range(skip):
            ret, frame = videoCapture.read()
            
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
    if circles is None:
        return [0,0,99999]
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

def clear_dir(dirpath, delete = False):
    if os.path.isdir( dirpath):
        print('Folder %s found...' % dirpath)
        if delete:
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
    
