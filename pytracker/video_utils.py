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
    def __init__(self, starting_position, t0=0, c=None, dtype='uint16'):
        self.t0= t0
        self.x = starting_position[0]*np.ones((4,), dtype=dtype)
        self.y = starting_position[1]*np.ones((4,), dtype=dtype)
        self.c = [c]
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
        
    def expected_position(self, alpha=1, inertia=1):
        # return self.coordinates()[-1] + alpha*self.speed()[-1]
        return  self.coordinates()[-1] + alpha*np.nanmean(self.speed()[-inertia:], axis=0)
        
    def update_contour(self, contour, autocenter=True):
        if autocenter:
            self.c.append( contour - np.min( contour, axis=0))
        else:
            self.c.append( contour )



#.............................................................................#
class MultiWormTracker:
    def __init__(self, max_step=100, inertia = 5, n_step=10, speed_damp=0.5, is_worm = None, keep_alive=-1, verbose=False):
        self.WORMS = []

        self.MAX_STEP = max_step
        self.INERTIA  = inertia
        self.N_STEP   = n_step
        self.SPEED_DAMP= speed_damp
        self.IS_WORM   = is_worm
        self.VERBOSE   = verbose
        self.N_CALLS   = -1
        self.KEEP_ALIVE = keep_alive
        
        
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
                _worm_expected = self.WORMS[worm_jj].expected_position( alpha = self.SPEED_DAMP, inertia=self.INERTIA)
                dist = np.sqrt( np.sum( ( _worm_expected - blobs_t_disp )**2, axis=1) )
                
                #... if any blob is closer than _radius
                if np.any( dist < _radius ):
                    
                    #... identify the most proximal blob
                    idx = np.argmin(dist)
                    
                    #... link worm_jj to blob idx
                    self.WORMS[ worm_jj ].update( blobs_t[idx] )
                    self.WORMS[ worm_jj ].update_contour( contours[idx] )
                    
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
            _worm_expected = self.WORMS[worm_jj].expected_position( alpha = self.SPEED_DAMP, inertia=self.INERTIA)
            dist = np.sqrt( np.sum( ( _worm_expected - blobs_t )**2, axis=1) )
            
            if np.any( dist < self.MAX_STEP ):
                #... link worm_jj to blob idx
                idx = np.argmin(dist)
                worms_t_1_disp = np.delete( worms_t_1_disp, np.where(worms_t_1_disp==worm_jj) ) 

                self.WORMS[ worm_jj ].update( blobs_t[idx] )
                self.WORMS[ worm_jj ].update_contour( contours[idx] )

                if self.VERBOSE:
                    print('--- Linked worm %d to blob %d at distance %1.1f' % (worm_jj, idx, dist[idx]) )
    
    
        ###### 3. Keep worms alive or not ######
        # For all worms left, keep them alive only if their speed in the last 
        # self.KEEP_ALIVE number of frames is different than zero.
        # If they are to be ignored, first finish their position with a nan to
        # stop them being tracked.
        
        for worm_jj in np.random.permutation( worms_t_1_disp):
            _valid_pos= self.WORMS[ worm_jj ].x[-1] > 0
            _subspeed = self.WORMS[ worm_jj ].speed_module()[-self.KEEP_ALIVE:]
            t_without_moving = np.sum( _subspeed == 0 )
            
            to_ignore = (self.KEEP_ALIVE>0) and (t_without_moving>=self.KEEP_ALIVE)
            to_delete = self.WORMS[ worm_jj ].x[-1] > 0
            
            # Track only if it has a valid last position, and a valid speed
            to_track = (self.WORMS[worm_jj].x[-1]>0) and (t_without_moving<self.KEEP_ALIVE)
            if _valid_pos:
                if to_ignore:
                    expected_position = np.zeros((2,))*np.nan
                    self.WORMS[ worm_jj ].update( expected_position )
                else:
                    expected_position = self.WORMS[ worm_jj].expected_position(alpha = self.SPEED_DAMP , inertia=self.INERTIA)
                    self.WORMS[ worm_jj ].update( expected_position )
                    self.WORMS[ worm_jj ].update_contour( self.WORMS[worm_jj].c[-1] )
                    
            # Ignore worm when self.KEEP_ALIVE option is different than 0
            #             when t_without moving is equal than self.KEEP_ALIVE frames
            #   if ignored for the first time, you need to "delete" it, by placing nans
            #   if it already has .x[-1]=nan then just simply ignore it
            # If it should not be ignored, then update it with the expected position
            
            # if to_ignore and to_delete:
            #     expected_position = np.zeros((2,))*np.nan
            #     self.WORMS[ worm_jj ].update( expected_position )
            # elif to_ignore and not to_delete:
            # #    pass
            #     print('Success, ignored and not deleted')
            # else:
            #     expected_position = self.WORMS[ worm_jj].expected_position(alpha = self.SPEED_DAMP )
            #     self.WORMS[ worm_jj ].update( expected_position )
            #     self.WORMS[ worm_jj ].update_contour( self.WORMS[worm_jj].c[-1] )
            
            
            # print('Worm %d has been %d frames without moving.' % (worm_jj, t_without_moving) )
            
            # if (self.KEEP_ALIVE>0) and (self.N_CALLS>self.KEEP_ALIVE) and (t_without_moving>=self.KEEP_ALIVE):
            #     if not np.isnan(self.WORMS[ worm_jj ].x[-1]):
            #         expected_position = np.zeros((2,))*np.nan
            #         self.WORMS[ worm_jj ].update( expected_position )
            #         self.WORMS[ worm_jj ].update_contour( self.WORMS[worm_jj].c[-1] )
            #     else:
            #         print('Worm %d not updated with a nan')
            # else:
            #     expected_position = self.WORMS[ worm_jj].expected_position(alpha = self.SPEED_DAMP )
            
            #     self.WORMS[ worm_jj ].update( expected_position )
            #     self.WORMS[ worm_jj ].update_contour( self.WORMS[worm_jj].c[-1] )


        
        ###### 4. New worm identification ######
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
                    self.WORMS.append( Worm(_position, t0 = self.N_CALLS, c=cnt) )

         
        

def resize_video(filename, formfactor):
    # Open video
    cv  = cv2.VideoCapture( filename )
    nframes = cv.get( cv2.CAP_PROP_FRAME_COUNT )
    
    #... get first frame
    suc, frame = cv.read()
    
    #... and resize it to find its final size
    ff_frame = cv2.resize( frame, None , fx=formfactor, fy=formfactor )
    
    
    height , width , c = ff_frame.shape
    
    
    
    # Create Video Writer
    out_filename = '.'.join( [ filename.split('.')[-2]+'_ff%1.2f' % formfactor, filename.split('.')[-1]] )
    out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))
    
    
    #... write first frame
    out.write(ff_frame)
    
    
    # Then loop...
    for _ in tqdm(range( int( nframes-1) )):
        
        suc, frame = cv.read()
        if not suc: break
    
        ff_frame = cv2.resize( frame, None , fx=formfactor, fy=formfactor )    
        out.write(ff_frame)
        
    cv.release()
    out.release()


def cut_video(filename, start_frame=0, end_frame = -1 ):
    # Open video
    cv  = cv2.VideoCapture( filename )
    nframes = cv.get( cv2.CAP_PROP_FRAME_COUNT )
    
    if (end_frame < 0) or (end_frame > nframes):
        end_frame= nframes
        
    if start_frame>nframes:
        print('< E > Start frame needs to be leq total frames.')
        return None
    
    
    #... get first frame
    suc, frame = cv.read()
    
    height , width , c = frame.shape
    
    
    
    # Create Video Writer
    out_filename = '.'.join( [ filename.split('.')[-2]+'_cut', filename.split('.')[-1]] )
    out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc('M','P','4','2'), 30, (width,height))
    
    print('Exporting to %s' % out_filename)
    # Then loop...
    for i_frame in tqdm(range( int( nframes-1) )):
        if  start_frame < i_frame:
            out.write(frame)
        
        if i_frame > end_frame:
            break
        
        if not suc: 
            break   
        
        suc, frame = cv.read()
        
    cv.release()
    out.release()
    





def traj2matrix(trajectories):
    nworms = len(trajectories)
    ntimes = len(trajectories[0].x)
    
    output = np.zeros( (2, nworms, ntimes) )
    for worm_jj, worm in enumerate(trajectories):
        l = len(worm.x)
        output[0, worm_jj, -l:] = worm.x
        output[1, worm_jj, -l:] = worm.y
    return output






def generate_background( videoCapture, n_imgs = 0, skip=0, processing = None, mad=0 ):
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
    fondo_ = np.zeros( (width, height, n_imgs), dtype='uint8' )
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
    if mad != 0:
        fondo_mad = np.median( np.abs( np.expand_dims(fondo,axis=-1).astype('float64') - fondo_ ), axis=2)
        fondo = ( np.clip( fondo.astype('int32')-mad*fondo_mad, 0, 255) ).astype('uint8') 
        
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

def detect_plate(frame, size_ratio=1.0, r0=[0.5,0.5], mindist=100, blur_kernel=10):
    height, width = frame.shape
    exp_radius = height/2*size_ratio
    
    frame = cv2.blur( frame, (blur_kernel,blur_kernel))

    circles = cv2.HoughCircles(frame,cv2.HOUGH_GRADIENT,
                               dp=2,
                               minDist=mindist,
                               param1=50,
                               param2=50,
                               minRadius= int(0.8*exp_radius) ,
                               maxRadius= int(1.2*exp_radius) )
    
    if circles is not None:
        print('Found %d good candidates' % circles.shape[1] )

        # Find closest circle to r0
        dist = (circles[0,:,0]/width - r0[0] )**2 + (circles[0,:,1]/height - r0[1])**2
        idx = np.argmin( dist)
        return circles[0,idx,:].astype('int')
    else:
        return np.array([0,0,999]).astype('int')
    # true_circles=[]
    # if circles is None:
    #     return [[0,0,99999]]
    # for c in circles[0,:]:            
    #     if (0.8*width/2)<c[0]<(1.2*width/2):
    #         if (0.8*height/2)<c[1]<(1.2*height/2):
    #             true_circles.append(c)
    # true_circles= np.array(true_circles)
    # print('Found %d good candidates' % len(true_circles))

    # if len(true_circles)>0:
    #     idx = np.argmin( true_circles[:,2]-exp_radius*size_ratio)
    #     return true_circles[idx].astype('int')
    # else:
    #     return [0,0,99999]
    
def zoom_in(frame, center, formfactor):
    if formfactor==1.0:
        return frame
    
    height, width = frame.shape[:2]    
    
    DeltaX = width/2.0/formfactor
    DeltaY = height/2.0/formfactor
    
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
    
