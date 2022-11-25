# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:26:05 2021

@author: Pablo
"""

import cv2
import numpy as np
from datetime import datetime
from time import sleep
from threading import Thread
from sys import platform
from matplotlib import pyplot as plt


class myCamera:
    '''
        This is to streamline the use of different cameras in the same scripts.
        Common methods of a myCamera are:
            open() and  close()
            set()  and  get()
            snapshot()
            start_ and stop_streaming()
            start_ and stop_preview()
            start_ and stop_recording()
        However, the list of properties that can be set/get are specific to each model. 
        Then refer to the dictionnary properties to get the list of valid values.
    '''
    def __init__(self, camera_id = None, backend = cv2.CAP_DSHOW ):
        
        #...define initial variables
        self.camera_id = camera_id
        self.backend   = backend
        self.vcapture  = cv2.VideoCapture() # cap object is created but it is not assigned to any camera or anything
        
        
        self._verbose      = False
        self._is_open      = False
        self._frame        = []
        
        self.thread_recording_running = False
        self.thread_streaming_running = False
        self.preview_running          = False
        
        self.recording_filename = 'video_NVIDEO.avi'
        self.recording_format   = 'MJPG'
        self.recording_fps      = 5
        self.recording_totaltime= 99999 #about 2.7h
        self.recording_maxtime  = 5
        self.recording_nvideo   = -1
        
        self.properties = {}
        self.__define_prop2cv()
        
        
        # INITIALIZATION #
        if camera_id is not None:
            self.open( self.camera_id )
            self.__update_properties()
            
        
        
    def __del__(self):
        self.close()


    def close(self):
        self.stop_recording() if self.thread_recording_running else None
        self.stop_preview()   if self.preview_running          else None
        self.stop_streaming() if self.thread_streaming_running else None
            
        self.vcapture.release()
        self._is_open = False
        print('Camera is now closed') if self._verbose else None
        
        
    def open(self, camera_id=None):
        if self._is_open:
            print('<< E >> Camera %d is already open.' % camera_id)
            return 
        
        self.vcapture.open( camera_id , self.backend)
        self._is_open = True
        self.snapshot()
        print('Camera is now open') if self._verbose else None
        
        
    def snapshot(self, formfactor = 1):
        if not self._is_open:
            print("<< E >> Please open a camera before taking any snapshot.")
            return None
            
        suc, self.frame = self.vcapture.read()
        
        if not suc:
            print("<< E >> Camera is open, but VideoCapture object could not take any picture.")
            return None       

        if formfactor != 1:    
            print('Snapshot taken and resized by %1.2f.' % formfactor ) if self._verbose else None
            return cv2.resize( self.frame , None , fx=formfactor , fy=formfactor)
        else:
            print('Snapshot taken.') if self._verbose else None
            return self.frame




    #################################
    ####### MODIFY PROPERTIES #######
    def __define_prop2cv(self):
        self.prop2cv = {}
        self.prop2cv.update( {'width' : cv2.CAP_PROP_FRAME_WIDTH         } )
        self.prop2cv.update( {'height' : cv2.CAP_PROP_FRAME_HEIGHT       } )
        self.prop2cv.update( {'brightness' : cv2.CAP_PROP_BRIGHTNESS     } )
        self.prop2cv.update( {'contrast' : cv2.CAP_PROP_CONTRAST         } )
        self.prop2cv.update( {'exposure' : cv2.CAP_PROP_EXPOSURE         } )
        self.prop2cv.update( {'saturation' : cv2.CAP_PROP_SATURATION     } )
        self.prop2cv.update( {'autoexposure' :cv2.CAP_PROP_AUTO_EXPOSURE } )
        self.prop2cv.update( {'hue' : cv2.CAP_PROP_HUE                   } )
        self.prop2cv.update( {'gain' : cv2.CAP_PROP_GAIN                 } )

        
    def __update_properties(self):
        for propName, cv2_code in self.prop2cv.items():
            self.properties.update( { propName: self.vcapture.get(cv2_code   ) } )        
        
    def get(self, propertyName):
        self.__update_properties()
        return self.properties[propertyName]
        
        
    def set(self, propertyName, value):
        if propertyName == 'resolution':
            
            self.vcapture.set( cv2.CAP_PROP_FRAME_WIDTH,  value[0] )
            self.vcapture.set( cv2.CAP_PROP_FRAME_HEIGHT, value[1] )    
            self.__update_properties()
            
            if self.properties['width'] != value[0] or self.properties['height']!= value[1]:
                print('Could not set the specified resolution. Using %dx%d instead.' %
                      (self.properties['width'],self.properties['height'] ) )
                return False
        
        else:
            cv2_code = self.prop2cv[propertyName]
            self.vcapture.set( cv2_code , value )
            self.__update_properties()
            
            if self.properties[propertyName] != value:
                print('Could not set the specified %s. Using %d instead.' % (propertyName, self.properties[propertyName]) )
                return False
        
    def summary(self):
        print('----- Summary -----')
        for key, value in self.properties.items():
            print("%12s - %1.1f" % (key, value) )
            
        
        

    #################################
    ######## STREAMING THREAD #######
    def start_streaming(self):
        
        if not self._is_open:
            print("<< E >> Please open a camera before starting the stream.")
            return False
        
        if self.thread_streaming_running:
            print("<< W >> A stream is already running.")
            return True
            
        self.thread_streaming = Thread(target = self.__streaming_fun, daemon = True) 
        self.thread_streaming.start()
        sleep(0.5)
        
        return True


    def stop_streaming(self):
        if self.thread_streaming_running:
            self.thread_streaming_running = False
            self.thread_streaming.join()
        else:
            print('<< W >> Streaming not found, it could not be stopped.')
        
        return True


    def __streaming_fun(self):
        self.thread_streaming_running = True
        print('Streaming started.') if self._verbose else None

        self.streaming_myfps    = myFPS( averaging = 30 )
        
        while self.thread_streaming_running:
            self.snapshot()
            self.streaming_myfps.tick()
        
        self.thread_streaming_running = False
        print('Streaming stopped.') if self._verbose else None


    def streaming_fps(self):
        return self.streaming_myfps.get()



    #################################
    ########## PREVIEW EASY #########
    def start_preview(self, formfactor=1, vlim=[10,230] ):
        #... start stream if there is none
        if self.start_streaming():
            print('Starting camera preview.')
            print('Press R to start recording with default filename.')
            print('Press Z to zoom in a rectangle.')
            print('Press T to track moving objects.')
            print('Press Q to exit.')

            self.preview_running = True
            self.preview_zoom_bbox = None
            self.preview_ffactor_main = formfactor
            self.preview_ffactor_zoom = 1
            self.preview_winname_main = 'Preview. Press Q to exit.'
            self.preview_winname_zoom = 'Zoom region. Press Z to delete.'
            self.preview_autotrack = False
            backSub = cv2.createBackgroundSubtractorMOG2(  detectShadows= False )

            while self.preview_running:
                frame    = self._resize( self.frame, self.preview_ffactor_main)
                fgMask = backSub.apply(frame)
                
                # Prepare output
                output = frame.copy()
                gray   = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY) 
                
                # Detect (un)burnt pixels
                _, thresh_high = cv2.threshold( gray, int(vlim[1]), 255,cv2.THRESH_BINARY)
                cnt_high, _ = cv2.findContours(thresh_high, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                _, thresh_low = cv2.threshold( gray, vlim[0],255,cv2.THRESH_BINARY_INV)
                cnt_low, _ = cv2.findContours(thresh_low, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, cnt_high, -1, (255,0,0), -1)                  
                cv2.drawContours(output, cnt_low , -1, (0,0,255), -1)                  

                
                
                # Draw some text with fps information on the main frame
                fps_text = '%1.1f +/- %1.1f' % self.streaming_fps() 
                color    = (0,0,255)
                coords   = (20,20)
                font     = cv2.FONT_HERSHEY_SIMPLEX
                output    = cv2.putText(output, fps_text, coords, font, 0.5, color, 1, cv2.LINE_AA )
                
                # Draw some text with information on the main frame
                if self.thread_recording_running:
                    rec_text = '[REC %ds]' % self.recording_time
                    cv2.putText(output, rec_text, (20,40), font, 0.5, color, 1, cv2.LINE_AA )
                
                # Create zoom window
                if self.preview_zoom_bbox:
                    x0,y0,w,h = self.preview_zoom_bbox
                    ff        = self.preview_ffactor_main
                    
                    #... crop original frame and display
                    zoom_frame = self.frame[int(y0):int(y0+h), int(x0):int(x0+w)]
                    zoom_frame = self._resize( zoom_frame, self.preview_ffactor_zoom)

                    cv2.imshow( self.preview_winname_zoom , zoom_frame )
                    cv2.setMouseCallback(self.preview_winname_zoom, lambda event,x,y,flags,params : self.__preview_callback(event,x,y,flags,'zoom') )
                    
                    #... draw on the main frame the zoomed rectangle
                    output = cv2.rectangle( output, ( int(x0*ff) ,int(y0*ff) ), ( int(x0*ff+w*ff), int(y0*ff+h*ff) ), (255,0,0), 1 )
                
                # Track and highlight moving objects with a MOG background detector
                if self.preview_autotrack:
                    ret, thresh = cv2.threshold( fgMask,127,255,cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(output, contours, -1, (0,255,0), 2)                  
                
                         
                cv2.imshow(self.preview_winname_main, output  )
                cv2.setMouseCallback(self.preview_winname_main, lambda event,x,y,flags,params : self.__preview_callback(event,x,y,flags,'main') )
                
                
                key = cv2.waitKey(1)
                
                if key==ord('q'):
                    break
                elif key == ord('r'):
                    self.toggle_recording()
                elif key == ord('t'):
                    self.preview_autotrack = not self.preview_autotrack
                elif key == ord('z'):
                    if self.preview_zoom_bbox:
                        self.preview_zoom_bbox=None
                        cv2.destroyAllWindows()
                    else:
                        cv2.destroyAllWindows()
                        self.preview_zoom_bbox = cv2.selectROI(frame, showCrosshair=False)
                        self.preview_zoom_bbox = [ value/self.preview_ffactor_main for value in self.preview_zoom_bbox]
                        cv2.destroyAllWindows()
                    

            cv2.destroyAllWindows()
            self.preview_running = False
           
     
    def __preview_callback(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if param =='main':
                self.preview_ffactor_main = 1.2*self.preview_ffactor_main
            if param =='zoom':
                self.preview_ffactor_zoom = 1.2*self.preview_ffactor_zoom
          
            
        if event == cv2.EVENT_RBUTTONDBLCLK:
            if param =='main':
                self.preview_ffactor_main = 0.833*self.preview_ffactor_main
            if param =='zoom':
                self.preview_ffactor_zoom = 0.833*self.preview_ffactor_zoom
           
            
    def stop_preview(self):
        self.preview_running = False
        sleep(0.5)
            
    
    
    #################################
    ########## RECORDINGS ###########
    def start_recording(self, filename=None, fmt=None, total_time=None, fps=None ):
        if not self._is_open:
            print("<< E >> Please open a camera before recording any video.")
            return False
        
        if not self.start_streaming():
            print("<< E >> Could not start a video stream.")
            return False
        
        if self.thread_recording_running:
            print('<< E >> A recording is already running, please stop it first.')
            return False
        
        if filename   is None: filename   = self.recording_filename
        if fmt        is None: fmt        = self.recording_format
        if total_time is None: total_time = self.recording_totaltime
        if fps        is None: fps        = self.recording_fps
        
        self.recording_time    = 0
        self.recording_nframes = 0
        self.recording_start   = datetime.now()

        filename = filename.replace( 'NVIDEO', '%03d' % self.recording_nvideo ).replace('DATETIME', datetime.now().strftime('%y%m%d%H%M') )
        THREADFUN = lambda: self.__recording_fun(filename, fmt, total_time, fps)
        
        self.thread_recording = Thread(target = THREADFUN, daemon = True) 
        self.thread_recording.start()
        sleep(0.5)
        
        
    def stop_recording(self):
        if self.thread_recording_running:
            self.thread_recording_running = False
            self.thread_recording.join()
        else:
            print('<< W >> Recording not found, it could not be stopped.')
        return True
        
        
    def __recording_fun(self, filename, fmt, total_time, fps ):
        color      = int(0)
        resolution = ( self.frame.shape[1], self.frame.shape[0] )
        
        fourcc     = cv2.VideoWriter_fourcc( *fmt )
        speed      = 30
        autorestart= False
        started_chunk = datetime.now()
        
        
        timer       = myTimer( 1.0/fps)
        videoWriter = cv2.VideoWriter(filename, fourcc, speed,  resolution, color )
        
        self.recording_nvideo += 1
        self.thread_recording_running = True
        
        #... then record every 1/FPS seconds
        while self.recording_time <= total_time and self.thread_recording_running:           
            self.recording_time = (datetime.now() - self.recording_start).total_seconds()
            recording_chunk     = (datetime.now() - started_chunk ).total_seconds()
             
            if timer.isTime():
                videoWriter.write(  cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)   )
                self.recording_nframes += 1
                 
            if recording_chunk >= self.recording_maxtime:
                self.thread_recording_running = False
                autorestart = True
                
        # signal that the recording has finished
        self.thread_recording_running = False
        
        if autorestart:
            self.__restart_recording()


    def __restart_recording(self):
        filename = self.recording_filename.replace( 'NVIDEO', '%03d' % self.recording_nvideo ).replace('DATETIME', datetime.now().strftime('%y%m%d%H%M') )
        fmt        = self.recording_format
        total_time = self.recording_totaltime
        fps        = self.recording_fps
       
        THREADFUN = lambda: self.__recording_fun(filename, fmt, total_time, fps)

        self.thread_recording = Thread(target = THREADFUN, daemon = True) 
        self.thread_recording.start()
        sleep(0.5)
        


    def toggle_recording(self):
        if self.thread_recording_running and self.thread_recording.is_alive():
            self.stop_recording()
        else:
            self.start_recording()
            
        
    
    
    #################################
    ##### CONVENIENCE FUNCTIONS ##### 
    def _resize(self, frame, formfactor):
        return cv2.resize( frame, None , fx=formfactor, fy=formfactor )
        





class myFPS:
    def __init__(self, averaging = 10):
        self.lastTick=datetime(2,1,1,1,1)            #this would be tick number n
        self.previousTick=datetime(1,1,1,1,1)       #this would be tick n-1
        self.historic = np.ones((averaging,))
        self._ii = 0
        
    def tick(self):
        # Update ticks
        self.previousTick=self.lastTick
        self.lastTick=datetime.now()
        
        # Update FPS historic
        self.historic[ self._ii ] = 1/( (self.lastTick-self.previousTick).total_seconds() + 1e-6)
        self._ii += 1
        if self._ii == len( self.historic ):
            self._ii = 0
        
    def get(self):
        return np.mean(self.historic), np.std(self.historic)



class myTimer:
    def __init__(self,DeltaTime):
        self.DeltaTime=DeltaTime
        self.previous=datetime.now()
        
    def isTime(self):
        tFromPrevious=(datetime.now()-self.previous).total_seconds()
        if tFromPrevious>self.DeltaTime:
            self.previous = datetime.now()
            return True
        else :
            return False   



def number_of_cameras():
    NumCams=0    
    while True:
        cap = cv2.VideoCapture(NumCams)
        if cap.read()[0]:
            NumCams=NumCams+1
            cap.release()
        else:
            # cap.release()
            break
    if NumCams>0:
        print('Total cameras found: '+str(NumCams)+'.')
        return NumCams
    else:
        print('No cameras were found.')
        return NumCams


























