# -*- coding: utf-8 -*-
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
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

# Variables que no van a cambiar
VIDEO_PATH  = 'video_grad_5mM_sinCond_50ulOP50_2206091243_000.avi'#'video_2206071634_001.mkv'
BG_FRAMES   = 99
MIN_AREA    = 10
MAX_AREA    = 40000
BLUR_SIZE   = 5
FORMFACTOR  = 1
MAX_FRAMES  = 1000
USE_MOG = False

# Output variables
CONTOURS  = [] 
POSITIONS = []
WORMS     = []

def generate_background( video, NIMGS = 0, processing = None ):
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT)/5)
    if NIMGS == 0:
        NIMGS = nframes
    elif NIMGS > nframes:
        NIMGS = nframes

    # Cargamos el primer frame
    ret, frame = video.read()

    #... y lo pasamos a grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if processing is not None:
        gray = processing(gray)


    #... y obtenemos el tamaño del video
    width, height = gray.shape

    # Creamos el modelo del fondo
    fondo_ = np.zeros((width, height, NIMGS))
    fondo_[:,:,0] = gray
    
    for ii in tqdm( range( NIMGS-1) ):
        ret, frame = video.read()                      # leer siguiente frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasar a grises
        
        if processing is not None:
            gray = processing(gray)
            
        fondo_[:,:,1+ii] = gray
    
    # 2. Calcular el fondo como la mediana de fondo_ sobre el eje temporal (axis=2)
    fondo = np.median( fondo_, axis=2).astype('uint8')
    
    # 3. Guardar modelo del fondo
    cv2.imwrite( './fondo.png', fondo )
    return fondo

def load_background( filename = None ):
    if filename is None:
        filename = './fondo.png'
        
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


# Style of annotations
font     = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 1
color    = (0, 255, 0)
thickness = 1

wait_time = 1 #... wait time of each frame during the preview



# Cargamos el video
video = cv2.VideoCapture( VIDEO_PATH )
ret, frame = video.read()
_h , _w , _ = frame.shape


# ... y cargamos el fondo
# fondo = generate_background(video, NIMGS = 300 )
fondo = load_background()
fondo = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
plate = detect_plate( fondo , size_ratio=1.4)
print('Scale is: %1.2f px/mm' % (plate[2]/55) )

# ... crear una máscara de la placa
plate[2] = int(0.95*plate[2])
mask_plate = np.zeros_like(fondo)
mask_plate = cv2.circle(mask_plate, (plate[0],plate[1]), plate[2], (255,), -1)

# Uncomment to show masked ROI on the background
# output = cv2.resize( fondo, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
# cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),2)
# cv2.imshow('w', cv2.bitwise_and(output, mask_plate) )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if USE_MOG:
    backSub = cv2.createBackgroundSubtractorMOG2()
    
tStart = datetime.now()
while ret:
    ret, frame = video.read()

    if not ret:
        break
    
    
    
    if FORMFACTOR != 1:
        frame = cv2.resize( frame, ( int(_w*FORMFACTOR) , int(_h*FORMFACTOR) ) )
        
    curr_frame = video.get( cv2.CAP_PROP_POS_FRAMES )
    if curr_frame >= MAX_FRAMES:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )   
    
    if USE_MOG:
        gray = backSub.apply(gray)
    else:    
        gray = cv2.subtract(fondo, gray)
        gray = adjust_gamma( gray, gamma=0.8)
        gray = cv2.medianBlur(gray, 5)
        gray = auto_BC(gray)
        gray = adjust_gamma( gray, gamma=2.0)
        gray = cv2.bitwise_and(gray, mask_plate)

    _, thresh = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
    
    thresh = cv2.medianBlur(thresh, 3)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8) )

    
    cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntArea = [cv2.contourArea(c) for c in cnt ]
    
    
    filtered_contours = [ c for c in cnt if MIN_AREA < cv2.contourArea(c) < MAX_AREA ]
    CONTOURS.append(  filtered_contours) 
    print( curr_frame, len(cnt), len(filtered_contours) )
    
    
    
    
    output = gray.copy()
    if len( output.shape ) ==2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR )  
    # output = cv2.drawContours( output, filtered_contours, -1, color, thickness, cv2.LINE_AA )
    # _= [cv2.putText(output, 
    #                 "%d" % cv2.contourArea(c),
    #                 (int(cv2.moments(c)["m10"] / cv2.moments(c)["m00"]),int(cv2.moments(c)["m01"] / cv2.moments(c)["m00"])),
    #                 font, fontsize, color, thickness, cv2.LINE_AA ) for c in filtered_contours ]
    _= [cv2.circle( output,
                   (int(cv2.moments(c)["m10"] / cv2.moments(c)["m00"]),int(cv2.moments(c)["m01"] / cv2.moments(c)["m00"])),
                   20,(0,255,0),1)  for c in filtered_contours]
    
    cv2.circle( output,(plate[0], plate[1]), plate[2],(0,255,0),2)

    output = zoom_in( output, [0.5, 0.5], 2)
    output = cv2.putText( output, "%d" % curr_frame, (20,20), font, fontsize, color, thickness, cv2.LINE_AA)
    output = cv2.resize( output, (1280, 960) )
    cv2.imshow( 'window', output)
    
    
    key = cv2.waitKey(wait_time)
    
    if key==ord('q'):
        break
    elif key==ord('p'):
        wait_time = (1-wait_time)


cv2.destroyAllWindows()

speed = video.get( cv2.CAP_PROP_POS_FRAMES)/(datetime.now()-tStart).total_seconds()
print('Analysis speed: %1.2f fps' % (speed) )


print('Data exported!')
with open('data_analysis.pkl', 'wb') as f:
    pickle.dump(CONTOURS, f)   
   
    
   
if USE_MOG :
    print('Frames in background history: %d' % backSub.getHistory())
    print('Number of gaussians in mix:   %d' % backSub.getNMixtures())

# # 5. Vaya, parece que funciona bien, pero los gusanos son muy oscuros!
# # Mas info en: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# # Otsu's thresholding after Gaussian filtering, and some thinning
# blur = cv2.GaussianBlur(final,(BLUR_SIZE,BLUR_SIZE),0)
# ret,thresh = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # Si los gusanos quedan muy thicc, se puede probar a erosionar la imagen binaria
# # Mas info en https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
# #kernel = np.ones((3,3),np.uint8)
# #eroded = cv2.erode( thresh , kernel , iterations = 1)

# cv2.imshow( 'wdw', thresh )
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 6. Intenta contar cuantos gusanos hay:
# # Mas info en https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
# # ... obtenemos los blobs/contornos
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # ... contamos el area total y filtramos los contornos por area
# total_contoured_area = np.sum([ cv2.contourArea(c) for c in contours])
# filtered_contours = [c for c in contours if MIN_AREA<cv2.contourArea(c)<MAX_AREA ]
# worm_areas = [ cv2.contourArea(c) for c in filtered_contours]

# count_naive = len(filtered_contours)
# count_better= total_contoured_area/np.mean(worm_areas)
# error_better= count_better*np.std(worm_areas)/np.mean(worm_areas)
# print('We find approximately %d worms.' % count_naive )
# print('But computing by size we estimate n_worms = %1.2f +/- %1.2f' % (count_better, error_better) )





# # 7. Reproducir video desde el principio restando el fondo
# video = cv2.VideoCapture( VIDEO_PATH )
# ret, frame = video.read()
# for _ in tqdm(range( nframes )):
#     # Convert to gray and subtract
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     final = cv2.subtract(fondo, gray)
    
#     # Blur and threshold
#     blur = cv2.GaussianBlur(final,(BLUR_SIZE,BLUR_SIZE),0)
#     ret,thresh = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
#     # Find contours
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
#     # Count worms after filtering by area
#     total_contoured_area = np.sum([ cv2.contourArea(c) for c in contours])
#     filtered_contours = [c for c in contours if MIN_AREA<cv2.contourArea(c) ]
#     worm_areas = [ cv2.contourArea(c) for c in filtered_contours]
    
#     count_naive = len(filtered_contours)
#     count_better= total_contoured_area/np.mean(worm_areas)
#     error_better= count_better*np.std(worm_areas)/np.mean(worm_areas)
    
#     NWORMS.append( count_better )
    
    
#     newdim = (1280,960)
#     output = frame
#     cv2.drawContours( output, filtered_contours, -1, (0,255,0), cv2.LINE_4, 1 )
#     cv2.imshow( 'wdw', cv2.resize( output, newdim) )
#     if cv2.waitKey(1)==ord('q'):
#         break
    
#     #... read next frame
#     ret, frame = video.read()


# cv2.destroyAllWindows()

# plt.plot(NWORMS)
# plt.xlabel('Time (frame)')
# plt.ylabel('Number of moving worms')

