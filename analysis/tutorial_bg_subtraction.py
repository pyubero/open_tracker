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
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

# Variables que no van a cambiar
VIDEO_PATH  = '../video_2206071634_001.mkv'#'video_2206071634_001.mkv'
BG_FRAMES   = 100
MIN_AREA    = 8
MAX_AREA    = 300
BLUR_SIZE   = 5
NWORMS      = []

# Cargamos el video
video   = cv2.VideoCapture( VIDEO_PATH )
nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#... y el primer frame
ret, frame = video.read()

#... y lo pasamos a grises
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#... y obtenemos el tamaño del video
width, height = gray.shape


###############################
######### Trad method #########

# 1. Primero hacemos una pasada a NFRAMES del video para obtener una imagen que sea
# la mediana de esos frames, esta es una manera facil y rapida para tener un
# modelo del fondo.
fondo_ = np.zeros((width, height, BG_FRAMES))
fondo_[:,:,0] = gray

for ii in tqdm( range( BG_FRAMES-1) ):
    ret, frame = video.read()                      # leer siguiente frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasar a grises
    gray = cv2.GaussianBlur( gray,(BLUR_SIZE,BLUR_SIZE),0)
    fondo_[:,:,1+ii] = gray

# 2. Calcular el fondo como la mediana de fondo_ sobre el eje temporal (axis=2)
fondo = np.median( fondo_, axis=2).astype('uint8')

# 3. Guardar modelo del fondo
cv2.imwrite( './fondo.png', fondo )

# 4. Ver un frame ejemplo al que le quitamos el fondo a ver que tal funciona...
final = cv2.subtract(fondo, gray)
cv2.imshow( 'wdw', final )
cv2.waitKey(0)
cv2.destroyAllWindows()


# 5. Vaya, parece que funciona bien, pero los gusanos son muy oscuros!
# Mas info en: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# Otsu's thresholding after Gaussian filtering, and some thinning
blur = cv2.GaussianBlur(final,(BLUR_SIZE,BLUR_SIZE),0)
ret,thresh = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Si los gusanos quedan muy thicc, se puede probar a erosionar la imagen binaria
# Mas info en https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
#kernel = np.ones((3,3),np.uint8)
#eroded = cv2.erode( thresh , kernel , iterations = 1)

cv2.imshow( 'wdw', thresh )
cv2.waitKey(0)
cv2.destroyAllWindows()


# 6. Intenta contar cuantos gusanos hay:
# Mas info en https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
# ... obtenemos los blobs/contornos
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ... contamos el area total y filtramos los contornos por area
total_contoured_area = np.sum([ cv2.contourArea(c) for c in contours])
filtered_contours = [c for c in contours if MIN_AREA<cv2.contourArea(c)<MAX_AREA ]
worm_areas = [ cv2.contourArea(c) for c in filtered_contours]

count_naive = len(filtered_contours)
count_better= total_contoured_area/np.mean(worm_areas)
error_better= count_better*np.std(worm_areas)/np.mean(worm_areas)
print('We find approximately %d worms.' % count_naive )
print('But computing by size we estimate n_worms = %1.2f +/- %1.2f' % (count_better, error_better) )







# 7. Reproducir video desde el principio restando el fondo
video = cv2.VideoCapture( VIDEO_PATH )
ret, frame = video.read()
for _ in tqdm(range( nframes )):
    # Convert to gray and subtract
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    final = cv2.subtract(fondo, gray)
    
    # Blur and threshold
    blur = cv2.GaussianBlur(final,(BLUR_SIZE,BLUR_SIZE),0)
    ret,thresh = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Count worms after filtering by area
    total_contoured_area = np.sum([ cv2.contourArea(c) for c in contours])
    filtered_contours = [c for c in contours if MIN_AREA<cv2.contourArea(c) ]
    worm_areas = [ cv2.contourArea(c) for c in filtered_contours]
    
    count_naive = len(filtered_contours)
    count_better= total_contoured_area/np.mean(worm_areas)
    error_better= count_better*np.std(worm_areas)/np.mean(worm_areas)
    
    NWORMS.append( count_better )
    
    
    newdim = (1280,960)
    output = frame
    cv2.drawContours( output, filtered_contours, -1, (0,255,0), cv2.LINE_4, 1 )
    cv2.imshow( 'wdw', cv2.resize( output, newdim) )
    if cv2.waitKey(1)==ord('q'):
        break
    
    #... read next frame
    ret, frame = video.read()


cv2.destroyAllWindows()

plt.plot(NWORMS)
plt.xlabel('Time (frame)')
plt.ylabel('Number of moving worms')


