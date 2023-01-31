# open_tracker :bug:

The `open_tracker` is a suite of Computer Vision scripts that automatizes the recording and analysis of *C. elegans* behavioral videos, although it can be used with any animal/object in general. It was developed by Pablo Yubero at the Centro Nacional de Biotecnología (CSIC) in Madrid, Spain for the completion of the doctoral thesis entitled "Trade-offs in the architecture and predictability of complex phenotypes" funded by the Ministerio de Ciencia e Innovación and the Europen Social Fund.


## Protocol :bug:
The intended use of these scripts is sequential. I here include a brief description of each script.

* `s0_Recording.py` On execution, it will show a live preview of your camera stream, thus the user can easily adjust the lighting conditions, the focus of the camera and all that jazz before pressing "R" and starting to record a video. If the video is too long, the video will be recorded in fixed-length chunks (limits can be tuned). 
* `s1_Blob_extraction.py` On execution it will load a video, and do all the preprocessing to then export a pickle file with all the moving blobs found. If the results are not satisfactory, the user can easily tune all preprocessing steps. 
* `s2a_Ref_contour_finder.py` As worms are kind of small (to maximize field of view), they can sometimes be mixed up with noise. This scripts is aimed at identifying a model worm to later reliably identify worms in every frame. On execution you will be presented with the first frame and its blobs, you can loop through them until finding the "ideal worm" that will be used as reference. Once found, press "S" to save it, then "Q" to exit.
* `s2b_Metric_selection.py` On execution, it will analyse all blobs in the first frames of the videos to build a model (Gaussian Mixture) to automatically set up a metric that guarantees the correct identification of most (and best) worms.
* `s3_Tracking.py` This is the real deal, it will use all the previous data to finally export an .npz matrix  of size (n\_worms x n\_frames x 2) that stores the coordinates (x and y) of all worms in all frames, filled with np.nan's when necessary. 


## Materials and methods :telescope:
#### Image processing
For image processing we strongly recommend  using OpenCV, although we seldom use our own algorithms. OpenCV includes for example automatic background subtraction, complex shape/animal recognition, brightness and contrast control, thresholding, image denoising...

#### Tracking algorithm
The tracker is able to track a variable number of worms in any video, thus a predefined number is not necessary. At frame t+1, identity matching proceeds in four steps.

1. All worms present in frame t are matched by a progressive scanning algorithm and a Kalman filter, this facilitates the identification of worm forking, that is,when two initially superimposed worms separate).
2. The remaining worms are linked to blobs that are already in use to facilitate the identification of overlapping worms, that is, when two initially distinct worms touch each other). 
3. The remaining unused blobs are then candidates to be newly identified worms. Those are selected according to some user-defined rule, in our case, we used similarity of both Hu's invariant image descriptors and worm size.
4. (Optional) The tracking of some worms can be stopped. This can be used when worms can escape the field of view. 

#### Environmental control 
The environmental conditions are controlled by an Arduino Nano on a homemade controlling board. The board supports 3x high power LEDs (we are currently using the triplet 400nm, 650nm and 400K); a 60W Thermoelectric Couple that works in both heat/cold directions; 2x small fans to dissipate heat and accelerate cooling; automatic zoom/focus via two stepper motors (currently unused); two small indicating bicolor LEDs; and a temperature sensor.

#### Recording platform
The recording platform and a case for the controlling board were both designed using FreeCAD. Original and print-ready .stl files are available in the `/docs/models_3d` folder. All pieces were printed nicely using either ABS or PLA by standard protocols in a Ultimaker 2 printer. We strongly recommend using black (mate if possible) plastic to prevent leaks of light and internal reflections.

#### Webcam support
In this repo you will find support for two types of webcam, those that use the Universal Video Class (UVC) and those that rely on AppliedVision's Vimba SDK. While UVC works using OpenCV, AppliedVision's cameras need the Vimba SDK for Python installed.

There are different classes to interact with each type called `/pytracker/myCamera_UVC` and `/pytracker/myCamera_Alvium`, be sure to use the most appropiate one. These classes have a few fundamental functions: `open()`,`close()`,`snapshot()`,`start_preview()`, `get()`, and `set()`, you can find more info on how to use them in the scripts contained in the `\test` folder.



## Reference :bell:
Should you use this repo in any way, please reference the PhD thesis: 

Yubero P. (2022). _Trade-offs in the architecture and predictability of complex phenotypes_. (Doctoral dissertation, Universidad Autónoma de Madrid, Spain). Retrieved from repositorio.uam.es


## Contact and license :incoming_envelope:
If your have any issue with this repo please contact me through github rather than email. Feel free to ask for updated versions, or latest developments.

This code is distributed under an MIT license.



