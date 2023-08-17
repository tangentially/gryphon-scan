
# Gryphon Scan
## This is highly customisable version of Horus 3D Scan software with some advanced features support

This project was created to support custom built Ciclop-style 3D scanners with different hardware configurations and some more features like:
- photo lights control
- saving photos for photogrammetry
- enhanced calibration
- etc...

Also, to enhance the existing functions and to fix bugs.  


###### For the people who expect software with the single "Make Cool Scan" button
Sorry guys. This is DIY project not the finished commercial software. 
If you are not ready for some manual installation and adjustments than please don't waste your time.  
This project need a bit of understanding how things are work. Here can be some bug's. 
This is platform for learning and experiments.  

I will try to make things work smooth but i can not warranty everything will work out of the box in every environment. 
Most likely you'll need to spend some time and use brain to make things work.  

Please don't write that shitty comments as you do for Ciclop/Horus. This is not the out-of-the-box solution at the moment.  


###### Dear Visitors
I need some feedback for this project including usage (beta testing) experience.
Also, please point me to the right place where Ciclop/Horus based scanners are discussed at the moment as there is completely no activity on the original Ciclop/Horus repositories.
Is there any 3D scanner project that supersedes Horus?

###### At the moment:
- customisable camera height/angle/distance
- support for autofocus cameras in calibration->video panel (camera driver has to support set focus), autodetect camera resolution
- customisable scan area (turntable) size
- photo lights control support at laser pins 3 and 4 (be aware that board hardware pins current is limited and you need extra hardware for powerful lights)
- fixed turntable firmware hello string detection (horus-fw 2.0 support or any custom grbl based firmwares if same G-codes are supported)
- save frames for Photogrammetry
- enhanced calibration:
    - augmented visualization allow visually check calibration quality both for platform and lasers
    - augmented laser lines draw over pattern allow to manually move pattern and compare actual laser beam and calibrated
    - better pattern usage in laser calibration
    - draw trace of detected laser lines during laser triangulation
    - multipass calibration mode for laser triangulation to increase accuracy
    - more informative calibration directions pages
    - reworked platform calibration
    - save/read camera calibration frames (autosave to 'camera_intrisics' folder. read with 'r' button in "Camera Intrinsics")
- HSV laser line detection method, Green and Blue laser colors support
- experimental "Laser background" filter to remove laser line detected at background objects
- laser id saved as "Original_cloud_index" field at .ply export so point cloud can be separated by lasers and additionally aligned
- some builtin constant values moved to settings or estimated automatically
- movement toolbar
- auto save/read camera calibration images (use 'r' to read previous image in current frame)
- calibration (camera,platform,lasers) and scanning (laser,angle) data saved to .ply file to use in future mesh processing
- some bugs fixed
- support for OpenCV 4.x


P.S. Project is work in progress so at some moments the latest commits can be non-functional or contain some debug console output.


Discussion board:  
https://vk.com/topic-99790498_40015734  
  
  
Night Gryphon  
ngryph@gmail.com  
http://vk.com/wingcatlab  


------------------------------------------

## Calibration

__!!! WARNING !!!__ Current machine defaults are set for my custom hardware. 
To use with Ciclop hardware you need to adjust 'settings.json'. 
At least: platform diameter=200, all offsets = 0, machine_model_path=ciclop_platform.stl
Take a look at README_custom_machine.md  
  


### General calibration process is:  

- if needed, set machine custom parameters (turntable geometry, board init string, etc...) in 'settings.json' file.
It is created automatically after first run with default values. Check README_custom_machine.md  
- make sure you can turn on/off lasers and move platform in correct direction with toolbar buttons. Use "Control workbench"  
- check camera resolution and set focus for autofocus cameras at "Calibration"->"Video settings"  
- using "Calibration"->"Pattern settings" setup pattern cols/rows/dimensions  
- using "Adjustment" workbench adjust camera exposure/brightness/contrast to clearly capture pattern and lasers  
- using "Adjustment" workbench adjust segmentation parameters to detect laser lines with minimum noise  
- using "Calibration"->"Camera intrinsics" calibrate your camera focal length and distortion  
- using "Calibration"->"Laser triangulation" detect the laser planes position  
- using "Calibration"->"Platform extrinsics" calibrate platform position  
- check calibration quality by calibration visualization for pattern/platform/laser lines positions  

##### Notes

1. Do not oversharp/overtight images in laser capture adjustments. 
A bit blurry laser line before segmentation filters provide subpixel position information.
Over-sharped image create wobbly "pixel stairs" style artifacts at the 3D scan.

2. Using ROI increase overall scan precision by removing noisy surroundings from laser point detection input

3. If you wish to use segmentation equivalent to Horus 0.1 than set: YCrCb filter, blur and window filters off, refinement off  

4. C270 doesn't have automatic focus. By default, it focuses better on object 2 meters away than 20 cm away. 
To fix that remove the screws and open camera. Then remove glue around the lens with a box cutter and adjust the zoom.

### Camera calibration
To get better scanning results you has to calibrate your camera using "Calibration"->"Camera intrinsics".  
Gryphon Scan use multistep camera calibration to get better precision. Each next camera calibration use previous results as starting point.  

#### Steps to calibrate camera:

1. Measure rough initial camera intrinsics.
For Logitech C270 you can use the default initial values and skip to next step.  
If your camera is not Logitech C270 it is better to estimate rough focal length using ruler and some linear target. 
As calibration target you can use another ruler or just something straight. 
- Measure your target length and fill in "Target length" field
- Put target horizontally and parallel to camera. Move target back/front, so it exactly fit in to camera frame.
Measure distance from camera to target and fill "Target horizontal dist" field
- Do the same for vertical target and fill "Target vertical dist" field
- Check the calculated camera matrix and if it looks good, then apply with "Apply calculated camera data" button.  
2. Using your chessboard pattern calibrate precise camera matrix and distortion.  
The chessboard pattern has to be flat and rigid as possible.
Capture 15 frames of calibration data moving pattern all around the camera view. 
Keep patten parallel to camera. To capture frame press \[space\]. 
Frames are only captured if pattern is detected within frame.  

All captured frames are automatically saved to `camera_intrisics` folder. You can recall them one by one pressing [r] button during frames capture. 

##### Some reading
The Intrinsic Matrix:  
http://ksimek.github.io/2013/08/13/intrinsic/  
  
The distortion vector:  
https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html  
https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/camera%20distortion.pdf  


### Laser calibration
Using "Calibration"->"Laser triangulation" make two calibration sequences with different chessboard pattern positions.  

- Put the pattern slightly in front of platform center. 
Run first calibration. 
You should see laser traces forming two planes during calibration.  

- Put the pattern slightly below of platform center. 
Run second calibration choosing to continue previous calibration.
You should see laser traces append to previous ones forming cross of planes.  

This will make calibration more precise by using larger lasers point cloud area.


------------------------------------------

## Point clouds misalignment
Still investigating. As general workaround try to keep things symmetrical/parallel/perpendicular.  

**NOTE** If cloud misalignment is _very big_ than check your hardware and connections. Check is platform rotation direction correspond the buttons? Maybe your stepper cable turned upside down, or you misconfigured rotation direction flag?  Is there any not well fixed parts? Etc...  

### Lasers
For Gryphon Scan laser planes position and tilt does not affect on point cloud misalignment. 
During some tests i move lasers left-right, back-front, tilt, slightly change focus etc and this does not cause any noticible point clouds misalignment.  
I plan to do more investigations later.

~~- laser to camera distance should be equival~~  
~~- laser lines should be vertical~~  
~~- laser lines should cross at platform center~~

### Platform
Platform calibration is the most critical. 
- to get better platform calibration put pattern in the middle of turntable.
- platform center should be in the middle of horizontal axis of camera image  
- platform rotation axis should be vertical in camera image and parallel to image plane  
- platform should be perpendicular to rotation axis  

_Here is how platform offset vector affect the clouds alignment:_  

#### Change X value
The X coordinate (horizontal) mostly affect clouds scaling artifacts  

<p><img height="300px" src="https://github.com/nightgryphon/gryphon-scan/raw/develop/doc/images/platform_offset_x-.png"> <img  height="300px" src="https://github.com/nightgryphon/gryphon-scan/raw/develop/doc/images/platform_offset_x+.png"></p>

#### Change Y value
The Y coordinate (vertical) mostly affect general position of mesh on the vertical axis  

#### Change Z value
The Z coordinate (depth) mostly affect clouds rotation artifacts  

<p><img height="300px" src="https://github.com/nightgryphon/gryphon-scan/raw/develop/doc/images/platform_offset_z-.png"> <img  height="300px" src="https://github.com/nightgryphon/gryphon-scan/raw/develop/doc/images/platform_offset_z+.png"></p>


### Model correction tool  
You can manually adjust model after scan or load and correct existing .ply. 
Tool allow to apply platform center offset to manually compensate calibration error.  
Also you can do a test scan and manually find offset to adjust platform calibration for further scans.  
To adjust saved .ply it has to be saved with Gryphon Scan and contain scanning metadata. 

<p><img height="270px" src="https://github.com/nightgryphon/gryphon-scan/raw/develop/doc/images/ScanCorrection_in.png"> <img  height="270px" src="https://github.com/nightgryphon/gryphon-scan/raw/develop/doc/images/ScanCorrection_out.png"></p>


------------------------------------------

## Capture performance 
### Video flush values
Most web cam drivers use buffer of few captured frames to enchance performance. 
But in case of scanning we need _current_ frame not the buffered one.
To flush the buffer and get the current frame the specified amount of frames are pre-read from camera driver. 
But this also slowdown the image capture because there is no way to surely detect which frame is actual not the buffered one.  

You can adjust the number of flushed frames in 'settings.json' to optimize performance or to fix 'jump' artifacts in scan.
The parameters are: `flush_stream_*` and `flush_*`  
The `flush_stream_` is values for user interface preview mode. The `flush_` is for calibrating/scanning mode.  
Values are:  
```
    [ texture capture flush,
      laser capture flush,
      pattern capture flush,
      change mode flush ]  
```
- If the value is positive number it is treated as the number of frames to pre-read. Typically 3-4 is enough.  
- If the value is negative it is treated as number of milliseconds for automatic flush mode. 
It will try to detect an actual frame by measuring frame grabbing time. 
If frame grab time is more than specified number of milliseconds than frame treated as actual frame. 
But this timings are depend on overall system performance etc so is not perfect. 
For logitech C270 typical actual frame grab time is 25-30ms so you can try -30 value. 
The number of auto flush frames is limited to 4 frames to prevent possible lockup.  

### Multithreading
During scanning Horus use two separate threads to capture data and to process data. 
If automatic flush is used the processing thread can affect frame grabbing time causing false positive detection of current frame 
which will cause slices "jumping" out of the model. To minimize this effects you can pause processing during capture frames.
To enable set `"scan_sync_threads": true` in 'settings.json'  


------------------------------------------

## Installation
At the moment there is no automatic installer, so you need to install Python and required libraries manually then download and run Gryphon Scan.  
Yes, there is bundler scripts in Horus but there is a lot of broken download links and old software versions.
Also I plan to switch to new OpenCV with contrib package (4.1.0 at the moment)  
  
  
### Windows: Installing Python 2.7 for Gryphon Scan
This notes can be incomplete. This is my experience for my environment (Win 8.1)

1. Get and install the latest Python 2.7  
https://www.python.org/downloads/release/python-2716/  
Windows x86-64 MSI installer  

2. Get and install Microsoft Visual C++ Compiler for Python 2.7  
Required to compile some lib's during pip install  
http://aka.ms/vcpython27  
http://www.microsoft.com/en-us/download/details.aspx?id=44266  

3. Get and install wxWidgets 3.0.4.  
The 3.0.4 is required for matplotlib==1.4.0. 
Will switch to new versions later if required  
https://sourceforge.net/projects/wxpython/files/wxPython/3.0.2.0/  
wxPython3.0-win64-3.0.2.0-py27.exe  

4. Install OpenCV-contrib  
```
pip install opencv-contrib-python  
```
Note: do NOT install opencv-python. Only install opencv-contrib-python  

5. Install OpenGL 
```
pip install pyopengl pyopengl-accelerate
```
Also pip package require GLUT DLLs to be installed separately  
http://freeglut.sourceforge.net/index.php#download  
https://www.transmissionzero.co.uk/software/freeglut-devel/  
Download "freeglut 3.0.0 MSVC Package"  
Extract and copy `\freeglut\bin\\[x64\\]freeglut.dll` -> `\Python27\Lib\site-packages\OpenGL\DLLS\freeglut[64].dll`  
For 64 bit python you need to rename DLL to `freeglut64.dll`

6. Install packages with PIP  
```
pip install -U pyserial numpy scipy matplotlib==1.4.0
```

7. Download and run Gryphon Scan  
Download or clone Gryphon Scan project from GitHub (use green "Clone or Download" button)  
Unzip archive if required  
Open command line and change current dir to project folder.  
Run app with  
```
python horus
```
  
### MacOS: Setting Python for Gryphon Scan
I'm new in Mac world so please correct me if i miss something.  

1. Mac OS already have python 2.7 installed

2. Install package managers and stuff
```
sudo easy_install pip
sudo pip install tornado   <- this will ask to install developers tools which is ok
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

3. Install wxPython
```
brew install wxpython
```

This will also install numpy

4. Install packages
```
sudo pip install opencv-contrib-python
sudo pip install pyopengl pyopengl-accelerate
sudo pip install -U pyserial scipy 
```

5. Setup matplotlib 1.4.0
```
brew install freetype
sudo pip install matplotlib
```

6. Download Gryphon Scan
```
git clone https://github.com/nightgryphon/gryphon-scan.git
```
7. Build USB camera controls library
```
cd gryphon-scan\src\horus\engine\driver\uvc\mac
make
```
8. Run the app  
```
gryphon-scan\horus
```
You may need to allow this app to run in settings->security because there is no digital signature  

------------------------------------------
For the original Horus README look at README_Horus.md  
