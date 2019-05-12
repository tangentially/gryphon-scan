# Gryphon Scan
## This is highly customizible version of Horus 3D Scan software with some advanced features support

This project was created to support custom built Ciclop-style 3D scanners with different hardware configurations and some more features like:
- photo lights control
- saving photos for photogrammetry
- enchanced calibration
- etc...

Also to enhance the existing functions and to fix bugs.  


###### For the people who expect software with the single "Make Cool Scan" button
Sorry guys. This is DIY project not the finished commercial software. 
If you are not ready for some manual install and adjustments than please don't waste your time.  
This project need a bit of understanding how things are work. Here can be some bug's. 
This is platform for learning and experiments.  

I will try to make things work smooth but i can not warranty everything will work out of the box in every environment. 
Most likely you'll need to spend some time and use brain to make things work.  

Please don't write that shitty comments as you do for Ciclop/Horus. This is not expected to be out of the box solution.  


###### Dear Visitors
I need some feedback for this project including usage (beta testing) experience.
Also please point me to the right place where Ciclop/Horus based scanners are discussed at the moment as there is completely no activity on the original Ciclop/Horus repositories.
Is there any 3D scanner project that superseeds Horus?

###### At the moment:
- customizible camera height/angle/distance
- support for autofocus cameras in calibration->video panel (camera driver has to support set focus), autodetect camera resolution
- customizible scan area (turntable) size
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
- HSV laser line detection method, Green and Blue laser colors support
- experimental "Laser background" filter to remove laser line detected at background objects
- laser id saved as "Original_cloud_index" field at .ply export so point cloud can be separated by lasers and additionally aligned
- some builtin constant values moved to settings or estimated automatically
- movement toolbar
- auto save/read camera calibration images (use 'r' to read previous image in current frame)
- some bugs fixed


###### Notes

1. Do not oversharp/overtight images in laser capture adjustments. 
A bit blurry laser line before segmentation filters provide subpixel position information.
Oversharped image create wobbly "pixel stairs" style artifacts at the 3D scan.

2. Using ROI increase overall scan precision by removing noisy surroundings from laser point detection input


Discussion board:  
https://vk.com/topic-99790498_40015734  
  
  
Night Gryphon  
ngryph@gmail.com  
http://vk.com/wingcatlab  


------------------------------------------
### Installing Python 2.7.16 (latest 2.7) for Gryphon Scan
This notes can be incomplete. This is my experience for my environment (Win 8.1)
Yes, there is bundler scripts in Horus but there is a lot of broken download links and old software versions.
I plan to switch to new OpenCV with contrib package (4.1.0 at the moment)  

1. Get and install latest Python 2.7  
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
Extract and copy \freeglut\bin\\[x64\\]freeglut.dll -> \Python27\Lib\site-packages\OpenGL\DLLS\freeglut[64].dll  
For 64 bit python you need to rename DLL to freeglut64.dll

6. Install packages with PIP  
```
pip install -U pyserial numpy scipy matplotlib==1.4.0
```


### Camera calibration
To get better scanning results you has to calibrate your camera using "Calibration"->"Camera intrinsics".  
There is two steps to calibrate camera:

1. Measure rough initial camera intrinsics.
For Logitech C270 you can use the default initial values ans skip to next step.  
If your camera is not Logitech C270 it is better to estimate rough focal length using ruler and some linear target. 
You can use another ruler or just something straight as target. 
- Measure your target length and fill in "Targrt length" field
- Put target horosontally and parallel to camera. Move target back/front so it exactly fit in to camera frame.
Measure distance from camera to target and fill "Target horisontal dist" field
- Do the same for vertical target and fill "Target vertical dist" field
Check the calculated camera matrix and if it look good than apply with "Apply calculated camera data" button.  

2. Using your chessboard pattern calibrate precise camera matrix and distortion.  
The chessboard pattern has to be flat and rigid as possible.
Capture 15 frames of calibration data moving pattern all around the camera view. 
Keep patten parallel to camera. To capture frame press \[space\]. 
Frames are only captured if pattern is detected within in frame.  

##### Some reading
The Intrinsic Matrix:  
http://ksimek.github.io/2013/08/13/intrinsic/  
  
The distortion vector:  
https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html  
https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/camera%20distortion.pdf  



------------------------------------------
# Horus

[![R&D](https://img.shields.io/badge/-R%26D-brightgreen.svg)](https://github.com/bqlabs/horus)
[![License](http://img.shields.io/:license-gpl-blue.svg)](http://opensource.org/licenses/GPL-2.0)
[![Documentation Status](https://readthedocs.org/projects/horus/badge/?version=release-0.2)](http://horus.readthedocs.io/en/release-0.2/?badge=release-0.2)

Horus is a general solution for 3D laser scanning. It provides graphic user interfaces for connection, configuration, control, calibration and scanning with Open Source [Ciclop 3D Scanner](https://github.com/bqlabs/ciclop).

This is a research project to explore the 3D laser scan with free tools. Feel free to use it for experiments, modify and adapt it to new devices and contribute new features or ideas.

This project has been developed in [Python](https://www.python.org/) language and it is distributed under [GPL v2](https://www.gnu.org/licenses/gpl-2.0.html) license.

## Installation

#### Supported

###### Current version: 0.2rc1

| Logo              | Name     | Instructions                        |
|:-----------------:|:--------:|:-----------------------------------:|
| ![][ubuntu-logo]  | Ubuntu   | [[en]](http://horus.readthedocs.io/en/release-0.2/source/installation/ubuntu.html)  [[es]](http://horus.readthedocs.io/es/release-0.2/source/installation/ubuntu.html) |
| ![][windows-logo] | Windows  |  [[en]](http://horus.readthedocs.io/en/release-0.2/source/installation/windows.html)  [[es]](http://horus.readthedocs.io/es/release-0.2/source/installation/windows.html) |
| ![][macosx-logo]  | Mac OS X |  [[en]](http://horus.readthedocs.io/en/release-0.2/source/installation/macosx.html)  [[es]](http://horus.readthedocs.io/es/release-0.2/source/installation/macosx.html) |

#### Experimental

**Horus 0.2 is not supported for the following distributions**.

However, anyone can test it and contribute to its support.

| Logo               | Name      | Instructions                          |
|:------------------:|:---------:|:-------------------------------------:|
| ![][debian-logo]   | Debian    | [[en]](doc/installation/debian.md)    |
| ![][fedora-logo]   | Fedora    | [[en]](doc/installation/fedora.md)    |

## Documentation

Here you will find the official documentation of the application:

* [User's manual](http://horus.readthedocs.io/en/release-0.2/) [[es](http://horus.readthedocs.io/es/release-0.2/)]

And also all the scientific background of the project in nice Jupyter notebooks:

* [Notebooks](http://nbviewer.jupyter.org/github/Jesus89/3DScanScience/tree/master/notebooks/)
* [Repository](https://github.com/Jesus89/3DScanScience)

## Development

Horus is an Open Source Project. Anyone has the freedom to use, modify, share and distribute this software. If you want to:
* run the source code
* make your own modifications
* contribute to the project
* build packages

follow the next instructions

#### GNU/Linux

Horus has been developed using [Ubuntu Gnome](http://ubuntugnome.org/), that is based on [Debian](https://www.debian.org/), like [Raspbian](https://www.raspbian.org/), [Mint](http://linuxmint.com/), etc. All instructions provided in this section probably work for most of these systems.

* [Ubuntu development](doc/development/ubuntu.md)

NOTE: *deb* and *exe* packages can be generated in *debian like* systems

#### Mac OS X

* [Darwin development](doc/development/darwin.md)

NOTE: *dmg* packages only can be generated in Mac OS X


More interest links are shown below:

* [Presentation](http://diwo.bq.com/en/presentacion-ciclop-horus/) [[es](http://diwo.bq.com/presentacion-ciclop-horus/)]
* [3D Design](http://diwo.bq.com/en/ciclop-released/) [[es](http://diwo.bq.com/ciclop-released/)]
* [Electronics](http://diwo.bq.com/en/zum-scan-released/) [[es](http://diwo.bq.com/zum-scan-released/)]
* [Firmware](http://diwo.bq.com/en/horus-fw-released/) [[es](http://diwo.bq.com/horus-fw-released/)]
* [Software](http://diwo.bq.com/en/horus-released/) [[es](http://diwo.bq.com/horus-released/)]
* [Product documentation](http://diwo.bq.com/en/documentation-ciclop-and-horus/) [[es](http://diwo.bq.com/documentation-ciclop-and-horus/)]
* [Google group](https://groups.google.com/forum/?hl=en#!forum/ciclop-3d-scanner)

[ubuntu-logo]: doc/images/ubuntu.png
[windows-logo]: doc/images/windows.png
[macosx-logo]: doc/images/macosx.png
[debian-logo]: doc/images/debian.png
[raspbian-logo]: doc/images/raspbian.png
[fedora-logo]: doc/images/fedora.png
