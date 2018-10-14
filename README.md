# Gryphon Scan
## This is highly customizible version of Horus 3D Scan software with some advanced features support

This project was created to support custom built Ciclop-style 3D scanners with different hardware configurations and some more features like:
- photo lights control
- saving photos for photogrammetry
- etc?...

Also to enhance the existing functions and to fix bugs.

###### Dear Visitors
I need some feedback for this project including usage (beta testing) experience.
Also please point me to the right place where Ciclop/Horus based scanners are discussed at the moment as there is completely no activity on the original Ciclop/Horus repositories.
Is there any 3D scanner project that superseeds Horus?

###### At the moment:
- customizible camera height/angle/distance
- customizible scan area (turntable) size
- photo lights control support at laser pins 3 and 4 (be aware that board hardware pins current is limited and you need extra hardware for powerful lights)
- fixed turntable firmware hello string detection (horus-fw 2.0 support or any custom grbl based firmwares if same G-codes are supported)
- enhanced calibration:
    - augmented visualization allow visually check calibration quality both for platform and lasers
    - augmented laser lines draw over pattern allow to manually move pattern and compare actual laser beam and calibrated
    - better pattern usage in laser calibration
    - draw trace of detected laser lines during laser triangulation
    - multipass calibration mode for laser triangulation to increase accuracy
    - more informative calibration directions pages
- HSV laser line detection method
- experimental "Laser background" filter to remove laser line detected at background objects
- laser id saved as "Original_cloud_index" field at .ply export so point cloud can be separated by lasers and additionally aligned
- some builtin constant values moved to settings or estimated automatically
- movement toolbar
- some bugs fixed



Night Gryphon

ngryph@gmail.com

http://vk.com/wingcatlab


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
