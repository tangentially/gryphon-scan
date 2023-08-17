# coding=utf-8

import os
import sys
from setuptools import setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from horus import __version__

APP = ['horus']
DATA_FILES = ['res']

PLIST = {
    'CFBundleName': 'Horus',
    'CFBundleShortVersionString': __version__,
    'CFBundleVersion': __version__,
    'CFBundleIdentifier': 'com.bq.Horus-' + __version__,
    'LSMinimumSystemVersion': '10.8',
    'LSApplicationCategoryType': 'public.app-category.graphics-design'
}

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'res/horus.icns',
    'plist': PLIST
}

setup(name='Horus',
      version=__version__,
      author='Jesús Arroyo Torrens',
      author_email='jesus.arroyo@bq.com',
      description='Horus is a full software solution for 3D scanning',
      license='GPLv2',
      keywords="horus ciclop scanning 3d",
      url='https://www.diwo.bq.com/tag/ciclop',
      app=APP,
      data_files=DATA_FILES,
      options={'py2app': OPTIONS},
      setup_requires=['py2app'])
