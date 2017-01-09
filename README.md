gmpe-smtk
=========

Python and OpenQuake-based Toolkit for Analysis of Strong Motions and Interpretation of GMPEs

GMPE Strong Motion Modeller's Toolkit (gmpe-smtk)
====

This is the web repository of the GMPE Strong Motion Modeller's Toolkit
(gmpe-smtk). The gmpe-smtk is a suite of tools developed by Scientists 
working at the GEM (i.e. Global Earthquake Model) Model Facility. 

The GMPE Strong Motion Modeller's Toolkit is free software: you can redistribute 
it and/or modify it under the terms of the GNU Affero General Public 
License as published by the Free Software Foundation, either version 
3 of the License, or (at your option) any later version. Please take 
a minute of your time to read the disclaimer below.

Copyright (c) 2014, GEM Foundation


Disclaimer
----

The software GMPE Strong Motion Modeller's Toolkit (gmpe-smtk) provided herein 
is released as a prototype implementation on behalf of scientists and 
engineers working within the GEM Foundation (Global Earthquake Model). 

It is distributed for the purpose of open collaboration and in the 
hope that it will be useful to the scientific, engineering, disaster
risk and software design communities. 

The software is NOT distributed as part of GEM’s OpenQuake suite 
(http://www.globalquakemodel.org/openquake) and must be considered as a 
separate entity. The software provided herein is designed and implemented 
by scientific staff. It is not developed to the design standards, nor 
subject to same level of critical review by professional software 
developers, as GEM’s OpenQuake software suite.  

Feedback and contribution to the software is welcome, and can be 
directed to the hazard scientific staff of the GEM Model Facility 
(hazard@globalquakemodel.org). 

The GMPE Strong Motion Modeller's Toolkit (gmpe-smtk) is therefore
distributed WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

The GEM Foundation, and the authors of the software, assume no 
liability for use of the software.


dependencies
============

The gmpe-smtk currently requires the following dependencies:


* OpenQuake Hazard Library (oq-hazardlib)
* Numpy (1.6.1 or later) (installed with oq-hazardlib)
* Scipy (0.11.0 or later) (installed with oq-hazardlib)
* Shapely (installed with oq-hazardlib)
* Matplotlib (1.3.x or later)
* h5py (2.2.0)

installation
============

* Windows

Windows users should install the PythonXY software package (https://code.google.com/p/pythonxy/), which will install all of the dependencies except oq-hazardlib 
To install oq-hazardlib it is recommended to install MinGW or Github for Windows.

If using Github for Windows simply open a bash shell, clone the oq-hazardlib
repository using:

>> git clone https://github.com/gem/oq-hazardlib.git

Then type

>> cd oq-hazardlib
>> python setup.py install build --compiler=mingw32

To install the gmpe-smtk simply download the zipped code from the repository,
unzip it to a location of your choice then add the directory path to
the Environment Variables found in:

My Computer -> Properties -> System Properties -> Advanced -> Environment Variables

In the Environment Variables you will see a list of System Variables. Select
"Path" and then "Edit". Then simply add the directory of the gmpe-smtk to the
list of directories.

* OSX/Linux

To install oq-hazardlib simply clone the oq-hazardlib repository into a folder
of your choice.

>> git clone https://github.com/gem/oq-hazardlib.git

Then run

>> cd oq-hazardlib
>> python setup.py install

Matplotlib and h5py can both be installed from the native package managers,
although it is recommended to use pip for this purpose.

To install the gmpe-smtk, clone the code from the repository and then
add the following line to your bash or profile script:

export PYTHONPATH=/path/to/gmpe-smtk/:$PYTHONPATH
