gmpe-smtk
=========
[![Build Status](https://travis-ci.org/GEMScienceTools/gmpe-smtk.svg?branch=master)](https://travis-ci.org/GEMScienceTools/gmpe-smtk)

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

Copyright (c) 2014-2018 GEM Foundation


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


Installation
============

The gmpe-smtk is built on top of the [OpenQuake-engine](https://github.com/gem/oq-engine).

To install the toolkit, and ensure that it is aligned with the OpenQuake-engine
we recommend to install the OpenQuake-engine according to the instructions
given for your specific operating system. The full installation instructions for
OpenQuake can be found here: [https://github.com/gem/oq-engine](https://github.com/gem/oq-engine).

Once the OpenQuake-engine is installed, the gmpe-smtk repository is cloned by:
```bash
git clone https://github.com/GEMScienceTools/gmpe-smtk
```

Then:
```bash
cd gmpe-smtk
python setup.py install
```

For users interested in developing new features or contributing code to the
repository we strongly recommend to install the OpenQuake-engine according to
the instructions for [installing OpenQuake for development](https://github.com/gem/oq-engine/blob/master/doc/installing/development.md).

The gmpe-smtk is currently written in Python 2.7 and work is now underway to
port this to Python 3.5.

