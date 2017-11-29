#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from datetime import datetime


def positive_float(value, key, verbose=False):
    """
    Returns True if the value is positive or zero, false otherwise
    """
    value = value.strip()
    if value and float(value) >= 0.0:
        return float(value)
    if verbose:
        print("Positive float value (or 0.0) is needed for %s - %s is given"
              % (key, str(value)))
    return None

def vfloat(value, key):
    """
    Returns value or None if not possible to calculate
    """
    value = value.strip()
    if value:
        try:
            return float(value)
        except:
            print("Invalid float value %s for %s" % (value, key))
    return None

def vint(value, key):
    """
    Returns value or None if not possible to calculate
    """
    value = value.strip()
    if "." in value:
        value = value.split(".")[0]
    if value:
        try:
            return int(value)
        except:
            print("Invalid int value %s for %s" % (value, key))
    return None


def positive_int(value, key):
    """
    Returns True if the value is positive or zero, false otherwise
    """
    value = value.strip()
    if value and int(value) >= 0.0:
        return int(value)
    print("Positive float value (or 0.0) is needed for %s - %s is given"
          % (key, str(value)))
    return False

def longitude(value):
    """
    Returns True if the longitude is valid, False otherwise
    """
    lon = float(value.strip())
    if not lon:
        return False
    if (lon >= -180.0) and (lon <= 180.0):
        return lon
    print("Longitude %s is outside of range -180 <= lon <= 180" % str(lon))
    return False

def latitude(value):
    """
    Returns True if the latitude is valid, False otherwise
    """
    lat = float(value.strip())
    if not lat:
        print("Latitude is missing")
        return False
    if (lat >= -90.0) and (lat <= 90.0):
        return lat 
    print("Latitude %s is outside of range -90 <= lat <= 90" % str(lat))
    return False

def date(year, month, day):
    """
    Checks that the year is given and greater than 0, that month is in the
    range 1 - 12, and day is in the range 1 - 31
    """
    if all([year > 0, month > 0, month <= 12, day > 0, day <= 31]):
        return True
    print("Date %s/%s/%s is not valid" % (str(year), str(month), str(day)))
    return False

def date_time(value, dt_format="%Y-%m-%d %H:%M:%S"):
    """
    Returns a valid date time
    """
    dt = value.strip()
    try:
        output = datetime.strptime(dt, dt_format)
        return output
    except ValueError:
        print("Date-time %s not valid under format %s" % (dt, dt_format))
        return False

def strike(value):
    """
    Returns a float value in range 0 - 360.0
    """
    strike = value.strip()
    if not strike:
        return None
    strike = float(strike)
    if strike and (strike >= 0.0) and (strike <= 360.0):
        return strike
    print("Strike %s is not in range 0 - 360" % value)
    return None

def dip(value):
    """
    Returns a float value in range 0 - 90.
    """
    dip = value.strip()
    if not dip:
        return None
    dip = float(dip)
    if dip and (dip > 0.0) and (dip <= 90.0):
        return dip
    print("Dip %s is not in range 0 - 90" % value)
    return None

def rake(value):
    """
    Returns a float value in range -180 - 180
    """
    rake = value.strip()
    if not rake:
        return None
    rake = float(rake)
    if rake and (rake >= -180.0) and (rake <= 180.0):
        return rake
    print("Rake %s is not in range -180 - 180" % value)
    return None
