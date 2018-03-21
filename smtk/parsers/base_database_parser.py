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

"""
Abstract base class for a strong motion database reader
"""
import os
import abc
from openquake.baselib.python3compat import with_metaclass


def get_float(xval):
    """
    Returns a float value, or none
    """
    if xval.strip():
        try:
            return float(xval)
        except:
            return None
    else:
        return None


def get_int(xval):
    """
    Returns an int value or none
    """
    if xval.strip():
        try:
            return int(xval)
        except:
            return None
    else:
        return None


def get_positive_float(xval):
    """
    Returns a float value if valid and positive - or else None
    """
    if xval.strip():
        try:
            value = float(xval)
        except:
            return None
        if value >= 0.0:
            return value
        else:
            return None
    else:
        return None


def get_positive_int(xval):
    """
    Returns an int value or none
    """
    if xval.strip():
        try:
            value = int(xval)
        except:
            return None
        if value >= 0:
            return value
        else:
            return None
    else:
        return None


class SMDatabaseReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for strong motion database parser
    """

    def __init__(self, db_id, db_name, filename, record_folder=None):
        """
        Instantiate and conduct folder checks
        """
        self.id = db_id
        self.name = db_name
        self.filename = filename
        self.database = None
        if record_folder:
            self.record_folder = record_folder
        else:
            self.record_folder = self.filename

    @abc.abstractmethod
    def parse(self):
        """
        Parses the database
        """


class SMTimeSeriesReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for a reader of a ground motion time series
    """
    def __init__(self, input_files, folder_name=None, units="cm/s/s"):
        """
        Instantiate and conduct folder checks
        """
        self.input_files = []
        for fname in input_files:
            if folder_name:
                filename = os.path.join(folder_name, fname)
                if os.path.exists(filename):
                    self.input_files.append(filename)
            else:
                if os.path.exists(fname):
                    self.input_files.append(fname)
        self.time_step = None
        self.number_steps = None
        self.units = units
        self.metadata = None

    @abc.abstractmethod
    def parse_records(self, record=None):
        """
        Parse the strong motion record
        """


class SMSpectraReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract Base Class for a reader of a ground motion spectra record
    """
    def __init__(self, input_files, folder_name=None):
        """
        Intantiate with basic file checks
        """
        self.input_files = []
        for fname in input_files:
            if folder_name:
                filename = os.path.join(folder_name, fname)
                if os.path.exists(filename):
                    self.input_files.append(filename)
            else:
                if os.path.exists(fname):
                    self.input_files.append(fname)

    @abc.abstractmethod
    def parse_spectra(self):
        """
        Parses the spectra
        """
