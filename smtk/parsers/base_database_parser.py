#!/usr/bin/env python

"""
Abstract base class for a strong motion database reader
"""
import os
import abc

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
    

class SMDatabaseReader(object):
    """
    Abstract base class for strong motion database parser
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, db_id, db_name, filename, record_folder=None):
        """

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
        """

class SMTimeSeriesReader(object):
    """
    Abstract base class for a reader of a ground motion time series
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_files, folder_name=None, units="cm/s/s"):
        """

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
        """


#    def _build_fileset(self, input_file, folder_name):
#        """
#
#        """
#        self.input_files = []
#        for fname in input_files:
#            if folder_name:
#                filename = os.path.join(folder_name, fname)
#                if os.path.exists(filename):
#                    self.input_files.append(filename)
#            else:
#                if os.path.exists(fname):
#                    self.input_files.append(fname)
#

class SMSpectraReader(object):
    """
    Abstract Base Class for a reader of a ground motion spectra record
    """
    def __init__(self, input_files, folder_name=None):
        """

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
        """
