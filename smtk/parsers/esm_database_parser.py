#!/usr/bin/env/python
"""
Parser set for the European Strong Motion database format.
Detailed documentation on the format is here: 
http://esm.mi.ingv.it/static_stage/doc/manual_ESM.pdf
"""
import h5py
import os
import csv
import numpy as np
from collections import OrderedDict
from datetime import datetime
from linecache import getline
from math import sqrt
from openquake.hazardlib.geo.geodetic import geodetic_distance
from smtk.sm_database import *
from smtk.parsers.base_database_parser import (get_float, get_int,
                                               SMDatabaseReader,
                                               SMTimeSeriesReader,
                                               SMSpectraReader)

FILE_INFO_KEY = ["Net", "Station", "Location", "Channel", "DM", "Date", "Time",
                 "Processing", "Waveform", "Format"]

def _to_float(string):
    """
    Returns float or None
    """
    if string:
        return float(string)
    else:
        return None

def _to_int(string):
    """
    Returns integer or None
    """
    if string:
        return int(string)
    else:
        return None

ESMD_MECHANISM_TYPE = {"NF": -90., "SS": 0.0, "TF": 90.0, "U": 0.0}

class ESMDatabaseMetadataReader(SMDatabaseReader):
    """
    Reader for the metadata database of the European Strong Motion Database
    """
    ORGANIZER = []
    def parse(self):
        """
        Parses the record
        """
        file_list = os.listdir(self.filename)
        num_files = len(file_list)
        self.database = GroundMotionDatabase(self.id, self.name)
        self._sort_files()
        assert (len(self.ORGANIZER) > 0)
        for file_dict in self.ORGANIZER:
            metadata = self._get_xyz_metadata(file_dict)
#        for file_str in file_list:
#            if "DS_Store" in file_str:
#                continue
#            if file_str in self.SKIP_FILES:
#               # Records already organised
#            fname_info = self._get_filename_info(self, file_str)
#            metadata = self._get_metadata_from_file(self, file_str)
#            xyz_metadata = self._get_xyz_metadata(
#            # Metadata comes from lines 1 - 64
#            metadata = []
#            for i in range(1, 65):
#                row = (getline(file_str, i).rstrip("\n")).split(":")
#                if len(row) > 2:
#                    # The character : occurs somewhere in the datastring
#                    metadata.append((row[0].strip(), ":".join(row[1:]).strip()))
#                else:
#                    # Parse as normal
#                    metadata.append((row[0].strip(), row[1].strip()))


    def _get_filename_info(self, filename):
        """
        ESMD follows a specific naming convention. Return this information in
        a dictionary
        """
        file_info = filename.split(".")
        return OrderedDict([
            (FILE_INFO_KEY[i], file_info[i]) for i in range(len(file_info))
            ])

    def _get_metadata_from_file(self, file_str):
        """
        Pulls the metadata from lines 1 - 64 of a file and returns a cleaned
        version as an ordered dictionary
        """
        metadata = []
        for i in range(1, 65):
            row = (getline(file_str, i).rstrip("\n")).split(":")
            if len(row) > 2:
                # The character : occurs somewhere in the datastring
                metadata.append((row[0].strip(), ":".join(row[1:]).strip()))
            else:
                # Parse as normal
                metadata.append((row[0].strip(), row[1].strip()))
        return OrderedDict(metadata)

    def _sort_files(self):
        """
        Searches through the directory and organise the files associated
        with a particular recording into a dictionary
        """
        skip_files = []
        for file_str in os.listdir(self.filename):
            file_dict = {"Time-Series": {"X": None, "Y": None, "Z": None},
                         "PSV": {"X": None, "Y": None, "Z": None},
                         "SA": {"X": None, "Y": None, "Z": None}}

            if (file_str in skip_files) or ("DS_STORE" in file_str):
                continue
            #print file_str, skip_files
            file_info = self._get_filename_info(file_str)
            code1 = ".".join([file_info[key] for key in ["Net", "Station",
                                                         "Location"]])
            code2 = ".".join([file_info[key] for key in ["DM", "Date", "Time",
                                                         "Processing",
                                                         "Waveform"]])
            #print code1, code2
            for x_term in ["HNE", "HN2", "HLE", "HL2", "HGE", "HG2"]:
                if file_dict["Time-Series"]["X"]:
                    continue
                test_filename = "{:s}.{:s}.{:s}.ASC".format(code1,
                                                            x_term,
                                                            code2)
                test_filename = os.path.join(self.filename, test_filename)
                if os.path.exists(test_filename):
                    file_dict["Time-Series"]["X"] = test_filename
                    skip_files.append(test_filename)
                    # Get SA and PSV
                    sa_filename = test_filename.replace("ACC", "SA")
                    if os.path.exists(sa_filename):
                        file_dict["SA"]["X"] = sa_filename
                        skip_files.append(sa_filename)
                    psv_filename = test_filename.replace("ACC", "PSV")
                    if os.path.exists(psv_filename):
                        file_dict["PSV"]["X"] = psv_filename
                        skip_files.append(psv_filename)
                    for y_term in ["N", "1", "3"]:
                        y_filename = test_filename.replace(
                            x_term, "{:s}{:s}".format(x_term[:2], y_term))
                        if os.path.exists(y_filename):
                            file_dict["Time-Series"]["Y"] = y_filename
                            skip_files.append(y_filename)
                            sa_filename = y_filename.replace("ACC", "SA")
                            if os.path.exists(sa_filename):
                                file_dict["SA"]["Y"] = sa_filename
                                skip_files.append(sa_filename)
                            psv_filename = y_filename.replace("ACC", "PSV")
                            if os.path.exists(psv_filename):
                                file_dict["PSV"]["Y"] = psv_filename
                                skip_files.append(psv_filename)
                    # Get vertical files
                    v_filename = test_filename.replace(x_term,
                        "{:s}Z".format(x_term[:2]))
                    if os.path.exists(v_filename):
                        file_dict["Time-Series"]["Z"] = v_filename
                        skip_files.append(v_filename)
                        sa_filename = v_filename.replace("ACC", "SA")
                        if os.path.exists(sa_filename):
                            file_dict["SA"]["Z"] = sa_filename
                            skip_files.append(sa_filename)
                        psv_filename = v_filename.replace("ACC", "PSV")
                        if os.path.exists(psv_filename):
                            file_dict["PSV"]["Z"] = psv_filename
                            skip_files.append(psv_filename)
            self.ORGANIZER.append(file_dict)

    def _get_xyz_metadata(self, file_dict):
        """
        The ESM is a bit messy mixing the station codes. Returns the metadata
        corrsponding to the x-, y- and z-component of each of the records
        """
        metadata = {}
        if file_dict["Time-Series"]["X"]:
            metadata["X"] = self._get_metadata_from_file(
                file_dict["Time-Series"]["X"])
        
        if file_dict["Time-Series"]["Y"]:
            metadata["Y"] = self._get_metadata_from_file(
                file_dict["Time-Series"]["Y"])
        
        if file_dict["Time-Series"]["Z"]:
            metadata["Z"] = self._get_metadata_from_file(
                file_dict["Time-Series"]["Z"])
        return metadata


#        # X-component
#        x_terms = ["HNE", "HN2", "HLE", "HL2", "HGE", "HG2"]
#        # Determine if a file exists with one of the terms
#        for x_term in x_terms:
#            test_filename = os.path.join(self.filename,
#                                         ".".join([file_info["Net"],
#                                                   file_info["Station"],
#                                                   file_info["Location"],
#                                                   x_term,
#                                                   file_info["Date"],
#                                                   file_info["Time"],
#                                                   file_info["Processing"],
#                                                   file_info["Waveform"],
#                                                   file_info["Format"]])
#            if os.path.exists(test_filename):
#                x_metadata = self._get_metadata_from_file(test_filename)
#                break
#            else:
#                x_metadata = None
#        if not x_metadata:
#            raise ValueError(
#                "No x-component found for %s" %
#                ".".join([file_info[key] for key in FILE_INFO_KEY]))
#        # Y-component
#        y_terms = ["HNN", "HN1", "HN3", "HLN", "HL1", "HL3", "HGN", "HG1",
#                   "HG3"]
#        # Determine if a file exists with one of the terms
#        for y_term in y_terms:
#            test_filename = os.path.join(self.filename,
#                                         ".".join([file_info["Net"],
#                                                   file_info["Station"],
#                                                   file_info["Location"],
#                                                   y_term,
#                                                   file_info["Date"],
#                                                   file_info["Time"],
#                                                   file_info["Processing"],
#                                                   file_info["Waveform"],
#                                                   file_info["Format"]])
#            if os.path.exists(test_filename):
#                y_metadata = self._get_metadata_from_file(test_filename)
#                break
#            else:
#                y_metadata = None
#        if not y_metadata:
#            raise ValueError(
#                "No y-component found for %s" %
#                ".".join([file_info[key] for key in FILE_INFO_KEY]))
#        # Z-component (if it exists)
#        z_terms = ["HNZ", "HLZ", "HGZ"]
#        for z_term in z_terms:
#            test_filename = os.path.join(self.filename,
#                                         ".".join([file_info["Net"],
#                                                   file_info["Station"],
#                                                   file_info["Location"],
#                                                   z_term,
#                                                   file_info["Date"],
#                                                   file_info["Time"],
#                                                   file_info["Processing"],
#                                                   file_info["Waveform"],
#                                                   file_info["Format"]])
#            if os.path.exists(test_filename):
#                z_metadata = self._get_metadata_from_file(test_filename)
#                break
#            else:
#                z_metadata = None
#        return {"X": x_metadata, "Y": y_metadata, "Z": z_metadata}


    def parse_metadata(self, metadata, file_str):
        """
        Parses the metadata dictionary
        """
        # Waveform ID - in this case we use the file info string
        wfid = "|".format([
            file_info[key]
            for key in ["Net", "Station", "Location", "Date", "Time"]])
        # Get event information
        event = self._parse_event(metadata["X"], file_str)
        # Get Distance information
        distance = self._parse_distance_data(metadata["X"], file_str)
        # Get site data
        site = self._parse_site_data(metadata["X"])
        


    def _parse_event(self, metadata, file_str):
        """
        Parses the event metadata to return an instance of the :class:
        smtk.sm_database.Earthquake
        """
        # Date and time
        year, month, day = (_to_int(metadata["EVENT_DATE_YYYYMMDD"][:4]),
                            _to_int(metadata["EVENT_DATE_YYYYMMDD"][4:6]),
                            _to_int(metadata["EVENT_DATE_YYYYMMDD"][6:]))
        hour, minute, second = (_to_int(metadata["EVENT_TIME_HHMMSS"][:4]),
                                _to_int(metadata["EVENT_TIME_HHMMSS"][4:6]),
                                _to_int(metadata["EVENT_TIME_HHMMSS"][6:]))
        eq_datetime = datetime(year, month, day, hour, minute, second)
        # Event ID and Name
        eq_id = metadata["EVENT_ID"]
        eq_name = metadata["EVENT_NAME"]
        # Get magnitudes
        m_w = _to_float(metadata["MAGNITUDE_W"])
        mag_list = []
        if m_w:
            moment_mag = Magnitude(m_w, "Mw",
                                   source=metadata["MAGNITUDE_W_REFERENCE"])
            mag_list.append(moment_mag)
        else:
            moment_mag = None
        m_l = _to_float(metadata["MAGNITUDE_L"])
        if m_l:
            local_mag = Magnitude(m_l, "ML",
                                   source=metadata["MAGNITUDE_L_REFERENCE"])
            mag_list.append(local_mag)
        else:
            local_mag = None
        if moment_mag:
            pref_mag = moment_mag
        elif local_mag:
            pref_mag = local_mag
        else:
            raise ValueError("Record %s has no magnitude!" % file_str)
        # Get focal mechanism data - here only the general type is reported
        foc_mech = FocalMechanism(eq_id, eq_name, None, None,
            mechanism_type=ESMD_MECHANISM_TYPE[metadata["FOCAL_MECHANISM"]])
        # Build event
        eqk = Earthquake(eq_id, eq_name, eq_datetime,
            _to_float(metadata["EVENT_LONGITUDE_DEGREE"]),
            _to_float(metadata["EVENT_LATITUDE_DEGREE"]),
            _to_float(metadata["EVENT_DEPTH_KM"]),
            pref_mag,
            foc_mech)
        eqk.magnitude_list = mag_list
        return eqk

    def _parse_distance_data(self, metadata, file_str, eqk):
        """
        Parses the event metadata to return an instance of the :class:
        smtk.sm_database.RecordDistance
        """
        repi = _to_float(metadata["EPICENTRAL_DISTANCE_KM"])
        # No hypocentral distance in file - calculate from event
        if eqk.depth:
            rhypo = sqrt(repi ** 2. + eqk.depth ** 2.)
        else:
            rhypo = copy(repi)
        azimuth = Point(eqk.longitude, eqk.latitude, eqk.depth).azimuth(Point(
            _to_float(metadata["STATION_LONGITUDE_DEGREE"]),
            _to_float(metadata["STATION_LATITUDE_DEGREE"])))
        dists = RecordDistance(repi, rhypo)
        dists.azimuth = azimuth
        return dists

    
    def _parse_site_data(self, metadata):
        """
        Parses the site metadata
        """
        site = RecordSite(
            "|".join([metadata["NETWORK"], metadata["STATION_CODE"]]),
            metadata["STATION_CODE"],
            metadata["STATION_NAME"],
            _to_float(metadata["STATION_LONGITUDE_DEGREE"]),
            _to_float(metadata["STATION_LATITUDE_DEGREE"]),
            _to_float(metadata["STATION_ELEVATION_M"]),
            vs30=_to_float(metadata["VS30"]))
        site.morphology = metadata["MORPHOLOGIC_CLASSIFICATION"]
        if metadata["SITE_CLASSIFICATION_EC8"]:
            if "*" in metadata["SITE_CLASSIFICATION_EC8"]:
                site.ec8 = metadata["SITE_CLASSIFICATION_EC8"][:-1]
                site.vs30_measured = False
            else:
                site.ec8 = metadata["SITE_CLASSIFICATION_EC8"]
        elif site.vs30:
            site.ec8 = site.get_ec8_class()
            site.nehrp = site.get_nehrp_class()
        else:
            pass
        return site
            
            
    def _parse_processing_data(self, metadata):
        """
        Parses the information regarding the record processing
        """
        

    def _get_component_data(self, metadata):
        """
        Returns the information specific to a component
        """
        units = "cm/s/s" if metadata["UNITS"] == "cm/s^2" else\
            metadata["UNITS"]
        sampling_interval = _to_float(metadata["SAMPLING_INTERVAL_S"])
        nsamp = _to_int(metadata["NDATA"])
        # Baseline correction
        baseline = {"Type": metadata["BASELINE_CORRECTION"]}
        filter_info = {"Type": metadata["FILTER_TYPE"],
                       "Order": _to_int(metadata["FILTER_ORDER"]),
                       "Low-Cut": _to_float(metadata["LOW_CUT_FREQUENCY_HZ"]),
                       "High-Cut": _to_float(metadata["HIGH_CUT_FREQUENCY_HZ"])
                       }
        
                       




