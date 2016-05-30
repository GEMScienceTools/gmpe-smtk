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
from copy import copy
from openquake.hazardlib.geo.geodetic import geodetic_distance
from smtk.sm_utils import convert_accel_units, get_time_vector
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


def _get_filename_info(filename):
    """
    ESMD follows a specific naming convention. Return this information in
    a dictionary
    """
    file_info = filename.split(".")
    # Sometimes there are consecutive dots in the delimiter
    return OrderedDict([
        (FILE_INFO_KEY[i], file_info[i]) for i in range(len(file_info))
        ])


def _get_filename_from_info(self, file_info):
    """
    Given a file info dictionary return the corresponding filename
    """
    return ".".join([file_info[key] for key in file_info.keys()])


def _get_metadata_from_file(file_str):
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

def _get_xyz_metadata(file_dict):
    """
    The ESM is a bit messy mixing the station codes. Returns the metadata
    corrsponding to the x-, y- and z-component of each of the records
    """
    metadata = {}
    if file_dict["Time-Series"]["X"]:
        metadata["X"] = _get_metadata_from_file(
            file_dict["Time-Series"]["X"])
    
    if file_dict["Time-Series"]["Y"]:
        metadata["Y"] = _get_metadata_from_file(
            file_dict["Time-Series"]["Y"])
    
    if file_dict["Time-Series"]["Z"]:
        metadata["Z"] = _get_metadata_from_file(
            file_dict["Time-Series"]["Z"])
    return metadata

ESMD_MECHANISM_TYPE = {"NF": -90., "SS": 0.0, "TF": 90.0, "U": 0.0}
DATA_TYPE_KEYS = {
    "ACCELERATION": "PGA_",
    "VELOCITY": "PGV_",
    "DISPLACEMENT": "PGD_",
    "ACCELERATION RESPONSE SPECTRUM": "PGA_" ,
    "PSEUDO-VELOCITY RESPONSE SPECTRUM": "PGV_" ,
    "DISPLACEMENT RESPONSE SPECTRUM": "PGD_"}


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
            metadata = _get_xyz_metadata(file_dict)
            self.database.records.append(self.parse_metadata(metadata,
                                                             file_dict))
        return self.database

    def _sort_files(self):
        """
        Searches through the directory and organise the files associated
        with a particular recording into a dictionary
        """
        skip_files = []
        for file_str in os.listdir(self.filename):
            if (file_str in skip_files) or ("ds_store" in file_str.lower()) or\
                ("DIS.ASC" in file_str[-7:]) or ("VEL.ASC" in file_str[-7:]):
                continue
            file_dict = {"Time-Series": {"X": None, "Y": None, "Z": None},
                         "PSV": {"X": None, "Y": None, "Z": None},
                         "SA": {"X": None, "Y": None, "Z": None},
                         "SD": {"X": None, "Y": None, "Z": None}}


            file_info = _get_filename_info(file_str)
            code1 = ".".join([file_info[key] for key in ["Net", "Station",
                                                         "Location"]])
            code2 = ".".join([file_info[key] for key in ["DM", "Date", "Time",
                                                         "Processing",
                                                         "Waveform"]])
            for x_term in ["HNE", "HN2", "HLE", "HL2", "HGE", "HG2"]:
                if file_dict["Time-Series"]["X"]:
                    continue
                test_filestring = "{:s}.{:s}.{:s}.ASC".format(code1,
                                                            x_term,
                                                            code2)

                test_filename = os.path.join(self.filename, test_filestring)

                if os.path.exists(test_filename):
                    file_dict["Time-Series"]["X"] = test_filename
                    skip_files.append(os.path.split(test_filename)[-1])
                    # Get SA, SD and PSV
                    # SA - x-component
                    sa_filename = test_filename.replace("ACC", "SA")
                    if os.path.exists(sa_filename):
                        file_dict["SA"]["X"] = sa_filename
                        skip_files.append(os.path.split(sa_filename)[-1])
                    # SD - x-component
                    sd_filename = test_filename.replace("ACC", "SD")
                    if os.path.exists(sd_filename):
                        file_dict["SD"]["X"] = sd_filename
                        skip_files.append(os.path.split(sd_filename)[-1])
                    # PSV - x-component
                    psv_filename = test_filename.replace("ACC", "PSV")
                    if os.path.exists(psv_filename):
                        file_dict["PSV"]["X"] = psv_filename
                        skip_files.append(os.path.split(psv_filename)[-1])
                    for y_term in ["N", "1", "3"]:
                        y_filename = test_filename.replace(
                            x_term, "{:s}{:s}".format(x_term[:2], y_term))
                        if os.path.exists(y_filename):
                            # Get y-time series
                            file_dict["Time-Series"]["Y"] = y_filename
                            skip_files.append(os.path.split(y_filename)[-1])
                            # SA
                            sa_filename = y_filename.replace("ACC", "SA")
                            if os.path.exists(sa_filename):
                                file_dict["SA"]["Y"] = sa_filename
                                skip_files.append(
                                    os.path.split(sa_filename)[-1])
                            # SD
                            sd_filename = y_filename.replace("ACC", "SD")
                            if os.path.exists(sd_filename):
                                file_dict["SD"]["Y"] = sd_filename
                                skip_files.append(
                                    os.path.split(sd_filename)[-1])
                            # PSV
                            psv_filename = y_filename.replace("ACC", "PSV")
                            if os.path.exists(psv_filename):
                                file_dict["PSV"]["Y"] = psv_filename
                                skip_files.append(
                                    os.path.split(psv_filename)[-1])
                    # Get vertical files
                    v_filename = test_filename.replace(x_term,
                        "{:s}Z".format(x_term[:2]))
                    if os.path.exists(v_filename):
                        # Get z-time series
                        file_dict["Time-Series"]["Z"] = v_filename
                        skip_files.append(os.path.split(v_filename)[-1])
                        # Get SA 
                        sa_filename = v_filename.replace("ACC", "SA")
                        if os.path.exists(sa_filename):
                            file_dict["SA"]["Z"] = sa_filename
                            skip_files.append(os.path.split(sa_filename)[-1])
                        # Get SD
                        sd_filename = v_filename.replace("ACC", "SD")
                        if os.path.exists(sd_filename):
                            file_dict["SD"]["Z"] = sd_filename
                            skip_files.append(os.path.split(sd_filename)[-1])
                        # Get PSV
                        psv_filename = v_filename.replace("ACC", "PSV")
                        if os.path.exists(psv_filename):
                            file_dict["PSV"]["Z"] = psv_filename
                            skip_files.append(os.path.split(psv_filename)[-1])
            self.ORGANIZER.append(file_dict)

    def parse_metadata(self, metadata, file_dict):
        """
        Parses the metadata dictionary
        """
        # Get the file info dictionary for the X-record
        file_str = file_dict["Time-Series"]["X"]
        file_info = _get_filename_info(file_str)
        # Waveform ID - in this case we use the file info string
        wfid = "_".join([
            file_info[key]
            for key in ["Net", "Station", "Location", "Date", "Time"]])
        wfid = wfid.replace(os.sep, "_")
        # Get event information
        event = self._parse_event(metadata["X"], file_str)
        # Get Distance information
        distance = self._parse_distance_data(metadata["X"], file_str, event)
        # Get site data
        site = self._parse_site_data(metadata["X"])
        # Get component and processing data
        xcomp, ycomp, zcomp = self._parse_processing_data(wfid, metadata)
        # 

        return GroundMotionRecord(wfid,
            [os.path.split(file_dict["Time-Series"]["X"])[1],
             os.path.split(file_dict["Time-Series"]["Y"])[1],
             os.path.split(file_dict["Time-Series"]["Z"])[1]],
            event,
            distance,
            site,
            xcomp,
            ycomp,
            vertical=zcomp,
            ims=None,
            spectra_file=[os.path.split(file_dict["SA"]["X"])[1],
                          os.path.split(file_dict["SA"]["Y"])[1],
                          os.path.split(file_dict["SA"]["Z"])[1]])

    def _parse_event(self, metadata, file_str):
        """
        Parses the event metadata to return an instance of the :class:
        smtk.sm_database.Earthquake
        """
        # Date and time
        year, month, day = (_to_int(metadata["EVENT_DATE_YYYYMMDD"][:4]),
                            _to_int(metadata["EVENT_DATE_YYYYMMDD"][4:6]),
                            _to_int(metadata["EVENT_DATE_YYYYMMDD"][6:]))
        hour, minute, second = (_to_int(metadata["EVENT_TIME_HHMMSS"][:2]),
                                _to_int(metadata["EVENT_TIME_HHMMSS"][2:4]),
                                _to_int(metadata["EVENT_TIME_HHMMSS"][4:]))
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
            vs30=_to_float(metadata["VS30_M/S"]))
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
            
    def _parse_processing_data(self, wfid, metadata):
        """
        Parses the information regarding the record processing
        """
        xcomp = self._parse_component_data(wfid, metadata["X"])
        ycomp = self._parse_component_data(wfid, metadata["Y"])
        if "Z" in metadata.keys():
            zcomp = self._parse_component_data(wfid, metadata["Z"])
        else:
            zcomp = None
        return xcomp, ycomp, zcomp

    def _parse_component_data(self, wfid, metadata):
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
        data_type = metadata["DATA_TYPE"]
        if data_type == "ACCELERATION":
            intensity_measures = {"PGA": _to_float(metadata[
                DATA_TYPE_KEYS[data_type] + metadata["UNITS"].upper()])}
        elif data_type == "VELOCITY":
            intensity_measures = {"PGV": _to_float(metadata[
                DATA_TYPE_KEYS[data_type] + metadata["UNITS"].upper()])}
        elif data_type == "DISPLACEMENT":
            intensity_measures = {"PGD": _to_float(metadata[
                DATA_TYPE_KEYS[data_type] + metadata["UNITS"].upper()])}
        else:
            # Unknown
            pass
        component = Component(wfid, metadata["STREAM"],
                              ims=intensity_measures,
                              waveform_filter=filter_info,
                              baseline=baseline,
                              units=units)
        if metadata["LATE/NORMAL_TRIGGERED"] == "LT":
            component.late_trigger = True
        return component
                         

class ESMTimeSeriesParser(SMTimeSeriesReader):
    """
    Parses time series in the European Strong Motion Database Format
    """ 
        
    def parse_records(self, record=None):
        """
        Parses the time series
        """
        time_series = OrderedDict([
            ("X", {"Original": {}, "SDOF": {}}),
            ("Y", {"Original": {}, "SDOF": {}}),
            ("V", {"Original": {}, "SDOF": {}})])
             
        target_names = time_series.keys()
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                continue
            else:
                time_series[target_names[iloc]]["Original"] = \
                    self._parse_time_history(ifile)
        return time_series

    def _parse_time_history(self, ifile):
        """
        Parses the time history
        """
        print ifile
        # Build the metadata dictionary again
        metadata = _get_metadata_from_file(ifile)
        self.number_steps = _to_int(metadata["NDATA"])
        self.time_step = _to_float(metadata["SAMPLING_INTERVAL_S"])
        self.units = metadata["UNITS"]
        # Get acceleration data
        accel = np.genfromtxt(ifile, skip_header=64)
        if "DIS" in ifile:
            pga = None
            pgd = np.fabs(_to_float(metadata["PGD_" +
                          metadata["UNITS"].upper()]))
        else:
            pga = np.fabs(_to_float(
                          metadata["PGA_" + metadata["UNITS"].upper()]))
            pgd = None
            if "s^2" in self.units:
                self.units = self.units.replace("s^2", "s/s")
        
        output = {
            # Although the data will be converted to cm/s/s internally we can
            # do it here too
            "Acceleration": convert_accel_units(accel, self.units),
            "Time": get_time_vector(self.time_step, self.number_steps),
            "Time-step": self.time_step,
            "Number Steps": self.number_steps,
            "Units": self.units,
            "PGA": pga,
            "PGD": pgd
        }
        return output

class ESMSpectraParser(SMSpectraReader):
    """
    Parses response spectra in the European Strong Motion Database Format
    """
    def parse_spectra(self):
        """
        Parses the response spectra - 5 % damping is assumed
        """
        sm_record = OrderedDict([
            ("X", {"Scalar": {}, "Spectra": {"Response": {}}}), 
            ("Y", {"Scalar": {}, "Spectra": {"Response": {}}}), 
            ("V", {"Scalar": {}, "Spectra": {"Response": {}}})])
        target_names = sm_record.keys()
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                continue
            metadata = _get_metadata_from_file(ifile)
            data = np.genfromtxt(ifile, skip_header=64)

            units = metadata["UNITS"]
            if "s^2" in units:
                units = units.replace("s^2", "s/s")
            pga = convert_accel_units(
                _to_float(metadata["PGA_" + metadata["UNITS"].upper()]),
                units)
            periods = data[:, 0]
            s_a = convert_accel_units(data[:, 1], units)
                                      
            sm_record[target_names[iloc]]["Spectra"]["Response"] = { 
                "Periods": periods,
                "Number Periods" : len(periods),
                "Acceleration" : {"Units": "cm/s/s"},
                "Velocity" : None,
                "Displacement" : None,
                "PSA" : None,
                "PSV" : None}
            sm_record[target_names[iloc]]["Spectra"]["Response"]\
                     ["Acceleration"]["damping_05"] = s_a
            # If the displacement file exists - get the data from that directly
            sd_file = ifile.replace("SA.ASC", "SD.ASC")
            if os.path.exists(sd_file):
                # SD data
                sd_data = np.genfromtxt(sd_file, skip_header=64)
                # Units should be cm
                sm_record[target_names[iloc]]["Spectra"]["Response"]\
                    ["Displacement"] = {"damping_05": sd_data[:, 1],
                                        "Units": "cm"}
        return sm_record

               



            
        
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




