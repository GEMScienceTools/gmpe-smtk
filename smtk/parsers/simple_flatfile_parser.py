#!/usr/bin/env/python

"""
Parser from a "Simple Flatfile + ascii format" to SMTK
"""
import os
import csv
import numpy as np
from sets import Set
from linecache import getline
from collections import OrderedDict
from datetime import datetime
import h5py
from smtk.sm_database import *
from smtk.sm_utils import convert_accel_units
from smtk.parsers.base_database_parser import (get_float, get_int,
                                               SMDatabaseReader,
                                               SMTimeSeriesReader,
                                               SMSpectraReader)


HEADER_LIST = Set(['Record Sequence Number', 'EQID', 'Station Sequence Number',
    'Earthquake Name', 'Country', 'Tectonic environement', 'YEAR', 'MODY',
    'HRMN', 'Hypocenter Latitude (deg)', 'Hypocenter Longitude (deg)',
    'Hypocenter Depth (km)', 'Earthquake Magnitude', 'Magnitude Type',
    'Uncertainty', 'Mo (dyne.cm)', 'Strike (deg)', 'Dip (deg)',
    'Rake Angle (deg)', 'Mechanism Based on Rake Angle', 'Style-of-Faulting',
    'Depth to Top Of Fault Rupture Model', 'Fault Rupture Length (km)',
    'Fault Rupture Width (km)', 
    'Earthquake in Extensional Regime: 1=Yes; 0=No', 'Station ID  No.',
    'Preferred NEHRP Based on Vs30', 'Preferred Vs30 (m/s)',
    'Measured/Inferred Class', 'Sigma of Vs30 (in natural log Units)',
    'Geological Unit', 'Geology', 'Owner', 'Station Latitude',
    'Station Longitude', 'Depth to Basement Rock', 'Z1 (m)', 'Z1.5 (m)',
    'Z2.5 (m)', 'EpiD (km)', 'HypD (km)', 'Joyner-Boore Dist. (km)',
    'Campbell R Dist. (km)', 'FW/HW Indicator', 'Source to Site Azimuth (deg)',
    'Forearc/Backarc for subduction events', 'File Name (Horizontal 1)',
    'File Name (Horizontal 2)', 'File Name (Vertical)', 'Type of Recording', 
    'Acceleration/Velocity', 'Unit', 'File format', 'Processing flag',
    'Type of Filter', 'npass', 'nroll', 'HP-H1 (Hz)', 'HP-H2 (Hz)',
    'LP-H1 (Hz)', 'LP-H2 (Hz)', 'Factor', 'Lowest Usable Freq - H1 (Hz)', 
    'Lowest Usable Freq - H2 (Hz)', 'Lowest Usable Freq - Ave. Component (Hz)',
    'Maximum Usable Freq - Ave. Component (Hz)', 'Comments'])


FILTER_TYPE = {"BW": "Butterworth",
               "OR": "Ormsby"}

class SimpleFlatfileParser(SMDatabaseReader):
    """
    Parser for strong motio database stored in a "Simple" flatfile format.
    Typically this format is an Excel spreadsheet, though for the current
    purposes it is assumed the spreadsheet is formatted as csv
    """
    def parse(self):
        """
        Parses the database
        """
        self._header_check()
        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"))
        metadata = []
        self.database = GroundMotionDatabase(self.id, self.name)
        for row in reader:
            self.database.records.append(self._parse_record(row))
        return self.database

    def _header_check(self):
        """
        Checks to see if any of the headers are missing, raises error if so.
        Also informs the user of redundent headers in the input file.
        """
        # Check which of the pre-defined headers are missing
        headers = Set((getline(self.filename, 1).rstrip("\n")).split(","))
        missing_headers = HEADER_LIST.difference(headers)
        if len(missing_headers) > 0:
            output_string = ", ".join([value for value in missing_headers])
            raise IOError("The following headers are missing from the input "
                          "file: %s" % output_string)

        additional_headers = headers.difference(HEADER_LIST)
        if len(additional_headers) > 0:
            for header in additional_headers:
                print "Header %s not recognised - ignoring this data!" % header
        return

    def _parse_record(self, metadata):
        """
        Parses the record information and returns an instance of the
        :class: smtk.sm_database.GroundMotionRecord 
        """
        # Waveform ID
        wfid = metadata["Record Sequence Number"]
        # Event information
        event = self._parse_event_data(metadata)
        # Distance Information
        distances = self._parse_distance_data(metadata)
        # Site information
        site = self._parse_site_data(metadata)
        # Components
        x_comp, y_comp, vertical = self._parse_processing_data(wfid, metadata)
        # Return record metadata
        lup = get_float(metadata["Lowest Usable Freq - Ave. Component (Hz)"])
        if lup:
            lup = 1. / lup
        sup = get_float(metadata["Maximum Usable Freq - Ave. Component (Hz)"])
        if sup:
            sup = 1. / sup
        return GroundMotionRecord(wfid,
            [metadata["File Name (Horizontal 1)"],
             metadata["File Name (Horizontal 2)"],
             metadata["File Name (Vertical)"]],
            event,
            distances,
            site,
            x_comp,
            y_comp,
            longest_period=lup,
            shortest_period=sup)

    def _parse_event_data(self, metadata):
        """
        Read in the distance related metadata and return an instance of the
        :class: smtk.sm_database.Earthquake

        """
        metadata["MODY"] = metadata["MODY"].zfill(4)
        metadata["HRMN"] = metadata["HRMN"].zfill(4)
        # Date and Time
        year = get_int(metadata["YEAR"])
        month = get_int(metadata["MODY"][:2])
        day = get_int(metadata["MODY"][2:])
        hour = get_int(metadata["HRMN"][:2])
        minute = get_int(metadata["HRMN"][2:])
        eq_datetime = datetime(year, month, day, hour, minute)
        # Event ID and Name
        eq_id = metadata["EQID"]
        eq_name = metadata["Earthquake Name"]
        # Focal Mechanism
        focal_mechanism = self._get_focal_mechanism(eq_id, eq_name, metadata)
        focal_mechanism.scalar_moment = get_float(metadata["Mo (dyne.cm)"]) *\
            1E-7
        # Read magnitude
        pref_mag = Magnitude(get_float(metadata["Earthquake Magnitude"]),
                             metadata["Magnitude Type"],
                             sigma=get_float(metadata["Uncertainty"]))
        # Create Earthquake Class
        eqk = Earthquake(eq_id, eq_name, eq_datetime,
            get_float(metadata["Hypocenter Longitude (deg)"]),
            get_float(metadata["Hypocenter Latitude (deg)"]),
            get_float(metadata["Hypocenter Depth (km)"]),
            pref_mag,
            focal_mechanism,
            metadata["Country"])
        # Rupture
        eqk.rupture = Rupture(eq_id,
            get_float(metadata["Fault Rupture Length (km)"]),
            get_float(metadata["Fault Rupture Width (km)"]),
            get_float(metadata["Depth to Top Of Fault Rupture Model"]))
        eqk.rupture.get_area()
        return eqk
            
    def _get_focal_mechanism(self, eq_id, eq_name, metadata):
        """
        Returns the focal mechanism information as an instance of the
        :class: smtk.sigma_database.FocalMechanism 
        """
        nodal_planes = GCMTNodalPlanes()
        nodal_planes.nodal_plane_1 = {
            "strike": get_float(metadata["Strike (deg)"]),
            "dip": get_float(metadata["Dip (deg)"]),
            "rake": get_float(metadata["Rake Angle (deg)"])}

        nodal_planes.nodal_plane2 = {"strike": None,
                                     "dip": None,
                                     "rake": None}
        principal_axes = GCMTPrincipalAxes()
        return FocalMechanism(eq_id, eq_name, nodal_planes, principal_axes,
            mechanism_type=metadata["Mechanism Based on Rake Angle"])

    def _parse_distance_data(self, metadata):
        """
        Read in the distance related metadata and return an instance of the
        :class: smtk.sm_database.RecordDistance
        """
        distance = RecordDistance(
            get_float(metadata["EpiD (km)"]),
            get_float(metadata["HypD (km)"]),
            get_float(metadata["Joyner-Boore Dist. (km)"]),
            get_float(metadata["Campbell R Dist. (km)"]))
        distance.azimuth = get_float(metadata["Source to Site Azimuth (deg)"])
        distance.hanging_wall = get_float(metadata["FW/HW Indicator"])
        return distance

    def _parse_site_data(self, metadata):
        """
        Returns the site data as an instance of the :class:
        smtk.sm_database.RecordSite
        """
        site = RecordSite(metadata["Station Sequence Number"],
                          metadata["Station ID  No."],
                          metadata["Station ID  No."],
                          get_float(metadata["Station Longitude"]),
                          get_float(metadata["Station Latitude"]),
                          None, # Elevation data not given
                          get_float(metadata["Preferred Vs30 (m/s)"]),
                          network_code=metadata["Owner"])
        site.nehrp = metadata["Preferred NEHRP Based on Vs30"]
        site.vs30_measured_type = metadata["Measured/Inferred Class"]
        if site.vs30_measured_type in ["0", "5"]:
            site.vs30_measured = True
        else:
            site.vs30_measured = False
        site.vs30_uncertainty = get_float(
            metadata["Sigma of Vs30 (in natural log Units)"])
        site.z1pt0 = get_float(metadata["Z1 (m)"])
        site.z1pt5 = get_float(metadata["Z1.5 (m)"])
        site.z2pt5 = get_float(metadata["Z2.5 (m)"])
        site.arc_location = metadata["Forearc/Backarc for subduction events"]
        site.instrument_type = metadata["Type of Recording"]
        return site
                           
    def _parse_processing_data(self, wfid, metadata):
        """
        Parses the information for each component
        """
        filter_params1 = {
            'Type': FILTER_TYPE[metadata["Type of Filter"]] ,
            'Order': None,
            'Passes': get_int(metadata['npass']),
            'NRoll': get_int(metadata['nroll']),
            'High-Cut': get_float(metadata["LP-H1 (Hz)"]),
            'Low-Cut': get_float(metadata["HP-H1 (Hz)"])}

        filter_params2 = {
            'Type': FILTER_TYPE[metadata["Type of Filter"]] ,
            'Order': None,
            'Passes': get_int(metadata['npass']),
            'NRoll': get_int(metadata['nroll']),
            'High-Cut': get_float(metadata["LP-H2 (Hz)"]),
            'Low-Cut': get_float(metadata["HP-H2 (Hz)"])}
        intensity_measures = {
            # All m - convert to cm
            'PGA': None,
            'PGV': None,
            'PGD': None
            }

        lup1 = 1. / get_float(metadata["Lowest Usable Freq - H1 (Hz)"])
        lup2 = 1. / get_float(metadata["Lowest Usable Freq - H2 (Hz)"])
        xcomp = Component(wfid, "1",
            ims=intensity_measures,
            longest_period=lup1,
            waveform_filter=filter_params1,
            units=metadata["Unit"])
        ycomp = Component(wfid, "2",
            ims=intensity_measures,
            longest_period=lup2,
            waveform_filter=filter_params2,
            units=metadata["Unit"]) 
        return xcomp, ycomp, None


class SimpleAsciiTimeseriesReader(SMTimeSeriesReader):
    """
    Parses a simple ascii representation of a record in which the first line
    contains the number of values and the time-step. Whilst the rest of the
    file contains the acceleration record
    """
    def parse_records(self):
        """
        Parses the record set
        """
        time_series = OrderedDict([
            ("X", {"Original": {}, "SDOF": {}}),
            ("Y", {"Original": {}, "SDOF": {}}),
            ("V", {"Original": {}, "SDOF": {}})])
             
        target_names = time_series.keys()
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                if iloc < 2:
                    # Expected horizontal component is missing - raise error
                    raise ValueError("Horizontal record %s is expected but "
                        "not found!" % ifile)
                else:
                    print "Vertical record file %s not found" % ifile
                    del time_series["V"]
                    continue
            else:
                time_series[target_names[iloc]]["Original"] = \
                    self._parse_time_history(ifile)
        if iloc < 2:
            del time_series["V"]

        return time_series

    def _parse_time_history(self, ifile):
        """
        Parses the time history from the file and returns a dictionary of
        time-series properties
        """
        output = {}
        accel = np.genfromtxt(ifile, skip_header=1)
        output["Acceleration"] = convert_accel_units(accel, self.units)
        nvals, time_step = (getline(ifile, 1).rstrip("\n")).split()
        output["Time-step"] = float(time_step)
        output["Number Steps"] = int(nvals)
        output["Units"] = "cm/s/s"
        output["PGA"] = np.max(np.fabs(output["Acceleration"]))
        return output
