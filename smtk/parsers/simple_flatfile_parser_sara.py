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
# In order to define default fault dimension import scaling relationships
from openquake.hazardlib.scalerel.strasser2010 import StrasserInterface, StrasserIntraslab
from openquake.hazardlib.scalerel.wc1994 import WC1994

HEADER_LIST = Set([
    'Record Sequence Number', 'EQID', 'Earthquake Name', 'Country', 'Year', 
    'Month', 'Day', 'Hour', 'Minute', 'Second', 
    'Epicenter Latitude (deg; positive N)',
    'Epicenter Longitude (deg; positive E)', 'Hypocenter Depth (km)',
    'Magnitude', 'Magnitude type', 'Magnitude uncertainty', 'Mo (dyne.cm)',
    'Tectonic environment (Crustal; Inslab; Interface; Stable; Geothermal; Volcanic; Oceanic_crust)',
    'Earthquake in Extensional Regime: 1=Yes; 0=No',
    'Nodal Plane 1 Strike (deg)',
    'Nodal Plane 1 Dip (deg)', 'Nodal Plane 1 Rake Angle (deg)', 
    'Nodal Plane 2 Strike (deg)', 'Nodal Plane 2 Dip (deg)',
    'Nodal Plane 2 Rake Angle (deg)', 'Fault Plane (1; 2; X)',
    'Style-of-Faulting (S; R; N; U)', 'Fault Name',
    'Depth to Top Of Fault Rupture Model',
    'Fault Rupture Length (km)', 'Fault Rupture Width (km)',
    'Along-strike Hypocenter location on the fault (fraction between 0 and 1)',
    'Along-width Hypocenter location on the fault (fraction between 0 and 1)',
    'Static Stress Drop (bars)', 'Type Static Stress Drop',
    'Directivity flag (Y; N; U)', 'Station ID', 'Station Code', 
    'Station Latitude (deg positive N)', 'Station Longitude (deg positive E)',
    'Station Elevation (m)',
    'Site Class (Hard Rock; Rock; Stiff Soil; Soft Soil)',
    'Preferred NEHRP Based on Vs30', 'Preferred Vs30 (m/s)',
    'Measured(1)/Inferred(2) Class', 'Sigma of Vs30 (in natural log Units)',
    'Depth to Basement Rock', 'Z1 (m)','Z2.5 (m)', 'Epicentral Distance (km)',
    'Hypocentral Distance (km)', 'Joyner-Boore Distance (km)', 
    'Rupture Distance (km)', 'Source to Site Azimuth (deg)', 'FW/HW Indicator',
    'Forearc/Backarc for subduction events', 'File Name (Horizontal 1)',
    'File Name (Horizontal 2)','File Name (Vertical)',
    'Digital (D)/Analog (A) Recording', 'Acceleration (A)/Velocity (V)',
    'Unit (cm/s/s; m/s/s; g)', 'File format', 'Processing flag',
    'Type of Filter','npass','nroll','HP-H1 (Hz)','HP-H2 (Hz)',
    'LP-H1 (Hz)', 'LP-H2 (Hz)', 'Factor', 'Lowest Usable Freq - H1 (Hz)',
    'Lowest Usable Freq - H2 (Hz)', 'Lowest Usable Freq - Ave. Component (Hz)',
    'Maximum Usable Freq - Ave. Component (Hz)', 'HP-V (Hz)', 'LP-V (Hz)',
    'Lowest Usable Freq - V (Hz)','Comments'])

FILTER_TYPE = {"BW": "Butterworth",
               "OR": "Ormsby"}

NEW_MECHANISM_TYPE = {"N": "Normal",
                      "S": "Strike-Slip",
                      "R": "Reverse",
                      "U": "Unknown"}

class SimpleFlatfileParserV9(SMDatabaseReader):
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
        self._get_site_id = self.database._get_site_id
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
        data_fields = ['Month', 'Day', 'Hour', 'Minute', 'Second']
        for f in data_fields:
            metadata[f] = metadata[f].zfill(2)

        # Date and Time
        year = get_int(metadata["Year"])
        month = get_int(metadata["Month"])
        day = get_int(metadata["Day"])
        hour = get_int(metadata["Hour"])
        minute = get_int(metadata["Minute"])
        second = get_int(metadata["Second"])
        eq_datetime = datetime(year, month, day, hour, minute, second)
        # Event ID and Name
        eq_id = metadata["EQID"]
        eq_name = metadata["Earthquake Name"]
        # Focal Mechanism
        focal_mechanism = self._get_focal_mechanism(eq_id, eq_name, metadata)
        
        focal_mechanism.scalar_moment = get_float(metadata["Mo (dyne.cm)"]) *\
            1E-7
        # Read magnitude
        pref_mag = Magnitude(get_float(metadata["Magnitude"]),
                             metadata["Magnitude type"],
                             sigma=get_float(metadata["Magnitude uncertainty"]))
        # Create Earthquake Class
        eqk = Earthquake(eq_id, eq_name, eq_datetime,
            get_float(metadata["Epicenter Longitude (deg; positive E)"]),
            get_float(metadata["Epicenter Latitude (deg; positive N)"]),
            get_float(metadata["Hypocenter Depth (km)"]),
            pref_mag,
            focal_mechanism,
            metadata["Country"])

        # hypocenter location
        f1 = get_float(metadata[
            "Along-strike Hypocenter location " +
            "on the fault (fraction between 0 and 1)"])
        f2 = get_float(metadata[
            "Along-width Hypocenter location " +
            "on the fault (fraction between 0 and 1)"])
        if f1 is None or f2 is None:
            hypo_loc = (0.5, 0.7)
        else:
            hypo_loc = (f1, f2)

        evt_tectonic_region = metadata["Tectonic environment (Crustal; Inslab; Interface; Stable; Geothermal; Volcanic; Oceanic_crust)"]
        if evt_tectonic_region == "Stable" or evt_tectonic_region == "Crustal":
            msr=WC1994()
        elif evt_tectonic_region == "Inslab":
            msr=StrasserIntraslab()
        elif evt_tectonic_region == "Interface":
            msr=StrasserInterface()

        # Warning rake set to 0.0 in scaling relationship
        area = msr.get_median_area(pref_mag.value,0.0)
        aspect_ratio = 1.5
        width_model = np.sqrt(area / aspect_ratio)
        length_model = aspect_ratio * width_model
        ztor_model = eqk.depth - width_model/2
        if ztor_model < 0:
            ztor_model = 0.0

        length = get_float(metadata["Fault Rupture Length (km)"])
        if length is None:
            length = length_model
        width = get_float(metadata["Fault Rupture Width (km)"])
        if width is None:
            width = width_model
        ztor = get_float(metadata["Depth to Top Of Fault Rupture Model"])
        if ztor is None:
            ztor=ztor_model
        
        # Rupture
        eqk.rupture = Rupture(eq_id,
            length,
            width,
            ztor,
            hypo_loc=hypo_loc)
        eqk.rupture.get_area()
        return eqk

    def _get_focal_mechanism(self, eq_id, eq_name, metadata):
        """
        Returns the focal mechanism information as an instance of the
        :class: smtk.sigma_database.FocalMechanism
        """
        nodal_planes = GCMTNodalPlanes()
        # By default nodal plane 1 is assumed to be the fault plane in smtk. Depending on 
        # parameter fault_plane import correct angles in nodal planes 1 and 2 (1 being the
        # fault plane)
        if metadata['Fault Plane (1; 2; X)'] == '1':
            nodal_planes.nodal_plane_1 = {
                "strike": get_float(metadata['Nodal Plane 1 Strike (deg)']),
                "dip": get_float(metadata['Nodal Plane 1 Dip (deg)']),
                "rake": get_float(metadata['Nodal Plane 1 Rake Angle (deg)'])}

            nodal_planes.nodal_plane_2 = {
                "strike": get_float(metadata['Nodal Plane 2 Strike (deg)']),
                "dip": get_float(metadata['Nodal Plane 2 Dip (deg)']),
                "rake": get_float(metadata['Nodal Plane 2 Rake Angle (deg)'])}
        elif metadata['Fault Plane (1; 2; X)'] == '2':
            nodal_planes.nodal_plane_1 = {
                "strike": get_float(metadata['Nodal Plane 2 Strike (deg)']),
                "dip": get_float(metadata['Nodal Plane 2 Dip (deg)']),
                "rake": get_float(metadata['Nodal Plane 2 Rake Angle (deg)'])}

            nodal_planes.nodal_plane_2 = {
                "strike": get_float(metadata['Nodal Plane 1 Strike (deg)']),
                "dip": get_float(metadata['Nodal Plane 1 Dip (deg)']),
                "rake": get_float(metadata['Nodal Plane 1 Rake Angle (deg)'])}
        elif metadata['Fault Plane (1; 2; X)'] == 'X': # Set default vertical fault with strike to North
            nodal_planes.nodal_plane_1 = {"strike": 0.0,
                                 "dip": 90.0,
                                 "rake": None}
            nodal_planes.nodal_plane_2 = {"strike": None,
                                 "dip": None,
                                 "rake": None}

        principal_axes = GCMTPrincipalAxes()
        mech_type =\
            NEW_MECHANISM_TYPE[metadata["Style-of-Faulting (S; R; N; U)"]]
        return FocalMechanism(eq_id, eq_name, nodal_planes, principal_axes,
            mechanism_type=mech_type)


    def _parse_distance_data(self, metadata):
        """
        Read in the distance related metadata and return an instance of the
        :class: smtk.sm_database.RecordDistance
        """
        distance = RecordDistance(
            get_float(metadata["Epicentral Distance (km)"]),
            get_float(metadata["Hypocentral Distance (km)"]),
            get_float(metadata["Joyner-Boore Distance (km)"]),
            get_float(metadata["Rupture Distance (km)"]))
        distance.azimuth = get_float(metadata["Source to Site Azimuth (deg)"])
        distance.hanging_wall = get_float(metadata["FW/HW Indicator"])
        return distance

    def _parse_site_data(self, metadata):
        """
        Returns the site data as an instance of the :class:
        smtk.sm_database.RecordSite
        """
        site = RecordSite(
            self._get_site_id(metadata["Station ID"]),
            metadata["Station ID"],
            metadata["Station Code"],
            get_float(metadata["Station Longitude (deg positive E)"]),
            get_float(metadata["Station Latitude (deg positive N)"]),
            get_float(metadata["Station Elevation (m)"]),
            get_float(metadata["Preferred Vs30 (m/s)"]),
            site_class=metadata['Site Class (Hard Rock; Rock; Stiff Soil; Soft Soil)'],
            network_code=None, # not provided
            )
            # network_code=metadata["Owner"])
        site.nehrp = metadata["Preferred NEHRP Based on Vs30"]
        site.vs30_measured_type = metadata["Measured(1)/Inferred(2) Class"]
        if site.vs30_measured_type in ["0", "5"]:
            site.vs30_measured = True
        else:
            site.vs30_measured = False
        site.vs30_uncertainty = get_float(
            metadata["Sigma of Vs30 (in natural log Units)"])
        site.z1pt0 = get_float(metadata["Z1 (m)"])
        site.z1pt5 = None
        # site.z1pt5 = get_float(metadata["Z1.5 (m)"])
        site.z2pt5 = get_float(metadata["Z2.5 (m)"])
        site.arc_location = metadata["Forearc/Backarc for subduction events"]
        site.instrument_type = metadata["Digital (D)/Analog (A) Recording"]
        return site
                           
    def _parse_processing_data(self, wfid, metadata):
        """
        Parses the information for each component
        """
        filter_params1 = {
            'Type': FILTER_TYPE[metadata["Type of Filter"]],
            'Order': None,
            'Passes': get_int(metadata['npass']),
            'NRoll': get_int(metadata['nroll']),
            'High-Cut': get_float(metadata["LP-H1 (Hz)"]),
            'Low-Cut': get_float(metadata["HP-H1 (Hz)"])}

        filter_params2 = {
            'Type': FILTER_TYPE[metadata["Type of Filter"]],
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
            units=metadata["Unit (cm/s/s; m/s/s; g)"])

        ycomp = Component(wfid, "2",
            ims=intensity_measures,
            longest_period=lup2,
            waveform_filter=filter_params2,
            units=metadata["Unit (cm/s/s; m/s/s; g)"])

        if get_float(metadata["Lowest Usable Freq - V (Hz)"]) is None:
            return xcomp, ycomp, None
        else:
            filter_params3 = {
                'Type': FILTER_TYPE[metadata["Type of Filter"]],
                'Order': None,
                'Passes': get_int(metadata['npass']),
                'NRoll': get_int(metadata['nroll']),
                'High-Cut': get_float(metadata["LP-V (Hz)"]),
                'Low-Cut': get_float(metadata["HP-V (Hz)"])}

            lup3 = 1. / get_float(metadata["Lowest Usable Freq - V (Hz)"])

            zcomp = Component(wfid, "V",
                ims=intensity_measures,
                longest_period=lup3,
                waveform_filter=filter_params3,
                units=metadata["Unit (cm/s/s; m/s/s; g)"])

            return xcomp, ycomp, zcomp


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
