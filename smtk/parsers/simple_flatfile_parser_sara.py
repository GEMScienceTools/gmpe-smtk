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
Parser from a "Simple Flatfile + ascii format" to SMTK
"""
import os
import csv
import numpy as np
import copy
import h5py
from sets import Set
from linecache import getline
from collections import OrderedDict
from datetime import datetime
# In order to define default fault dimension import scaling relationships
from openquake.hazardlib.scalerel.strasser2010 import (StrasserInterface,
                                                       StrasserIntraslab)
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.geo.mesh import Mesh, RectangularMesh
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
import smtk.trellis.configure as rcfg
from smtk.sm_database import *
from smtk.sm_utils import convert_accel_units
from smtk.parsers.base_database_parser import (get_float, get_int,
                                               get_positive_float,
                                               get_positive_int,
                                               SMDatabaseReader,
                                               SMTimeSeriesReader,
                                               SMSpectraReader)

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
               "OR": "Ormsby",
               "Ormsby": "Ormsby",
               "Acausal Butterworth": "Acausal Butterworth",
               "Causal Butterworth": "Causal Butterworth"}

NEW_MECHANISM_TYPE = {"N": "Normal",
                      "S": "Strike-Slip",
                      "R": "Reverse",
                      "U": "Unknown",
                      "RO": "Reverse-Oblique",
                      "NO": "Normal-Oblique"}

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
        HEADER_LIST1 = copy.deepcopy(HEADER_LIST)
        self._header_check(HEADER_LIST1)
        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"))
        metadata = []
        self.database = GroundMotionDatabase(self.id, self.name)
        self._get_site_id = self.database._get_site_id
        for row in reader:
            self.database.records.append(self._parse_record(row))
        return self.database

    def _header_check(self, headerslist):
        """
        Checks to see if any of the headers are missing, raises error if so.
        Also informs the user of redundent headers in the input file.
        """
        # Check which of the pre-defined headers are missing
        headers = Set((getline(self.filename, 1).rstrip("\n")).split(","))
        missing_headers = headerslist.difference(headers)
        if len(missing_headers) > 0:
            output_string = ", ".join([value for value in missing_headers])
            raise IOError("The following headers are missing from the input "
                          "file: %s" % output_string)

        additional_headers = headers.difference(headerslist)
        if len(additional_headers) > 0:
            for header in additional_headers:
                print("Header %s not recognised - ignoring this data!" %
                      header)
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
        # Site information
        site = self._parse_site_data(metadata)
        # Distance Information
        distances = self._parse_distance_data(event, site, metadata)
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
        year, month, day, hour, minute, second = self._validate_datetime(
            metadata)
        #year = get_int(metadata["Year"])
        #month = get_int(metadata["Month"])
        #day = get_int(metadata["Day"])
        #hour = get_int(metadata["Hour"])
        #minute = get_int(metadata["Minute"])
        #second = get_int(metadata["Second"])
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

        eqk.tectonic_region = metadata["Tectonic environment (Crustal; Inslab; Interface; Stable; Geothermal; Volcanic; Oceanic_crust)"]
        if (eqk.tectonic_region == "Stable" or
            eqk.tectonic_region == "Crustal" or
            eqk.tectonic_region == "Oceanic_crust"):
            msr=WC1994()
        elif eqk.tectonic_region == "Inslab":
            msr=StrasserIntraslab()
        elif eqk.tectonic_region == "Interface":
            msr=StrasserInterface()

        # Warning rake set to 0.0 in scaling relationship - applies only
        # to WC1994
        area = msr.get_median_area(pref_mag.value, 0.0)
        aspect_ratio = 1.5
        width_model = np.sqrt(area / aspect_ratio)
        length_model = aspect_ratio * width_model
        ztor_model = eqk.depth - width_model / 2.
        if ztor_model < 0:
            ztor_model = 0.0

        length = get_positive_float(metadata["Fault Rupture Length (km)"])
        if length is None:
            length = length_model
        width = get_positive_float(metadata["Fault Rupture Width (km)"])
        if width is None:
            width = width_model
        ztor = get_float(metadata["Depth to Top Of Fault Rupture Model"])
        if ztor is None:
            ztor=ztor_model

        # Rupture
        eqk.rupture = Rupture(eq_id,
                              eq_name,
                              pref_mag,
                              length,
                              width,
                              ztor,
                              hypo_loc=hypo_loc)
#            get_float(metadata["Fault Rupture Length (km)"]),
#            get_float(metadata["Fault Rupture Width (km)"]),
#            get_float(metadata["Depth to Top Of Fault Rupture Model"]),
#            hypo_loc=hypo_loc)
        eqk.rupture.get_area()
        return eqk
    
    @staticmethod
    def _validate_datetime(metadata):
        """
        SARA flatfile should be formatted correctly but other flatfiles
        are prone to bad datetime values - these will be overwritten
        """
        return (get_int(metadata["Year"]), get_int(metadata["Month"]),
                get_int(metadata["Day"]), get_int(metadata["Hour"]),
                get_int(metadata["Minute"]), get_int(metadata["Second"]))

    def _get_focal_mechanism(self, eq_id, eq_name, metadata):
        """
        Returns the focal mechanism information as an instance of the
        :class: smtk.sigma_database.FocalMechanism
        """
        nodal_planes = GCMTNodalPlanes()
        # By default nodal plane 1 is assumed to be the fault plane in smtk.
        # Depending on parameter fault_plane import correct angles in nodal
        # planes 1 and 2 (1 being the fault plane)
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
        elif metadata['Fault Plane (1; 2; X)'] == 'X':
            # Check if values for strike or dip are given otherwise set 
            # strike=0 and dip=90 and fill strike and dip for fault plane 1
            # What can we do for rake?
            strike = get_float(metadata['Nodal Plane 1 Strike (deg)'])
            if strike is None:
                strike = get_float(metadata['Nodal Plane 2 Strike (deg)'])
            if strike is None:
                strike = 0.0
            dip = get_float(metadata['Nodal Plane 1 Dip (deg)'])
            if dip is None:
                dip = get_float(metadata['Nodal Plane 2 Dip (deg)'])
            if dip is None:
                dip = 90.0
            nodal_planes.nodal_plane_1 = {"strike": strike,
                                 "dip": dip,
                                 "rake": None}
            nodal_planes.nodal_plane_2 = {"strike": None,
                                 "dip": None,
                                 "rake": None}

        nodal_planes = self._check_mechanism(nodal_planes)
        principal_axes = GCMTPrincipalAxes()
        mech_type =\
            NEW_MECHANISM_TYPE[metadata["Style-of-Faulting (S; R; N; U)"]]
        return FocalMechanism(eq_id, eq_name, nodal_planes, principal_axes,
            mechanism_type=mech_type)

    def _check_mechanism(self, nodal_planes):
        """
        Verify that the nodal planes are valid, and default nodal plane 1
        to a "null" plane if not
        """
        if nodal_planes.nodal_plane_1["strike"]:
            nodal_planes.nodal_plane_1["strike"] = \
                nodal_planes.nodal_plane_1["strike"] % 360.0
        if nodal_planes.nodal_plane_2["strike"]:
            nodal_planes.nodal_plane_2["strike"] = \
                nodal_planes.nodal_plane_2["strike"] % 360.0

        valid_plane_1 = nodal_planes.nodal_plane_1["strike"] >= 0.0 and\
                        nodal_planes.nodal_plane_1["strike"] < 360.0 and\
                        nodal_planes.nodal_plane_1["dip"] > 0.0 and\
                        nodal_planes.nodal_plane_1["dip"] <= 90.0 and\
                        nodal_planes.nodal_plane_1["rake"] >= -180.0 and\
                        nodal_planes.nodal_plane_1["rake"] <= 180.0
        valid_plane_2 = nodal_planes.nodal_plane_2["strike"] >= 0.0 and\
                        nodal_planes.nodal_plane_2["strike"] <= 360.0 and\
                        nodal_planes.nodal_plane_2["dip"] > 0.0 and\
                        nodal_planes.nodal_plane_2["dip"] <= 90.0 and\
                        nodal_planes.nodal_plane_2["rake"] >= -180.0 and\
                        nodal_planes.nodal_plane_2["rake"] <= 180.0
        if valid_plane_1:
            return nodal_planes
        
        # If nodal plane 2 is valid then swap over
        if valid_plane_2:
            np1 = copy.deepcopy(nodal_planes.nodal_plane_1)
            np2 = copy.deepcopy(nodal_planes.nodal_plane_2)
            nodal_planes.nodal_plane_1 = np2
            nodal_planes.nodal_plane_2 = np1
        else:
            nodal_planes.nodal_plane_1 = {"strike": 0.0,
                                          "dip": 90.0,
                                          "rake": 0.0}
        return nodal_planes


                                

    def _parse_distance_data(self, event, site, metadata):
        """
        Read in the distance related metadata and return an instance of the
        :class: smtk.sm_database.RecordDistance
        """
        # Compute various distance metrics
        # Add calculation of Repi, Rhypo from event and station localizations 
        # (latitudes, longitudes, depth, elevation)?
        target_site = Mesh(np.array([site.longitude]),
                           np.array([site.latitude]),
                           np.array([-site.altitude / 1000.0]))
        # Warning ratio fixed to 1.5
        ratio=1.5
        surface_modeled = rcfg.create_planar_surface(
            Point(event.longitude, event.latitude, event.depth),
            event.mechanism.nodal_planes.nodal_plane_1['strike'],
            event.mechanism.nodal_planes.nodal_plane_1['dip'],
            event.rupture.area,
            ratio)
        hypocenter = rcfg.get_hypocentre_on_planar_surface(
            surface_modeled,
            event.rupture.hypo_loc)
        try:
            surface_modeled._create_mesh()
        except:
            dip = surface_modeled.get_dip()
            dip_dir = (surface_modeled.get_strike() - 90.) % 360.
            ztor = surface_modeled.top_left.depth
            d_x = ztor * np.tan(np.radians(90.0 - dip))
            top_left_surface = surface_modeled.top_left.point_at(d_x,
                                                                 -ztor,
                                                                 dip_dir)
            top_left_surface.depth = 0.
            top_right_surface = surface_modeled.top_right.point_at(d_x,
                                                                   -ztor,
                                                                   dip_dir)
            top_right_surface.depth = 0.
            surface_modeled = SimpleFaultSurface.from_fault_data(
                Line([top_left_surface, top_right_surface]),
                surface_modeled.top_left.depth,
                surface_modeled.bottom_left.depth,
                surface_modeled.get_dip(),
                1.0)
        
        # Rhypo
        Rhypo = get_float(metadata["Hypocentral Distance (km)"])
        if Rhypo is None:
            Rhypo = hypocenter.distance_to_mesh(target_site)
        # Repi
        Repi = get_float(metadata["Epicentral Distance (km)"])
        if Repi is None:
            Repi= hypocenter.distance_to_mesh(target_site, with_depths=False)
        # Rrup
        Rrup = get_float(metadata["Rupture Distance (km)"])
        if Rrup is None:
            Rrup = surface_modeled.get_min_distance(target_site)
        # Rjb
        Rjb = get_float(metadata["Joyner-Boore Distance (km)"])
        if Rjb is None:
            Rjb = surface_modeled.get_joyner_boore_distance(target_site)
        # Need to check if Rx and Ry0 are consistant with the other metrics
        # when those are coming from the flatfile?
        # Rx
        Rx = surface_modeled.get_rx_distance(target_site)
        # Ry0
        Ry0 = surface_modeled.get_ry0_distance(target_site)
        
        distance = RecordDistance(
            repi = float(Repi),
            rhypo = float(Rhypo),
            rjb = float(Rjb),
            rrup = float(Rrup),
            r_x = float(Rx),
            ry0 = float(Ry0))
        distance.azimuth = get_float(metadata["Source to Site Azimuth (deg)"])
        #distance.hanging_wall = get_float(metadata["FW/HW Indicator"])
        if metadata["FW/HW Indicator"] == "HW":
            distance.hanging_wall = True
        elif metadata["FW/HW Indicator"] == "FW":
            distance.hanging_wall = False
        else:
            pass
        
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
        if metadata["Measured(1)/Inferred(2) Class"] == "1":
            site.vs30_measured = True
        else:
            site.vs30_measured = False
        site.vs30_uncertainty = get_float(
            metadata["Sigma of Vs30 (in natural log Units)"])
        site.z1pt0 = get_float(metadata["Z1 (m)"])
        site.z1pt5 = None
        # site.z1pt5 = get_float(metadata["Z1.5 (m)"])
        site.z2pt5 = get_float(metadata["Z2.5 (m)"])
        # Implement default values for z1pt0 and z2pt5
        if site.z1pt0 is None:
            site.z1pt0 = rcfg.vs30_to_z1pt0_as08(site.vs30)
        if site.z2pt5 is None:
            site.z2pt5 = rcfg.z1pt0_to_z2pt5(site.z1pt0)        
        else:
            # Need to convert z2pt5 from m to km
            site.z2pt5 = site.z2pt5/1000.0
        if "Backarc" in metadata["Forearc/Backarc for subduction events"]:
            site.backarc = True
        site.instrument_type = metadata["Digital (D)/Analog (A) Recording"]
        return site
                           
    def _parse_processing_data(self, wfid, metadata):
        """
        Parses the information for each component
        """
        if metadata["Type of Filter"]:
            filter_params1 = {
                'Type': FILTER_TYPE[metadata["Type of Filter"]],
                'Order': None,
                'Passes': get_positive_int(metadata['npass']),
                'NRoll': get_positive_int(metadata['nroll']),
                'High-Cut': get_positive_float(metadata["LP-H1 (Hz)"]),
                'Low-Cut': get_positive_float(metadata["HP-H1 (Hz)"])}

            filter_params2 = {
                'Type': FILTER_TYPE[metadata["Type of Filter"]],
                'Order': None,
                'Passes': get_positive_int(metadata['npass']),
                'NRoll': get_positive_int(metadata['nroll']),
                'High-Cut': get_positive_float(metadata["LP-H2 (Hz)"]),
                'Low-Cut': get_positive_float(metadata["HP-H2 (Hz)"])}
        else:
            filter_params1, filter_params2 = None, None

        intensity_measures = {
            # All m - convert to cm
            'PGA': None,
            'PGV': None,
            'PGD': None
            }
        luf1 = get_float(metadata["Lowest Usable Freq - H1 (Hz)"])
        if luf1 and luf1 > 0.0:
           lup1 = 1. / luf1
        else:
           lup1 = None
        luf2 = get_float(metadata["Lowest Usable Freq - H2 (Hz)"])
        if luf2 and luf2 > 0.0:
            lup2 = 1. / luf2
        else:
            lup2 = None
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
        
        luf3 = get_float(metadata["Lowest Usable Freq - V (Hz)"])
        if luf3 and luf3 > 0.0:
            filter_params3 = {
                'Type': FILTER_TYPE[metadata["Type of Filter"]],
                'Order': None,
                'Passes': get_int(metadata['npass']),
                'NRoll': get_int(metadata['nroll']),
                'High-Cut': get_float(metadata["LP-V (Hz)"]),
                'Low-Cut': get_float(metadata["HP-V (Hz)"])}
            lup3 = 1. / luf3
            zcomp = Component(wfid, "V",
                ims=intensity_measures,
                longest_period=lup3,
                waveform_filter=filter_params3,
                units=metadata["Unit (cm/s/s; m/s/s; g)"])
            return xcomp, ycomp, zcomp
        else:
            return xcomp, ycomp, None




class NearFaultFlatFileParser(SimpleFlatfileParserV9):

    def parse(self):
        """
        Parses the database
        """
        HEADER_LIST1 = copy.deepcopy(HEADER_LIST)
        HEADER_LIST1.add("Rcdpp")
        self._header_check(HEADER_LIST1)
        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"))
        metadata = []
        self.database = GroundMotionDatabase(self.id, self.name)
        self._get_site_id = self.database._get_site_id
        for row in reader:
            self.database.records.append(self._parse_record(row))
        return self.database

    def _parse_distance_data(self, event, site, metadata):
        """
        Read in the distance related metadata and return an instance of the
        :class: smtk.sm_database.RecordDistance
        """
        # Compute various distance metrics
        # Add calculation of Repi, Rhypo from event and station localizations 
        # (latitudes, longitudes, depth, elevation)?
        target_site = Mesh(np.array([site.longitude]),
                           np.array([site.latitude]),
                           np.array([-site.altitude / 1000.0]))
        # Warning ratio fixed to 1.5
        ratio=1.5
        surface_modeled = rcfg.create_planar_surface(
            Point(event.longitude, event.latitude, event.depth),
            event.mechanism.nodal_planes.nodal_plane_1['strike'],
            event.mechanism.nodal_planes.nodal_plane_1['dip'],
            event.rupture.area,
            ratio)
        hypocenter = rcfg.get_hypocentre_on_planar_surface(
            surface_modeled,
            event.rupture.hypo_loc)
        try:
            surface_modeled._create_mesh()
        except:
            dip = surface_modeled.get_dip()
            dip_dir = (surface_modeled.get_strike() - 90.) % 360.
            ztor = surface_modeled.top_left.depth
            d_x = ztor * np.tan(np.radians(90.0 - dip))
            top_left_surface = surface_modeled.top_left.point_at(d_x,
                                                                 -ztor,
                                                                 dip_dir)
            top_left_surface.depth = 0.
            top_right_surface = surface_modeled.top_right.point_at(d_x,
                                                                   -ztor,
                                                                   dip_dir)
            top_right_surface.depth = 0.
            surface_modeled = SimpleFaultSurface.from_fault_data(
                Line([top_left_surface, top_right_surface]),
                surface_modeled.top_left.depth,
                surface_modeled.bottom_left.depth,
                surface_modeled.get_dip(),
                1.0)
            
        # Rhypo
        Rhypo = get_float(metadata["Hypocentral Distance (km)"])
        if Rhypo is None:
            Rhypo = hypocenter.distance_to_mesh(target_site)
        # Repi
        Repi = get_float(metadata["Epicentral Distance (km)"])
        if Repi is None:
            Repi= hypocenter.distance_to_mesh(target_site, with_depths=False)
        # Rrup
        Rrup = get_float(metadata["Rupture Distance (km)"])
        if Rrup is None:
            Rrup = surface_modeled.get_min_distance(target_site)
        # Rjb
        Rjb = get_float(metadata["Joyner-Boore Distance (km)"])
        if Rjb is None:
            Rjb = surface_modeled.get_joyner_boore_distance(target_site)
        # Rcdpp
        Rcdpp = get_float(metadata["Rcdpp"])
        if Rcdpp is None:
            Rcdpp = surface_modeled.get_cdppvalue(target_site)
        # Need to check if Rx and Ry0 are consistant with the other metrics
        # when those are coming from the flatfile?
        # Rx
        Rx = surface_modeled.get_rx_distance(target_site)
        # Ry0
        Ry0 = surface_modeled.get_ry0_distance(target_site)
        
        distance = RecordDistance(
            repi = float(Repi),
            rhypo = float(Rhypo),
            rjb = float(Rjb),
            rrup = float(Rrup),
            r_x = float(Rx),
            ry0 = float(Ry0),
            rcdpp = float(Rcdpp) )
        distance.azimuth = get_float(metadata["Source to Site Azimuth (deg)"])
        #distance.hanging_wall = get_float(metadata["FW/HW Indicator"])
        if metadata["FW/HW Indicator"] == "HW":
            distance.hanging_wall = True
        elif metadata["FW/HW Indicator"] == "FW":
            distance.hanging_wall = False
        else:
            pass
        
        return distance

class SimpleAsciiTimeseriesReader(SMTimeSeriesReader):
    """
    Parses a simple ascii representation of a record in which the first line
    contains the number of values and the time-step. Whilst the rest of the
    file contains the acceleration record
    """
    def parse_records(self, record):
        """
        Parses the record set
        """
        time_series = OrderedDict([
            ("X", {"Original": {}, "SDOF": {}}),
            ("Y", {"Original": {}, "SDOF": {}}),
            ("V", {"Original": {}, "SDOF": {}})])

        target_names = list(time_series)
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                if iloc < 2:
                    # Expected horizontal component is missing - raise error
                    raise ValueError("Horizontal record %s is expected but "
                        "not found!" % ifile)
                else:
                    print("Vertical record file %s not found" % ifile)
                    del time_series["V"]
                    continue
            else:
                time_series[target_names[iloc]]["Original"] = \
                    self._parse_time_history(ifile, record.xrecord.units)
        if iloc < 2:
            del time_series["V"]

        return time_series

    def _parse_time_history(self, ifile, units="cm/s/s"):
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
        output["Units"] = units
        output["PGA"] = np.max(np.fabs(output["Acceleration"]))
        return output
