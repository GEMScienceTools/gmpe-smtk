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
General Flatfile Only Parser - to handle the cases in which the only strong
motion input is in the form of a flatfile
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
from smtk.parsers.simple_flatfile_parser_sara import SimpleFlatfileParserV9

SCALAR_LIST = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "D5-95"]

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
    'Rupture Distance (km)', 'Rx (km)', 'Ry0 (km)', 
    'Source to Site Azimuth (deg)', 'FW/HW Indicator',
    'Forearc/Backarc for subduction events', 'File Name (Horizontal 1)',
    'File Name (Horizontal 2)','File Name (Vertical)',
    'Digital (D)/Analog (A) Recording', 'Acceleration (A)/Velocity (V)',
    'Unit (cm/s/s; m/s/s; g)', 'File format', 'Processing flag',
    'Type of Filter','npass','nroll','HP-H1 (Hz)','HP-H2 (Hz)',
    'LP-H1 (Hz)', 'LP-H2 (Hz)', 'Factor', 'Lowest Usable Freq - H1 (Hz)',
    'Lowest Usable Freq - H2 (Hz)', 'Lowest Usable Freq - Ave. Component (Hz)',
    'Maximum Usable Freq - Ave. Component (Hz)', 'HP-V (Hz)', 'LP-V (Hz)',
    'Lowest Usable Freq - V (Hz)','Comments'])


def valid_positive(value):
    """
    Returns True if the value is positive or zero, false otherwise
    """
    if value and value >= 0.0:
        return True
    print("Positive value (or 0) is needed - %s is given" % str(value))
    return False

def valid_longitude(lon):
    """
    Returns True if the longitude is valid, False otherwise
    """
    if not lon:
        return False
    if (lon >= -180.0) and (lon <= 180.0):
        return True
    print("Longitude %s is outside of range -180 <= lon <= 180" % str(lon))
    return False

def valid_latitude(lat):
    """
    Returns True if the latitude is valid, False otherwise
    """
    if not lat:
        print("Latitude is missing")
        return False
    if (lat >= -90.0) and (lat <= 90.0):
        return True 
    print("Latitude %s is outside of range -90 <= lat <= 90" % str(lat))
    return False

def valid_date(year, month, day):
    """
    Checks that the year is given and greater than 0, that month is in the
    range 1 - 12, and day is in the range 1 - 31
    """
    if all([year > 0, month > 0, month <= 12, day > 0, day <= 31]):
        return True
    print("Date %s/%s/%s is not valid" % (str(year), str(month), str(day)))
    return False



class GeneralFlatfileParser(SimpleFlatfileParserV9):
    """
    Operates in the same manner as for SARA Simple Flatfile parser, albeit
    with Rx and Ry0 added as explicit columns
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
        counter = 0
        for row in reader:
            #print(row["Record Sequence Number"])
            #if (counter % 100) == 0:
            #    print("%s - %s" % (str(counter), row["Record Sequence Number"]))

            if self._sanitise(row, reader):
                record = self._parse_record(row)
                if record:
                    self.database.records.append(record)
            else:
                print("Record with sequence number %s is null/invalid"
                      % str(row["Record Sequence Number"]))
            counter += 1
        return self.database


    def _sanitise(self, row, reader):
        """
        If all of the strong motion values are negative the record is null
        and should be removed
        """
        iml_vals = []
        for fname in reader.fieldnames:
            if fname.startswith("SA(") or fname in SCALAR_LIST:
                # Ground motion value
                iml = get_float(row[fname])
                if iml:
                    iml_vals.append(iml)
                else:
                    iml_vals.append(-999.0)
        # If all ground motion values are negative then the record is null
        # return false
        if np.all(np.array(iml_vals) < 0.0):
            return False
        else:
            return True
                    
    def _parse_record(self, metadata):
        """
        Parses the record information and returns an instance of the
        :class: smtk.sm_database.GroundMotionRecord
        """
        # Waveform ID
        wfid = metadata["Record Sequence Number"]
        # Event information
        # Verify datetime
        event = self._parse_event_data(metadata)
        if not self._verify_event(event, wfid):
            print("Record Number %s has invalid event!" % wfid)
            return False
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

    def _verify_event(self, event, wfid):
        """
        Some basic checks are needed to ensure that the event is usable!
        1. Year, month and day should be valid
        2. Magnitude should be present and valid > -10.0!
        3. Epicentre should be in valid range (-180.0 <= lon <= 180,
                                               -90.0 <= lat <= 180)
        4. Depth should be in valid range (>= 0)
        """
        is_valid = valid_date(event.datetime.year, event.datetime.month,
                              event.datetime.day) and\
            valid_longitude(event.longitude) and\
            valid_latitude(event.latitude) and\
            valid_positive(event.depth)
        if not is_valid:
            return False
        
        # Magnitude check
        if not event.magnitude.value or event.magnitude.value < -10.0:
            # Invalid or missing event
            print("Record Number %s has invalid magnitude value %s"
                  % (str(wfid), str(event.magnitude.value)))
            return False
        return True


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
                if ("SA(" in header) or ("PGA" in header) or ("PGV" in header)\
                    or ("PGD" in header):
                    continue
                else:
                    print "Header %s not recognised - ignoring this data!" %\
                        header

#    @staticmethod
#    def _validate_datetime(metadata):
#        """
#        NGA West 2 flatfile is badly formatted w.r.t. date and time - this
#        bit of defensive coding should prevent it from crashing out with
#        errors
#        """
#        year = get_int(metadata["Year"])
#        month = get_int(metadata["Month"])
#        # Month values given incorrectly for many records
#        if month > 12:
#            month = (month % 12)
#            if month == 0:
#                month = 12
#            
#        day = get_int(metadata["Day"])
#        hour = get_int(metadata["Hour"])
#        minute = get_int(metadata["Minute"])
#        second = get_int(metadata["Second"])
#        return year, month, day, hour, minute, second

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
        if not event.rupture.area:
            event.rupture.area = WC1994().get_median_area(event.magnitude.value,
                                                          None)
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
        Rhypo, Repi, Rrup, Rjb, Ry0 = tuple(map(
            get_positive_float, [metadata[key] for key in [
                "Hypocentral Distance (km)", "Epicentral Distance (km)",
                "Rupture Distance (km)", "Joyner-Boore Distance (km)",
                "Ry0 (km)"]]))
        Rx = get_float(metadata["Rx (km)"]) # Rx can be negative

        #Rhypo = get_float(metadata["Hypocentral Distance (km)"])
        if Rhypo is None or Rhypo < 0.0:
            Rhypo = hypocenter.distance_to_mesh(target_site)
        # Repi
        #Repi = get_float(metadata["Epicentral Distance (km)"])
        if Repi is None or Repi < 0.0:
            Repi= hypocenter.distance_to_mesh(target_site, with_depths=False)
        # Rrup
        #Rrup = get_float(metadata["Rupture Distance (km)"])
        if Rrup is None or Rrup < 0.0:
            Rrup = surface_modeled.get_min_distance(target_site)[0]
        # Rjb
        #Rjb = get_float(metadata["Joyner-Boore Distance (km)"])
        if Rjb is None or Rjb < 0.0:
            Rjb = surface_modeled.get_joyner_boore_distance(
                target_site)[0]
        # Need to check if Rx and Ry0 are consistant with the other metrics
        # when those are coming from the flatfile?
        # Rx
        #Rx = get_float(metadata["Rx (km)"])
        if Rx is None or Rx < 0.0:
            Rx = surface_modeled.get_rx_distance(target_site)[0]
        # Ry0
        Ry0 = get_float(metadata["Ry0 (km)"])
        if Ry0 is None or Ry0 < 0.0:
            Ry0 = surface_modeled.get_ry0_distance(target_site)[0]
        
        distance = RecordDistance(
            repi = Repi,
            rhypo = Rhypo,
            rjb = Rjb,
            rrup = Rrup,
            r_x = Rx,
            ry0 = Ry0)
        distance.azimuth = get_float(metadata["Source to Site Azimuth (deg)"])
        #distance.hanging_wall = get_float(metadata["FW/HW Indicator"])
        if metadata["FW/HW Indicator"] == "HW":
            distance.hanging_wall = True
        elif metadata["FW/HW Indicator"] == "FW":
            distance.hanging_wall = False
        else:
            pass
        
        return distance
    
        return
#class GeneralFlatfileSpectraReader(SMSpectraReader):
#    """
#    This
#    """
