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
Parser from the ESM Flatfile to SMTK
"""
import os, sys
import csv
import numpy as np
import copy
import h5py
from math import sqrt
#from sets import Set
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
from smtk.parsers import valid
from smtk.parsers.base_database_parser import (get_float, get_int,
                                               get_positive_float,
                                               get_positive_int,
                                               SMDatabaseReader,
                                               SMTimeSeriesReader,
                                               SMSpectraReader)
from smtk.trellis.configure import (vs30_to_z1pt0_cy14,
                                    vs30_to_z2pt5_cb14)

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

# Import the ESM dictionaries
from .esm_dictionaries import *
#from smtk.parsers.simple_flatfile_parser_sara import SimpleFlatfileParserV9

SCALAR_LIST = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "D5-95"]

HEADER_STR = "event_id;event_time;ISC_ev_id;USGS_ev_id;INGV_ev_id;"\
             "EMSC_ev_id;ev_nation_code;ev_latitude;ev_longitude;"\
             "ev_depth_km;ev_hyp_ref;fm_type_code;ML;ML_ref;Mw;Mw_ref;Ms;"\
             "Ms_ref;EMEC_Mw;EMEC_Mw_type;EMEC_Mw_ref;event_source_id;"\
             "es_strike;es_dip;es_rake;es_strike_dip_rake_ref;es_z_top;"\
             "es_z_top_ref;es_length;es_width;es_geometry_ref;network_code;"\
             "station_code;location_code;instrument_code;sensor_depth_m;"\
             "proximity_code;housing_code;installation_code;st_nation_code;"\
             "st_latitude;st_longitude;st_elevation;ec8_code;"\
             "ec8_code_method;ec8_code_ref;vs30_m_sec;vs30_ref;"\
             "vs30_calc_method;vs30_meas_type;slope_deg;vs30_m_sec_WA;"\
             "epi_dist;epi_az;JB_dist;rup_dist;Rx_dist;Ry0_dist;"\
             "instrument_type_code;late_triggered_flag_01;U_channel_code;"\
             "U_azimuth_deg;V_channel_code;V_azimuth_deg;W_channel_code;"\
             "U_hp;V_hp;W_hp;U_lp;V_lp;W_lp"

HEADERS = set(HEADER_STR.split(";"))

COUNTRY_CODES = {"AL": "Albania", "AM": "Armenia", "AT": "Austria",
                 "AZ": "Azerbaijan", "BA": "Bosnia and Herzegowina",
                 "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus",
                 "CZ": "Czech Republic", "DE": "Germany",  "DZ": "Algeria",
                 "ES": "Spain", "FR": "France", "GE": "Georgia",
                 "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
                 "IL": "Israel", "IR": "Iran", "IS": "Iceland", "IT": "Italy",
                 "JO": "Jordan",  "LI": "Lichtenstein", "MA": "Morocco",
                 "MC": "Monaco", "MD": "Moldova", "ME": "Montenegro",
                 "MK": "Macedonia", "MT": "Malta", "PL": "Poland",
                 "PT": "Portugal", "RO": "Romania", "RS": "Serbia",
                 "RU": "Russia", "SI": "Slovenia", "SM": "San Marino",
                 "SY": "Syria", "TM": "Turkmenistan", "TR": "Turkey",
                 "UA": "Ukraine", "UZ": "Uzbekistan", "XK": "Kosovo"}


class ESMFlatfileParser(SMDatabaseReader):
    """
    Parses the ESM metadata from the flatfile to a set of metadata objects
    """
    M_PRECEDENCE = ["EMEC_Mw", "Mw", "Ms", "ML"]
    BUILD_FINITE_DISTANCES = False

    def parse(self, location="./"):
        """

        """
        headers = getline(self.filename, 1).rstrip("\n").split(";")
        for hdr in HEADERS:
            if hdr not in headers:
                raise ValueError("Required header %s is missing in file"
                                 % hdr)
        # Read in csv
        reader = csv.DictReader(open(self.filename, "r"), delimiter=";")
        metadata = []
        self.database = GroundMotionDatabase(self.id, self.name)
        counter = 0
        for row in reader:
            if self._sanitise(row, reader):
                # Build the metadata
                record = self._parse_record(row)
                if record:
                    # Parse the strong motion
                    record = self._parse_ground_motion(
                        os.path.join(location, "records"),
                        row, record, headers)
                    self.database.records.append(record)

                else:
                    print("Record with sequence number %s is null/invalid"
                          % "{:s}-{:s}".format(row["event_id"],
                                               row["station_code"]))
            if (counter % 100) == 0:
                print("Processed record %s - %s" % (str(counter),
                                                    record.id))

            counter += 1

    @classmethod
    def autobuild(cls, dbid, dbname, output_location, flatfile_location):
        """
        Quick and dirty full database builder!
        """
        if os.path.exists(output_location):
            raise IOError("Target database directory %s already exists!"
                          % output_location)
        os.mkdir(output_location)
        # Add on the records folder
        os.mkdir(os.path.join(output_location, "records"))
        # Create an instance of the parser class
        database = cls(dbid, dbname, flatfile_location)
        # Parse the records
        print("Parsing Records ...")
        database.parse(location=output_location)
        # Save itself to file
        metadata_file = os.path.join(output_location, "metadatafile.pkl")
        print("Storing metadata to file %s" % metadata_file)
        with open(metadata_file, "wb+") as f:
            pickle.dump(database.database, f)
        return database

    def _sanitise(self, row, reader):
        """
        TODO - Not implemented yet!
        """
        return True

    def _parse_record(self, metadata):
        # Waveform ID not provided in file so concatenate Event and Station ID
        wfid = "_".join([metadata["event_id"], metadata["network_code"],
                         metadata["station_code"], metadata["location_code"]])
        wfid = wfid.replace("-", "_")
        # Parse the event metadata
        event = self._parse_event_data(metadata)
        # Parse the distance metadata
        distances = self._parse_distances(metadata, event.depth)
        # Parse the station metadata
        site = self._parse_site_data(metadata)
        # Parse waveform data
        xcomp, ycomp, vertical = self._parse_waveform_data(metadata, wfid)
        return GroundMotionRecord(wfid,
                                  [None, None, None],
                                  event, distances, site,
                                  xcomp, ycomp,
                                  vertical=vertical)


    def _parse_event_data(self, metadata):
        """
        Parses the event metadata
        """
        # ID and Name (name not in file so use ID again)
        eq_id = metadata["event_id"]
        eq_name = metadata["event_id"]
        # Country
        cntry_code = metadata["ev_nation_code"].strip()
        if cntry_code and cntry_code in COUNTRY_CODES:
            eq_country = COUNTRY_CODES[cntry_code]
        else:
            eq_country = None
        # Date and time
        eq_datetime = valid.date_time(metadata["event_time"],
                                     "%Y-%m-%d %H:%M:%S")
        # Latitude, longitude and depth
        eq_lat = valid.latitude(metadata["ev_latitude"])
        eq_lon = valid.longitude(metadata["ev_longitude"])
        eq_depth = valid.positive_float(metadata["ev_depth_km"], "ev_depth_km")
        if not eq_depth:
            eq_depth = 0.0
        eqk = Earthquake(eq_id, eq_name, eq_datetime, eq_lon, eq_lat, eq_depth,
                         None, # Magnitude not defined yet
                         eq_country=eq_country)
        # Get preferred magnitude and list
        pref_mag, magnitude_list = self._parse_magnitudes(metadata)
        eqk.magnitude = pref_mag
        eqk.magnitude_list = magnitude_list
        eqk.rupture, eqk.mechanism = self._parse_rupture_mechanism(metadata,
                                                                   eq_id,
                                                                   eq_name,
                                                                   pref_mag,
                                                                   eq_depth)
        return eqk

    def _parse_magnitudes(self, metadata):
        """
        So, here things get tricky. Up to four magnitudes are defined in the
        flatfile (EMEC Mw, MW, Ms and ML). An order of precedence is required
        and the preferred magnitude will be the highest found
        """
        pref_mag = None
        mag_list = []
        for key in self.M_PRECEDENCE:
            mvalue = metadata[key].strip()
            if mvalue:
                if key == "EMEC_Mw":
                    mtype = "Mw"
                    msource = "EMEC({:s}|{:s})".format(
                        metadata["EMEC_Mw_type"],
                        metadata["EMEC_Mw_ref"])
                else:
                    mtype = key
                    msource = metadata[key + "_ref"].strip()
                mag = Magnitude(float(mvalue),
                                mtype,
                                source=msource)
                if not pref_mag:
                    pref_mag = copy.deepcopy(mag)
                mag_list.append(mag)
        return pref_mag, mag_list

    def _parse_rupture_mechanism(self, metadata, eq_id, eq_name, mag, depth):
        """
        If rupture data is available - parse it, otherwise return None
        """
        sof = metadata["fm_type_code"]
        if not metadata["event_source_id"].strip():
            # No rupture model available. Mechanism is limited to a style
            # of faulting only
            rupture = Rupture(eq_id, eq_name, mag, 5., 5., depth)
            mechanism = FocalMechanism(
                eq_id, eq_name, GCMTNodalPlanes(), None,
                mechanism_type=MECHANISM_TYPE[sof])
            # See if focal mechanism exists
            fm_set = []
            for key in ["strike_1", "dip_1", "rake_1"]:
                if key in metadata:
                    fm_param = valid.vfloat(metadata[key], key)
                    if fm_param is not None:
                        fm_set.append(fm_param)
            if len(fm_set) == 3:
                # Have one valid focal mechanism
                mechanism.nodal_planes.nodal_plane_1 = {"strike": fm_set[0],
                                                        "dip": fm_set[1],
                                                        "rake": fm_set[2]}
            fm_set = []
            for key in ["strike_2", "dip_2", "rake_2"]:
                if key in metadata:
                    fm_param = valid.vfloat(metadata[key], key)
                    if fm_param is not None:
                        fm_set.append(fm_param)
            if len(fm_set) == 3:
                # Have one valid focal mechanism
                mechanism.nodal_planes.nodal_plane_2 = {"strike": fm_set[0],
                                                        "dip": fm_set[1],
                                                        "rake": fm_set[2]}

            if not mechanism.nodal_planes.nodal_plane_1 and not\
                mechanism.nodal_planes.nodal_plane_2:
                # Absolutely no information - base on stye-of-faulting
                mechanism.nodal_planes.nodal_plane_1 = {
                    "strike": 0.0,  # Basically unused
                    "dip": DIP_TYPE[sof],
                    "rake": MECHANISM_TYPE[sof]
                    }
            return rupture, mechanism
        #print(metadata["es_strike"], metadata["es_dip"], metadata["es_rake"])
        strike = valid.strike(metadata["es_strike"])
        dip = valid.dip(metadata["es_dip"])
        rake = valid.rake(metadata["es_rake"])
        ztor = valid.positive_float(metadata["es_z_top"], "es_z_top")
        length = valid.positive_float(metadata["es_length"], "es_length")
        width = valid.positive_float(metadata["es_width"], "es_width")
        rupture = Rupture(eq_id, eq_name, mag, length, width, ztor)

        # Get mechanism type and focal mechanism
        # No nodal planes, eigenvalues moment tensor initially
        mechanism = FocalMechanism(
            eq_id, eq_name, GCMTNodalPlanes(), None,
            mechanism_type=MECHANISM_TYPE[metadata["fm_type_code"]])
        if strike is None:
            strike = 0.0
        if dip is None:
            dip = DIP_TYPE[sof]
        if rake is None:
            rake = MECHANISM_TYPE[sof]
        #if strike is not None and dip is not None and rake is not None:
        mechanism.nodal_planes.nodal_plane_1 = {"strike": strike,
                                                "dip": dip,
                                                "rake": rake}
        return rupture, mechanism

    def _parse_distances(self, metadata, hypo_depth):
        """
        Parse the distances
        """
        repi = valid.positive_float(metadata["epi_dist"], "epi_dist")
        razim = valid.positive_float(metadata["epi_az"], "epi_az")
        rjb = valid.positive_float(metadata["JB_dist"], "JB_dist")
        rrup = valid.positive_float(metadata["rup_dist"], "rup_dist")
        r_x = valid.positive_float(metadata["Rx_dist"], "Rx_dist")
        ry0 = valid.positive_float(metadata["Ry0_dist"], "Ry0_dist")
        rhypo = sqrt(repi ** 2. + hypo_depth ** 2.)
        if not isinstance(rjb, float):
            # In the first case Rjb == Repi
            rjb = copy.copy(repi)

        if not isinstance(rrup, float):
            # In the first case Rrup == Rhypo
            rrup = copy.copy(rhypo)

        if not isinstance(r_x, float):
            # In the first case Rx == -Repi (collapse to point and turn off
            # any hanging wall effect)
            r_x = copy.copy(-repi)
        
        if not isinstance(ry0, float):
            # In the first case Ry0 == Repi
            ry0 = copy.copy(repi)
        distances = RecordDistance(repi, rhypo, rjb, rrup, r_x, ry0)
        distances.azimuth = razim
        return distances

    def _parse_site_data(self, metadata):
        """
        Parses the site information
        """
        network_code = metadata["network_code"].strip()
        station_code = metadata["station_code"].strip()
        site_id = "{:s}-{:s}".format(network_code, station_code)
        location_code = metadata["location_code"].strip()
        site_lon = valid.longitude(metadata["st_longitude"])
        site_lat = valid.latitude(metadata["st_latitude"])
        elevation = valid.vfloat(metadata["st_elevation"], "st_elevation")

        vs30 = valid.vfloat(metadata["vs30_m_sec"], "vs30_m_sec")
        vs30_topo = valid.vfloat(metadata["vs30_m_sec_WA"], "vs30_m_sec_WA")
        if vs30:
            vs30_measured = True
        elif vs30_topo:
            vs30 = vs30_topo
            vs30_measured = False
        else:
            vs30_measured = False
        st_nation_code = metadata["st_nation_code"].strip()
        if st_nation_code:
            st_country = COUNTRY_CODES[st_nation_code]
        else:
            st_country = None
        site = RecordSite(site_id, station_code, station_code, site_lon,
                          site_lat, elevation, vs30, vs30_measured,
                          network_code=network_code,
                          country=st_country)
        site.slope = valid.vfloat(metadata["slope_deg"], "slope_deg")
        site.sensor_depth = valid.vfloat(metadata["sensor_depth_m"],
                                         "sensor_depth_m")
        site.instrument_type = metadata["instrument_code"].strip()
        if site.vs30:
            site.z1pt0 = vs30_to_z1pt0_cy14(vs30)
            site.z2pt5 = vs30_to_z2pt5_cb14(vs30)
        housing_code = metadata["housing_code"].strip()
        if housing_code and (housing_code in HOUSING):
            site.building_structure = HOUSING[housing_code]
        return site

    def _parse_waveform_data(self, metadata, wfid):
        """
        Parse the waveform data
        """
        late_trigger = valid.vint(metadata["late_triggered_flag_01"],
                                  "late_triggered_flag_01")
        # U channel - usually east
        xorientation = metadata["U_channel_code"].strip()
        xazimuth = valid.vfloat(metadata["U_azimuth_deg"], "U_azimuth_deg")
        xfilter = {"Low-Cut": valid.vfloat(metadata["U_hp"], "U_hp"),
                   "High-Cut": valid.vfloat(metadata["U_lp"], "U_lp")}
        xcomp = Component(wfid, xazimuth, waveform_filter=xfilter,
                          units="cm/s/s")
        xcomp.late_trigger = late_trigger
        # V channel - usually North
        vorientation = metadata["V_channel_code"].strip()
        vazimuth = valid.vfloat(metadata["V_azimuth_deg"], "V_azimuth_deg")
        vfilter = {"Low-Cut": valid.vfloat(metadata["V_hp"], "V_hp"),
                   "High-Cut": valid.vfloat(metadata["V_lp"], "V_lp")}
        vcomp = Component(wfid, vazimuth, waveform_filter=vfilter,
                          units="cm/s/s")
        vcomp.late_trigger = late_trigger
        zorientation = metadata["W_channel_code"].strip()
        if zorientation:
            zfilter = {"Low-Cut": valid.vfloat(metadata["W_hp"], "W_hp"),
                       "High-Cut": valid.vfloat(metadata["W_lp"], "W_lp")}
            zcomp = Component(wfid, None, waveform_filter=zfilter,
                              units="cm/s/s")
            zcomp.late_trigger = late_trigger
        else:
            zcomp = None
        
        return xcomp, vcomp, zcomp

    def _parse_ground_motion(self, location, row, record, headers):
        """
        In this case we parse the information from the flatfile directly
        to hdf5 at the metadata stage
        """
        # Get the data
        scalars, spectra = self._retreive_ground_motion_from_row(row, headers)
        # Build the hdf5 files
        filename = os.path.join(location, "{:s}.hdf5".format(record.id))
        fle = h5py.File(filename, "w-")
        ims_grp = fle.create_group("IMS")
        for comp, key in [("X", "U"), ("Y", "V"), ("V", "W")]:
            comp_grp = ims_grp.create_group(comp)
            # Add on the scalars
            scalar_grp = comp_grp.create_group("Scalar")
            for imt in scalars[key]:
                if imt in ["ia", "housner"]:
                    # In the smtk convention it is "Ia" and "Housner"
                    ikey = imt[0].upper() + imt[1:]
                else:
                    # Everything else to upper case (PGA, PGV, PGD, T90, CAV)
                    ikey = imt.upper()
                dset = scalar_grp.create_dataset(ikey, (1,), dtype="f")
                dset[:] = scalars[key][imt]
            # Add on the spectra
            spectra_grp = comp_grp.create_group("Spectra")
            response = spectra_grp.create_group("Response")
            accel = response.create_group("Acceleration")
            accel.attrs["Units"] = "cm/s/s"
            # Add on the periods
            pers = spectra[key]["Periods"]
            periods = response.create_dataset("Periods", pers.shape, dtype="f")
            periods[:] = pers
            periods.attrs["Low Period"] = np.min(pers)
            periods.attrs["High Period"] = np.max(pers)
            periods.attrs["Number Periods"] = len(pers)

            # Add on the values
            values = spectra[key]["Values"]
            spectra_dset = accel.create_dataset("damping_05", values.shape,
                                                dtype="f")
            spectra_dset[:] = np.copy(values)
            spectra_dset.attrs["Damping"] = 5.0
        # Add on the horizontal values
        hcomp = ims_grp.create_group("H")
        # Scalars - just geometric mean for now
        hscalar = hcomp.create_group("Scalar")
        for imt in scalars["Geometric"]:
            if imt in ["ia", "housner"]:
                # In the smtk convention it is "Ia" and "Housner"
                key = imt[0].upper() + imt[1:]
            else:
                # Everything else to upper case (PGA, PGV, PGD, T90, CAV)
                key = imt.upper()
            dset = hscalar.create_dataset(key, (1,), dtype="f")
            dset[:] = scalars["Geometric"][imt]
        # For Spectra - can support multiple components
        hspectra = hcomp.create_group("Spectra")
        hresponse = hspectra.create_group("Response")
        pers = spectra["Geometric"]["Periods"]
        hpers_dset = hresponse.create_dataset("Periods", pers.shape, dtype="f")
        hpers_dset[:] = np.copy(pers)
        hpers_dset.attrs["Low Period"] = np.min(pers)
        hpers_dset.attrs["High Period"] = np.max(pers)
        hpers_dset.attrs["Number Periods"] = len(pers)
        haccel = hresponse.create_group("Acceleration")
        for htype in ["Geometric", "rotD00", "rotD50", "rotD100"]:
            if np.all(np.isnan(spectra[htype]["Values"])):
                # Component not determined
                continue
            if not (htype == "Geometric"):
                key = htype[0].upper() + htype[1:]
            else:
                key = copy.deepcopy(htype)
            htype_grp = haccel.create_group(htype)
            hvals = spectra[htype]["Values"]
            hspec_dset = htype_grp.create_dataset("damping_05", hvals.shape,
                                                  dtype="f")
            hspec_dset[:] = hvals
            hspec_dset.attrs["Units"] = "cm/s/s"
        record.datafile = filename
        return record


    def _retreive_ground_motion_from_row(self, row, header_list):
        """

        """
        imts = ["U", "V", "W", "rotD00", "rotD100", "rotD50"]
        spectra = []
        scalar_imts = ["pga", "pgv", "pgd", "T90", "housner", "ia", "CAV"]
        scalars = []
        for imt in imts:
            periods = []
            values = []
            key = "{:s}_T".format(imt)
            scalar_dict = {}
            for header in header_list:
                # Deal with the scalar case
                for scalar in scalar_imts:
                    if header == "{:s}_{:s}".format(imt, scalar):
                        # The value is a scalar
                        value = row[header].strip()
                        if value:
                            scalar_dict[scalar] = np.fabs(float(value))
                        else:
                            scalar_dict[scalar] = None
            scalars.append((imt, scalar_dict))
            for header in header_list:
                if key in header:
                    if header == "{:s}90".format(key):
                        # Not a spectral period but T90
                        continue
                    iky = header.replace(key, "").replace("_", ".")
                    #print imt, key, header, iky
                    periods.append(float(iky))
                    value = row[header].strip()
                    if value:
                        values.append(np.fabs(float(value)))
                    else:
                        values.append(np.nan)
                    #values.append(np.fabs(float(row[header].strip())))
            periods = np.array(periods)
            values = np.array(values)
            idx = np.argsort(periods)
            spectra.append((imt, {"Periods": periods[idx],
                                   "Values": values[idx]}))
        # Add on the as-recorded geometric mean
        spectra = OrderedDict(spectra)
        scalars = OrderedDict(scalars)
        spectra["Geometric"] = {
            "Values": np.sqrt(spectra["U"]["Values"] *
                              spectra["V"]["Values"]),
            "Periods": np.copy(spectra["U"]["Periods"])
            }
        scalars["Geometric"] = dict([(key, None) for key in scalars["U"]])
        for key in scalars["U"]:
            if scalars["U"][key] and scalars["V"][key]:
                scalars["Geometric"][key] = np.sqrt(
                    scalars["U"][key] * scalars["V"][key])
        return scalars, spectra

