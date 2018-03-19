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
Parser from the Sigma data format to SMTK
"""
import os
import csv
import numpy as np
from collections import OrderedDict
from datetime import datetime
import h5py
from smtk.sm_database import *
from smtk.parsers.base_database_parser import (get_float, get_int,
                                               SMDatabaseReader,
                                               SMTimeSeriesReader,
                                               SMSpectraReader)

class SigmaDatabaseMetadataReader(SMDatabaseReader):
    """
    Reader for the Sigma Database
    """
    XCOMP_STR = "_H1.cor.acc"
    YCOMP_STR = "_H2.cor.acc"
    ZCOMP_STR = "_V.cor.acc"
    XSPEC_STR = "_H1.rs"
    YSPEC_STR = "_H2.rs"
    ZSPEC_STR = "_V.rs"

    def parse(self):
        """

        """
        file_list = os.listdir(self.filename)
        num_files = len(file_list)
        self.database = GroundMotionDatabase(self.id, self.name)
        for file_str in file_list:
            if "DS_Store" in file_str:
                continue
            metafile = os.path.join(self.filename,
                                    file_str + "/" + file_str + ".metadata")
            if not os.path.exists(metafile):
                continue
            csv_data = csv.DictReader(open(metafile, 'r'),
                                      delimiter=",",
                                      quotechar='"')
            metadata = next(csv_data)
            self.database.records.append(self.parse_metadata(metadata, 
                                                             file_str))

        return self.database

    def parse_metadata(self, metadata, file_str):
        """
        Parses the metadata dictionary
        """
        # Waveform id
        wfid = metadata['waveform_sourceid']
        # Get event information
        event = self._parse_event_data(metadata)
        # Get distance information
        distances = self._parse_distance_data(metadata)
        # Get site data
        site = self._parse_site_data(metadata)
        # Get processing data
        x_comp, y_comp, z_comp = self._parse_processing_data(wfid, metadata)
        # Create and return record
        rec_file = os.path.join(file_str + "/" + file_str)
                                
        return GroundMotionRecord(wfid,
            [rec_file + self.XCOMP_STR, 
             rec_file + self.YCOMP_STR,
             rec_file + self.ZCOMP_STR],
            event,
            distances,
            site,
            x_comp,
            y_comp,
            vertical=z_comp,
            ims=None,
            longest_period=get_float(metadata['up4horiz_components']),
            spectra_file=[rec_file + self.XSPEC_STR,
                          rec_file + self.YSPEC_STR,
                          rec_file + self.ZSPEC_STR])


    def _parse_event_data(self, metadata):
        """
        """
        # Get datetime
        if len(metadata['event.datetime']) > 20:
            eq_datetime = datetime.strptime(metadata['event.datetime'],
                                        "%Y-%m-%d %H:%M:%S.%f")
        else:
            eq_datetime = datetime.strptime(metadata['event.datetime'],
                                        "%Y-%m-%d %H:%M:%S")
        # Event ID and Name
        if metadata['event.unid']:
            eq_id = metadata['event.unid']
        else:
            eq_id = eq_datetime.isoformat()
        
        eq_name = metadata['event.name']
        
        # Get focal mechanism
        focal_mech = self._parse_focal_mechanism(eq_id, eq_name, metadata)

        # Get preferred magnitude
        pref_mag = Magnitude(get_float(metadata['event.pref_mag']),
                             metadata['event.pref_mag_type'])
        # Get magnitude list
        mag_list = []
        for kstr in ['0', '1', '2', '3', '4']:
            mag_list.append(Magnitude(
                get_float(metadata['event.magnitudes/' + kstr + '.value']),
                metadata['event.magnitudes/' + kstr + '.magtype']))

        eqk = Earthquake(eq_id, eq_name, eq_datetime,
            get_float(metadata['event.longitude']),
            get_float(metadata['event.latitude']),
            get_float(metadata['event.focaldepth']),
            pref_mag,
            focal_mech,
            metadata['event.country.name'])
        eqk.magnitude_list = mag_list
        # Get Rupture data
        eqk.rupture = self._parse_rupture(eq_id, eq_name, pref_mag, metadata)
        return eqk

    def _parse_focal_mechanism(self, eq_id, eq_name, metadata):
        """
        Parses the focal mechanism returning an instance of FocalMechanism
        """
        nodal_planes = GCMTNodalPlanes()
        nodal_planes.nodal_plane_1 = {
            'strike': get_float(metadata['event.strike1']),
            'dip': get_float(metadata['event.dip1']),
            'rake': get_float(metadata['event.slip1'])
            }
        nodal_planes.nodal_plane_2 = {
            'strike': get_float(metadata['event.strike2']),
            'dip': get_float(metadata['event.dip2']),
            'rake': get_float(metadata['event.slip2'])
            }

        principal_axes = GCMTPrincipalAxes()
        principal_axes.t_axes = {
            'eigenvalue': None,
            'plunge': get_float(metadata['event.t_axes_plg']),
            'azimuth': get_float(metadata['event.t_axes_az'])}
        principal_axes.p_axes = {
            'eigenvalue': None,
            'plunge': get_float(metadata['event.p_axes_plg']),
            'azimuth': get_float(metadata['event.p_axes_az'])}
        principal_axes.b_axes = {
            'eigenvalue': None,
            'plunge': None,
            'azimuth': None}

        return FocalMechanism(eq_id, eq_name, nodal_planes, principal_axes,
            mechanism_type=metadata['event.fault_mechanism.name'])

    def _parse_rupture(self, eq_id, eq_name, magnitude, metadata):
        """

        """
        return Rupture(eq_id,
                       eq_name,
                       magnitude,
                       get_float(metadata['event.fault_rupture_length']),
                       get_float(metadata['event.fault_rupture_width']),
                       get_float(metadata['event.fault_rupture_depth']))

    def _parse_distance_data(self, metadata):
        """
        Parses the distance data
        """
        return RecordDistance(get_float(metadata['distance_repi']),
                              get_float(metadata['distance_rhyp']),
                              rjb = get_float(metadata['distance_rjb']),
                              rrup = get_float(metadata['distance_rrup']),
                              r_x = None,
                              flag = get_int(metadata['distance_flag']))
                        
                        
    def _parse_site_data(self, metadata):
        """
        Parse the site data
        """
        site = RecordSite(
            metadata['station.oid'],
            metadata['station.name'],
            metadata['station.code'],
            get_float(metadata['station.longitude']),
            get_float(metadata['station.latitude']),
            get_float(metadata['station.altitude']),
            vs30=get_float(metadata['station.vs30']),
            vs30_measured=get_int(metadata['station.vs30_measured']),
            network_code=metadata['station.agency.name'],
            country=metadata['station.country.name'])
        site.vs30_measured_type = metadata['station.vs30_measured_type']
        site.instrument_type = metadata['recording_type']
        site.digitiser = metadata['digitalizer']
        site.building_structure = metadata['station.building_struct.name']
        site.number_floors = get_int(metadata['station.number_of_floor'])
        site.floor = metadata['station.installed_on_floor']
        site.ec8 = metadata['station.ec8']
        return site

    def _parse_processing_data(self, wfid, metadata):
        """
        Parses the record processing information
        """
        proc_x = {}
        proc_y = {}
        proc_z = {}
        for key in metadata:
            if 'comp_ordered()/0' in key:
                proc_x[key.replace("()/0", "")] = metadata[key]
            elif 'comp_ordered()/1' in key:
                proc_y[key.replace("()/1", "")] = metadata[key]
            elif 'comp_ordered()/2' in key:
                proc_z[key.replace("()/2", "")] = metadata[key]
            else:
                pass
        x_comp = self._parse_component_data(wfid, proc_x)
        y_comp = self._parse_component_data(wfid, proc_y)
        if len(proc_z) > 0:
            z_comp = self._parse_component_data(wfid, proc_z)
        else:
            z_comp = None
        return x_comp, y_comp, z_comp
            
    def _parse_component_data(self, wfid, proc_data):
        """
        Parses the information for each component
        """
        filter_params = {
            'Type': proc_data['comp_ordered.filter_type'],
            'Order': proc_data['comp_ordered.filter_order'],
            'Passes': get_int(
                proc_data['comp_ordered.filter_number_of_passes']),
            'NRoll': get_int(proc_data['comp_ordered.nroll']),
            'High-Cut': get_float(proc_data['comp_ordered.high_cut_freq']),
            'Low-Cut': get_float(proc_data['comp_ordered.low_cut_freq'])}

        intensity_measures = {
            # All m - convert to cm
            'PGA': get_float(proc_data['comp_ordered.pga']),
            'PGV': get_float(proc_data['comp_ordered.pgv']),
            'PGD': get_float(proc_data['comp_ordered.pgd'])
            }

        for imkey in intensity_measures:
            if intensity_measures[imkey]:
                intensity_measures[imkey] = 100.0 * intensity_measures[imkey]
            
        comp = Component(wfid,
            proc_data['comp_ordered.orientation'],
            ims=intensity_measures,
            longest_period=None,
            waveform_filter=filter_params,
            baseline=proc_data['comp_ordered.baseline_correction'])
        return comp


class SigmaSpectraParser(SMSpectraReader):
    """
    Class to parse a triplet of strong motion
    """
    def parse_spectra(self):
        """
        Parses the Spectra to an instance of the database dictionary
        """
        
        damping_list = ["damping_02", "damping_05", "damping_07", 
                        "damping_10", "damping_20", "damping_30"]
        sm_record = OrderedDict([
            ("X", {"Scalar": {}, "Spectra": {"Response": {}}}), 
            ("Y", {"Scalar": {}, "Spectra": {"Response": {}}}), 
            ("V", {"Scalar": {}, "Spectra": {"Response": {}}})])
        target_names = list(sm_record)
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                continue
            data = np.genfromtxt(ifile, skip_header=1)
            per = data[:-1, 0]
            spec_acc = data[:-1, 1:]
            pgv = 100.0 * data[-1, 1]
            num_per = len(per)

            sm_record[target_names[iloc]]["Scalar"]["PGA"] =\
                {"Value": 100.0  * spec_acc[0, 0], "Units": "cm/s/s"}
            sm_record[target_names[iloc]]["Scalar"]["PGV"] =\
                {"Value": pgv, "Units": "cm/s"}
            sm_record[target_names[iloc]]["Spectra"]["Response"] = {
                "Periods": per,
                "Number Periods" : num_per,
                "Acceleration" : {"Units": "cm/s/s"},
                "Velocity" : None,
                "Displacement" : None,
                "PSA" : None,
                "PSV" : None}
            for jloc, damping in enumerate(damping_list):
                sm_record[target_names[iloc]]["Spectra"]["Response"]\
                    ["Acceleration"][damping] = 100.0 * data[:-1, jloc + 1]
        return sm_record

class SigmaRecordParser(SMTimeSeriesReader):
    """

    """
    def parse_records(self, record=None):
        """

        """
        time_series = OrderedDict([
            ("X", {"Original": {}, "SDOF": {}}),
            ("Y", {"Original": {}, "SDOF": {}}),
            ("V", {"Original": {}, "SDOF": {}})])
             
        target_names = list(time_series)
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                continue
            else:
                time_series[target_names[iloc]]["Original"] = \
                    self._parse_time_history(ifile)
        return time_series

    def _parse_time_history(self, ifile):
        """

        """
        f = open(ifile, "r")
        output = {}
        acc_hist = []
        cases = {0: self._get_datetime,
                 1: self._get_station_code_name,
                 2: self._get_network,
                 3: self._get_orientation,
                 4: self._get_processing_info,
                 5: self._get_filter_cutoffs,
                 6: self._get_timestep,
                 7: self._get_number_steps,
                 8: self._get_pga,
                 9: self._get_units}
        
        for iloc, line in enumerate(f.readlines()):
            if iloc in cases:
                output = cases[iloc](output, line.rstrip("\n"))
            else:
                acc_hist.extend(self._get_timehist_line(line.rstrip("\n")))
            
        output["Acceleration"] = 100.0 * np.array(acc_hist)
        output["Time"] = np.cumsum(
            self.time_step * np.ones(self.number_steps)) - self.time_step
        return output

    def _get_datetime(self, output, line):
        """
        Adds the date and time information
        """
        date_line = line[32:]
        if len(date_line.split(":")) < 3:
            dtime = datetime.strptime(date_line, "%Y-%m-%d %H:%M")
        else:
            dtime = datetime.strptime(date_line.split(".")[0], 
                                      "%Y-%m-%d %H:%M:%S")
        output['Year'] = dtime.year
        output['Month'] = dtime.month
        output['Day'] = dtime.day
        output['Hour'] = dtime.hour
        output['Minute'] = dtime.minute
        output['Second'] = dtime.second
        return output

    def _get_station_code_name(self, output, line):
        """
        Adds the station code name
        """
        stat_code = line[32:].split("/")
        output['Station Code'] = int(stat_code[0])
        output['Station Name'] = stat_code[1].lstrip(" ")
        return output

    def _get_network(self, output, line):
        """
        Add the network
        """
        output['Network'] = line[32:]
        return output

    def _get_orientation(self, output, line):
        """
        Adds the orientation
        """
        output['Orientation'] = line[32:].strip(" ")
        return output

    def _get_processing_info(self, output, line):
        """
        Adds information about the filter
        """
        output['Processing'] = line[32:]
        return output

    def _get_filter_cutoffs(self, output, line):
        """
        
        """
        cofs = list(map(float, line[32:].split("-")))
        output['Low Frequency Cutoff'] = cofs[0]
        output['High Frequency Cutoff'] = cofs[1]
        return output

    def _get_timestep(self, output, line):
        """
        Extracts timestep
        """
        self.time_step = float(line[32:])
        output['Time-step'] = self.time_step
        return output

    def _get_number_steps(self, output, line):
        """
        Extracts number of acceleration measures
        """
        self.number_steps = int(line[32:])
        output["Number Steps"] = self.number_steps
        return output
        
    def _get_pga(self, output, line):
        """
        Adds PGA
        """
        output['PGA'] = 100.0 * float(line[32:])
        return output
    
    def _get_units(self, output, line):
        """

        """
        output['Units'] = "cm/s/s"
        return output

    def _get_timehist_line(self, line):
        """

        """
        idx = np.arange(0, len(line) + 14, 14)
        return [float(line[idx[i]:idx[i + 1]]) for i in range(len(idx) - 1)]
