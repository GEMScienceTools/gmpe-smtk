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
Parser set for the Mexican Standard Acceleration file, ASA ver 2.0
Detailed documentation on the format is here:
https://aplicaciones.iingen.unam.mx/AcelerogramasRSM/DscAsa.aspx

Each file contains the acceleration time series for all 3 components
and the corresponding metadata
"""


import os
import re
from collections import OrderedDict
from linecache import getline
from datetime import datetime
from math import sqrt
from openquake.hazardlib.geo import *
from smtk.sm_database import *
from smtk.sm_utils import convert_accel_units, get_time_vector
from smtk.parsers.base_database_parser import (get_float,
                                               get_int,
                                               SMDatabaseReader,
                                               SMTimeSeriesReader)


def _get_info_from_archive_name(aname):
    """
    Extract the network and event information from the filename and return
    a dictionary. Event information is the year, month, day, and identifier.
    For UNAM the identifier is the event number of that day, and for CICESE
    it is the event time.
    """
    FILE_INFO_KEY = ["Net", "Year", "Month", "Day", "Identifier"]

    # CICESE
    if "dat" in aname[-3:] or "Dat" in aname[-3:]:
        file_info = [aname[:3], aname[8:12], aname[6:8],
                     aname[4:6], aname[12:-4]]
    # UNAM
    else:
        file_info = [aname[:4], aname[4:6], aname[6:8],
                     aname[9:11], aname[11:12]]

    return OrderedDict([
        (FILE_INFO_KEY[i], file_info[i]) for i in range(len(file_info))
    ])


def _get_metadata_from_file(file_str):
    """
    Pulls the metadata from lines 7 - 80 of ASA file and returns a cleaned
    version as an ordered dictionary. Note that every file contains
    the metadata corresponding to all 3 components.
    """

    metadata = {}
    for i in range(7, 81):
        # Exclude lines with "==", websites, and with lenghts < 42
        exclude = ["==", "www"]
        if not any(x in getline(file_str, i) for x in exclude) and\
                len(getline(file_str, i)) > 41:
            # Delete newlines at end and split on ":"
            row = (getline(file_str, i).rstrip("\n")).split(":")
            if len(row) > 2:
                # The character ":" occurs somewhere in the datastring
                if len(row[0].strip()) != 0:
                    metadata[row[0].strip()] = ":".join(row[1:]).strip()
            else:
                # Parse as normal
                if len(row[0].strip()) != 0:
                    metadata[row[0].strip()] = row[1].strip()
                    recentkey = row[0].strip()
                elif len(row[0].strip()) == 0:
                    # When values continue on a new line
                    metadata[recentkey] = (
                        metadata[recentkey] + ' ' + row[1].strip())

    return metadata


class ASADatabaseMetadataReader(SMDatabaseReader):

    """
    Reader for the metadata database of the UNAM and CICESE files (ASA format)
    """
    ORGANIZER = []

    def parse(self):
        """
        Parses the record
        """
        self.database = GroundMotionDatabase(self.id, self.name)
        self._sort_files()
        assert (len(self.ORGANIZER) > 0)
        for file_dict in self.ORGANIZER:
            # metadata for all componenets comes from the same file
            metadata = _get_metadata_from_file(file_dict["Time-Series"]["X"])
            self.database.records.append(self.parse_metadata(metadata,
                                                             file_dict))
        return self.database

    def _sort_files(self):
        """
        Searches through the directory and organise the files associated
        with a particular recording into a dictionary
        """

        for file_str in sorted(os.listdir(self.filename)):

            file_dict = {"Time-Series": {"X": None, "Y": None, "Z": None},
                         "PSV": {"X": None, "Y": None, "Z": None},
                         "SA": {"X": None, "Y": None, "Z": None},
                         "SD": {"X": None, "Y": None, "Z": None}}

            file_loc = os.path.join(self.filename, file_str)

            # Filepath to each of the time series (same path for each component)
            file_dict["Time-Series"]["X"] = file_loc
            file_dict["Time-Series"]["Y"] = file_loc
            file_dict["Time-Series"]["Z"] = file_loc

            self.ORGANIZER.append(file_dict)

    def parse_metadata(self, metadata, file_dict):
        """
        Parses the metadata dictionary
        """

        file_str = metadata["NOMBRE DEL ARCHIVO"]
        file_info = _get_info_from_archive_name(file_str)

        # create Waveform ID (unique ID)
        wfid = "".join([file_info[key]for key in ["Net",
                                                  "Year",
                                                  "Month",
                                                  "Day",
                                                  "Identifier"]])
        # Get event information
        event = self._parse_event(metadata, file_str)
        # Get Distance information
        distance = self._parse_distance_data(metadata, file_str, event)
        # Get site data
        site = self._parse_site_data(metadata)
        # Get component and processing data
        xcomp, ycomp, zcomp = self._parse_component_data(wfid, metadata)

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
            spectra_file=None)

    def _parse_event(self, metadata, file_str):
        """
        Parses the event metadata to return an instance of the :class:
        smtk.sm_database.Earthquake
        """

        months = {'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5,
                 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9,
                 'OCTUBURE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12}

        # Date and time
        try:  # UNAM DATES
            year, month, day = (
                get_int(metadata["FECHA DEL SISMO [GMT]"].split("/")[0]),
                get_int(metadata["FECHA DEL SISMO [GMT]"].split("/")[1]),
                get_int(metadata["FECHA DEL SISMO [GMT]"].split("/")[2]))
        except:  # CICESE DATES
            year, month, day = (
                get_int(metadata["FECHA DEL SISMO (GMT)"][-4:]),
                months[metadata["FECHA DEL SISMO (GMT)"].split()[2]],
                get_int(metadata["FECHA DEL SISMO (GMT)"][:2]))

        # Get event time, naming is not consistent (e.g. 07.1, 00, 17,1)
        for i in metadata:
            if 'HORA EPICENTRO (GMT)' in i:
                hour, minute, second = (get_int(metadata[i].split(":")[0]),
                                        get_int(metadata[i].split(":")[1]),
                                        int(float(metadata[i].split(":")[2].
                                            replace("0", "", 1).
                                            replace(",", "."))))

        try:
            eq_datetime = datetime(year, month, day, hour, minute, second)
        except:
            raise ValueError("Record %s is missing event time" % file_str)

        # Event ID - No EVID, so use the date and time of the event
        eq_id = str(eq_datetime).replace(" ", "_")

        # Event Name
        eq_name = None

        # Get magnitudes, below are the different types given in ASA files
        moment_mag = None
        surface_mag = None
        body_mag = None
        c_mag = None
        l_mag = None
        e_mag = None
        a_mag = None
        m_mag = None

        mag_list = []
        mag = metadata["MAGNITUD(ES)"].split("/")
        for i in range(0, len(mag)):
            if mag[i][0:2] == "":
                continue
            if mag[i][0:2] == "Mw":
                m_w = get_float(mag[i][3:])
                moment_mag = Magnitude(m_w, "Mw")
                mag_list.append(moment_mag)
            if mag[i][0:2] == "Ms":
                m_s = get_float(mag[i][3:])
                surface_mag = Magnitude(m_s, "Ms")
                mag_list.append(surface_mag)
            if mag[i][0:2] == "Mb":
                m_b = get_float(mag[i][3:])
                body_mag = Magnitude(m_b, "Mb")
                mag_list.append(body_mag)
            if mag[i][0:2] == "Mc":
                m = get_float(mag[i][3:])
                c_mag = Magnitude(m, "Mc")
                mag_list.append(c_mag)
            if mag[i][0:2] == "Ml":
                m = get_float(mag[i][3:])
                l_mag = Magnitude(m, "Ml")
                mag_list.append(l_mag)
            if mag[i][0:2] == "Me" or mag[i][0:2] == "ME":
                m = get_float(mag[i][3:])
                e_mag = Magnitude(m, "Me")
                mag_list.append(e_mag)
            if mag[i][0:2] == "Ma":
                m = get_float(mag[i][3:])
                a_mag = Magnitude(m, "Ma")
                mag_list.append(a_mag)
            if mag[i][0:2] == "M=":
                m = get_float(mag[i][2:])
                m_mag = Magnitude(m, "M")
                mag_list.append(m_mag)

        # magnitude hierarchy for defining pref_mag
        if moment_mag is not None:
            pref_mag = moment_mag
        elif surface_mag is not None:
            pref_mag = surface_mag
        elif body_mag is not None:
            pref_mag = body_mag
        elif c_mag is not None:
            pref_mag = c_mag
        elif l_mag is not None:
            pref_mag = l_mag
        elif e_mag is not None:
            pref_mag = e_mag
        elif a_mag is not None:
            pref_mag = a_mag
        elif m_mag is not None:
            pref_mag = m_mag
        else:
            raise ValueError("Record %s has no magnitude!" % file_str)

        # Get focal mechanism data (not given in ASA file)
        foc_mech = FocalMechanism(eq_id,
                                  eq_name,
                                  None,
                                  None,
                                  mechanism_type=None)

        # Get depths, naming is not consistent so allow for variation
        for i in metadata:
                if 'PROFUNDIDAD ' in i:
                    # assume <5km = 5km
                    evtdepth = get_float(re.sub('[ <>]', '', metadata[i]))
        if evtdepth is None:
            raise ValueError("Record %s is missing event depth" % file_str)

        # Build event
        eqk = Earthquake(
            eq_id,
            eq_name,
            eq_datetime,
            get_float(metadata["COORDENADAS DEL EPICENTRO"].split(" ")[3]),
            get_float(metadata["COORDENADAS DEL EPICENTRO"].split(" ")[0]),
            evtdepth,
            pref_mag,
            foc_mech)

        eqk.magnitude_list = mag_list

        return eqk

    def _parse_distance_data(self, metadata, file_str, eqk):
        """
        Parses the event metadata to return an instance of the :class:
        smtk.sm_database.RecordDistance
        """

        epi_lon = get_float(
            metadata["COORDENADAS DEL EPICENTRO"].split(" ")[3])
        epi_lat = get_float(
            metadata["COORDENADAS DEL EPICENTRO"].split(" ")[0])
        sta_lon = get_float(
            metadata["COORDENADAS DE LA ESTACION"].split(" ")[3])
        sta_lat = get_float(
            metadata["COORDENADAS DE LA ESTACION"].split(" ")[0])

        p = Point(longitude=epi_lon, latitude=epi_lat)
        repi = p.distance(Point(longitude=sta_lon, latitude=sta_lat))

        # No hypocentral distance in file - calculate from event
        rhypo = sqrt(repi ** 2. + eqk.depth ** 2.)

        azimuth = Point(epi_lon, epi_lat, eqk.depth).azimuth(
            Point(sta_lon, sta_lat))

        dists = RecordDistance(repi, rhypo)
        dists.azimuth = azimuth
        return dists

    def _parse_site_data(self, metadata):
        """
        Parses the site metadata
        """
        try:
            altitude = get_float(metadata["ALTITUD (msnm)"])
        except:
            altitude = 0

        site = RecordSite(
            "|".join([metadata["INSTITUCION RESPONSABLE"],
                    metadata["CLAVE DE LA ESTACION"]]),
            metadata["CLAVE DE LA ESTACION"],
            metadata["NOMBRE DE LA ESTACION"],
            get_float(metadata["COORDENADAS DE LA ESTACION"].split(" ")[3]),
            get_float(metadata["COORDENADAS DE LA ESTACION"].split(" ")[0]),
            altitude)

        if "UNAM" in metadata["INSTITUCION RESPONSABLE"]:
            site.network_code = "UNAM"
        elif "CICESE" in metadata["INSTITUCION RESPONSABLE"]:
            site.network_code = "CICESE"
        else:
            site.network_code = "unknown"

        try:
            site.morphology = metadata["TIPO DE SUELO"]
        except:
            site.morphology = None

        site.instrument_type = metadata["MODELO DEL ACELEROGRAFO"]

        return site

    def _parse_component_data(self, wfid, metadata):
        """
        Returns the information specific to a component
        """
        units_provided = metadata["UNIDADES DE LOS DATOS"]
        # consider only units within parenthesis
        units = units_provided[units_provided.find("(") + 1:
                               units_provided.find(")")]

        xcomp = Component(
            wfid,
            orientation=None,
            ims=None,
            waveform_filter=None,
            baseline=None,
            units=units)

        ycomp = Component(
            wfid,
            orientation=None,
            ims=None,
            waveform_filter=None,
            baseline=None,
            units=units)

        zcomp = Component(
            wfid,
            orientation=None,
            ims=None,
            waveform_filter=None,
            baseline=None,
            units=units)

        return xcomp, ycomp, zcomp


class ASATimeSeriesParser(SMTimeSeriesReader):
    """
    Parses time series. Each file contains 3 components
    """

    def parse_records(self, record=None):
        """
        Parses the time series
        """

        time_series = OrderedDict([
            ("X", {"Original": {}, "SDOF": {}}),
            ("Y", {"Original": {}, "SDOF": {}}),
            ("V", {"Original": {}, "SDOF": {}})])

        target_names = list(time_series.keys())
        for iloc, ifile in enumerate(self.input_files):
            if not os.path.exists(ifile):
                continue
            else:
                component2parse = target_names[iloc]
                time_series[target_names[iloc]]["Original"] = \
                    self._parse_time_history(ifile, component2parse)
        return time_series

    def _parse_time_history(self, ifile, component2parse):
        """
        Parses the time history and returns the time history of the specified
        component. All 3 components are provided in every ASA file. Note that
        components are defined with various names, and are not always
        given in the same order
        """

        # The components are definied using the following names
        comp_names = {'X': ['N90E', 'N90E;', 'N90W', 'N90W;',
                            'S90E', 'S90W', 'E--W', 'S9OE'],
                      'Y': ['N00E', 'N00E;', 'N00W', 'N00W;',
                            'S00E', 'S00W', 'N--S', 'NOOE'],
                      'V': ['V', 'V;+', '+V', 'Z', 'VERT']}

        # Read component names, which are given on line 107
        o = open(ifile, "r")
        r = o.readlines()
        components = list(r[107].split())

        # Check if any component names are repeated
        if any(components.count(x) > 1 for x in components):
            raise ValueError(
                "Some components %s in record %s have the same name"
                % (components, ifile))
        # Check if more than 3 components are given
        if len(components) > 3:
            raise ValueError(
                "More than 3 components %s in record %s"
                % (components, ifile))

        # Get acceleration data from correct column
        column = None
        for i in comp_names[component2parse]:
            if i == components[0]:
                column = 0
                try:
                    accel = np.genfromtxt(ifile, skip_header=109,
                                          usecols=column, delimiter='')
                except:
                    raise ValueError(
                        "Check %s has 3 equal length time-series columns"
                        % ifile)
                break
            elif i == components[1]:
                column = 1
                try:
                    accel = np.genfromtxt(ifile, skip_header=109,
                        usecols=column, delimiter='')
                except:
                    raise ValueError(
                        "Check %s has 3 equal length time-series columns"
                        % ifile)
                break
            elif i == components[2]:
                column = 2
                try:
                    accel = np.genfromtxt(ifile, skip_header=109,
                                          usecols=column, delimiter='')
                except:
                    raise ValueError(
                        "Check %s has 3 equal length time-series columns"
                        % ifile)
                break
        if column is None:
                raise ValueError(
                    "None of the components %s were found to be \n\
                    the %s component of file %s" %
                    (components, component2parse, ifile))

        # Build the metadata dictionary again
        metadata = _get_metadata_from_file(ifile)

        # Get units
        units_provided = metadata["UNIDADES DE LOS DATOS"]
        units = units_provided[units_provided.find("(") + 1:
                               units_provided.find(")")]

        # Get time step, naming is not consistent so allow for variation
        for i in metadata:
            if 'INTERVALO DE MUESTREO, C1' in i:
                self.time_step = get_float(metadata[i].split("/")[1])

        # Get number of time steps, use len(accel) because
        # sometimes "NUM. TOTAL DE MUESTRAS, C1-C6" is wrong
        self.number_steps = len(accel)

        output = {
            "Acceleration": convert_accel_units(accel, self.units),
            "Time": get_time_vector(self.time_step, self.number_steps),
            "Time-step": self.time_step,
            "Number Steps": self.number_steps,
            "Units": self.units,
            "PGA": max(abs(accel)),
            "PGD": None
        }

        return output
