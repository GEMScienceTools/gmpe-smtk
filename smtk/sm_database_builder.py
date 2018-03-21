#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation and G. Weatherill
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
Constructs the HDF5 database
"""

import os
import sys
import re
import csv
import numpy as np
import h5py
import smtk.intensity_measures as ims
import smtk.sm_utils as utils
from smtk.parsers.base_database_parser import get_float
SCALAR_LIST = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "D5-95", "Housner"]

if sys.version_info[0] >= 3:
    # In Python 3 pickle uses cPickle by default
    import pickle
else:
    # In Python 2 use cPickle
    import cPickle as pickle


def _get_fieldnames_from_csv(reader):
    """
    """
    scalar_fieldnames = []
    spectra_fieldnames = []
    periods = []
    for fname in reader.fieldnames:
        if fname.startswith('SA('):
            match = re.match(r'^SA\(([^)]+?)\)$', fname)
            periods.append(float(match.group(1)))
            spectra_fieldnames.append(fname)
            continue
        for imt in SCALAR_LIST:
            if imt in fname:
                scalar_fieldnames.append((fname, imt))
    return scalar_fieldnames, spectra_fieldnames, np.array(periods)


class SMDatabaseBuilder(object):
    """
    Constructor of an hdf5-pkl pseudo-database.

    :param dbtype:
        Metadata reader as instance of class :class: SMDatabaseReader
    :param str location:
        Path to location of metadata (either directory name or file name)
    :param database:
        Strong motion database as instance of :class: SMDatabase
    :param time_series_parser:
        Parser for time series files, as instance of :class: SMTimeSeriesReader
    :param spectra_parser:
        Parser for spectra files, as instance of :class: SMSpectraReader
    :param str metafile:
        Path to output metadata file
    """
    TS_ATTRIBUTE_LIST = ["Year", "Month", "Day", "Hour", "Minute", "Second",
                         "Station Code", "Station Name", "Orientation",
                         "Processing", "Low Frequency Cutoff",
                         "High Frequency Cutoff"]

    IMS_SCALAR_LIST = SCALAR_LIST

    SPECTRA_LIST = ["Acceleration", "Velocity", "Displacement", "PSA", "PSV"]

    def __init__(self, dbtype, db_location):
        """
        Instantiation will create target database directory

        :param dbtype:
            Instance of :class:
                smtk.parsers.base_database_parser.SMDatabaseReader
        :param str db_location:
            Path to database to be written
        """
        self.dbtype = dbtype
        self.dbreader = None
        if os.path.exists(db_location):
            raise IOError("Target database directory %s already exists!"
                          % db_location)
        self.location = db_location
        os.mkdir(self.location)
        self.database = None
        self.time_series_parser = None
        self.spectra_parser = None
        self.metafile = None

    def build_database(self, db_id, db_name, metadata_location,
                       record_location=None):
        """
        Constructs the metadata database and exports to a .pkl file
        :param str db_id:
            Unique ID string of the database
        :param str db_name:
            Name of the database
        :param str metadata_location:
            Path to location of metadata
        :param str record_directory:
            Path to directory containing records (if different from metadata)
        """
        self.dbreader = self.dbtype(db_id, db_name, metadata_location,
                                    record_location)
        # Build database
        print("Reading database ...")
        self.database = self.dbreader.parse()
        self.metafile = os.path.join(self.location, "metadatafile.pkl")
        print("Storing metadata to file %s" % self.metafile)
        with open(self.metafile, "wb+") as f:
            pickle.dump(self.database, f)

    def parse_records(self, time_series_parser, spectra_parser=None,
                      units="cm/s/s"):
        """
        Parses the strong motion records to hdf5
        :param time_series_parser:
            Reader of the time series as instance of :class:
            smtk.parsers.base_database_parser.SMTimeSeriesReader
        :param spectra_parser:
            Reader of the spectra files as instance of :class:
            smtk.parsers.base_database_parser.SMSpectraReader
        :param str units:
            Units of the records
        """
        record_dir = os.path.join(self.location, "records")
        os.mkdir(record_dir)
        print("Creating repository for strong motion hdf5 records ... %s"
              % record_dir)
        nrecords = self.database.number_records()
        valid_records = []
        for iloc, record in enumerate(self.database.records):
            print("Processing record %s of %s" % (iloc, nrecords))
            has_spectra = isinstance(record.spectra_file, list) and\
                (spectra_parser is not None)
            # Parse strong motion record
            sm_parser = time_series_parser(record.time_series_file,
                                           self.dbreader.record_folder,
                                           units)
            if len(sm_parser.input_files) < 2:
                print("Record contains < 2 components - skipping!")
                continue
            sm_data = sm_parser.parse_records(record)
            if not sm_data.get("X", {}).get("Original", {}):
                print('No processed records - skipping')
                continue

            # Create hdf file and parse time series data
            fle, output_file = self.build_time_series_hdf5(record, sm_data,
                                                           record_dir)

            if has_spectra:
                # Parse spectra data
                spec_parser = spectra_parser(record.spectra_file,
                                             self.dbreader.filename)
                spec_data = spec_parser.parse_spectra()
                fle = self.build_spectra_hdf5(fle, spec_data)
            else:
                # Build the data structure for IMS
                self._build_hdf5_structure(fle, sm_data)
            fle.close()
            print("Record %s written to output file %s" % (record.id,
                                                           output_file))
            record.datafile = output_file
            valid_records.append(record)
        self.database.records = valid_records
        print("Updating metadata file")
        os.remove(self.metafile)
        with open(self.metafile, "wb+") as f:
            pickle.dump(self.database, f)
        print("Done!")

    def build_spectra_from_flatfile(self, component, damping="05",
                                    units="cm/s/s"):
        """
        In the case in which the spectra data is defined in the
        flatfile we construct the hdf5 from this information
        :param str component:
            Component to which the horizontal (or vertical!) records refer
        :param str damping"
            Percent damping
        """

        # Flatfile name should be stored in database parser
        # Get header

        reader = csv.DictReader(open(self.dbreader.filename, "r"))
        # Fieldnames
        scalar_fieldnames, spectra_fieldnames, periods =\
            _get_fieldnames_from_csv(reader)
        # Setup records folder
        record_dir = os.path.join(self.location, "records")
        os.mkdir(record_dir)
        print("Creating repository for strong motion hdf5 records ... %s"
              % record_dir)
        valid_idset = [rec.id for rec in self.database.records]
        for i, row in enumerate(reader):
            # Build database file
            # Waveform ID
            if not row["Record Sequence Number"] in valid_idset:
                # The record being passed has already been flagged as bad
                # skipping
                continue
            idx = valid_idset.index(row["Record Sequence Number"])
            wfid = self.database.records[idx].id
            output_file = os.path.join(record_dir, wfid + ".hdf5")
            self._build_spectra_hdf5_from_row(output_file, row, periods,
                                              scalar_fieldnames,
                                              spectra_fieldnames,
                                              component, damping, units)
            self.database.records[idx].datafile = output_file
            if (i % 100) == 0:
                print("Record %g written" % i)
        print("Updating metadata file")
        os.remove(self.metafile)
        with open(self.metafile, "wb+") as f:
            pickle.dump(self.database, f)
        print("Done!")

    def _build_spectra_hdf5_from_row(self, output_file, row, periods,
                                     scalar_fields, spectra_fields, component,
                                     damping, units):
        fle = h5py.File(output_file, "w-")
        ts_grp = fle.create_group("Time Series")
        ims_grp = fle.create_group("IMS")
        h_grp = ims_grp.create_group("H")
        scalar_grp = h_grp.create_group("Scalar")
        # Create Scalar values
        for f_attr, imt in scalar_fields:
            dset = scalar_grp.create_dataset(imt, (1,), dtype="f")
            dset.attrs["Component"] = component
            input_units = re.search('\((.*?)\)', f_attr).group(1)
            if imt == "PGA":
                # Convert acceleration from reported units to cm/s/s
                dset.attrs["Units"] = "cm/s/s"
                dset[:] = utils.convert_accel_units(get_float(row[f_attr]),
                                                    input_units)
            else:
                # For other values take direct from spreadsheet
                # Units should be given in parenthesis from fieldname
                dset.attrs["Units"] = input_units
                dset[:] = get_float(row[f_attr])

        spectra_grp = h_grp.create_group("Spectra")
        rsp_grp = spectra_grp.create_group("Response")
        # Setup periods dataset
        per_dset = rsp_grp.create_dataset("Periods",
                                          (len(periods),),
                                          dtype="f")
        per_dset.attrs["High Period"] = np.max(periods)
        per_dset.attrs["Low Period"] = np.min(periods)
        per_dset.attrs["Number Periods"] = len(periods)
        per_dset[:] = periods
        # Get response spectra
        spectra = np.array([get_float(row[f_attr])
                            for f_attr in spectra_fields])
        acc_grp = rsp_grp.create_group("Acceleration")
        comp_grp = acc_grp.create_group(component)
        spectra_dset = comp_grp.create_dataset("damping_{:s}".format(damping),
                                               (len(spectra),),
                                               dtype="f")
        spectra_dset.attrs["Units"] = "cm/s/s"
        spectra_dset[:] = utils.convert_accel_units(spectra, units)
        fle.close()

    def build_time_series_hdf5(self, record, sm_data, record_dir):
        """
        Constructs the hdf5 file for storing the strong motion record
        :param record:
            Strong motion record as instance of :class: GroundMotionRecord
        :param dict sm_data:
            Data dictionary for the strong motion record
        :param str record_dir:
            Directory in which to save the record
        """
        output_file = os.path.join(record_dir, record.id + ".hdf5")
        fle = h5py.File(output_file, "w-")
        grp = fle.create_group("Time Series")
        for key in sm_data.keys():
            if not sm_data[key]["Original"]:
                continue
            grp_comp = grp.create_group(key)
            grp_orig = grp_comp.create_group("Original Record")
            for attribute in self.TS_ATTRIBUTE_LIST:
                if attribute in sm_data[key]["Original"]:
                    grp_orig.attrs[attribute] =\
                        sm_data[key]["Original"][attribute]
            ts_dset = grp_orig.create_dataset(
                "Acceleration",
                (sm_data[key]["Original"]["Number Steps"],),
                dtype="f")
            ts_dset.attrs["Units"] = "cm/s/s"
            time_step = sm_data[key]["Original"]["Time-step"]
            ts_dset.attrs["Time-step"] = time_step
            number_steps = sm_data[key]["Original"]["Number Steps"]
            ts_dset.attrs["Number Steps"] = number_steps
            ts_dset.attrs["PGA"] = utils.convert_accel_units(
                sm_data[key]["Original"]["PGA"],
                sm_data[key]["Original"]["Units"])
            # Store acceleration as cm/s/s
            ts_dset[:] = utils.convert_accel_units(
                sm_data[key]["Original"]["Acceleration"],
                sm_data[key]["Original"]["Units"])
            # Get velocity and displacement
            vel, dis = utils.get_velocity_displacement(
                time_step,
                ts_dset[:],
                "cm/s/s")
            # Build velocity data set
            v_dset = grp_orig.create_dataset("Velocity",
                                             (number_steps,),
                                             dtype="f")
            v_dset.attrs["Units"] = "cm/s"
            v_dset.attrs["Time-step"] = time_step
            v_dset.attrs["Number Steps"] = number_steps
            v_dset[:] = vel
            # Build displacement data set
            d_dset = grp_orig.create_dataset("Displacement",
                                             (number_steps,),
                                             dtype="f")
            d_dset.attrs["Units"] = "cm"
            d_dset.attrs["Time-step"] = time_step
            d_dset.attrs["Number Steps"] = number_steps
            d_dset[:] = dis
                
        # Get the velocity and displacement time series and build scalar IMS
        return fle, output_file

    def _build_hdf5_structure(self, fle, data):
        """
        :param fle:
            Datastream of hdf file
        :param data:
            Strong motion database
        """
        grp0 = fle.create_group("IMS")
        for key in data.keys():
            grp_comp0 = grp0.create_group(key)
            grp_scalar = grp_comp0.create_group("Scalar")
            pga_dset = grp_scalar.create_dataset("PGA", (1,), dtype="f")
            pga_dset.attrs["Units"] = "cm/s/s"
            pgv_dset = grp_scalar.create_dataset("PGV", (1,), dtype="f")
            pgv_dset.attrs["Units"] = "cm/s"
            pgd_dset = grp_scalar.create_dataset("PGD", (1,), dtype="f")
            pgd_dset.attrs["Units"] = "cm"
            locn = "/".join(["Time Series", key, "Original Record"])
            pga_dset[:] = np.max(np.fabs(fle[locn + "/Acceleration"].value))
            pgv_dset[:] = np.max(np.fabs(fle[locn + "/Velocity"].value))
            pgd_dset[:] = np.max(np.fabs(fle[locn + "/Displacement"].value))

    def build_spectra_hdf5(self, fle, data):
        """
        Adds intensity measure data (scalar and spectra) to hdf5 datafile
        :param fle:
            h5py.File object for storing record data
        :param dict data:
            Intensity MEasure Data dictionary
        """
        grp0 = fle.create_group("IMS")
        for key in data.keys():
            if not data[key]["Spectra"]["Response"]:
                continue
            grp_comp0 = grp0.create_group(key)
            grp_scalar = grp_comp0.create_group("Scalar")
            for scalar_im in self.IMS_SCALAR_LIST:
                if scalar_im in data[key]["Scalar"]:
                    #print scalar_im, data[key]["Scalar"][scalar_im]
                    dset_scalar = grp_scalar.create_dataset(scalar_im, (1,),
                                                            dtype="f")
                    dset_scalar.attrs["Units"] =\
                        data[key]["Scalar"][scalar_im]["Units"]
                    dset_scalar[:] = data[key]["Scalar"][scalar_im]["Value"]
            grp_spectra = grp_comp0.create_group("Spectra")
            grp_four = grp_spectra.create_group("Fourier")
            grp_resp = grp_spectra.create_group("Response")
            # Add periods
            periods = data[key]["Spectra"]["Response"]["Periods"]
            num_per = len(data[key]["Spectra"]["Response"]["Periods"])
            dset_per = grp_resp.create_dataset("Periods", (num_per,),
                                               dtype="f")
            dset_per.attrs["Number Periods"] = num_per
            dset_per.attrs["Low Period"] = np.min(periods)
            dset_per.attrs["High Period"] = np.max(periods)
            dset_per[:] = periods
            # Add spectra
            for spec_type in self.SPECTRA_LIST:
                if not data[key]["Spectra"]["Response"][spec_type]:
                    continue
                # Parser spectra
                spec_data = data[key]["Spectra"]["Response"][spec_type]
                grp_spec = grp_resp.create_group(spec_type)
                grp_spec.attrs["Units"] = spec_data["Units"]
                for spc_key in spec_data.keys():
                    if spc_key == "Units":
                        continue
                    resp_dset = grp_spec.create_dataset(spc_key, (num_per,),
                                                        dtype="f")
                    resp_dset.attrs["Damping"] = float(spc_key.split("_")[1])
                    resp_dset[:] = spec_data[spc_key]
        return fle


def get_name_list(fle):
    """
    Returns structure of the hdf5 file as a list
    """
    name_list = []

    def append_name_list(name, obj):
        name_list.append(name)
    fle.visititems(append_name_list)
    return name_list


def add_recursive_nameset(fle, string):
    """
    For an input structure (e.g. AN/INPUT/STRUCTURE) will create the
    the corresponding name space at the level.
    """
    if string in get_name_list(fle):
        return
    levels = string.split("/")
    current_level = levels[0]
    if current_level not in fle:
        fle.create_group(current_level)

    for iloc in range(1, len(levels)):
        new_level = levels[iloc]
        if new_level not in fle[current_level]:
            fle[current_level].create_group(new_level)
            current_level = "/".join([current_level, new_level])


SCALAR_IMS = ["PGA", "PGV", "PGD", "CAV", "CAV5", "Ia", "T90", "Housner"]


SPECTRAL_IMS = ["Geometric", "Arithmetic", "Envelope", "Larger PGA"]


SCALAR_XY = {"Geometric": lambda x, y : np.sqrt(x * y),
             "Arithmetic": lambda x, y : (x + y) / 2.,
             "Larger": lambda x, y: np.max(np.array([x, y])),
             "Vectorial": lambda x, y : np.sqrt(x ** 2. + y ** 2.)}


ORDINARY_SA_COMBINATION = {
    "Geometric": ims.geometric_mean_spectrum,
    "Arithmetic": ims.arithmetic_mean_spectrum,
    "Envelope": ims.envelope_spectrum,
    "Larger PGA": ims.larger_pga
    }


class HorizontalMotion(object):
    """
    Base Class to implement methods to add horizontal motions to database
    """
    def __init__(self, fle, component="Geometric", periods=[], damping=0.05):
        """
        :param fle:
            Opem datastream of hdf5 file
        :param str component:
            The component of horizontal motion
        :param np.ndarray periods:
            Spectral periods
        :param float damping:
            Fractional coefficient of damping
        """
        self.fle = fle
        self.periods = periods
        self.damping = damping
        self.component = component

    def add_data(self):
        """
        Adds the data
        """


class AddPGA(HorizontalMotion):
    """
    Adds the resultant Horizontal PGA to the database
    """
    def add_data(self):
        """
        Takes PGA from X and Y component and determines the resultant
        horizontal component
        """
        if "PGA" not in self.fle["IMS/X/Scalar"]:
            x_pga = self._get_pga_from_time_series(
                "Time Series/X/Original Record/Acceleration",
                "IMS/X/Scalar")
        else:
            x_pga = self.fle["IMS/X/Scalar/PGA"].value

        if "PGA" not in self.fle["IMS/Y/Scalar"]:
            y_pga = self._get_pga_from_time_series(
                "Time Series/Y/Original Record/Acceleration",
                "IMS/Y/Scalar")
        else:
            y_pga = self.fle["IMS/Y/Scalar/PGA"].value

        h_pga = self.fle["IMS/H/Scalar"].create_dataset("PGA", (1,),
                                                        dtype=float)
        h_pga.attrs["Units"] = "cm/s/s"
        h_pga.attrs["Component"] = self.component
        h_pga[:] = SCALAR_XY[self.component](x_pga, y_pga)

    def _get_pga_from_time_series(self, time_series_location, target_location):
        """
        If PGA is not found as an attribute of the X or Y dataset then
        this extracts them from the time series.
        """
        pga = np.max(np.fabs(self.fle[time_series_location].value))
        pga_dset = self.fle[target_location].create_dataset("PGA", (1,),
                                                            dtype=float)
        pga_dset.attrs["Units"] = "cm/s/s"
        pga_dset[:] = pga
        return pga


class AddPGV(HorizontalMotion):
    """
    Adds the resultant Horizontal PGV to the database
    """
    def add_data(self):
        """
        Takes PGV from X and Y component and determines the resultant
        horizontal component
        """
        if "PGV" not in self.fle["IMS/X/Scalar"]:
            x_pgv = self._get_pgv_from_time_series(
                "Time Series/X/Original Record/",
                "IMS/X/Scalar")
        else:
            x_pgv = self.fle["IMS/X/Scalar/PGV"].value

        if "PGV" not in self.fle["IMS/Y/Scalar"]:
            y_pgv = self._get_pgv_from_time_series(
                "Time Series/Y/Original Record",
                "IMS/Y/Scalar")
        else:
            y_pgv = self.fle["IMS/Y/Scalar/PGV"].value

        h_pgv = self.fle["IMS/H/Scalar"].create_dataset("PGV", (1,),
                                                        dtype=float)
        h_pgv.attrs["Units"] = "cm/s"
        h_pgv.attrs["Component"] = self.component
        h_pgv[:] = SCALAR_XY[self.component](x_pgv, y_pgv)

    def _get_pgv_from_time_series(self, time_series_location, target_location):
        """
        If PGV is not found as an attribute of the X or Y dataset then
        this extracts them from the time series.
        """
        if "Velocity" not in self.fle[time_series_location]:
            accel_loc = time_series_location + "/Acceleration"
            # Add velocity to the record
            velocity, _ = ims.get_velocity_displacement(
                self.fle[accel_loc].attrs["Time-step"],
                self.fle[accel_loc].value)

            vel_dset = self.fle[time_series_location].create_dataset(
                "Velocity",
                (len(velocity),),
                dtype=float)

        else:
            velocity = self.fle[time_series_location + "/Velocity"].value

        pgv = np.max(np.fabs(velocity))
        pgv_dset = self.fle[target_location].create_dataset("PGV", (1,),
                                                            dtype=float)
        pgv_dset.attrs["Units"] = "cm/s/s"
        pgv_dset[:] = pgv
        return pgv


SCALAR_IM_COMBINATION = {"PGA": AddPGA,
                         "PGV": AddPGV}


class AddResponseSpectrum(HorizontalMotion):
    """
    Adds the resultant horizontal response spectrum to the database
    """
    def add_data(self):
        """
        Adds the response spectrum
        """
        if len(self.periods) == 0:
            self.periods = self.fle["IMS/X/Spectra/Response/Periods"].value[1:]

        x_acc = self.fle["Time Series/X/Original Record/Acceleration"]
        y_acc = self.fle["Time Series/Y/Original Record/Acceleration"]
        sax, say = ims.get_response_spectrum_pair(x_acc.value,
                                                  x_acc.attrs["Time-step"],
                                                  y_acc.value,
                                                  y_acc.attrs["Time-step"],
                                                  self.periods,
                                                  self.damping)
        sa_hor = ORDINARY_SA_COMBINATION[self.component](sax, say)
        dstring = "damping_" + str(int(100.0 * self.damping)).zfill(2)
        nvals = len(sa_hor["Acceleration"])
        self._build_group("IMS/H/Spectra/Response", "Acceleration",
                          "Acceleration", sa_hor, nvals, "cm/s/s", dstring)
        self._build_group("IMS/H/Spectra/Response", "Velocity",
                          "Velocity", sa_hor, nvals, "cm/s", dstring)
        self._build_group("IMS/H/Spectra/Response", "Displacement",
                          "Displacement", sa_hor, nvals, "cm", dstring)
        self._build_group("IMS/H/Spectra/Response", "PSA",
                          "Pseudo-Acceleration", sa_hor, nvals, "cm/s/s",
                          dstring)
        self._build_group("IMS/H/Spectra/Response", "PSV",
                          "Pseudo-Velocity", sa_hor, nvals, "cm/s", dstring)
        self._add_periods()

    def _build_group(self, base_string, key, im_key, sa_hor, nvals, units,
                     dstring):
        """
        Builds the group corresponding to the full definition of the
        resultant component
        """
        if key not in self.fle[base_string]:
            base_grp = self.fle[base_string].create_group(key)
        else:
            base_grp = self.fle["/".join([base_string, key])]
        base_cmp_grp = base_grp.create_group(self.component)
        dset = base_cmp_grp.create_dataset(dstring, (nvals,), dtype=float)
        dset.attrs["Units"] = units
        dset[:] = sa_hor[im_key]

    def _add_periods(self):
        """
        Adds the periods to the database
        """
        if "Periods" in self.fle["IMS/H/Spectra/Response"]:
            return
        dset = self.fle["IMS/H/Spectra/Response"].create_dataset(
            "Periods",
            (len(self.periods),),
            dtype="f")
        dset.attrs["High Period"] = np.max(self.periods)
        dset.attrs["Low Period"] = np.min(self.periods)
        dset.attrs["Number Periods"] = len(self.periods)
        dset[:] = self.periods


class AddGMRotDppSpectrum(AddResponseSpectrum):
    """
    Adds the GMRotDpp spectrum to the database
    """
    def add_data(self, percentile=50.0):
        """
        :param float percentile:
            Percentile (pp)
        """
        if len(self.periods) == 0:
            self.periods = self.fle["IMS/X/Spectra/Response/Periods"].value[1:]

        x_acc = self.fle["Time Series/X/Original Record/Acceleration"]
        y_acc = self.fle["Time Series/Y/Original Record/Acceleration"]

        gmrotdpp = ims.gmrotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                y_acc.value, y_acc.attrs["Time-step"],
                                self.periods, percentile, self.damping)
        dstring = "damping_" + str(int(100.0 * self.damping)).zfill(2)
        nvals = len(gmrotdpp)
        # Acceleration
        if not "Acceleration" in self.fle["IMS/H/Spectra/Response"]:
            acc_grp = self.fle["IMS/H/Spectra/Response"].create_group(
                "Acceleration")
        else:
            acc_grp = self.fle["IMS/H/Spectra/Response/Acceleration"]
        acc_cmp_grp = acc_grp.create_group("GMRotD" + 
                                           str(int(percentile)).zfill(2))
        acc_dset = acc_cmp_grp.create_dataset(dstring, (nvals,), dtype=float)
        acc_dset.attrs["Units"] = "cm/s/s"
        acc_dset[:] = gmrotdpp["GMRotDpp"]
        self._add_periods()


class AddRotDppSpectrum(AddResponseSpectrum):
    """
    Adds the RotDpp spectrum to the database
    """
    def add_data(self, percentile=50.0):
        """
        :param float percentile:
            Percentile (pp)
        """
        if len(self.periods) == 0:
            self.periods = self.fle["IMS/X/Spectra/Response/Periods"].value[1:]

        x_acc = self.fle["Time Series/X/Original Record/Acceleration"]
        y_acc = self.fle["Time Series/Y/Original Record/Acceleration"]
        rotdpp = ims.rotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                y_acc.value, y_acc.attrs["Time-step"],
                                self.periods, percentile, self.damping)[0]
        dstring = "damping_" + str(int(100.0 * self.damping)).zfill(2)
        nvals = len(rotdpp["Pseudo-Acceleration"])
        # Acceleration
        if not "Acceleration" in self.fle["IMS/H/Spectra/Response"]:
            acc_grp = self.fle["IMS/H/Spectra/Response"].create_group(
                "Acceleration")
        else:
            acc_grp = self.fle["IMS/H/Spectra/Response/Acceleration"]
        acc_cmp_grp = acc_grp.create_group("RotD" + 
                                           str(int(percentile)).zfill(2))
        acc_dset = acc_cmp_grp.create_dataset(dstring, (nvals,), dtype=float)
        acc_dset.attrs["Units"] = "cm/s/s"
        acc_dset[:] = rotdpp["Pseudo-Acceleration"]
        self._add_periods()


class AddGMRotIppSpectrum(AddResponseSpectrum):
    """
    Adds the GMRotIpp spectrum to the database
    """
    def add_data(self, percentile=50.0):
        """
        :param float percentile:
            Percentile (pp)
        """
        if len(self.periods) == 0:
            self.periods = self.fle["IMS/X/Spectra/Response/Periods"].value[1:]

        x_acc = self.fle["Time Series/X/Original Record/Acceleration"]
        y_acc = self.fle["Time Series/Y/Original Record/Acceleration"]
        sa_hor = ims.gmrotipp(x_acc.value, x_acc.attrs["Time-step"],
                              y_acc.value, y_acc.attrs["Time-step"],
                              self.periods, percentile, self.damping)
        nvals = len(sa_hor["Acceleration"])
        dstring = "damping_" + str(int(100.0 * self.damping)).zfill(2)
        # Acceleration
        self._build_group("IMS/H/Spectra/Response", "Acceleration", 
                          "Acceleration", sa_hor, nvals, "cm/s/s", dstring)
        # Velocity
        self._build_group("IMS/H/Spectra/Response", "Velocity", 
                          "Velocity", sa_hor, nvals, "cm/s", dstring)
        # Displacement
        self._build_group("IMS/H/Spectra/Response", "Displacement", 
                          "Displacement", sa_hor, nvals, "cm", dstring)
        # Pseudo-Acceletaion
        self._build_group("IMS/H/Spectra/Response", "PSA", 
                          "Pseudo-Acceleration", sa_hor, nvals,
                          "cm/s/s", dstring)
        # Pseudo-Velocity
        self._build_group("IMS/H/Spectra/Response", "PSV", 
                          "Pseudo-Velocity", sa_hor, nvals, "cm/s", dstring)
        self._add_periods()


SPECTRUM_COMBINATION = {"Geometric": AddResponseSpectrum,
                        "Arithmetic": AddResponseSpectrum,  
                        "Envelope": AddResponseSpectrum,  
                        "Larger PGA": AddResponseSpectrum} 


def add_horizontal_im(database, intensity_measures, component="Geometric",
        damping="05", periods=[]):
    """
    For a database this adds the resultant horizontal components to the
    hdf databse for each record
    :param database:
        Strong motion databse as instance of :class:
        smtk.sm_database.GroundMotionDatabase
    :param list intensity_measures:
        List of strings of intensity measures
    :param str Geometric:
        For scalar measures only, defines the resultant horizontal component
    :param str damping:
        Percentile damping
    :param list/np.ndarray periods:
        Periods
    """
    nrecs = len(database.records)
    for iloc, record in enumerate(database.records):
        print("Processing %s (Record %s of %s)" % (record.datafile, 
                                                   iloc + 1,
                                                   nrecs))
        fle = h5py.File(record.datafile, "r+")
        add_recursive_nameset(fle, "IMS/H/Spectra/Response")
        fle["IMS/H/"].create_group("Scalar")
        for intensity_measure in intensity_measures:
            if len(intensity_measure.split("GMRotI")) > 1:
                # GMRotIpp
                percentile = float(intensity_measure.split("GMRotI")[1])
                i_m = AddGMRotIppSpectrum(fle, intensity_measure, periods, 
                                          float(damping) / 100.)
                i_m.add_data(percentile)
            elif len(intensity_measure.split("GMRotD")) > 1:
                # GMRotDpp
                percentile = float(intensity_measure.split("GMRotD")[1])
                i_m = AddGMRotDppSpectrum(fle, intensity_measure, periods, 
                                          float(damping) / 100.)
                i_m.add_data(percentile)
            elif len(intensity_measure.split("RotD")) > 1:
                # RotDpp
                percentile = float(intensity_measure.split("RotD")[1])
                i_m = AddRotDppSpectrum(fle, intensity_measure, periods, 
                                          float(damping) / 100.)
                i_m.add_data(percentile)
            elif intensity_measure in SCALAR_IMS:
                # Is a scalar value
                i_m = SCALAR_IM_COMBINATION[intensity_measure](fle,
                    component,
                    periods,
                    float(damping) / 100.)
                i_m.add_data()
            elif intensity_measure in SPECTRAL_IMS:
                # Is a normal spectrum combination
                i_m = SPECTRUM_COMBINATION[intensity_measure](fle,
                    component,
                    periods,
                    float(damping) / 100.)
                i_m.add_data()
            else:
                raise ValueError("Unrecognised Intensity Measure!")
        fle.close()
