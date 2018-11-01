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
Basic classes for the GMDatabase (HDF5 database) and parsers
"""

import os
import sys
import re
import csv
import hashlib
from datetime import datetime
from itertools import chain
from contextlib import contextmanager
from collections import defaultdict
import tables
from tables.table import Table
from tables.group import Group
from tables.exceptions import HDF5ExtError, NoSuchNodeError
import numpy as np
import h5py
from tables.description import IsDescription, Int64Col, StringCol, \
    Int16Col, UInt16Col, Float32Col, Float16Col, TimeCol, BoolCol, \
    UInt8Col, Float64Col, Int8Col, UInt64Col, UInt32Col, EnumCol
import smtk.intensity_measures as ims
from smtk import sm_utils
from smtk.parsers.base_database_parser import get_float

# rewrite pytables description column types to account for default
# values meaning MISSING, and bounds (min and max)
def _col(col_class, **kwargs):
    '''utility function returning a pytables Column. The rationale behind
    this simple wrapper (`_col(StringCol, ...)` equals `StringCol(...)`)
    is twofold:

    1. pytables do not allow Nones, which would be perfect to specify missing
    values. Therefore, a Gm database table column needs to use the
    column default ('dflt') as missing value.
    This function builds the default automatically, when not explicitly
    provided as 'dflt' argument in `kwargs`: 0 for Unsigned integers, the
    empty string for Enum (if the empty string is not provided in the `enum`
    argument, it will be inserted), nan for floats and complex types, the
    empty string for `StringCol`s.
    This way, a default value which can be reasonably also considered as
    missing value is set (only exception: `BoolCol`s, for which we do not
    have a third value which can be considered missing)
    2. pytables columns do not allow bounds (min, max), which can be
    specified here as 'min' and 'max' arguments. None or missing values will
    mean: no check on the relative bound (any value allowed)

    :param: col_class: the pytables column class, e.g. StringCol
    :param kwargs: keyword argument to be passed to `col_class` during
        initialization. Note thtat the `dflt` parameter, if provided
        will be overridden. See the `atom` module of pytables for a list
        of arguments for each Column class
    '''
    if 'dflt' not in kwargs:
        if col_class == StringCol:
            dflt = b''
        elif col_class == EnumCol:
            dflt = ''
            if dflt not in kwargs['enum']:
                kwargs['enum'].insert(0, dflt)
        elif col_class == BoolCol:
            dflt = False
        elif col_class.__name__.startswith('Complex'):
            dflt = complex(float('nan'), float('nan'))
        elif col_class.__name__.startswith('Float'):
            dflt = float('nan')
        elif col_class.__name__.startswith('UInt'):
            dflt = 0
        elif col_class.__name__.startswith('Int8'):
            dflt = np.iinfo(np.int8).min
        elif col_class.__name__.startswith('Int16'):
            dflt = np.iinfo(np.int16).min
        elif col_class.__name__.startswith('Int32'):
            dflt = np.iinfo(np.int32).min
        elif col_class.__name__.startswith('Int64'):
            dflt = np.iinfo(np.int64).min
        elif col_class.__name__.startswith('Int'):
            dflt = np.iinfo(np.int).min

        kwargs['dflt'] = dflt
    min_, max_ = kwargs.pop('min', None), kwargs.pop('max', None) 
    ret = col_class(**kwargs)
    ret.min_value, ret.max_value = min_, max_
    return ret


class GMDatabaseTable(IsDescription):
    """
    Implements a GMDatabase as `pytable.IsDescription` class.
    This class is the skeleton of the data structure of HDF5 tables, which
    map flatfiles data (in CSV) in an HDF5 file.

    **Remember that, with the exception of `BoolCol`s, default values
    are interpreted as 'missing'. Usually, no dflt argument has to be passed
    here as it will be set by default (see `_col` function)
    """
    # FIXME: check DEFAULTS. nan for floats, empty string for strings,
    # and -999? for integers? what for datetimes? empty strings? think about it?
    # Also, what about max length strings?

    # CONVENTION: StringCols default is '' (fine)
    # FloatCol is float('nan') by default is 0 (NOT fine)
    # IntCol will be set as the minimum allowed value (default is 0, not fine)
    # TimeCol: see int col

    # what = column(StringCol, 16)
    # id = UInt32Col()  # no default. Starts from 1 incrementally
    # max id: 4,294,967,295
    record_id = _col(StringCol, itemsize=20)
    event_id = _col(StringCol, itemsize=20)
    event_name = _col(StringCol, itemsize=40)
    event_country = _col(StringCol, itemsize=30)
    event_time = _col(StringCol, itemsize=19)  # In ISO Format YYYY-MM-DDTHH:mm:ss
    # Note: if we want to support YYYY-MM-DD only be aware that:
    # YYYY-MM-DD == YYYY-MM-DDT00:00:00
    # Note2: no support for microseconds for the moment
    event_latitude = _col(Float64Col, min=-90, max=90)
    event_longitude = _col(Float64Col, min=-180, max=180)
    hypocenter_depth = _col(Float32Col)
    magnitude = _col(Float16Col)
    magnitude_type = _col(StringCol, itemsize=5)
    magnitude_uncertainty = _col(Float32Col)
    tectonic_environment = _col(StringCol, itemsize=30)
    strike_1 = _col(Float32Col)
    strike_2 = _col(Float32Col)
    dip_1 = _col(Float32Col)
    dip_2 = _col(Float32Col)
    rake_1 = _col(Float32Col)
    rake_2 = _col(Float32Col)
    style_of_faulting = _col(Float32Col)
    depth_top_of_rupture = _col(Float32Col)
    rupture_length = _col(Float32Col)
    rupture_width = _col(Float32Col)
    station_id = _col(StringCol, itemsize=20)
    station_name = _col(StringCol, itemsize=40)
    station_latitude = _col(Float64Col, min=-90, max=90)
    station_longitude = _col(Float64Col, min=-180, max=180)
    station_elevation = _col(Float32Col)
    vs30 = _col(Float32Col)
    vs30_measured = _col(BoolCol, dflt=True)
    vs30_sigma = _col(Float32Col)
    depth_to_basement = _col(Float32Col)
    z1 = _col(Float32Col)
    z2pt5 = _col(Float32Col)
    repi = _col(Float32Col)  # epicentral_distance
    rhypo = _col(Float32Col)  # Float32Col
    rjb = _col(Float32Col)  # joyner_boore_distance
    rrup = _col(Float32Col)  # rupture_distance
    rx = _col(Float32Col)
    ry0 = _col(Float32Col)
    azimuth = _col(Float32Col)
    digital_recording = _col(BoolCol, dflt=True)
#     acceleration_unit = _col(EnumCol, enum=['cm/s/s', 'm/s/s', 'g'],
#                              base='uint8')
    type_of_filter = _col(StringCol, itemsize=25)
    npass = _col(Int8Col)
    nroll = _col(Float32Col)
    hp_h1 = _col(Float32Col)
    hp_h2 = _col(Float32Col)
    lp_h1 = _col(Float32Col)
    lp_h2 = _col(Float32Col)
    factor = _col(Float32Col)
    lowest_usable_frequency_h1 = _col(Float32Col)
    lowest_usable_frequency_h2 = _col(Float32Col)
    lowest_usable_frequency_avg = _col(Float32Col)
    highest_usable_frequency_h1 = _col(Float32Col)
    highest_usable_frequency_h2 = _col(Float32Col)
    highest_usable_frequency_avg = _col(Float32Col)
    pga = _col(Float64Col)
    pgv = _col(Float64Col)
    pgd = _col(Float64Col)
    duration_5_75 = _col(Float64Col)
    duration_5_95 = _col(Float64Col)
    arias_intensity = _col(Float64Col)
    cav = _col(Float64Col)
    sa = _col(Float64Col, shape=(111,))


class GMDatabaseParser(object):
    '''
    Implements a base class for parsing flatfiles in csv format into
    GmDatabase files in HDF5 format. The latter are Table-like heterogeneous
    datasets (each representing a flatfile) organized in subfolders-like
    structures called groups.
    See the :class:`GmDatabaseTable` for a description of the Table columns
    and types.

    The parsing is done in the `parse` method. The typical workflow
    is to implement a new subclass for each new flatfile release.
    Subclasses should override the `mapping` dict where flatfile
    specific column names are mapped to :class:`GmDatabaseTable` column names
    and optionally the `parse_row` method where additional operation
    is performed in-place on each flatfile row. For more details, see
    the :method:`parse_row` method docstring
    '''
    _accel_units = ["g", "m/s/s", "m/s**2", "m/s^2",
                    "cm/s/s", "cm/s**2", "cm/s^2"]

    _ref_periods = [0.010, 0.020, 0.022, 0.025, 0.029, 0.030, 0.032,
                    0.035, 0.036, 0.040, 0.042, 0.044, 0.045, 0.046,
                    0.048, 0.050, 0.055, 0.060, 0.065, 0.067, 0.070,
                    0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.110,
                    0.120, 0.130, 0.133, 0.140, 0.150, 0.160, 0.170,
                    0.180, 0.190, 0.200, 0.220, 0.240, 0.250, 0.260,
                    0.280, 0.290, 0.300, 0.320, 0.340, 0.350, 0.360,
                    0.380, 0.400, 0.420, 0.440, 0.450, 0.460, 0.480,
                    0.500, 0.550, 0.600, 0.650, 0.667, 0.700, 0.750,
                    0.800, 0.850, 0.900, 0.950, 1.000, 1.100, 1.200,
                    1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900,
                    2.000, 2.200, 2.400, 2.500, 2.600, 2.800, 3.000,
                    3.200, 3.400, 3.500, 3.600, 3.800, 4.000, 4.200,
                    4.400, 4.600, 4.800, 5.000, 5.500, 6.000, 6.500,
                    7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000,
                    11.000, 12.000, 13.000, 14.000, 15.000, 20.000]

    # the regular expression used to parse SAs periods. Note capturing
    # group for the SA period:
    _sa_periods_re = re.compile(r'^\s*sa\s*\((.*)\)\s*$',
                                re.IGNORECASE)  # @UndefinedVariable

    # the regular expression used to parse PGA periods. Note capturing
    # group for the PGA unit
    _pga_unit_re = re.compile(r'^\s*pga\s*\((.*)\)\s*$',
                              re.IGNORECASE)  # @UndefinedVariable

    # this field is a list of strings telling which are the column names
    # of the event time. If:
    # 1. A list of a single item => trivial case, it denotes the event time
    # column, which must be supplied as ISO format
    # 2. A list of length 3: => then it denotes the column names of the year,
    # month and day, respectively, all three int-parsable strings
    # 3. A list of length 6: then it denotes the column names of the
    # year, month, day hour minutes seconds, repsectively, all six int-parsable
    # strings
    _event_time_colnames = ['year', 'month', 'day', 'hour', 'minute', 'second']

    # The csv column names will be then converted according to the
    # `_mappings` dict below, where a csv flatfile column is mapped to its
    # corresponding Gm database column name. The mapping should take care of
    # all 1 to 1 mappings. This is the first operation performed on any row.
    # When providing the mappings, keep in mind that the algorithm
    # after the mapping will perform the following operations:
    # 1. Columns matching `_sa_periods_re` will be parsed and log-log
    # interpolated with '_ref_periods'. The resulting data will be put in the
    # Gm database 'sa' column.
    # 2. If a column 'event_time' is missing, then the program searches
    # for '_event_time_colnames' (ignoring case) and parses the date. The
    # resulting date (in ISO formatted string) will be put in the Gm databse
    # column 'event_time'.
    # 3. If a column matching `_pga_unit_re` is found, then the unit is
    # stored and the Gm databse column 'pga' is filled with the PGA value,
    # converted to cm/s/s.
    # 4. The `parse_row` method is called. Therein, the user should
    # implement any more complex operation
    # 5 a row is written, the columns 'event_id' , 'station_id' and
    # 'record_id' are automatically filled to uniquely identify their
    # respective entitites
    mappings = {}

    @classmethod
    def parse(cls, flatfile_path, output_path, mode='a'):
        '''Parses a flat file and writes its content in the GM database file
        `output_path`, which is a HDF5 organized hierarchically in groups
        (sort of sub-directories) each of which identifies a parsed
        input flatfile. Each group's `table` attribute is where
        the actual GM database data is stored and can be accessed later
        with pytables `open_file` function (see
        https://www.pytables.org/usersguide/tutorials.html).
        The group will have the same name as `flatfile_path` (more precisely,
        the file basename without extension).

        :param flatfile_path: string denoting the path to the input CSV
            flatfile
        :param output_path: string: path to the output GM database file.
        :param mode: either 'w' or 'a'. It is NOT the `mode` option of the
            `open_file` function (which is always 'a'): 'a' means append to
            the existing **table**, if it exists (otherwise create a new one),
            'w' means write (i.e. overwrite the existing table, if any).
            In case of 'a' and the table exists, it is up to the user not to
            add duplicated entries
        :return: a dictionary holding information with keys:
            'total': the total number of csv rows
            'written': the number of parsed rows written on the db table
            'error': a list of integers denoting the position (from
                0 = first row) of the parsed rows not written on the db table
                because of errors
            'missing_values': a dict with table column names as keys, mapped
                to the number of rows which have missing values for that
                column (e.g., invalid/empty values in the csv, or most
                likely, a column not found, if the number of missing values
                equals 'total').
            'outofbound_values': a dict with table column names as keys,
                mapped to the number of rows which had out-of-bound values for
                that column.

            Missing and out-of-bound values are stored in the GM database with
            the column default, which is usually NaN for floats, the minimum
            possible value for integers, the empty string for strings
        '''
        dbname = os.path.splitext(os.path.basename(flatfile_path))[0]
        with cls.get_table(output_path, dbname, mode) as table:

            i, error, missing, outofbound = \
                -1, [], defaultdict(int), defaultdict(int)

            for i, rowdict in enumerate(cls._rows(flatfile_path)):

                if rowdict:
                    tablerow = table.row
                    missingcols, outofboundcols = \
                        cls._writerow(rowdict, tablerow, dbname)
                    tablerow.append()  # pylint: disable=no-member
                    table.flush()
                else:
                    missingcols, outofboundcols = [], []
                    error.append(i)

                # write statistics:
                for col in missingcols:
                    missing[col] += 1
                for col in outofboundcols:
                    outofbound[col] += 1

            return {'total': i+1, 'written': i+1-len(error), 'error': error,
                    'missing_values': missing, 'outofbound_values': outofbound}

    @staticmethod
    @contextmanager
    def get_table(filepath, name, mode='r'):
        '''Yields a pytable Table object representing a Gm database
        in the given hdf5 file `filepath`. Creates such a table if mode != 'r'
        and the table does not exists.

        Example:
        ```
            with GmDatabaseParser.get_table(filepath, name, 'r') as table:
                # ... do your operation here
        ```

        :param filepath: the string denoting the path to the hdf file
            previously created with this method. If `mode`
            is 'r', the file must exist
        :param name: the name of the database table
        :param mode: the mode ('a', 'r', 'w') whereby the **table** is opened.
            I.e., 'w' does not overwrites the whole file, but the table data.
            More specifically:
            'r': opens file in 'r' mode, raises if the file or the table in
                the file content where not found
            'w': opens file in 'a' mode, creates the table if it does not
                exists, clears all table data if it exists. Eventually it
                returns the table
            'a': open file in 'a' mode, creates the table if it does not
                exists, does nothing otherwise. Eventually it returns the table

        :raises: :class:`tables.exceptions.NoSuchNodeError` if mode is 'r'
            and the table was not found in `filepath`, IOError if the
            file does not exist
        '''
        with tables.open_file(filepath, mode if mode == 'r' else 'a') \
                as h5file:
            table = None
            tablename = 'table'
            tablepath = '/%s/%s' % (name, tablename)
            try:
                table = h5file.get_node(tablepath, classname=Table.__name__)
                if mode == 'w':
                    h5file.remove_node(tablepath, recursive=True)
                    table = None
            except NoSuchNodeError as _:
                if mode == 'r':
                    raise
                table = None
                # create parent group node
                try:
                    h5file.get_node("/%s" % name, classname=Group.__name__)
                except NoSuchNodeError as _:
                    h5file.create_group(h5file.root, name)

            if table is None:
                table = h5file.create_table("/%s" % name, tablename,
                                            description=GMDatabaseTable)
            yield table

    @classmethod
    def _rows(cls, flatfile_path):  # pylint: disable=too-many-locals
        '''Yields each row from the CSV file `flatfile_path` as
        dictionary, after performing SA conversion and running custom code
        implemented in `cls.parse_row` (if overridden by
        subclasses). Yields empty dict in case of exceptions'''
        ref_log_periods = np.log10(cls._ref_periods)
        mappings = getattr(cls, 'mappings', {})
        with cls._get_csv_reader(flatfile_path) as reader:

            newfieldnames = [mappings[f] if f in mappings else f for f in
                             reader.fieldnames]
            # get spectra fieldnames and priods:
            try:
                spectra_fieldnames, spectra_periods =\
                    cls._get_sa_columns(newfieldnames)
            except Exception as exc:
                raise ValueError('Unable to parse SA columns '
                                 '("sa(<period>)"): %s' % str(exc))

            # get event time fieldname(s):
            try:
                evtime_fieldnames = \
                    cls._get_event_time_columns(newfieldnames, 'event_time')
            except Exception as exc:
                raise ValueError('Unable to parse event '
                                 'time column(s): %s' % str(exc))

            # get pga fieldname and units:
            try:
                pga_col, pga_unit = cls._get_pga_column(newfieldnames)
            except Exception as exc:
                raise ValueError('Unable to parse PGA column '
                                 '("pga(<unit>)"): %s' % str(exc))

            for rowdict in reader:
                # re-map keys:
                for k in mappings:
                    rowdict[mappings[k]] = rowdict.pop(k)

                # assign values (sa, event time, pga):
                try:
                    rowdict['sa'] = cls._get_sa(rowdict, spectra_fieldnames,
                                                ref_log_periods,
                                                spectra_periods)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                try:
                    rowdict['event_time'] = \
                        cls._get_event_time(rowdict, evtime_fieldnames)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                try:
                    rowdict['pga'] = cls._get_pga(rowdict, pga_col, pga_unit)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                try:
                    # custom post processing, if needed in subclasses:
                    cls.parse_row(rowdict)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                if not cls._sanity_check(rowdict):
                    rowdict = {}

                # yield row as dict:
                yield rowdict

    @classmethod
    def _sanity_check(cls, rowdict):
        '''performs sanity checks on the csv row `rowdict` before
        writing it. Note that  pytables does not support roll backs,
        and when closing the file pending data is automatically flushed.
        Therefore, the data has to be checked before, on the csv row'''
        # for the moment, just do a pga/sa[0] check for unit consistency
        # other methods might be added in the future
        return cls._pga_sa_unit_ok(rowdict)

    @classmethod
    def _pga_sa_unit_ok(cls, rowdict):
        '''Checks that pga unit and sa unit are in accordance
        '''
        # if the PGA and the acceleration in the shortest period of the SA
        # columns differ by more than an order of magnitude then certainly
        # there is something wrong and the units of the PGA and SA are not
        # in agreement and an error should be raised.
        try:
            pga, sa0 = float(rowdict['pga']) / 981., float(rowdict['sa'][0])
            retol = abs(max(pga, sa0) / min(pga, sa0))
            if not np.isnan(retol) and round(retol) >= 10:
                return False
        except Exception as _:  # disable=bare-except
            # it might seem weird to return true on exceptions, but this method
            # should only check wheather there is certainly a unit
            # mismatch between sa and pga, and int that case only return True
            pass
        return True

    @staticmethod
    @contextmanager
    def _get_csv_reader(filepath, dict_reader=True):
        '''opends a csv file and yields the relative reader. To be used
        in a with statement to properly close the csv file'''
        # according to the docs, py3 needs the newline argument
        kwargs = {'newline': ''} if sys.version_info[0] >= 3 else {}
        with open(filepath, **kwargs) as csvfile:
            reader = csv.DictReader(csvfile) if dict_reader else \
                csv.reader(csvfile)
            yield reader

    @classmethod
    def _get_sa_columns(cls, csv_fieldnames):
        """Returns the field names, the spectra fieldnames and the periods
        (numoy array) of e.g., a parsed csv reader's fieldnames
        """
        spectra_fieldnames = []
        periods = []
        reg = cls._sa_periods_re
        for fname in csv_fieldnames:
            match = reg.match(fname)
            if match:
                periods.append(float(match.group(1)))
                spectra_fieldnames.append(fname)

        return spectra_fieldnames, np.array(periods)

    @staticmethod
    def _get_sa(rowdict, spectra_fieldnames, ref_log_periods, spectra_periods):
        '''gets sa values with log log interpolation if needed'''
        sa_values = np.array([rowdict.get(key) for key in spectra_fieldnames],
                             dtype=float)
        logx = np.log10(spectra_periods)
        logy = np.log10(sa_values)
        return np.power(10.0, np.interp(ref_log_periods, logx, logy))

    @classmethod
    def _get_event_time_columns(cls, csv_fieldnames, default_colname):
        '''returns the event time column names'''
        if default_colname in csv_fieldnames:
            return [default_colname]
        evtime_defnames = {_.lower(): i for i, _ in
                           enumerate(cls._event_time_colnames)}
        evtime_names = [None] * 6
        for fname in csv_fieldnames:
            index = evtime_defnames.get(fname.lower(), None)
            if index is not None:
                evtime_names[index] = fname

        for _, caption in zip(evtime_names, ['year', 'month', 'day']):
            if _ is None:
                raise Exception('column "%s" not found' % caption)

        return evtime_names

    @classmethod
    def _get_event_time(cls, rowdict, evtime_fieldnames):
        '''returns the event time column names'''
        args = []
        for i, fieldname in enumerate(evtime_fieldnames):
            args.append(int(rowdict[fieldname] if i < 3 else
                            rowdict.get(fieldname, 0)))

        dtm = None
        if len(evtime_fieldnames) == 1:
            formats_ = ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d']
            for frmt in formats_:
                try:
                    dtm = datetime.strptime(args[0], frmt)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError('unparsable date-time string')
        else:
            dtm = datetime(*args)

        return dtm.strftime('%Y-%m-%dT%H:%M:%S')

    @classmethod
    def _get_pga_column(cls, csv_fieldnames):
        reg = cls._pga_unit_re
        for fname in csv_fieldnames:
            match = reg.match(fname)
            if match:
                unit = match.group(1)
                if unit not in cls._accel_units:
                    raise Exception('unit not in %s' % str(cls._accel_units))
                return fname, unit
        raise Exception('no matching column found')

    @classmethod
    def _get_pga(cls, rowdict, pga_column, pga_unit):
        '''Returns the pga value from the given `rowdict[pga_column]`
        converted to cm/^2
        '''
        return sm_utils.convert_accel_units(float(rowdict[pga_column]),
                                            pga_unit)

    @classmethod
    def parse_row(cls, rowdict):
        '''This method is intended to be overridden by subclasses (by default
        is no-op) to perform any further operation on the given csv row
        `rowdict` before writing it to the GM databse file.

        Please **keep in mind that**:

        1. This method should process `rowdict` in place, the returned value is
           ignored. Any exception raised here is wrapped in the caller method.
        2. `rowdict` keys might not be the same as the csv
           field names (first csv row). See `mappings` class attribute
        3. The values of `rowdict` are all strings, i.e. they have still to be
           parsed to the correct column type, except those mapped to the keys
           'sa', 'pga' and 'event_time': if present in the csv field names,
           their values have been already set (in case of errors, their
           values are left as they were in the csv).
        4. the `rowdict` keys 'event_id', 'station_id' and 'record_id' are
           reserved and their values will be anyway overridden, as they
           must represent hash string whereby we might weant to compare same
           events, stations and records, respectively

        :param rowdict: a row of the csv flatfile, as Python dict
        '''
        pass

    @classmethod
    def _writerow(cls, csvrow, tablerow, dbname):
        '''writes the content of csvrow into tablerow. Returns two lists:
        The missing column names (a missing column is also a column for which
        the csv value is invalid, i.e. it raised during assignement), and
        the out-of-bounds column names (in case bounds were provided in the
        column class. In this case, the default of that column will be set
        in `tablerow`)'''
        missing_colnames, outofbounds_colnames = [], []
        for col, colobj in tablerow.table.coldescrs.items():
            if col not in csvrow:
                missing_colnames.append(col)
                continue
            try:
                # remember: if val is a castable string -> ok
                #   (e.g. table column float, val is '5.5')
                # if val is out of bounds for the specific type, -> ok
                #   (casted to the closest value)
                # if val is scalar and the table column is a N length array,
                # val it is broadcasted
                #   (val= 5, then tablerow will have a np.array of N 5s)
                # TypeError is raised when there is a non castable element
                #   (e.g. 'abc' for a Float column): in this case pass
                tablerow[col] = csvrow[col]

                bound = getattr(colobj, 'min_value', None)
                if bound is not None and \
                        (np.asarray(tablerow[col]) < bound).any():
                    tablerow[col] = colobj.dflt
                    outofbounds_colnames.append(col)
                    continue

                bound = getattr(colobj, 'max_value', None)
                if bound is not None and \
                        (np.asarray(tablerow[col]) > bound).any():
                    tablerow[col] = colobj.dflt
                    outofbounds_colnames.append(col)
                    continue  # actually useless, but if we add code below ...

            except (ValueError, TypeError):
                missing_colnames.append(col)

        # build a record hashes as ids:
        evid, staid, recid = cls.get_ids(tablerow, dbname)
        tablerow['event_id'] = evid
        tablerow['station_id'] = staid
        tablerow['record_id'] = recid

        return missing_colnames, outofbounds_colnames

    @classmethod
    def get_ids(cls, tablerow, dbname):
        '''Returns the tuple record_id, event_id and station_id from
        the given HDF5 row `tablerow`'''
        toint = cls._toint
        ids = (dbname,
               toint(tablerow['pga']/981., 2),  # convert from cm/s^2 to g
               toint(tablerow['event_longitude'], 5),
               toint(tablerow['event_latitude'], 5),
               toint(tablerow['hypocenter_depth'], 3),
               tablerow['event_time'],
               toint(tablerow['station_longitude'], 5),
               toint(tablerow['station_latitude'], 5))
        # return event_id, station_id, record_id:
        return cls._hash(*ids[2:6]), cls._hash(*ids[6:]), cls._hash(*ids)

    @classmethod
    def _toint(cls, value, decimals):
        '''returns an integer by multiplying value * 10^decimals
        and rounding the result to int. Returns nan if value is nan'''
        return value if np.isnan(value) else \
            int(round((10**decimals)*value))

    @classmethod
    def _hash(cls, *values):
        '''generates a 160bit (20bytes) hash bytestring which uniquely
        identifies the given tuple of `values`.
        The returned string is assured to be the same for equal `values`
        tuples (note that the order of values matters).
        Conversely, the probability of colliding hashes, i.e., returning
        the same bytestring for two different tuples of values, is 1 in
        100 billion for roughly 19000 hashes (roughly 10 flatfiles with
        all different records), and apporaches 50% for for 1.42e24 hashes
        generated (for info, see
        https://preshing.com/20110504/hash-collision-probabilities/#small-collision-probabilities)

        :param values: a list of values, either bytes, str or numeric
            (no support for other values sofar)
        '''
        hashalg = hashlib.sha1()
        # use the slash as separator s it is unlikely to be in value(s):
        hashalg.update(b'/'.join(cls._tobytestr(v) for v in values))
        return hashalg.digest()

    @classmethod
    def _tobytestr(cls, value):
        '''converts a value to bytes. value can be bytes, str or numeric'''
        if not isinstance(value, bytes):
            value = str(value).encode('utf8')
        return value


def rows(filepath, dbname, *condition_expression, limit=None):
    count = 0
    with GMDatabaseParser.get_table(filepath, dbname, mode='r') as tbl:
        for row in tbl.where(condition_expression):
            if limit is None or count < limit:
                yield row

#     @staticmethod
#     def _alreay_existing(rowdict, table):
#         '''NOT USED (FIXME: remove) yields an iterator over table with
#         elements equal to `rowdict`
#         according to event spatial and termporal coordinates and station
#         coordinates
#         :param rowdict: a dict representing a flatfile csv rowdict
#         '''
#         try:
#             condition_syntax = ('(event_time == %s) & '
#                                 '(event_latitude == %s) &'
#                                 '(event_longitude == %s) &'
#                                 '(station_latitude == %s) &'
#                                 '(station_longitude == %s)') % \
#                 (rowdict['event_time'].encode('utf8'),
#                  str(rowdict['event_latitude']),
#                  str(rowdict['event_longitude']),
#                  str(rowdict['station_latitude']),
#                  str(rowdict['station_longitude']))
# 
#             for tablerow in table.where(condition_syntax):
#                 # compare pga, convert to g (pga are assumed to be cm/s^2):
#                 pga1_g = rowdict['pga'] / 981.
#                 pga2_g = tablerow['pga'] / 981.
#                 if abs(pga1_g - pga2_g) < 0.01:
#                     yield tablerow
#         except KeyError:
#             return []  # mimic no rowdict found

#     # station code should be a static method
#     def get_station_id_columns(self, columns):
#         # this should be station_id+'.'+station_code for nga west2
#         # and net.sta.loc.cha for ESM
# 
#     def get_event_id_columns(self, columns):
#         
#     def _make_event_id(tablerow):
#         _tablerow['event_id'] = hash(tablerow['event_latitude'],
#                                      tablerow['event_longitude'],
#                                      tablerow['event_depth'],
#                                      )
#     def __init__(self, flatfile_location):
#         """
#         Instantiation will create target database directory
#
#         :param dbtype:
#             Instance of :class:
#                 smtk.parsers.base_database_parser.SMDatabaseReader
#         :param str db_location:
#             Path to database to be written.
#         """
#         self.dbreader = None
#         if os.path.exists(flatfile_location):
#             raise IOError("Target database directory %s already exists!"
#                           % flatfile_location)
#         self.location = db_location
#         os.mkdir(self.location)
#         self.database = None
#         self.time_series_parser = None
#         self.spectra_parser = None
#         self.metafile = None
# 
#     def build_database(self, db_id, db_name, metadata_location,
#                        record_location=None):
#         """
#         Constructs the metadata database and exports to a .pkl file
#         :param str db_id:
#             Unique ID string of the database
#         :param str db_name:
#             Name of the database
#         :param str metadata_location:
#             Path to location of metadata
#         :param str record_directory:
#             Path to directory containing records (if different from metadata)
#         """
#         self.dbreader = self.dbtype(db_id, db_name, metadata_location,
#                                     record_location)
#         # Build database
#         print("Reading database ...")
#         self.database = self.dbreader.parse()
#         self.metafile = os.path.join(self.location, "metadatafile.pkl")
#         print("Storing metadata to file %s" % self.metafile)
#         with open(self.metafile, "wb+") as f:
#             pickle.dump(self.database, f)
# 
#     def parse_records(self, time_series_parser, spectra_parser=None,
#                       units="cm/s/s"):
#         """
#         Parses the strong motion records to hdf5
#         :param time_series_parser:
#             Reader of the time series as instance of :class:
#             smtk.parsers.base_database_parser.SMTimeSeriesReader
#         :param spectra_parser:
#             Reader of the spectra files as instance of :class:
#             smtk.parsers.base_database_parser.SMSpectraReader
#         :param str units:
#             Units of the records
#         """
#         record_dir = os.path.join(self.location, "records")
#         os.mkdir(record_dir)
#         print("Creating repository for strong motion hdf5 records ... %s"
#               % record_dir)
#         nrecords = self.database.number_records()
#         valid_records = []
#         for iloc, record in enumerate(self.database.records):
#             print("Processing record %s of %s" % (iloc, nrecords))
#             has_spectra = isinstance(record.spectra_file, list) and\
#                 (spectra_parser is not None)
#             # Parse strong motion record
#             sm_parser = time_series_parser(record.time_series_file,
#                                            self.dbreader.record_folder,
#                                            units)
#             if len(sm_parser.input_files) < 2:
#                 print("Record contains < 2 components - skipping!")
#                 continue
#             sm_data = sm_parser.parse_records(record)
#             if not sm_data.get("X", {}).get("Original", {}):
#                 print('No processed records - skipping')
#                 continue
# 
#             # Create hdf file and parse time series data
#             fle, output_file = self.build_time_series_hdf5(record, sm_data,
#                                                            record_dir)
# 
#             if has_spectra:
#                 # Parse spectra data
#                 spec_parser = spectra_parser(record.spectra_file,
#                                              self.dbreader.filename)
#                 spec_data = spec_parser.parse_spectra()
#                 fle = self.build_spectra_hdf5(fle, spec_data)
#             else:
#                 # Build the data structure for IMS
#                 self._build_hdf5_structure(fle, sm_data)
#             fle.close()
#             print("Record %s written to output file %s" % (record.id,
#                                                            output_file))
#             record.datafile = output_file
#             valid_records.append(record)
#         self.database.records = valid_records
#         print("Updating metadata file")
#         os.remove(self.metafile)
#         with open(self.metafile, "wb+") as f:
#             pickle.dump(self.database, f)
#         print("Done!")
# 
#     def build_spectra_from_flatfile(self, component, damping="05",
#                                     units="cm/s/s"):
#         """
#         In the case in which the spectra data is defined in the
#         flatfile we construct the hdf5 from this information
#         :param str component:
#             Component to which the horizontal (or vertical!) records refer
#         :param str damping"
#             Percent damping
#         """
# 
#         # Flatfile name should be stored in database parser
#         # Get header
# 
#         reader = csv.DictReader(open(self.dbreader.filename, "r"))
#         # Fieldnames
#         scalar_fieldnames, spectra_fieldnames, periods =\
#             _get_fieldnames_from_csv(reader)
#         # Setup records folder
#         record_dir = os.path.join(self.location, "records")
#         os.mkdir(record_dir)
#         print("Creating repository for strong motion hdf5 records ... %s"
#               % record_dir)
#         valid_idset = [rec.id for rec in self.database.records]
#         for i, row in enumerate(reader):
#             # Build database file
#             # Waveform ID
#             if not row["Record Sequence Number"] in valid_idset:
#                 # The record being passed has already been flagged as bad
#                 # skipping
#                 continue
#             idx = valid_idset.index(row["Record Sequence Number"])
#             wfid = self.database.records[idx].id
#             output_file = os.path.join(record_dir, wfid + ".hdf5")
#             self._build_spectra_hdf5_from_row(output_file, row, periods,
#                                               scalar_fieldnames,
#                                               spectra_fieldnames,
#                                               component, damping, units)
#             self.database.records[idx].datafile = output_file
#             if (i % 100) == 0:
#                 print("Record %g written" % i)
#         print("Updating metadata file")
#         os.remove(self.metafile)
#         with open(self.metafile, "wb+") as f:
#             pickle.dump(self.database, f)
#         print("Done!")
# 
#     def _build_spectra_hdf5_from_row(self, output_file, row, periods,
#                                      scalar_fields, spectra_fields, component,
#                                      damping, units):
#         fle = h5py.File(output_file, "w-")
#         ts_grp = fle.create_group("Time Series")
#         ims_grp = fle.create_group("IMS")
#         h_grp = ims_grp.create_group("H")
#         scalar_grp = h_grp.create_group("Scalar")
#         # Create Scalar values
#         for f_attr, imt in scalar_fields:
#             dset = scalar_grp.create_dataset(imt, (1,), dtype="f")
#             dset.attrs["Component"] = component
#             input_units = re.search('\((.*?)\)', f_attr).group(1)
#             if imt == "PGA":
#                 # Convert acceleration from reported units to cm/s/s
#                 dset.attrs["Units"] = "cm/s/s"
#                 dset[:] = utils.convert_accel_units(get_float(row[f_attr]),
#                                                     input_units)
#             else:
#                 # For other values take direct from spreadsheet
#                 # Units should be given in parenthesis from fieldname
#                 dset.attrs["Units"] = input_units
#                 dset[:] = get_float(row[f_attr])
# 
#         spectra_grp = h_grp.create_group("Spectra")
#         rsp_grp = spectra_grp.create_group("Response")
#         # Setup periods dataset
#         per_dset = rsp_grp.create_dataset("Periods",
#                                           (len(periods),),
#                                           dtype="f")
#         per_dset.attrs["High Period"] = np.max(periods)
#         per_dset.attrs["Low Period"] = np.min(periods)
#         per_dset.attrs["Number Periods"] = len(periods)
#         per_dset[:] = periods
#         # Get response spectra
#         spectra = np.array([get_float(row[f_attr])
#                             for f_attr in spectra_fields])
#         acc_grp = rsp_grp.create_group("Acceleration")
#         comp_grp = acc_grp.create_group(component)
#         spectra_dset = comp_grp.create_dataset("damping_{:s}".format(damping),
#                                                (len(spectra),),
#                                                dtype="f")
#         spectra_dset.attrs["Units"] = "cm/s/s"
#         spectra_dset[:] = utils.convert_accel_units(spectra, units)
#         fle.close()
# 
#     def build_time_series_hdf5(self, record, sm_data, record_dir):
#         """
#         Constructs the hdf5 file for storing the strong motion record
#         :param record:
#             Strong motion record as instance of :class: GroundMotionRecord
#         :param dict sm_data:
#             Data dictionary for the strong motion record
#         :param str record_dir:
#             Directory in which to save the record
#         """
#         output_file = os.path.join(record_dir, record.id + ".hdf5")
#         fle = h5py.File(output_file, "w-")
#         grp = fle.create_group("Time Series")
#         for key in sm_data.keys():
#             if not sm_data[key]["Original"]:
#                 continue
#             grp_comp = grp.create_group(key)
#             grp_orig = grp_comp.create_group("Original Record")
#             for attribute in self.TS_ATTRIBUTE_LIST:
#                 if attribute in sm_data[key]["Original"]:
#                     grp_orig.attrs[attribute] =\
#                         sm_data[key]["Original"][attribute]
#             ts_dset = grp_orig.create_dataset(
#                 "Acceleration",
#                 (sm_data[key]["Original"]["Number Steps"],),
#                 dtype="f")
#             ts_dset.attrs["Units"] = "cm/s/s"
#             time_step = sm_data[key]["Original"]["Time-step"]
#             ts_dset.attrs["Time-step"] = time_step
#             number_steps = sm_data[key]["Original"]["Number Steps"]
#             ts_dset.attrs["Number Steps"] = number_steps
#             ts_dset.attrs["PGA"] = utils.convert_accel_units(
#                 sm_data[key]["Original"]["PGA"],
#                 sm_data[key]["Original"]["Units"])
#             # Store acceleration as cm/s/s
#             ts_dset[:] = utils.convert_accel_units(
#                 sm_data[key]["Original"]["Acceleration"],
#                 sm_data[key]["Original"]["Units"])
#             # Get velocity and displacement
#             vel, dis = utils.get_velocity_displacement(
#                 time_step,
#                 ts_dset[:],
#                 "cm/s/s")
#             # Build velocity data set
#             v_dset = grp_orig.create_dataset("Velocity",
#                                              (number_steps,),
#                                              dtype="f")
#             v_dset.attrs["Units"] = "cm/s"
#             v_dset.attrs["Time-step"] = time_step
#             v_dset.attrs["Number Steps"] = number_steps
#             v_dset[:] = vel
#             # Build displacement data set
#             d_dset = grp_orig.create_dataset("Displacement",
#                                              (number_steps,),
#                                              dtype="f")
#             d_dset.attrs["Units"] = "cm"
#             d_dset.attrs["Time-step"] = time_step
#             d_dset.attrs["Number Steps"] = number_steps
#             d_dset[:] = dis
#                 
#         # Get the velocity and displacement time series and build scalar IMS
#         return fle, output_file
# 
#     def _build_hdf5_structure(self, fle, data):
#         """
#         :param fle:
#             Datastream of hdf file
#         :param data:
#             Strong motion database
#         """
#         grp0 = fle.create_group("IMS")
#         for key in data.keys():
#             grp_comp0 = grp0.create_group(key)
#             grp_scalar = grp_comp0.create_group("Scalar")
#             pga_dset = grp_scalar.create_dataset("PGA", (1,), dtype="f")
#             pga_dset.attrs["Units"] = "cm/s/s"
#             pgv_dset = grp_scalar.create_dataset("PGV", (1,), dtype="f")
#             pgv_dset.attrs["Units"] = "cm/s"
#             pgd_dset = grp_scalar.create_dataset("PGD", (1,), dtype="f")
#             pgd_dset.attrs["Units"] = "cm"
#             locn = "/".join(["Time Series", key, "Original Record"])
#             pga_dset[:] = np.max(np.fabs(fle[locn + "/Acceleration"].value))
#             pgv_dset[:] = np.max(np.fabs(fle[locn + "/Velocity"].value))
#             pgd_dset[:] = np.max(np.fabs(fle[locn + "/Displacement"].value))
# 
#     def build_spectra_hdf5(self, fle, data):
#         """
#         Adds intensity measure data (scalar and spectra) to hdf5 datafile
#         :param fle:
#             h5py.File object for storing record data
#         :param dict data:
#             Intensity MEasure Data dictionary
#         """
#         grp0 = fle.create_group("IMS")
#         for key in data.keys():
#             if not data[key]["Spectra"]["Response"]:
#                 continue
#             grp_comp0 = grp0.create_group(key)
#             grp_scalar = grp_comp0.create_group("Scalar")
#             for scalar_im in self.IMS_SCALAR_LIST:
#                 if scalar_im in data[key]["Scalar"]:
#                     #print scalar_im, data[key]["Scalar"][scalar_im]
#                     dset_scalar = grp_scalar.create_dataset(scalar_im, (1,),
#                                                             dtype="f")
#                     dset_scalar.attrs["Units"] =\
#                         data[key]["Scalar"][scalar_im]["Units"]
#                     dset_scalar[:] = data[key]["Scalar"][scalar_im]["Value"]
#             grp_spectra = grp_comp0.create_group("Spectra")
#             grp_four = grp_spectra.create_group("Fourier")
#             grp_resp = grp_spectra.create_group("Response")
#             # Add periods
#             periods = data[key]["Spectra"]["Response"]["Periods"]
#             num_per = len(data[key]["Spectra"]["Response"]["Periods"])
#             dset_per = grp_resp.create_dataset("Periods", (num_per,),
#                                                dtype="f")
#             dset_per.attrs["Number Periods"] = num_per
#             dset_per.attrs["Low Period"] = np.min(periods)
#             dset_per.attrs["High Period"] = np.max(periods)
#             dset_per[:] = periods
#             # Add spectra
#             for spec_type in self.SPECTRA_LIST:
#                 if not data[key]["Spectra"]["Response"][spec_type]:
#                     continue
#                 # Parser spectra
#                 spec_data = data[key]["Spectra"]["Response"][spec_type]
#                 grp_spec = grp_resp.create_group(spec_type)
#                 grp_spec.attrs["Units"] = spec_data["Units"]
#                 for spc_key in spec_data.keys():
#                     if spc_key == "Units":
#                         continue
#                     resp_dset = grp_spec.create_dataset(spc_key, (num_per,),
#                                                         dtype="f")
#                     resp_dset.attrs["Damping"] = float(spc_key.split("_")[1])
#                     resp_dset[:] = spec_data[spc_key]
#         return fle


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
