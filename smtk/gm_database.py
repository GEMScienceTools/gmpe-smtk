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
from contextlib import contextmanager
from collections import defaultdict
import tables
from tables.file import File
from tables.table import Table
from tables.group import Group
from tables.exceptions import NoSuchNodeError
import numpy as np
from tables.description import IsDescription, Int64Col, StringCol, \
    Int16Col, UInt16Col, Float32Col, Float16Col, TimeCol, BoolCol, \
    UInt8Col, Float64Col, Int8Col, UInt64Col, UInt32Col, EnumCol
from smtk import sm_utils


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

    :param: col_class: the pytables column class, e.g. StringCol. You can
        also supply the String "DateTime" which will set default to StringCol
        adding the default 'itemsize' to `kwargs` and a custom attribute
        'is_datetime_str' to the returned object. The attribute will be used
        in the `expr` class to properly cast passed values into the correct
        date-time ISO-formated string
    :param kwargs: keyword argument to be passed to `col_class` during
        initialization. Note thtat the `dflt` parameter, if provided
        will be overridden. See the `atom` module of pytables for a list
        of arguments for each Column class
    '''
    is_iso_dtime = col_class == 'DateTime'
    if is_iso_dtime:
        col_class = StringCol
        if 'itemsize' not in kwargs:
            kwargs['itemsize'] = 19  # '1999-31-12T01:02:59'
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
    if is_iso_dtime:
        ret.is_datetime_str = True
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
    event_time = _col("DateTime")  # In ISO Format YYYY-MM-DDTHH:mm:ss
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
                raise ValueError('Unable to parse SA columns: %s' % str(exc))

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
                raise ValueError('Unable to parse PGA column: %s' % str(exc))

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
                    acc_unit = rowdict[pga_unit] \
                        if pga_unit == 'acceleration_unit' else pga_unit
                    rowdict['pga'] = cls._get_pga(rowdict, pga_col, acc_unit)
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
        except Exception as _:  # disable=broad-except
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
        dtime = rowdict[evtime_fieldnames[0]]
        if len(evtime_fieldnames) > 1:
            args = [int(rowdict[fieldname] if i < 3 else
                        rowdict.get(fieldname, 0))
                    for i, fieldname in enumerate(evtime_fieldnames)]
            dtime = datetime(*args)

        return cls.normalize_dtime(dtime)

    @staticmethod
    def normalize_dtime(dtime):
        '''Returns a datetime *string* in ISO format ('%Y-%m-%dT%H:%M:%S')
        representing `dtime`

        :param dtime: string or datetime. In the former case, it must be
            in any of these formats:
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y'
        :return: ISO formatted string representing `dtime`
        :raises: ValueError (string not parsable) or TypeError (`dtime`
            neither datetime not string)
        '''
        base_format = '%Y-%m-%dT%H:%M:%S'
        if not isinstance(dtime, datetime):
            formats_ = [base_format, '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y']
            for frmt in formats_:
                try:
                    dtime = datetime.strptime(dtime, frmt)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError('Unparsable as date-time: "%s"' % str(dtime))
        return dtime.strftime(base_format)

    @classmethod
    def _get_pga_column(cls, csv_fieldnames):
        '''returns the column name denoting the PGA and the PGA unit.
        The latter is usually retrieved in the PGA column name. Otherwise,
        if a column 'PGA' *and* 'acceleration_unit' are found, returns
        the names of those columns'''
        reg = cls._pga_unit_re
        # store fields 'pga' and 'acceleration_unit', if present:
        pgacol, pgaunitcol = None, None
        for fname in csv_fieldnames:
            if not pgacol and fname.lower().strip() == 'pga':
                pgacol = fname
                continue
            if not pgaunitcol and fname.lower().strip() == 'acceleration_unit':
                pgaunitcol = fname
                continue
            match = reg.match(fname)
            if match:
                unit = match.group(1)
                if unit not in cls._accel_units:
                    raise Exception('unit not in %s' % str(cls._accel_units))
                return fname, unit
        # no pga(<unit>) column found. Check if we had 'pga' and
        # 'acceleration_unit'
        pgacol_ok, pgaunitcol_ok = pgacol is not None, pgaunitcol is not None
        if pgacol_ok and not pgaunitcol_ok:
            raise ValueError("provide field 'acceleration_unit' or "
                             "specify unit in '%s'" % pgacol)
        elif not pgacol_ok and pgaunitcol_ok:
            raise ValueError("missing field 'pga'")
        elif pgacol_ok and pgaunitcol_ok:
            return pgacol, pgaunitcol
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
           ignored. Any exception raised here is hanlded in the caller method.
        2. `rowdict` keys might not be the same as the csv
           field names (first csv row). See `mappings` class attribute
        3. The values of `rowdict` are all strings, i.e. they have still to be
           parsed to the correct column type, except those mapped to the keys
           'sa', 'pga' and 'event_time', if present.
        4. the `rowdict` keys 'event_id', 'station_id' and 'record_id' are
           reserved and their values will be anyway overridden, as they
           must represent hash string whereby comparing same
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
        # use the slash as separator as it is unlikely to be in value(s):
        hashalg.update(b'/'.join(cls._tobytestr(v) for v in values))
        return hashalg.digest()

    @classmethod
    def _tobytestr(cls, value):
        '''converts a value to bytes. value can be bytes, str or numeric'''
        if not isinstance(value, bytes):
            value = str(value).encode('utf8')
        return value

#########################################
# Database selection / maniuplation
#########################################


def get_table(filepath, dbname):
    '''Returns a Gm database table from the given database name `dbname`
    located in the specific HDF5 file with path `filepath`. To be used within
    a "with" statement:
    ```
    with get_table(filepath, dbname):
        # do your stuff here
    '''
    return GMDatabaseParser.get_table(filepath, dbname, 'r')


def records_where(table, condition, limit=None):
    '''Returns an iterator yielding records (Python dicts) of the
    database table "dbname" stored inside the HDF5 file with path `filepath`.
    The records returned will be filtered according to `condition`.
    IMPORTANT: This function is designed to be used inside a `for ...` loop
    to avoid loading all data into memory. Do **not** do this as it fails:
    `list(records_where(...))`.
    If you want all records in a list (be aware of potential meory leaks
    for huge amount of data) use :function:`read_where`

    Example:
    ```
        condition = between("pga", 0.5, 0.8) | lt("pgv", 1.1)
        with get_table(...) as table:
            for rec in records_where(table, condition):
                # do your stuff with `rec`, e.g. access the fields:
                sa = rec["sa"]
                pga = rec['pga']  # and so on...
    ```
    The same can be obtained by specifying `condition` without built-in
    functions of this module (`eq ne lt gt le ge isin between isaval`) but
    with the default pytables string expression syntax. Note however that
    this approach has some caveats (see [1]) which the first approach solves.
    Example with standard string expression:
    ```
        condition = "((pga >= 0.5) & (pga <=0.8)) | (pgv <1.1)"
        # the remainder of the code is the same as the example above
    ```

    :param table: The pytables Table object. See module function `get_table`
    :param condition: a string expression denoting a selection condition.
        See https://www.pytables.org/usersguide/tutorials.html#reading-and-selecting-data-in-a-table

        Alternatively `condition` can be given with the safer and more
        flexible expression objects imoplemented in this module, which can be
        prepended with the negation operator ~ or concatenated with the logical
        operators & (and), | (or):
        ```
        eq(column, value)  # column equal to value (works if value is nan)
        ne(column, value)  # column not equal to value (works if value is nan)
        lt(column, value)  # column lower than value
        gt(column, value)  # column greathen than value
        le(column, value)  # column lower or equal to value
        ge(column, value)  # colum greater or equal to value
        isaval(column)  # column value is available (i.e. not missing)
            # (for boolean columns, isaval always returns all records)
        between(column, min, max)  # column between (or equal to) min and max
        isin(column, *values)  # column equals any of the given values
        ```
        Example: the following `condition` (select element with PGA lower
        than 0.14 or greater than 1.1, with available PGV (not nan) and whose
        earthquake happened before 2006):
        ```
        ~between('pga', 0.14, 1.1) & ne('pgv', 'nan') &
            lt('event_time', '2006')
        ```
        should be rendered as string with the less friendly:
        ```
        "(pga < 0.14) | (pga > 1.1) & (pgv == pgv) &
            (event_time < b'2006-01-01T00:00:00')"
        ```
        See note [1] below for details if you need to implement string
        expressions.

    :param limit: integer (defaults: None) implements a SQL 'limit'
        when provided, yields only the first `limit` matching rows

    --------------------------------------------------------------------------

    [1] The use of the module level expression objects in the `condition`
    argument avoids some issues that users implementing strings should be
    aware of:
    1. expressions concatenated with & or | should be put into brakets. This
    does *not* work:
        "pga <= 0.5 & pgv > 9.5"
    whereas this does:
        "(pga <= 0.5) & (pgv > 9.5)"
    2. NaNs might be tricky to compare. For instance, given a valid column
    name (e.g. ,"pga") and the variable `value = float("nan")`, this does
    not work:
        "pga == %s" % str(value)
        "pga != %s" % str(value)
    whereas these get what expected:
        "pga != pga"
        "pga == pga"
    3. String column types (e.g., 'event_country') should be compared with
    quoted strings. Given e.g. `value = 'Germany'`, this does not work:
        "event_country == %s" % str(value)
    whereas this work:
        "event_country == '%s'" % str(value)
        'event_country == "%s"' % str(value)
    (in pytables documentation, they claim that in Python3 the above do not
    work either, as they should be encoded into bytes:
    "event_country == %s" % str(value).encode('utf8')
    **but** when tested in Python3.6.2 these work, so the claim is false or
    incomplete. Maybe it works as long as `value` has ascii characters only?).
    '''
    if condition not in ('False', 'false'):
        for count, row in enumerate(table.iterrows()
                                    if condition in ('True', 'true', None)
                                    else table.where(condition)):
            if limit is None or count < limit:
                yield row


def read_where(table, condition, limit=None):
    '''Returns a list of records (Python dicts) of the
    database table "dbname" stored inside the HDF5 file with path `filepath`.
    The records returned will be filtered according to `condition`.
    IMPORTANT: This function loads all data into memory
    To avoid potential memory leaks (especially if for some reason
    `condition` is 'True' or 'true' or None), use :function:`records_where`.

    All parameters are the same as :function:`records_where`
    '''
    if condition in ('True', 'true', None):
        return table.read()[:limit]
    if condition in ('False', 'false'):
        return []
    return table.read_where(condition)[:limit]

#############################
# Database selection syntax #
#############################


class expr(str):  # pylint: disable=invalid-name
    '''expression class subclassing string
    All expression classes are lower case as they mimic functions rather than
    classes'''
    # dict of valid operators mapped to their negtation:
    _operators = {'==': '!=', '!=': '==', '>': '<=', '<': '>=',
                  '>=': '<', '<=': '>'}

    def __new__(cls, *args):
        '''Creates a string expression parsing the given arguments, e.g.:

            ```expr("pgv", "<=", 9)```
        where
        1st arg is the column name (string)
        2nd arg is the operator (string): <= >= < > == !=
        3rd arg is a Python value

        This class takes care of converting value to
        the proper column's Python type and handles some caveats
        (e.g., quoting strings, handling NaNs comparison, date-time conversion
        to ISO formatted strings, casting to float or int when needed)

        :raise: ValueError, TypeError when `value` or `operator` are invalid
        '''
        # Note: the constructor with one or two arguments shoule be used only
        # by module-level functions and not exposed publicly
        if len(args) == 3:  # 3 args: column, operator, value: process value
            col, operator, value = args
            cls._check_operator(operator)
            colobj = cls._getcolobj(col)
            colclass = colobj.__class__.__name__
            if getattr(colobj, 'is_datetime_str', False):  # aka datetime col
                value = GMDatabaseParser.normalize_dtime(value)
            if isinstance(colobj, StringCol):
                if not isinstance(value, bytes):
                    # encode in bytes (pytables claims is mandatory in py3,
                    # although their claim does not seem to be true in py3.6.2)
                    value = str(value).encode('utf8')
            elif colclass.startswith('Int') or colclass.startswith('UInt'):
                value = int(value)
            elif colclass.startswith('Float'):
                value = float(value)
                if np.isnan(value):
                    if operator not in ('==', '!='):
                        raise ValueError('only != and == supported with NaNs')
                    # swap col==nan with col!=col, col!=nan with col==col:
                    value, operator = col, cls._operators[operator]
            _str = "%s %s %s" % (col, operator, str(value))
            _negation = "%s %s %s" % (col, cls._operators[operator],
                                      str(value))
        elif len(args) == 2:  # 2 args: expression, negation (internal use)
            _str = str(args[0])
            _negation = str(args[1])
        else:  # 1 arg: expression (internal use)
            _str = str(args[0])
            _negation = args[0]._negation if isinstance(args[0], expr) \
                else "~(%s)" % _str
        _str, _negation = cls._final_check(_str, _negation)
        ret = str.__new__(cls, _str)
        # _negation is the logical negation of this expression. It makes
        # composition more readable (and probably also more efficient when
        # selecting with pytables): example: ~~expr('pga', '<', 9.5) (negate
        # twice) will return the same expression "pga < 9.5" instead of
        # "~(~(pga < 9.5))". The little memory overhead of storing an
        # additional string is negligible
        ret._negation = _negation  # pylint: disable=protected-access
        return ret

    @classmethod
    def _getcolobj(cls, colname):
        dbcolumns = GMDatabaseTable.columns  # pylint: disable=no-member
        if colname not in dbcolumns:
            raise ValueError("Unknown table field '%s'" % str(colname))
        return dbcolumns[colname]

    @classmethod
    def _check_operator(cls, operator):
        if operator not in cls._operators:
            raise ValueError("Unknown operator '%s'" % str(operator))

    @classmethod
    def _final_check(cls, expression, negation):
        if not expression or expression == 'None':
            raise ValueError('empty expression')
        if expression in ('True', 'true'):
            expression, negation = 'True', 'False'
        elif expression in ('False', 'false'):
            expression, negation = 'False', 'True'
        return expression, negation

    def __and__(self, other):
        '''Implements logical 'and' obtainable by means of the & operator'''
        if other in (None, '', True, 'True', 'true'):
            return self
        if other in (False, 'False', 'false'):
            return expr('False')
        expr2 = expr(other)
        neg = self._negation  # pylint: disable=no-member, protected-access
        neg2 = expr2._negation  # pylint: disable=no-member, protected-access
        return expr("(%s) & (%s)" % (self, expr2), "(%s) | (%s)" % (neg, neg2))

    def __or__(self, other):
        '''Implements logical 'or' obtainable by means of the | operator'''
        if other in (None, '', False, 'False', 'false'):
            return self
        if other in (True, 'True', 'true'):
            return expr('True')
        expr2 = expr(other)
        neg = self._negation  # pylint: disable=no-member, protected-access
        neg2 = expr2._negation  # pylint: disable=no-member, protected-access
        return expr("(%s) | (%s)" % (self, expr2), "(%s) & (%s)" % (neg, neg2))

    def __invert__(self):
        '''Implements logical negation obtainable by means of the ~ operator'''
        return expr(self._negation, self)  # pylint: disable=no-member


class _single_operator_expr(expr):  # pylint: disable=invalid-name
    '''abstract-like class implementing a single operator expression, e.g.:
    expr("pga", ">", 0.5)`
    '''
    operator = None

    def __new__(cls, col, value):  # pylint: disable=arguments-differ
        '''forwards the super-constructor with the class-operator'''
        return expr.__new__(cls, col, cls.operator, value)


class eq(_single_operator_expr):  # pylint: disable=invalid-name
    '''Equality expression: eq('pga', 0.5) translates to "pga == 0.5",
    eq('pga', float('nan')) translates to "pga != pga"
    '''
    operator = '=='


class ne(_single_operator_expr):  # pylint: disable=invalid-name
    '''Inequality expression: ne('pga', 0.5) translates to "pga != 0.5",
    ne('pga', float('nan')) translates to"pga == pga" '''
    operator = '!='


class lt(_single_operator_expr):  # pylint: disable=invalid-name
    '''Lower-than expression: lt('pga', 0.5) translates to "pga < 0.5" '''
    operator = '<'


class le(_single_operator_expr):  # pylint: disable=invalid-name
    '''Lower-equal-to expression: le('pga', 0.5) translates to "pga <= 0.5" '''
    operator = '<='


class gt(_single_operator_expr):  # pylint: disable=invalid-name
    '''Greater-than expression: gt('pga', 0.5) translates to "pga > 0.5" '''
    operator = '>'


class ge(_single_operator_expr):  # pylint: disable=invalid-name
    '''Greater-equal-to expression: ge('pga', 0.5) translates to "pga >= 0.5"
    '''
    operator = '>='


class isin(expr):  # pylint: disable=invalid-name
    '''is-in expression ("in" in SQL): isin('pga', 1, 4.4, 5) translates to:
    "(pga == 1) | (pga == 4.4) | (pga == 5)"
    '''
    def __new__(cls, col, *values):  # pylint: disable=arguments-differ
        if not values:
            raise TypeError('No values provided for %s' % cls.__name__)
        exp = None
        for val in values:
            if exp is None:
                exp = eq(col, val)
                continue
            exp |= eq(col, val)
        return expr.__new__(cls, exp)


class isaval(expr):  # pylint: disable=invalid-name
    '''available (not missing) expression: isaval('pga') translates to:
    "(pga == pga)" (pga is not nan), ~isaval('event_time') translates to
    "event_time == ''" (event_time empty), and so on.

    Note: As boolean columns cannot have missing values (either True or False)
    for boolean columns this class returns 'True'. Use `eq(col, False)` or
    `eq(col, True)` in case
    '''
    def __new__(cls, col):  # pylint: disable=arguments-differ
        colobj = GMDatabaseTable.columns[col]  # pylint: disable=no-member
        if isinstance(colobj, BoolCol):
            return expr.__new__(cls, "True")
        return expr.__new__(cls, col, '!=', colobj.dflt)


class between(expr):  # pylint: disable=invalid-name
    '''between expression ("between" in SQL): between('pga', 1, 4.4) translates
    to: "(pga >= 1) & (pga <= 4.4)"
    '''
    def __new__(cls, col, min_, max_):  # pylint: disable=arguments-differ
        exp1 = expr(col, '>=', min_) if min_ is not None else None
        exp2 = expr(col, '<=', max_) if max_ is not None else None
        if exp1 and exp2:
            exp = exp1 & exp2
        elif exp1:
            exp = exp1
        elif exp2:
            exp = exp2
        else:
            exp = 'True'
        return expr.__new__(cls, exp)
