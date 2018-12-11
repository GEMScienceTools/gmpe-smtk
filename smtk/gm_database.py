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
from collections import OrderedDict
import re
import csv
import hashlib
import shlex
import tokenize
from tokenize import generate_tokens, TokenError, untokenize
from io import StringIO
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
from scipy.constants import g
import tables
# from tables.file import File
from tables.table import Table
from tables.group import Group
from tables.exceptions import NoSuchNodeError
from tables.description import StringCol as _StringCol, \
    Float32Col as _Float32Col, Float16Col as _Float16Col, BoolCol, \
    Float64Col as _Float64Col, UInt32Col, EnumCol as _EnumCol, Int8Col
from openquake.hazardlib.contexts import SitesContext, DistancesContext, \
    RuptureContext
from openquake.hazardlib import imt
from smtk import sm_utils
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14


# implements subclasses to account for dflt = nan, plus min and max and
# allowing other custom attributes in the future. Note that the way Col
# classes are implemented in pytables is tricky and full of abstraction
# layers, so we have to manually subclass each specific type of interest,
# with some "hacks":
class Float64Col(_Float64Col):
    '''subclasses pytables Float64Col, with nan as default and optional min/max
    attributes'''
    def __init__(self, shape=(), min=None, max=None):  # @ReservedAssignment
        super(Float64Col, self).__init__(shape=shape, dflt=np.nan)
        self.min_value, self.max_value = min, max


class Float32Col(_Float32Col):
    '''subclasses pytables Float32Col, with nan as default and optional min/max
    attributes'''
    def __init__(self, shape=(), min=None, max=None):  # @ReservedAssignment
        super(Float32Col, self).__init__(shape=shape, dflt=np.nan)
        self.min_value, self.max_value = min, max


class Float16Col(_Float16Col):
    '''subclasses pytables Float16Col, with nan as default and optional min/max
    attributes'''
    def __init__(self, shape=(), min=None, max=None):  # @ReservedAssignment
        super(Float16Col, self).__init__(shape=shape, dflt=np.nan)
        self.min_value, self.max_value = min, max


class DateTimeCol(_StringCol):
    '''subclasses pytables StringCol, to provide a storage class for date
    times in iso format. Use :function:`normalize_dtime` before writing
    an element under this column (this is done by default for 'event_time'
    flat files column). Also implements optional min max attributes
    (to be given as byte strings in ISO format, in case)'''
    def __init__(self, shape=(), min=None, max=None):  # @ReservedAssignment
        super(DateTimeCol, self).__init__(itemsize=19, shape=shape, dflt=b'')
        self.min_value, self.max_value = min, max
        # needed when parsing a numexpr to distinguish from
        # StringCol:
        self.is_datetime_str = True

    def prefix(self):  # make pytables happy. See description.py line 2013
        return 'String'


class StringCol(_StringCol):
    '''subclasses pytables StringCol to allow optional min/max attributes'''
    def __init__(self, itemsize, shape=(),
                 min=None, max=None):  # @ReservedAssignment
        super(StringCol, self).__init__(itemsize, shape, dflt=b'')
        self.min_value, self.max_value = min, max


class EnumCol(_EnumCol):
    '''subclasses pytables EnumCol: accepts a list of strings, inserts the
    empty string (if not present) and sets it as default'''
    def __init__(self, values):
        dflt = ''
        if dflt not in values:
            values = [''] + list(values)
        # decide the type automatically according to the size of elements:
        type_ = 'uint64'
        if len(values) <= 255:
            type_ = 'uint8'
        elif len(values) <= 65535:
            type_ = 'uint16'
        elif len(values) <= 4294967295:
            type_ = 'uint32'
        super(EnumCol, self).__init__(values, dflt, type_)

    def prefix(self):  # make pytables happy. See description.py line 2013
        return 'Enum'


# defines a mechanism type to be associated to a rake
MECHANISM_TYPE = {"Normal": -90.0,
                  "Strike-Slip": 0.0,
                  "Reverse": 90.0,
                  "Oblique": 0.0,
                  "Unknown": 0.0,
                  "N": -90.0,  # Flatfile conventions
                  "S": 0.0,
                  "R": 90.0,
                  "U": 0.0,
                  "NF": -90.,  # ESM flatfile conventions
                  "SS": 0.,
                  "TF": 90.,
                  "NS": -45.,  # Normal with strike-slip component
                  "TS": 45.,  # Reverse with strike-slip component
                  "O": 0.0
                  }

# Instead of implementing a static GMDatabase as `pytable.IsDescription` class.
# which does not allow to dynamically set the length of array columns, write
# a dict here of SCALAR values only. Array columns (i.e., 'sa') will be added
# later. This also permits to have the scalar columns in one place, as scalar
# columns only are selectable in pytables by default
GMDatabaseTable = dict(
    # id=UInt32Col()  # no default. Starts from 1 incrementally
    # max id: 4,294,967,295
    record_id=UInt32Col(),
    event_id=StringCol(20),
    event_name=StringCol(itemsize=40),
    event_country=StringCol(itemsize=30),
    event_time=DateTimeCol(),  # In ISO Format YYYY-MM-DDTHH:mm:ss
    # Note: if we want to support YYYY-MM-DD only be aware that:
    # YYYY-MM-DD == YYYY-MM-DDT00:00:00
    # Note2: no support for microseconds for the moment
    event_latitude=Float64Col(min=-90, max=90),
    event_longitude=Float64Col(min=-180, max=180),
    hypocenter_depth=Float32Col(),
    magnitude=Float16Col(),
    magnitude_type=StringCol(itemsize=5),
    magnitude_uncertainty=Float32Col(),
    tectonic_environment=StringCol(itemsize=30),
    strike_1=Float32Col(),
    strike_2=Float32Col(),
    dip_1=Float32Col(),
    dip_2=Float32Col(),
    rake_1=Float32Col(),
    rake_2=Float32Col(),
    style_of_faulting=EnumCol(list(MECHANISM_TYPE.keys())),
    depth_top_of_rupture=Float32Col(),
    rupture_length=Float32Col(),
    rupture_width=Float32Col(),
    station_id=StringCol(itemsize=20),
    station_name=StringCol(itemsize=40),
    station_latitude=Float64Col(min=-90, max=90),
    station_longitude=Float64Col(min=-180, max=180),
    station_elevation=Float32Col(),
    vs30=Float32Col(),
    vs30_measured=BoolCol(dflt=True),
    vs30_sigma=Float32Col(),
    depth_to_basement=Float32Col(),
    z1=Float32Col(),
    z2pt5=Float32Col(),
    repi=Float32Col(),  # epicentral_distance
    rhypo=Float32Col(),  # Float32Col
    rjb=Float32Col(),  # joyner_boore_distance
    rrup=Float32Col(),  # rupture_distance
    rx=Float32Col(),
    ry0=Float32Col(),
    azimuth=Float32Col(),
    digital_recording=BoolCol(dflt=True),
    #     acceleration_unit=_col(EnumCol, enum=['cm/s/s', 'm/s/s', 'g'],
    #                              base='uint8')
    type_of_filter=StringCol(itemsize=25),
    npass=Int8Col(),
    nroll=Float32Col(),
    hp_h1=Float32Col(),
    hp_h2=Float32Col(),
    lp_h1=Float32Col(),
    lp_h2=Float32Col(),
    factor=Float32Col(),
    lowest_usable_frequency_h1=Float32Col(),
    lowest_usable_frequency_h2=Float32Col(),
    lowest_usable_frequency_avg=Float32Col(),
    highest_usable_frequency_h1=Float32Col(),
    highest_usable_frequency_h2=Float32Col(),
    highest_usable_frequency_avg=Float32Col(),
    pga=Float64Col(),
    pgv=Float64Col(),
    pgd=Float64Col(),
    duration_5_75=Float64Col(),
    duration_5_95=Float64Col(),
    arias_intensity=Float64Col(),
    cav=Float64Col()
)


class GMDatabaseParser(object):
    '''
    Implements a base class for parsing flatfiles in csv format into
    GmDatabase files in HDF5 format. The latter are Table-like heterogeneous
    datasets (each representing a flatfile) organized in subfolders-like
    structures called groups.
    See the :class:`GMDatabaseTable` for a description of the Table columns
    and types.

    The parsing is done in the `parse` method. The typical workflow
    is to implement a new subclass for each new flatfile released.
    Subclasses should override the `mapping` dict where flatfile
    specific column names are mapped to :class:`GMDatabaseTable` column names
    and optionally the `parse_row` method where additional operation
    is performed in-place on each flatfile row. For more details, see
    the :method:`parse_row` method docstring
    '''
    # the csv delimiter:
    csv_delimiter = ';'

    _accel_units = ["g", "m/s/s", "m/s**2", "m/s^2",
                    "cm/s/s", "cm/s**2", "cm/s^2"]

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
    # `mappings` dict below, where a csv flatfile column is mapped to its
    # corresponding Gm database column name. The mapping is the first
    # operation performed on any row. After that:
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
    # 5 a row is written as record of the output HDF5 file. The columns
    # 'event_id', 'station_id' and 'record_id' are automatically filled to
    # uniquely identify their respective entitites
    mappings = {}

    @classmethod
    def parse(cls, flatfile_path, output_path, mode='a',
              delimiter=None):
        '''Parses a flat file and writes its content in the GM database file
        `output_path`, which is a HDF5 organized hierarchically in groups
        (sort of sub-directories) each of which identifies a parsed
        input flatfile. Each group's `table` attribute is where
        the actual GM database data is stored and can be accessed later
        with the module's :function:`get_table`.
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
        :param delimiter: the delimiter used to parse the csv. If None
            (the default when missing) it is the class-attribute
            `csv_delimiter` (';' by default when not subclassed)
        :return: a dictionary holding information with keys:
            'total': the total number of csv rows
            'written': the number of parsed rows written on the db table
            'error': a list of integers denoting the position
                (0 = first row) of the parsed rows not written on the db table
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
        with GMdb(output_path, dbname, mode) as gmdb:

            i, error, missing, outofbound = \
                -1, [], defaultdict(int), defaultdict(int)

            for i, (rowdict, sa_periods) in \
                    enumerate(cls._rows(flatfile_path, delimiter)):

                # write sa_periods only the first time
                if rowdict:
                    missingcols, outofboundcols = \
                        gmdb.write_record(rowdict, sa_periods)
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

    @classmethod
    def _rows(cls, flatfile_path, delimiter=None):  # pylint: disable=too-many-locals
        '''Yields each row from the CSV file `flatfile_path` as
        dictionary, after performing SA conversion and running custom code
        implemented in `cls.parse_row` (if overridden by
        subclasses). Yields empty dict in case of exceptions'''
        # ref_log_periods = np.log10(cls._ref_periods)
        mappings = getattr(cls, 'mappings', {})
        with cls._get_csv_reader(flatfile_path, delimiter=delimiter) as reader:

            newfieldnames = [mappings[f] if f in mappings else f for f in
                             reader.fieldnames]
            # get spectra fieldnames and priods:
            try:
                sa_periods_odict =\
                    cls._get_sa_columns(newfieldnames)
                sa_periods = list(sa_periods_odict.values())
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
                    rowdict['sa'] = np.array([rowdict[p] for p in
                                              sa_periods_odict.keys()],
                                             dtype=float)
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
                yield rowdict, sa_periods

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
            pga, sa0 = float(rowdict['pga']) / (100*g),\
                float(rowdict['sa'][0])
            retol = abs(max(pga, sa0) / min(pga, sa0))
            if not np.isnan(retol) and round(retol) >= 10:
                return False
        except Exception as _:  # disable=broad-except
            # it might seem weird to return true on exceptions, but this method
            # should only check wheather there is certainly a unit
            # mismatch between sa and pga, and int that case only return True
            pass
        return True

    @classmethod
    @contextmanager
    def _get_csv_reader(cls, filepath, dict_reader=True, delimiter=None):
        '''opends a csv file and yields the relative reader. To be used
        in a with statement to properly close the csv file'''
        # according to the docs, py3 needs the newline argument
        if delimiter is None:
            delimiter = cls.csv_delimiter
        kwargs = {'newline': ''} if sys.version_info[0] >= 3 else {}
        with open(filepath, **kwargs) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter) \
                if dict_reader else \
                csv.reader(csvfile, delimiter=delimiter)
            yield reader

    @classmethod
    def _get_sa_columns(cls, csv_fieldnames):
        """Returns the field names, the spectra fieldnames and the periods
        (numoy array) of e.g., a parsed csv reader's fieldnames
        """
        periods_names = []
        # spectra_fieldnames = []
        # periods = []
        reg = cls._sa_periods_re
        for fname in csv_fieldnames:
            match = reg.match(fname)
            if match:
                periods_names.append((fname, float(match.group(1))))
#                 periods.append(float(match.group(1)))
#                 spectra_fieldnames.append(fname)

        periods_names.sort(key=lambda item: item[1])
        return OrderedDict(periods_names)

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


#########################################
# Database selection / maniuplation
#########################################


def get_dbnames(filepath):
    '''Returns he database names of the given Gm database (HDF5 file)
    The file should have been created with the `GMDatabaseParser.parse`
    method.

    :param filepath: the path to the HDF5 file
    :return: a list of strings identyfying the database names in the file
    '''
    with tables.open_file(filepath, 'r') as h5file:
        root = h5file.get_node('/')
        return [group._v_name for group in  # pylint: disable=protected-access
                h5file.list_nodes(root, classname=Group.__name__)]
        # note: h5file.walk_groups() might raise a ClosedNodeError.
        # This error is badly documented (as much pytables styff),
        # the only mention is (pytables pdf doc): "CloseNodeError: The
        # operation can not be completed because the node is closed. For
        # instance, listing the children of a closed group is not allowed".
        # I suspect it deals with groups deleted / overwritten and the way
        # hdf5 files mark portions of files to be "empty". However,
        # the list_nodes above seems not to raise anymore


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
        condition = ("(pga < 0.14) | (pga > 1.1) & (pgv != nan) &
                      (event_time < '2006-01-01T00:00:00'")

        for record in records_where(tabke, condition):
            # loop through matching records
    ```
    For user trying to build expressions from input variables as python
    objects, simply use the `str(object)` function which supports datetime's,
    strings, boolean, floats and ints (note that datetimes and strings must be
    "double" quoted: '"%s"' % str(object)):
    ```
        # given a datetime object `dtime` and two loats pgamin, pgamax:
        condition = \
            "(pga < %s) | (pga > %s) & (pgv != %s) & (event_time < '%s')" % \
            (str(pgamin), str(pgamax), str(float('nan')), str(dtime))
    ```

    :param table: The pytables Table object. See module function `get_table`
    :param condition: a string expression denoting a selection condition.
        See https://www.pytables.org/usersguide/tutorials.html#reading-and-selecting-data-in-a-table
        If None or the empty string, no filter is applied and all records are
        yielded

    :param limit: integer (defaults: None) implements a SQL 'limit'
        when provided, yields only the first `limit` matching rows
    '''
    iterator = enumerate(table.iterrows() if condition in ('', None)
                         else table.where(_normalize_condition(condition)))
    for count, row in iterator:
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
    return (table.read() if condition in ('', None)
            else table.read_where(_normalize_condition(condition)))[:limit]


def _normalize_condition(condition):
    '''normalizes the given `condition` string (numexpr syntax) to be used
    in record selection in order to handle some caveats:
    1. expressions concatenated with & or | should be put into brakets:
        "(pga <= 0.5) & (pgv > 9.5)". This function raises if the logical
        operators are not preceeded by a ")" or not followed by a "("
    1b. Recognizes date time **strings** (i.e. quoted) in any format recognized
        by GmDatabase parser: 2006-12-31T00:00:00 (with or without T),
        2006-12-31, or simply 2006.
    1c. Does a fast check on correct comparison columns (fields) types
    2. Accepts expressions like 'col_name != nan' or 'col_name == nan' by
        converting it to the numexpr correct syntax:
        "pga != pga"  (pga is nan)
        "pga == pga"  (pga is not nan)
    3. Converts string column type values (e.g., 'event_country') to
       bytes, as expected by numexpr syntax:
        "event_country == b'Germany'"
        Note: This conversion (reported in pytables documentation) is made for
        safety **but** when tested in Python3.6.2 these work, so the claim is
        false or incomplete. Maybe it works as long as `value` has ascii
        characters only?).
    '''
    dbcolumns = GMDatabaseTable
    py3 = sys.version_info[0] >= 3
    oprs = {'==', '!=', '<=', '>=', '<', '>'}
    nan_indices = []
    str_indices = []
    dtime_indices = []
    result = []

    def last_tokenstr():
        return '' if not result else result[-1][1]

    def raise_invalid_logical_op_if(bool_value):
        if bool_value:
            raise ValueError('Logical operators (&|~) allowed only with '
                             'parenthezised expressions')

    ttypes = {'STR': tokenize.STRING, 'OP': tokenize.OP,
              'NAME': tokenize.NAME, 'NUM': tokenize.NUMBER}
    colname = None
    try:
        for token in generate_tokens(StringIO(condition).readline):
            tokentype, tokenstr = token[0], token[1]

            raise_invalid_logical_op_if(tokenstr in ('&', '|')
                                        and last_tokenstr() != ')')
            raise_invalid_logical_op_if(last_tokenstr() in ('~', '|', '&')
                                        and tokenstr not in ('~', '('))

            if colname is not None:
                if colname != tokenstr or tokentype != ttypes['NAME']:
                    _type_check(tokentype, tokenstr, colname,
                                dbcolumns[colname], ttypes['STR'],
                                ttypes['NUM'])
                    if getattr(dbcolumns[colname], "is_datetime_str", False)\
                            and tokentype == ttypes['STR']:
                        dtime_indices.append(len(result))
                    elif py3 and tokentype == ttypes['STR']:
                        str_indices.append(len(result))
                    elif tokenstr == 'nan' and tokentype == ttypes['NAME']:
                        nan_indices.append(len(result))
                colname = None
            else:
                if tokentype == ttypes['OP'] and tokenstr in oprs \
                        and result and result[-1][1] in dbcolumns and \
                        result[-1][0] == ttypes['NAME']:
                    colname = result[-1][1]

            result.append(list(token))

    except TokenError as terr:
        # tokenizer seems to do some weird stuff at the end of the parsed
        # stringas, raising TokenErrors for "unclosed string or brakets".
        # We do not want to raise this kind of stuff, as the idea here is
        # to check only for logical operatorsm, nans, and bytes conversion
        if untokenize(result).strip() != condition.strip():
            raise ValueError(str(terr))

    raise_invalid_logical_op_if(last_tokenstr() in ('&', '|', '~'))
    # replace nans, datetimes and strings at the real end:
    _normalize_tokens(result, dtime_indices, str_indices, nan_indices)
    # return the new normalized string by untokenizing back: aside changed
    # variables, spaces are preserved except trailing ones (a the end):
    return untokenize(result)


def _type_check(tokentype, tokenstr,  # pylint: disable=too-many-arguments
                colname, colobj,  str_code, num_code):
    colobj_name = colobj.__class__.__name__
    if colobj_name.startswith('String') and tokentype != str_code:
        raise ValueError("'%s' value must be strings (quoted)" %
                         colname)
    elif (colobj_name.startswith('UInt')
          or colobj_name.startswith('Int')) and \
            tokentype != num_code:
        raise ValueError("'%s' values must be integers" %
                         colname)
    elif colobj_name.startswith('Float'):
        if tokentype != num_code and tokenstr != 'nan':
            raise ValueError("'%s' values must be floats or nan" %
                             colname)
    elif colobj_name.startswith('Bool') and tokenstr not in \
            ('True', 'False'):
        raise ValueError("Boolean required with '%s'" %
                         colname)


def _normalize_tokens(tokens, dtime_indices, str_indices, nan_indices):
    for i in dtime_indices:
        tokenstr = tokens[i][1]  # it is quoted, e.g. '"string"', so use shlex:
        if tokenstr[0:1] == 'b':
            tokenstr = tokenstr[1:]
        string = shlex.split(tokenstr)[0]
        tokens[i][1] = str(GMDatabaseParser.normalize_dtime(string).
                           encode('utf8'))

    for i in str_indices:
        tokenstr = tokens[i][1]  # it is quoted, e.g. '"string"', so use shlex:
        if tokenstr[0:1] != 'b':
            string = shlex.split(tokenstr)[0]
            tokens[i][1] = str(string.encode('utf8'))

    if nan_indices:
        nan_operators = {'==': '!=', '!=': '=='}
        for i in nan_indices:
            varname = tokens[i-2][1]
            operator = tokens[i-1][1]
            if operator not in nan_operators:
                raise ValueError('only != and == can be compared with nan')
            tokens[i-1][1] = nan_operators[operator]
            tokens[i][1] = varname


########################################
# Residuals calculation
########################################

class GMdb:
    '''Implements a Ground motion database. This class differs from
    :class:`smtk.sm_database.GroundMotionDatabase` in that flat files are
    stored as pytables tables in a single HDF file container. This should in
    principle have more efficient IO operations, exploit numexpr syntax for
    efficient and simpler record selection, and allow the integration of
    customized flat-files (via pytables pre-defined column classes).
    Support for time-series (non-scalar) data is still possible although this
    functionality has not been not tested. From
    :class:`smtk.residuals.gmpe_residuals.Residuals.get_residuals`, both
    databses can be passed as `database` argument.
    (FIXME: in the future the two databases should be merged into a single
    class providing the functionalities of both)
    '''
    def __init__(self, filepath, dbname, mode='r'):
        '''
        Creates a new database. The main functionality of a GMdb is to
        provide the contexts for the residuals calculations:
        ```
            contexts = GMdb(...).get_contexts(...)
        ```
        For all other records manipulation tasks, note that this object
        needs to be accessed inside a with statement like a normal Python
        file-like object, which opens and closes the underlying HDF file:
        ```
            with GMdb(filepath, name, 'r') as dbase:
                # ... do your operation here
                for record in dbase.records:
                    ...
        ```

        :param filepath: the string denoting the path to the hdf file
            previously created with this method. If `mode`
            is 'r', the file must exist
        :param dbname: the name of the database table. It will be the name
            of the group (kind of sub-folder) of the underlying HDF file
        :param mode: string (default: 'r'). The mode ('a', 'r', 'w') whereby
            the underlying hdf file will be opened **when this object
            is used in a with statement**.
            Note that 'w' does not overwrite the whole file, but the table
            data only. More specifically:
            'r': opens file in 'r' mode, raises if the file or the table in
                the file content where not found
            'w': opens file in 'a' mode, creates the table if it does not
                exists, clears all table data if it exists.
            'a': open file in 'a' mode, creates the table if it does not
                exists, does nothing otherwise
        '''
        self.filepath = filepath
        self.dbname = dbname
        self.mode = mode
        self._root = '/%s' % dbname
        self._condition = None
        self.__h5file = None
        self._table = None

    @property
    def is_open(self):
        return self.__h5file is not None

    def filter(self, condition):
        '''Returns a read-only copy of this database filtered according to
        the given condition (numexpr expression on the database scalar
        columns, see :class:`GMDatabaseTable`). Raises ValueError if this
        method is accessed while the underlying HDF file is open (e.g.,
        inside a with statement).

        See module's function `func`:`records_where` and :func:`read_where`

        Example
        ```
            condition = ("(pga < 0.14) | (pga > 1.1) & (pgv != nan) &
                          (event_time < '2006-01-01T00:00:00'")

            filtered_gmdb = GMdb(...).filter(condition)
        ```
        For user trying to build expressions from input variables as python
        objects, simply use the `str(object)` function which supports
        datetime's, strings, boolean, floats and ints (note that datetimes
        and strings must be "double" quoted: '"%s"' % str(object)):
        ```
            # given a datetime object `dtime` and two loats pgamin, pgamax:
            condition = \
                "(pga < %s) | (pga > %s) & (pgv != %s) & \
                (event_time < '%s')" % \
                (str(pgamin), str(pgamax), str(float('nan')), str(dtime))

            filtered_gmdb = GMdb(...).filter(condition)
        ```
        '''
        if self.is_open:
            raise ValueError('Cannot filter, underlying HDF5 file is open. '
                             'Do not call this method inside a with statement')
        gmdb = GMdb(self.filepath, self.dbname, 'r')
        gmdb._condition = condition  # pylint: disable=protected-access
        return gmdb

    def __enter__(self):
        '''Yields a pytable Group object representing a Gm database
        in the given hdf5 file `filepath`. If such a group does not exist
        and mode is either 'a' or 'w', creates the group. In any other case
        where such  a group does not exist, raises a :class:`NoSuchNodeError`

        Example:
        ```
            with GMdb(filepath, name, 'r') as dbase:
                # ... do your operation here
        ```

        :raises: :class:`tables.exceptions.NoSuchNodeError` if mode is 'r'
            and the table was not found in `filepath`, IOError if the
            file does not exist
        '''
        filepath, mode, name = self.filepath, self.mode, self.dbname
        h5file = self.__h5file = \
            tables.open_file(filepath, mode if mode == 'r' else 'a')
        grouppath = self._root
        try:
            group = h5file.get_node(grouppath, classname=Group.__name__)
            if mode == 'w':
                for node in group:  # make node empty
                    h5file.remove_node(node, recursive=True)
        except NoSuchNodeError as _:
            if mode == 'r':
                raise
            # create group node
            group = h5file.create_group(h5file.root, name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.__h5file is not None:
            self.__h5file.close()
        self.__h5file = None
        self._table = None

    def get_array(self, relative_path):
        '''
        Returns a saved array saved on the
        undelrying HDF file, which must be open (i.e., the user must be
        inside a with statement). Raises a :class:`NoSuchNode` if the
        path does not exist

        :param relative_path: string, the path of the group relative to
            the path of the undelrying database storage.
            E.g. 'my/arrays/array_1'
        '''
        path = self._fullpath(relative_path)
        return self._h5file.get_node("/".join(path[:-1]), path[-1]).read()

    def write_array(self, relative_path, values, create_path=True):
        '''
        Writes the given array on the
        undelrying HDF file, which must be open (i.e., the user must be
        inside a with statement)

        :param relative_path: string, the path of the group relative to
            the path of the undelrying database storage.
            E.g. 'my/arrays/array_1'
        :param values: the array to be saved. The value saved to the HDF
            file will be `numpy.asarray(values)`
        :param create_path: boolean (defaault: True) whether to create the
            array path (and all its ancestors) if it does not exists. If False,
            and the path does not exists, a :class:`NoSuchNode` exception is
            raised
        '''
        _splitpath = relative_path.split('/')
        group = self.get_group("/".join(_splitpath[:-1]), create_path)
        self._h5file.create_array(group, _splitpath[-1],
                                  obj=np.asarray(values))

    def get_group(self, relative_path, create=True):
        '''
        Returns the given Group (HDF directory-like structure) from the
        undelrying HDF file, which must be open (i.e., the user must be
        inside a with statement)

        :param relative_path: string, the path of the group relative to
            the path of the undelrying database storage. E.g. 'my/arrays'
        :param create: boolean (defaault: True) whether to create the group
            (and all its ancestors) if it does not exists. If False, and
            the group does not exists, a :class:`NoSuchNode` exception is
            raised
        '''
        try:
            fullpath = self._fullpath(relative_path)
            return self._h5file.get_node(fullpath,
                                         Group.__name__)
        except NoSuchNodeError:
            if not create:
                raise
            node = self._h5file.get_node(self._root,
                                         classname=Group.__name__)
            for path in relative_path.split('/'):
                if path not in node:
                    node = self._h5file.create_group(node, path)
                else:
                    node = self._h5file.get_node(node, path,
                                                 classname=Group.__name__)
            return node

    def write_record(self, csvrow, sa_periods):
        '''writes the content of csvrow into tablerow. Returns two lists:
        The missing column names (a missing column is also a column for which
        the csv value is invalid, i.e. it raised during assignement), and
        the out-of-bounds column names (in case bounds were provided in the
        column class. In this case, the default of that column will be set
        in `tablerow`)'''
        try:
            table = self.table
        except NoSuchNodeError:
            table = self._create_table(len(sa_periods))

        # build a record hashes as ids:
        evid, staid, recid = self._get_ids(csvrow)
        csvrow['event_id'] = evid
        csvrow['station_id'] = staid
        # do not use record id, rather an incremental integer:
        csvrow['record_id'] = table.attrs._current_row_id
        table.attrs._current_row_id += 1

        # write sa periods (if not already written):
        try:
            table.attrs.sa_periods
        except AttributeError:
            table.attrs.sa_periods = np.asarray(sa_periods, dtype=float)

        tablerow = table.row

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

        tablerow.append()  # pylint: disable=no-member
        table.flush()

        return missing_colnames, outofbounds_colnames

    @property
    def table(self):
        '''Returns the underlying hdf file's table'''
        tab = self._table
        if self._table is None:
            tablepath = "%s/%s" % (self._root, "table")
            tab = self._table = self._h5file.get_node(tablepath,
                                                      classname=Table.__name__)
        return tab

    # ----- IO PRIVATE METHODS  ----- #

    def _create_table(self, sa_length):
        desc = dict(GMDatabaseTable, sa=Float64Col(shape=(sa_length,)))
        self._table = self._h5file.create_table(self._root, "table",
                                                description=desc)
        self._table.attrs._current_row_id = 1
        return self._table

    @property
    def _h5file(self):
        h5file = self.__h5file
        if h5file is None:
            raise ValueError('The underlying HDF5 file is not open. '
                             'Are you inside a "with" statement?')
        return h5file

    def _fullpath(self, path):
        return "%s/%s" % (self._root, path)

    def _get_ids(self, csvrow):
        '''Returns the tuple record_id, event_id and station_id from
        the given HDF5 row `csvrow`'''
        dbname = self.dbname
        toint = self._toint
        ids = (dbname,
               toint(csvrow['pga'], 0),  # (first two decimals of pga in g)
               toint(csvrow['event_longitude'], 5),
               toint(csvrow['event_latitude'], 5),
               toint(csvrow['hypocenter_depth'], 3),
               csvrow['event_time'],
               toint(csvrow['station_longitude'], 5),
               toint(csvrow['station_latitude'], 5))
        # return event_id, station_id, record_id:
        evid, staid, recid = \
            self._hash(*ids[2:6]), self._hash(*ids[6:]), self._hash(*ids)
        return evid, staid, recid

    @classmethod
    def _toint(cls, value, decimals):
        '''returns an integer by multiplying value * 10^decimals
        and rounding the result to int. Returns nan if value is nan'''
        try:
            value = float(value)
        except ValueError:
            value = float('nan')
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

    # ---- RESIDUALS ANALYSIS ---- #

#     def get_contexts(self, nodal_plane_index=1):
#         """
#         Returns a list of dictionaries, each containing the site, distance
#         and rupture contexts for individual records
#         """
#         wfid_list = np.array([rec.event.id for rec in self.records])
#         eqid_list = self._get_event_id_list()
#         context_dicts = []
#         for eqid in eqid_list:
#             idx = np.where(wfid_list == eqid)[0]
#             context_dicts.append({
#                 'EventID': eqid,
#                 'EventIndex': idx.tolist(),
#                 'Sites': self._get_sites_context_event(idx),
#                 'Distances': self._get_distances_context_event(idx),
#                 'Rupture': self._get_event_context(idx, nodal_plane_index)})
#         return context_dicts
#
#     @staticmethod
#     def _get_event_id_list(self):
#         """
#         Returns the list of unique event keys from the database
#         """
#         event_list = []
#         for record in self.records:
#             if not record.event.id in event_list:
#                 event_list.append(record.event.id)
#         return np.array(event_list)

    @property
    def records(self):
        '''Yields an iterator of the records according to the specified filter
        `condition`. The underlying HDF file (including each yielded record)
        must not be modified while accessing this property, and thus must
        be opened in read mode.
        ```
            with GMdb(filepath, name, 'r').filter(condition) as dbase:
                # ... do your operation here
                for record in dbase.records:
                    ...
        ```
        '''
        return records_where(self.table, self._condition)

    def get_contexts(self, imts, nodal_plane_index=1, component="Geometric"):
        """
        Returns an iterable of dictionaries, each containing the site, distance
        and rupture contexts for individual records
        """
        # FIXME: nodal_plane_index and component not used. Remove?
        context_dicts = {}
        with self:
            sa_periods = self.table.sa_periods
            for rec in self.records:
                evt_id = rec['event_id']
                dic = context_dicts.get(evt_id, None)
                if dic is None:
                    # we might use defaultdict, but like this is more readable
                    dic = {'EventID': evt_id,
                           'EventIndex': [],
                           'Sites': SitesContext(),
                           'Distances': DistancesContext(),
                           "Observations": OrderedDict([(imtx, []) for imtx
                                                        in imts]),
                           "Num. Sites": 0}
                    # set Rupture only once:
                    dic['Rupture'] = RuptureContext()
                    self._set_event_context(rec, dic['Rupture'],
                                            nodal_plane_index)
                    context_dicts[evt_id] = dic
                dic['EventIndex'].append(rec['record_id'].item())
                self._set_sites_context_event(rec, dic['Sites'])
                self._set_distances_context_event(rec, dic['Distances'])
                self._add_observations(rec, imts, dic['Observations'],
                                       sa_periods)
                dic["Num. Sites"] += 1

        # converts to numeric arrays (once at the end is faster, see
        # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
        # get default attributes not to be changed:
        site_context_def_attrs = set(dir(SitesContext()))
        distances_attrs = set(dir(DistancesContext()))
        for dic in context_dicts.values():
            self._lists2numpy(dic['Sites'],
                              set(dir(dic['Sites'])) - site_context_def_attrs)
            self._lists2numpy(dic['Distances'],
                              set(dir(dic['Distances'])) - distances_attrs)
            observations = dic['Observations']
            for key, val in observations.items():
                observations[key] = np.asarray(val, dtype=float)

        return list(context_dicts.values())

    @staticmethod
    def _append(obj, att, value):
        ret = getattr(obj, att, None)
        if ret is None:
            ret = []
            setattr(obj, att, ret)
        ret.append(value)

    @staticmethod
    def _lists2numpy(obj, att_names):
        for att_name in att_names:
            att_val = getattr(obj, att_name, None)
            if isinstance(att_val, list):
                setattr(obj, att_name, np.array(att_val))

    @staticmethod
    def _set_sites_context_event(record, sctx):
        """
        Adds the record's data to the given site context

        :param record: a pytable record usually representing a flatfile row
        :param sctx: a :class:`SitesContext` object
        """
        # From:
        # smtk.sm_database.GroundMotionDatabase._get_sites_context_event
        # line: 1085

        # Please remember that the method above is called ONCE PER DB and
        # returns an openquake's SitesContext
        # whereas this method is called ONCE PER RECORD and appends records
        # data to an already created SitesContext

        # Attributes attached to sctx in the OLD IMPLEMENTATION:
        # setattr(sctx, 'vs30', np.array(vs30))
        # if len(longs) > 0:
        #     setattr(sctx, 'lons', np.array(longs))
        # if len(lats) > 0:
        #     setattr(sctx, 'lats', np.array(lats))
        # if len(depths) > 0:
        #     setattr(sctx, 'depths', np.array(depths))
        # if len(vs30_measured) > 0:
        #     setattr(sctx, 'vs30measured', np.array(vs30_measured))
        # if len(z1pt0) > 0:
        #     setattr(sctx, 'z1pt0', np.array(z1pt0))
        # if len(z2pt5) > 0:
        #     setattr(sctx, 'z2pt5', np.array(z2pt5))
        # if len(backarc) > 0:
        #     setattr(sctx, 'backarc', np.array(backarc))

        # FIXME:
        # backarc?
        # deal with non attached attributes

        append, isnan = GMdb._append, np.isnan

        append(sctx, 'lons', record['station_longitude'])
        append(sctx, 'lats', record['station_latitude'])
        append(sctx, 'depths',  0.0 if isnan(record['station_elevation'])
               else record['station_elevation'])
        vs30 = record['vs30']
        append(sctx, 'vs30', vs30)
        append(sctx, 'vs30_measured', record['vs30_measured'])
        append(sctx, 'z1pt0',  vs30_to_z1pt0_cy14(vs30)
               if isnan(record['z1pt0']) else record['z1pt0'])
        append(sctx, 'z2pt5',  vs30_to_z2pt5_cb14(vs30)
               if isnan(record['z2pt5']) else record['z2pt5'])

    @staticmethod
    def _set_distances_context_event(record, dctx):
        """
        Adds the record's data to the given distance context

        :param record: a pytable record usually representing a flatfile row
        :param dctx: a :class:`DistancesContext` object
        """
        # From:
        # smtk.sm_database.GroundMotionDatabase._get_distances_context_event
        # line: 1141

        # Please remember that the method above is called ONCE PER DB and
        # returns an openquake's SitesContext
        # whereas this method is called ONCE PER RECORD and appends records
        # data to an already created SitesContext

        # Attributes attached to sctx in the OLD IMPLEMENTATION:
        # setattr(dctx, 'repi', np.array(repi))
        # setattr(dctx, 'rhypo', np.array(rhypo))
        # if len(rjb) > 0:
        #     setattr(dctx, 'rjb', np.array(rjb))
        # if len(rrup) > 0:
        #     setattr(dctx, 'rrup', np.array(rrup))
        # if len(r_x) > 0:
        #     setattr(dctx, 'rx', np.array(r_x))
        # if len(ry0) > 0:
        #     setattr(dctx, 'ry0', np.array(ry0))
        # if len(rcdpp) > 0:
        #     setattr(dctx, 'rcdpp', np.array(rcdpp))
        # if len(azimuth) > 0:
        #     setattr(dctx, 'azimuth', np.array(azimuth))
        # if len(hanging_wall) > 0:
        #     setattr(dctx, 'hanging_wall', np.array(hanging_wall))
        # if len(rvolc) > 0:
        #     setattr(dctx, 'rvolc', np.array(rvolc))

        # FIXME:
        # 1) These three attributes are missing in current implementation!
        # - append(dctx, 'rcdpp', rup['rcdpp'])
        # - append(dctx, 'hanging_wall', rup['hanging_wall'])
        # - append(dctx, 'rvolc', rup['rvolc'])
        # 2) Old TODO maybe to be fixed NOW?

        append, isnan = GMdb._append, np.isnan

        # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
        # is a hack! Need feedback on how to fix
        if isnan(record['rjb']):
            record['rjb'] = record['repi']
        if isnan(record['rrup']):
            record['rrup'] = record['rhypo']
        append(dctx, 'repi', record['repi'])
        append(dctx, 'rhypo', record['rhypo'])
        append(dctx, 'rjb', record['rjb'])
        append(dctx, 'rrup', record['rrup'])
        append(dctx, 'rx', record['rx'])
        append(dctx, 'ry0', record['ry0'])
        append(dctx, 'azimuth', record['azimuth'])

    def _set_event_context(self, record, rctx, nodal_plane_index=1):
        """
        Adds the record's data to the given distance context

        :param record: a pytable record usually representing a flatfile row
        :param rctx: a :class:`RuptureContext` object
        """
        # From:
        # smtk.sm_database.GroundMotionDatabase._get_event_context
        # line: 1208

        # Please remember that the method above is called ONCE PER DB and
        # returns an openquake's SitesContext
        # whereas this method is called ONCE PER RECORD and appends records
        # data to an already created SitesContext

        # Attributes attached to sctx in the OLD IMPLEMENTATION:
        # if nodal_plane_index == 2:
        #     setattr(rctx, 'strike',
        #         rup.event.mechanism.nodal_planes.nodal_plane_2['strike'])
        #     setattr(rctx, 'dip',
        #         rup.event.mechanism.nodal_planes.nodal_plane_2['dip'])
        #     setattr(rctx, 'rake',
        #         rup.event.mechanism.nodal_planes.nodal_plane_2['rake'])
        # else:
        #     setattr(rctx, 'strike', 0.0)
        #     setattr(rctx, 'dip', 90.0)
        #     rctx.rake = rup.event.mechanism.get_rake_from_mechanism_type()
        # if rup.event.rupture.surface:
        #     setattr(rctx, 'ztor', rup.event.rupture.surface.get_top_edge_depth())
        #     setattr(rctx, 'width', rup.event.rupture.surface.width)
        #     setattr(rctx, 'hypo_loc', rup.event.rupture.surface.get_hypo_location(1000))
        # else:
        #     setattr(rctx, 'ztor', rup.event.depth)
        #     # Use the PeerMSR to define the area and assuming an aspect ratio
        #     # of 1 get the width
        #     setattr(rctx, 'width',
        #             np.sqrt(DEFAULT_MSR.get_median_area(rctx.mag, 0)))
        #     # Default hypocentre location to the middle of the rupture
        #     setattr(rctx, 'hypo_loc', (0.5, 0.5))
        # setattr(rctx, 'hypo_depth', rup.event.depth)
        # setattr(rctx, 'hypo_lat', rup.event.latitude)
        # setattr(rctx, 'hypo_lon', rup.event.longitude)

        # FIXME: is style_of_faulting only needed for n/a rake?
        # Then I would remove style_of_faulting and create a rake for each
        # specific flatfile case, when parsing
        # Missing attributes: ztor, width

        strike, dip, rake = \
            record['strike_1'], record['dip_1'], record['rake_1']

        if np.isnan([strike, dip, rake]).any():
            strike, dip, rake = \
                record['strike_2'], record['dip_2'], record['rake_2']

        if np.isnan([strike, dip, rake]).any():
            strike = 0.0
            dip = 90.0
            try:
                rake = MECHANISM_TYPE[record['style_of_faulting']]
            except KeyError:
                rake = 0.0

        setattr(rctx, 'mag', record['magnitude'])
        setattr(rctx, 'strike', strike)
        setattr(rctx, 'dip', dip)
        setattr(rctx, 'rake', rake)
        setattr(rctx, 'hypo_depth', record['event.depth'])
        setattr(rctx, 'hypo_lat', record['event_latitude'])
        setattr(rctx, 'hypo_lon', record['event_longitude'])

    def _add_observations(self, record, observations, sa_periods,
                          component="Geometric"):
        '''Fetches the given observations (IMTs) from `record` into
        the `observations` dict'''
        # FIXME: unused parameter component (obviously, but how to dealt
        # with it?)
        hundred_g = 100.0 * g  # for convertion from/to cm s^2
        for imtx in observations.keys():
            value = np.nan
            if imtx == "PGA":
                value = record["pga"] / hundred_g
            elif "SA(" in imtx:
                target_period = imt.from_string(imtx).period
                spectrum = record['sa']
                observations[imtx].append(get_interpolated_period(
                    target_period, sa_periods, spectrum) / hundred_g)
            else:
                value = record[imtx.lower()]

            observations[imtx].append(value)


def get_interpolated_period(target_period, periods, values):
    """
    Returns the spectra interpolated in loglog space
    :param float target_period:
        Period required for interpolation
    :param np.ndarray periods:
        Spectral Periods
    :param np.ndarray values:
        Ground motion values
    """
    # FIXME: copied from gmpe_residuals in order to avoid circular
    # imports
    if (target_period < np.min(periods)) or (target_period > np.max(periods)):
        return None, "Period not within calculated range %s"
    lval = np.where(periods <= target_period)[0][-1]
    uval = np.where(periods >= target_period)[0][0]
    if (uval - lval) == 0:
        return values[lval]
    deltay = np.log10(values[uval]) - np.log10(values[lval])
    deltax = np.log10(periods[uval]) - np.log10(periods[lval])
    return 10.0 ** (
        np.log10(values[lval]) +
        (np.log10(target_period) - np.log10(periods[lval])) * deltay / deltax
        )

#     def get_observations(self, context, component="Geometric"):
#         """
#         Get the obsered ground motions from the database
#         """
#         select_records = self.database.select_from_event_id(context["EventID"])
#         observations = OrderedDict([(imtx, []) for imtx in self.imts])
#         selection_string = "IMS/H/Spectra/Response/Acceleration/"
#         for record in select_records:
#             fle = h5py.File(record.datafile, "r")
#             for imtx in self.imts:
#                 if imtx in SCALAR_IMTS:
#                     if imtx == "PGA":
#                         observations[imtx].append(
#                             get_scalar(fle, imtx, component) / 981.0)
#                     else:
#                         observations[imtx].append(
#                             get_scalar(fle, imtx, component))
#
#                 elif "SA(" in imtx:
#                     target_period = imt.from_string(imtx).period
#                     spectrum = fle[selection_string + component +
#                                    "/damping_05"].value
#                     periods = fle["IMS/H/Spectra/Response/Periods"].value
#                     observations[imtx].append(get_interpolated_period(
#                         target_period, periods, spectrum) / 981.0)
#                 else:
#                     raise "IMT %s is unsupported!" % imtx
#             fle.close()
#         for imtx in self.imts:
#             observations[imtx] = np.array(observations[imtx])
#         context["Observations"] = observations
#         context["Num. Sites"] = len(select_records)
#         return context
