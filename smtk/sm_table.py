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
from itertools import chain
"""
Basic classes for the GroundMotionTable (HDF5 database) and parsers
"""

import os
import sys
import json
from collections import OrderedDict, defaultdict
import csv
import hashlib
import shlex
import tokenize
from tokenize import generate_tokens, TokenError, untokenize
from io import StringIO
from contextlib import contextmanager
import numpy as np
# from scipy.constants import g
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
from smtk.sm_utils import MECHANISM_TYPE, get_interpolated_period, SCALAR_XY,\
    DEFAULT_MSR, DIP_TYPE
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14
from smtk.rcrs import ResidualsCompliantRecordSet


# Implements here pytables column subclasses.
# Pytable column classes help defining a table structure in Python code,
# similarly to ORM for SQL databases. However, they generally lack simplicity,
# features and documentation. Subclasses defined below define a default
# dflt = nan for Float columns, and we allow min and max to be passed in the
# constructor (and any other custom attribute in the future).
# Final notes: please DO NOT USE EnumCol: to
# put it shortly, it's complex and useless. Rather, use a StringCol
# (see style_of_faulting). Also, TRY NOT to use Integer Columns, as they cannot
# have a clear missing value (0 is pytables default) as it happens
# for StringColumns (b"") and FloatCols (NaN):
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


class DateTimeCol(_Float64Col):
    '''subclasses pytables StringCol, to provide a storage class for date
    times in iso format. Use :method:`GmDatabaseParser.timestamp` before
    writing an element under this column (this is done by default for
    'event_time' flat files column). Also implements optional min max
    attributes (to be given as byte strings in ISO format, in case)'''
    def __init__(self, shape=(), min=None, max=None):  # @ReservedAssignment
        super(DateTimeCol, self).__init__(shape=shape, dflt=np.nan)
        if min is not None:
            min = GMTableParser.timestamp(min)
            if np.isnan(min):
                raise ValueError('"%s" is not a date-time' % str(min))
        if max is not None:
            max = GMTableParser.timestamp(max)
            if np.isnan(max):
                raise ValueError('"%s" is not a date-time' % str(max))
        self.min_value, self.max_value = min, max
        # needed when parsing a numexpr to distinguish from
        # _Float64Col:
        self.is_datetime_str = True

    def prefix(self):  # make pytables happy. See description.py line 2013
        return 'Float64'


class StringCol(_StringCol):
    '''subclasses pytables StringCol to allow optional min/max attributes'''
    def __init__(self, itemsize, shape=(),
                 min=None, max=None):  # @ReservedAssignment
        super(StringCol, self).__init__(itemsize, shape, dflt=b'')
        self.min_value, self.max_value = min, max


# Instead of implementing a static GMDatabase as `pytable.IsDescription` class,
# which does not allow to dynamically set the length of array columns, write
# a dict here of SCALAR values only. Array columns (i.e., 'sa') will be added
# later. This also permits to have the scalar columns in one place, as scalar
# columns only are selectable in pytables by default
GMTableDescription = dict(
    # These three columns are created and filled by default
    # (see GroundMotionTable.write_record):
    #     record_id=UInt32Col(),  # max id: 4,294,967,295
    #     event_id=StringCol(20),
    #     station_id=StringCol(itemsize=20),
    event_name=StringCol(itemsize=40),
    event_country=StringCol(itemsize=30),
    event_time=DateTimeCol(),
    event_latitude=Float64Col(min=-90, max=90),
    event_longitude=Float64Col(min=-180, max=180),
    hypocenter_depth=Float64Col(),
    magnitude=Float64Col(),
    magnitude_type=StringCol(itemsize=5),
    magnitude_uncertainty=Float32Col(),
    tectonic_environment=StringCol(itemsize=30),
    strike_1=Float64Col(),
    strike_2=Float64Col(),
    dip_1=Float64Col(),
    dip_2=Float64Col(),
    rake_1=Float64Col(),
    rake_2=Float64Col(),
    style_of_faulting=StringCol(itemsize=max(len(_) for _ in MECHANISM_TYPE)),
    depth_top_of_rupture=Float32Col(),
    rupture_length=Float32Col(),
    rupture_width=Float32Col(),
    station_name=StringCol(itemsize=40),
    station_country=StringCol(itemsize=30),
    station_latitude=Float64Col(min=-90, max=90),
    station_longitude=Float64Col(min=-180, max=180),
    station_elevation=Float32Col(),
    vs30=Float32Col(),
    vs30_measured=BoolCol(dflt=True),
    vs30_sigma=Float32Col(),
    depth_to_basement=Float32Col(),
    z1=Float64Col(),
    z2pt5=Float64Col(),
    repi=Float64Col(),  # epicentral_distance
    rhypo=Float64Col(),  # Float32Col
    rjb=Float64Col(),  # joyner_boore_distance
    rrup=Float64Col(),  # rupture_distance
    rx=Float64Col(),
    ry0=Float64Col(),
    azimuth=Float32Col(),
    digital_recording=BoolCol(dflt=True),
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
    backarc=BoolCol(dflt=False),
    # imts:
    pga=Float64Col(),
    pgv=Float64Col(),
    # sa=Float64Col(),  # this columns will be overridden with an array column
    # whose shape depends on the number of sa periods in the flat file
    pgd=Float64Col(),
    duration_5_75=Float64Col(),
    duration_5_95=Float64Col(),
    arias_intensity=Float64Col(),
    cav=Float64Col(),
)


class GMTableParser(object):  # pylint: disable=useless-object-inheritance
    '''
    Implements a base class for parsing flatfiles in csv format into
    GroundMotionTable files in HDF5 format. The latter are Table-like
    heterogeneous datasets (each representing a flatfile) organized in
    subfolders-like structures called groups.
    See the :class:`GMTableDescription` for a description of the Table columns
    and types.

    The parsing is done in the `parse` method. The typical workflow
    is to implement a new subclass for each new flatfile released.
    Subclasses should override the `mapping` dict where flatfile
    specific column names are mapped to :class:`GMTableDescription` column
    names and optionally the `parse_row` method where additional operation
    is performed in-place on each flatfile row. For more details, see
    the :method:`parse_row` method docstring
    '''
    # the csv delimiter:
    csv_delimiter = ';'

    # Tells if the parsed flat file provides components for any of the given
    # IMTs. This will store for each IMT an additional field:
    # <imt>_components (e.g., 'pga_components') of length 3 denoting the
    # two horizontal components and the vertical (in this order.
    # 'sa_components' will be an array of shape [3 x sa_periods]).
    # If components are specified, the associated scalar values
    # will be used for selection/filtering only, and the user should put
    # therein any meaningful value (usually geometric mean of the two
    # horizontal components)
    has_imt_components = False

    # The csv column names will be then converted according to the
    # `mappings` dict below, where a csv flatfile column is mapped to its
    # corresponding Gm database column name. The mapping is the first
    # operation performed on any row
    mappings = {}

    @classmethod
    def parse(cls, flatfile_path, output_path, dbname=None, delimiter=None):
        '''Parses a flat file and writes its content in the file
        `output_path`, which is a HDF file organized hierarchically in groups
        (sort of sub-directories) each of which identifies a
        :class:`GroundMotionTable`, later accessible
        with the module's :function:`get_dbnames`.

        :param flatfile_path: string denoting the path to the input CSV
            flatfile
        :param output_path: string: path to the output GM database file.
        :param dbname: string: the database name. If None (the default),
            it will be the basename of `flatfile_path` (without extension)
        :param delimiter: the delimiter used to parse the csv. If None
            (the default when missing) it is the class-attribute
            `csv_delimiter` (';' by default when not overridden in subclasses)
        :return: a dictionary holding information with keys:
            'total': the total number of csv rows
            'written': the number of parsed rows written on the db table
            'error': the **indices** (0 = first row) of the rows not written
                due to errors (e.g., unexpected exceptions, suspicious bad
                values). It always holds: `len(errors) + written = total`
            'missing_values': a dict of column names mapped
                to the number of rows which have missing values for that
                column (e.g., empty value, or column not in the csv)
            'bad_values': a dict of column names mapped to the number of rows
                which have bad values for that column (e.g., invalid numeric
                values)
            'outofbound_values': a dict of column names mapped to the number
                of rows which have out-of-bound values for that column, if
                the column was implemented to be bounded within a range

            Bad, missing or out-of-bound values are stored in the GM database
            with the column default, which is usually NaN for floats, the
            minimum possible value for integers, the empty string for strings.
        '''
        i, error, missing, bad, outofbound = \
            0, [], defaultdict(int), defaultdict(int), defaultdict(int)

        rows = cls._rows(flatfile_path, delimiter)
        for rowdict, sa_periods in rows:
            break

        with gettable(output_path, "w", dbname, sa_periods,
                      cls.has_imt_components) as table:
            iter_ = enumerate(chain([(rowdict, sa_periods)], rows))
            for i, (rowdict, sa_periods) in iter_:

                written, missingcols, badcols, outofboundcols = \
                    write_record(table, rowdict, sa_periods)

                if not written:
                    error.append(i)
                else:
                    # write statistics:
                    for col in missingcols:
                        missing[col] += 1
                    for col in badcols:
                        bad[col] += 1
                    for col in outofboundcols:
                        outofbound[col] += 1

            stats = {'total': i+1, 'written': i+1-len(error), 'error': error,
                     'bad_values': dict(bad), 'missing_values': dict(missing),
                     'outofbound_values': dict(outofbound)}
            table.attrs.parser = cls.__name__
            table.attrs.parser_stats = stats

        return stats

    @classmethod
    def _rows(cls, flatfile_path, delimiter=None):  # pylint: disable=too-many-locals
        '''Yields each row from the CSV file `flatfile_path` as
        dictionary, after performing SA conversion and running custom code
        implemented in `cls.parse_row` (if overridden by
        subclasses). Yields empty dict in case of exceptions'''
        # ref_log_periods = np.log10(cls._ref_periods)
        mappings = getattr(cls, 'mappings', {})
        with cls._get_csv_reader(flatfile_path, delimiter=delimiter) as reader:

            # get sa periods:
            sa_columns = list(cls.get_sa_columns(reader.fieldnames).items())
            sa_columns.sort(key=lambda item: item[1])
            sa_periods = [_[1] for _ in sa_columns]
            sa_colnames = [_[0] for _ in sa_columns]

            for rowdict in reader:
                # re-map keys:
                for k in mappings:
                    rowdict[mappings[k]] = rowdict.pop(k)

                # custom post processing, if needed in subclasses:
                cls.parse_row(rowdict, sa_colnames)

                # yield row as dict:
                yield rowdict, sa_periods

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
    def get_sa_columns(cls, csv_fieldnames):
        """This method is intended to be overridden by subclasses (by default
        it raises :class:`NotImplementedError`) to return a `dict` of SA
        column names (string), mapped to a numeric value representing the SA
        period. This class will then sort and save SA periods accordingly.

        You can also implement here operations which should be executed once
        at the beginning of the flatfile parsing, such as e.g.
        creating objects and storing them as class attributes later accessible
        in :method:`parse_row`

        :param csv_fieldnames: an iterable of strings representing the
            header of the persed csv file
        """
        raise NotImplementedError()

    @classmethod
    def parse_row(cls, rowdict, sa_colnames):
        '''This method is intended to be overridden by subclasses (by default
        is no-op) to perform any further operation on the given csv row
        `rowdict` before writing it to the GM databse file. Please note that:

        1. This method should process `rowdict` in place, the returned value is
           ignored. Any exception raised here is hanlded in the caller method.
        2. `rowdict` keys might not be the same as the csv
           field names (first csv row). See `mappings` class attribute
        3. The values of `rowdict` are all strings and they will be casted
           later according to the column type. However, if a cast is needed
           here for some custom operation, in order to convert strings to
           floats or timestamps (floats denoting date-times) you can use the
           static methods `timestamp` and `float`. Both methods accept also
           lists or tuples to convert arrays and silently coerce unparsable
           values to nan (Note that nan represents a missing value for any
           numeric or timestamp column)

        :param rowdict: a row of the csv flatfile, as Python dict

        :param sa_colnames: a list of strings of the column
            names denoting the SA values. The list is sorted ascending
            according to the relative numeric period defined in
            :method:`get_sa_columns`
        '''
        pass

    @staticmethod
    def timestamp(value):
        '''converts value to timestamp (numpy float64). Silently coerces
            erros to NaN(s) when needed

        :param value: string representing a UTC datetime in iso-8601 format,
            or datetime, or any list/tuple of the above two types.
            If string, the formats can be:
            1. '%Y' (e.g. '2006' == '2006-01-01T00:00:00')
            2. '%Y-%m-%d' (e.g., '2006-01-01' == '2006-01-01T00:00:00')
            3. '%Y-%m-%d %H:%M:%S'
            4. '%Y-%m-%dT%H:%M:%S'

        :return: a numpy float64 (array or scalar depending on the input type)
        '''
        dtime = GMTableParser._np_datetime64
        isarray = (hasattr(value, '__len__') and
                   not isinstance(value, (bytes, str)))
        # did not find any better way to make array case and scalar case
        # behave the same: np.array(value, dtype='datetime64') does not
        # produce the same result as below (and numpy doc are nebulous)
        newvalue = np.array([dtime(_) for _ in value]) if isarray \
            else dtime(value)
        unix_epoch = \
            np.datetime64(0, 's')  # pylint: disable=too-many-function-args
        one_second = \
            np.timedelta64(1, 's')  # pylint: disable=too-many-function-args
        seconds_since_epoch = (newvalue - unix_epoch) / one_second
        return seconds_since_epoch

    @staticmethod
    def _np_datetime64(value):
        '''returns np.datetime64(value), or np.datetime64('NaT') in case of
        ValueError'''
        try:
            # Note: np.datetime64('') -> np.datetime64('NaT')
            return np.datetime64(value)
        except ValueError:
            # e.g., np.datetime64('abc'):
            return np.datetime64('NaT')

    @staticmethod
    def float(value):
        '''Converts value to float (numpy float64). Silently coerces
            erros to NaN(s) when needed

        :param value: number, string representing a number, or any list/tuple
            of the above two types.

        :return: a numpy float64 (array or scalar depending on the input type)
        '''
        float64 = GMTableParser._np_float64
        isarray = (hasattr(value, '__len__') and
                   not isinstance(value, (bytes, str)))
        return np.array([float64(_) for _ in value]) if isarray \
            else float64(value)

    @staticmethod
    def _np_float64(value):
        '''Returns np.float64(value) or np.nan in case of ValueError'''
        try:
            return np.float64(value)
        except ValueError:
            return np.nan


@contextmanager
def openfile(filepath, mode):
    '''
    Returns the hdf5file form path

    mode: 'w', 'r', 'a' (a not tested)
    sa_periods must be given if mode='w'
    name=None: basename(filepath) woithout extension

    OLD DOC:

    Yields a pytable Group object representing a Gm database
    in the given hdf5 file `filepath`. If such a group does not exist
    and mode is 'w', creates the group. In any other case
    where such  a group does not exist, raises a :class:`NoSuchNodeError`

    Example:
    ```
        with GroundMotionTable(filepath, name, 'r') as dbase:
            # ... do your operation here
    ```

    :raises: :class:`tables.exceptions.NoSuchNodeError` if mode is 'r'
        and the table was not found in `filepath`, IOError if the
        file does not exist
    '''
    h5file, w_mark, was_undo_enabl = None, None, False
    try:
        h5file = tables.open_file(filepath, mode)
        was_undo_enabl = h5file.is_undo_enabled()
        if mode in ('a', 'w'):
            h5file.enable_undo()
            w_mark = h5file.mark()

        yield h5file

    except Exception as _:
        if w_mark is not None:
            h5file.undo(w_mark)
        raise
    finally:
        if h5file is not None:
            if was_undo_enabl is False and h5file.is_undo_enabled():
                h5file.disable_undo()
            h5file.close()


@contextmanager
def gettable(filepath, mode, name=None, sa_periods=None,
             has_imt_components=False):
    '''
    Returns the table form path

    mode: 'w', 'r', 'a' (a not tested)
    sa_periods must be given if mode='w'
    name=None: basename(filepath) woithout extension

    OLD DOC:

    Yields a pytable Group object representing a Gm database
    in the given hdf5 file `filepath`. If such a group does not exist
    and mode is 'w', creates the group. In any other case
    where such  a group does not exist, raises a :class:`NoSuchNodeError`

    Example:
    ```
        with GroundMotionTable(filepath, name, 'r') as dbase:
            # ... do your operation here
    ```

    :raises: :class:`tables.exceptions.NoSuchNodeError` if mode is 'r'
        and the table was not found in `filepath`, IOError if the
        file does not exist
    '''
    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    with openfile(filepath, mode) as h5file:

        rootpath = f"/{name}"
        try:
            group = h5file.get_node(rootpath, classname=Group.__name__)
            if mode == 'w':
                for node in group:  # make group empty
                    h5file.remove_node(node, recursive=True)
        except NoSuchNodeError as _:
            if mode == 'r':
                raise
            # create group node
            group = h5file.create_group(h5file.root, name)

        tablepath = f"{rootpath}/table"
        try:
            table = h5file.get_node(tablepath, classname=Table.__name__)
            if mode == 'w':
                raise ValueError('Table should not exists in "w" mode, it '
                                 'probably could not be deleted. Retry or '
                                 'delete the file manually')
            try:
                # sanity check: is it (most likely) a GroundMotionTable?
                table.attrs.sa_periods
                table.attrs._current_row_id
                table.attrs._has_imt_components
            except AttributeError:
                raise ValueError(f'The node "{tablepath}" seems not to be '
                                 'a valid GroundMotionTable')

        except NoSuchNodeError as _:
            if mode == 'r':
                raise
            if sa_periods is None:
                raise ValueError("You need to provide 'sa_periods' in order "
                                 f"to open the table file in '{mode}' mode")
            # create table node
            table = _create_table(h5file, rootpath, sa_periods,
                                  has_imt_components)
            table.attrs.filename = os.path.basename(filepath)

        yield table

        if mode != 'r':
            table.flush()


def _create_table(h5file, rootpath, sa_periods, has_imt_components=False):
    '''
        New doc: the table does not exist. called from the method above

        Creates the HDF table representing this Ground motion table
        where to write records.

        NOTE: This method must be called when this
        object is opened in 'w' mode and inside a `with` statement, PRIOR
        to any call to `self.write_record`. If a table mapping this object
        already exists on the underlying HDF file, it will be overwritten

        :param sa_periods: a **monotonically increasing** numeric array of
            the periods (x values) for the SA
        :param has_imt_components: boolean telling if this table supports
            imts with components, i.e. for each IMT a 3-length vector
            is asslocated where to store the imt components (two horizontal
            and vertical, in this order). The SA will be stored as a
            [3 X len(sa_periods)] matrix, in case
    '''
    comps = {}
    sa_length = len(sa_periods)
    comps['sa'] = Float64Col(shape=(sa_length,))
    if has_imt_components:
        comps = {comp + '_components': Float64Col(shape=(3,))
                 for comp in ('pga', 'pgv', 'sa', 'pgd', 'duration_5_75',
                              'duration_5_95', 'arias_intensity', 'cav')}
        comps['sa_components'] = Float64Col(shape=(3, sa_length))

    # add internal ids (populated automatically for each written record):
    comps['record_id'] = UInt32Col()  # max id: 4,294,967,295
    comps['station_id'] = StringCol(itemsize=20)
    comps['event_id'] = StringCol(20)

    desc = dict(GMTableDescription, **comps)
    table = h5file.create_table(rootpath, "table", description=desc)
    table.attrs.sa_periods = np.asarray(sa_periods, dtype=float)
    table.attrs._current_row_id = 1
    #  table.attrs.filename = os.path.basename(self.filepath)
    table.attrs._has_imt_components = has_imt_components
    return self


def write_record(table, csvrow, sa_periods, flush=False):
    '''writes the content of `csvrow` into tablerow  on the table mapped
    to this object in the undelrying HDF file, which must be open
    (i.e., the user must be inside a with statement) in write mode.

    NOTE: `self.create_table` must have been called ONCE prior to this
        method call

    Returns the tuple:
    ```written, missing_colnames, bad_colnames, outofbounds_colnames```
    where the last three elements are lists of strings (the record
    column names under the given categories) and the first element is a
    boolean inicating if the record has been written. A record might not
    been written if the sanity check did not pass

    :param csvrow: a dict representing a record, usually read froma  csv
    file.
    '''
    # NOTE: if parsed from a csv reader (the usual case),
    # values of `csvrow` the dict are all strings
    missing_colnames, bad_colnames, outofbounds_colnames = [], [], []
    if not _sanity_check(csvrow):
        return False, missing_colnames, bad_colnames, outofbounds_colnames

    # build a record hashes as ids:
    evid, staid, recid = _get_ids(csvrow)
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

    for col, colobj in tablerow.table.coldescrs.items():
        if col not in csvrow:
            missing_colnames.append(col)
            continue
        try:
            # remember: if val is a castable string -> ok
            #   (e.g. table column float, val is '5.5' or '5.5 ')
            # if val is out of bounds for the specific type, -> ok
            #   (casted to the closest value)
            # if val is scalar and the table column is a N length array,
            # val it is broadcasted
            #   (val= 5, then tablerow will have a np.array of N 5s)
            # TypeError is raised when there is a non castable element
            #   (e.g. 'abc' or '' for a Float column): in this case pass
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
            if isinstance(csvrow[col], (str, bytes)) and \
                    csvrow[col] in ('', b''):
                missing_colnames.append(col)
            else:
                bad_colnames.append(col)

    tablerow.append()  # pylint: disable=no-member
    if flush:
        table.flush()

    return True, missing_colnames, bad_colnames, outofbounds_colnames


def _sanity_check(csvrow):
    '''performs sanity checks on the dict `csvrow` before
    writing it. Note that  pytables does not support roll backs,
    and when closing the file pending data is automatically flushed.
    Therefore, the data has to be checked before, on the csv row

    :param csvrow: a row of a parsed csv file representing a record to add
    '''
    # for the moment, just do a pga/sa[0] check for unit consistency
    # other methods might be added in the future
    return _pga_sa_unit_ok(csvrow)


def _pga_sa_unit_ok(csvrow):
    '''Checks that pga unit and sa unit are in accordance

    :param csvrow: a row of a parsed csv file representing a record to add
    '''
    # if the PGA and the acceleration in the shortest period of the SA
    # columns differ by more than an order of magnitude then certainly
    # there is something wrong and the units of the PGA and SA are not
    # in agreement and an error should be raised.
    try:
        pga, sa0 = float(csvrow['pga']), float(csvrow['sa'][0])
        retol = abs(max(pga, sa0) / min(pga, sa0))
        if not np.isnan(retol) and round(retol) >= 10:
            return False
    except Exception as _:  # disable=broad-except
        # it might seem weird to return true on exceptions, but this method
        # should only check wheather there is certainly a unit
        # mismatch between sa and pga, when they are given (i.e., not in
        # this case)
        pass
    return True


def _get_ids(tablename, csvrow):
    '''Returns the tuple record_id, event_id and station_id from
    the given HDF5 row `csvrow`'''
    # FIXME: record_id (recid) NOT USED: remove?
    ids = (tablename,
           _toint(csvrow['pga'], 0),  # (first two decimals of pga in g)
           _toint(csvrow['event_longitude'], 5),
           _toint(csvrow['event_latitude'], 5),
           _toint(csvrow['hypocenter_depth'], 3),
           csvrow['event_time'],
           _toint(csvrow['station_longitude'], 5),
           _toint(csvrow['station_latitude'], 5))
    # return event_id, station_id, record_id:
    evid, staid, recid = \
        _hash(*ids[2:6]), _hash(*ids[6:]), _hash(*ids)
    return evid, staid, recid


def _toint(value, decimals):
    '''returns an integer by multiplying value * 10^decimals
    and rounding the result to int. Returns nan if value is nan'''
    try:
        value = float(value)
    except ValueError:
        value = float('nan')
    return value if np.isnan(value) else \
        int(round((10**decimals)*value))


def _hash(*values):
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
    hashalg.update(b'/'.join(_tobytestr(v) for v in values))
    return hashalg.digest()


def _tobytestr(value):
    '''converts a value to bytes. value can be bytes, str or numeric'''
    if not isinstance(value, bytes):
        value = str(value).encode('utf8')
    return value


#########################################
# Database selection / maniuplation
#########################################


def get_dbnames(filepath):
    '''Returns he database names of the given Gm database (HDF5 file)
    The file should have been created with the `GMTableParser.parse`
    method.

    :param filepath: the path to the HDF5 file
    :return: a list of strings identyfying the database names in the file
    '''
    with tables.open_file(filepath, 'r') as h5file:
        root = h5file.get_node('/')
        return [group._v_name for group in  # pylint: disable=protected-access
                h5file.list_nodes(root, classname=Group.__name__)]
        # note: h5file.walk_groups() might raise a ClosedNodeError.
        # This error is badly documented (as much pytables stuff),
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

        with GroundMotionTable(<intput_file>, <dbname>).filter(condition) \
                as gmdb:
            for record in records_where(gmdb.table, condition):
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

    :param table: The pytables Table object. E.g., the `table`
        property of a `GroundMotionTable` object
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
        by GmDatabase parser: '2006-12-31T00:00:00', '2006-12-31 00:00:00',
        '2006-12-31', or simply '2006'.
    1c. Does a fast check on correct comparison according to columns (fields)
        types
    2. Accepts expressions like 'col_name != nan' or 'col_name == nan' by
        converting it to the numexpr correct syntax, e.g.:
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
    dbcolumns = GMTableDescription
    py3 = sys.version_info[0] >= 3
    oprs = {'==', '!=', '<=', '>=', '<', '>'}
    nan_indices = []
    str_indices = []
    dtime_indices = []
    bool_indices = []
    result = []  # list of tokens

    def last_tokenstr():
        return '' if not result else result[-1][1]

    _logical_op_errmsg = ('Logical operators (&|~) allowed only with '
                          'parenthezised expressions')

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

            # check logical operators not around brackets. I do not know how
            # numexpr handles that but it might be misleading for the user as
            # these operator take priority and might lead to unwxpected results
            if (tokenstr in ('&', '|') and last_tokenstr() != ')') or \
                    (last_tokenstr() in ('~', '|', '&')
                     and tokenstr not in ('~', '(')):
                raise _syntaxerr(('Logical operators (&|~) allowed only with '
                                  'parenthezised expressions'), token)

            if colname is not None:
                if colname != tokenstr or tokentype != ttypes['NAME']:
                    is_dtime_col = getattr(dbcolumns[colname],
                                           "is_datetime_str", False)
                    if not is_dtime_col:
                        _type_check(token, colname,
                                    dbcolumns[colname], ttypes['STR'],
                                    ttypes['NUM'], ttypes['NAME'])
                    elif tokentype != ttypes['STR']:
                        raise _syntaxerr("'%s' value must be date time ISO "
                                         "strings (quoted)" % colname, token)
                    if is_dtime_col:
                        dtime_indices.append(len(result))
                    elif py3 and tokentype == ttypes['STR']:
                        str_indices.append(len(result))
                    elif tokenstr == 'nan' and tokentype == ttypes['NAME']:
                        nan_indices.append(len(result))
                    elif dbcolumns[colname].__class__.__name__.startswith('Bool'):
                        bool_indices.append(len(result))
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
            # provide a syntaxerror with the whole string to be parsed
            # (see function _syntaxerr below)
            raise SyntaxError(str(terr), ('', 1, len(condition), condition))

    if last_tokenstr() in ('&', '|', '~'):
        # provide a syntaxerror with the whole string to be parsed
        # (see function _syntaxerr below)
        raise SyntaxError(('Logical operators (&|~) allowed only with '
                           'parenthezised expressions'),
                          ('', 1, len(condition), condition))

    # replace nans, datetimes and strings at the real end:
    _normalize_tokens(result, dtime_indices, str_indices, nan_indices,
                      bool_indices)
    # return the new normalized string by untokenizing back: aside changed
    # variables, spaces are preserved except trailing ones (a the end):
    return untokenize(result)


def _type_check(token, colname, colobj,  str_code, num_code, name_code):
    tokentype, tokenstr = token[0], token[1]
    colobj_name = colobj.__class__.__name__
    if colobj_name.startswith('String') and tokentype != str_code:
        raise _syntaxerr("'%s' value must be strings (quoted)" % colname,
                         token)
    elif (colobj_name.startswith('UInt')
          or colobj_name.startswith('Int')) and \
            tokentype != num_code:
        raise _syntaxerr("'%s' values must be integers" % colname, token)
    elif colobj_name.startswith('Float'):
        if tokentype != num_code and tokenstr != 'nan':
            raise _syntaxerr("'%s' values must be floats / nan" % colname,
                             token)
    elif colobj_name.startswith('Bool') and (tokentype != name_code or
                                             tokenstr.lower() not in
                                             ('true', 'false')):
        raise _syntaxerr("'%s' values must be boolean (True or False, "
                         "case insensitive)" % colname, token)


def _normalize_tokens(tokens, dtime_indices, str_indices, nan_indices,
                      bool_indices):
    for i in dtime_indices:
        tokenstr = tokens[i][1]  # it is quoted, e.g. '"string"', so use shlex:
        if tokenstr[0:1] == 'b':
            tokenstr = tokenstr[1:]
        if tokenstr[0:1] in ('"', "'"):
            string = shlex.split(tokenstr)[0]
            value = GMTableParser.timestamp(string)
        else:
            value = np.nan
        if np.isnan(value):
            raise _syntaxerr('not a date-time formatted string', tokens[i])
        tokens[i][1] = str(value)

    for i in str_indices:
        tokenstr = tokens[i][1]  # it is quoted, e.g. '"string"', so use shlex:
        if tokenstr[0:1] != 'b':
            string = shlex.split(tokenstr)[0]
            tokens[i][1] = str(string.encode('utf8'))

    for i in bool_indices:
        # just make the boolean python compatible:
        tokens[i][1] = tokens[i][1].lower().title()

    if nan_indices:
        nan_operators = {'==': '!=', '!=': '=='}
        for i in nan_indices:
            varname = tokens[i-2][1]
            operator = tokens[i-1][1]
            if operator not in nan_operators:
                raise _syntaxerr('only != and == can be compared with nan',
                                 tokens[i])
            tokens[i-1][1] = nan_operators[operator]
            tokens[i][1] = varname


def _syntaxerr(message, token):
    '''Build a python syntax error from the given token, with the
    given message. Token can be a list or a Token object. If list, it must
    have 5 elements:
    0 type (int)
    1 string (the token string)
    2 start (2-element tuple: (line_start, offset_start))
    3 end (2-element tuple: (line_end, offset_end))
    4 line: (the currently parsed line, as string. It includes 'string')
    '''
    if isinstance(token, list):
        token_line_num, token_offset = token[3]
        token_line = token[-1]
    else:
        token_line_num, token_offset = token.end
        token_line = token.line
    # the tuple passed is (filename, lineno, offset, text)
    return SyntaxError(message, ('', token_line_num, token_offset,
                                 token_line[:token_offset]))

##########################################
# GroundMotionTable/ Residuals calculation
##########################################


def raises_if_file_is_open(meth):
    '''decorator to be attached to the methods of GroundMotionTable
    requiring the underlying hdf file not to be opened'''
    def wrapped(self, *args, **kwargs):
        if self.is_open:
            raise ValueError('Underlying HDF file is open. Did you call '
                             'the method inside a `with` statement?')
        return meth(self, *args, **kwargs)
    return wrapped


class GroundMotionTable(ResidualsCompliantRecordSet):
    '''Implements a Ground motion database in table format. This class
    differs from :class:`smtk.sm_database.GroundMotionDatabase` in that flat
    files are stored as pytables tables in a single HDF file container.
    This should in principle have more efficient IO operations,
    exploit numexpr syntax for efficient and simpler record selection,
    and allow the creation of customized flat-files (via pytables pre-defined
    column classes).
    Support for time-series (non-scalar) data is still possible although this
    functionality has not been not tested yet. From
    :class:`smtk.residuals.gmpe_residuals.Residuals.get_residuals`, both
    databses can be passed as `database` argument.
    '''
    # TODO: in the future the two databases might inherit from a single
    # abstract class providing the functionalities of both
    def __init__(self, filepath, dbname, mode='r'):
        '''
        Creates a new database. The main functionality of a GroundMotionTable
        is to provide the contexts for the residuals calculations:
        ```
            contexts = GroundMotionTable(...).get_contexts(...)
        ```
        For all other records manipulation tasks, note that this object
        needs to be accessed inside a with statement like a normal Python
        file-like object, which opens and closes the underlying HDF file:
        ```
            with GroundMotionTable(filepath, name) as dbase:
                # ... do your operation here
                for record in dbase.records:
                    ...
        ```

        :param filepath: the string denoting the path to the hdf file
            previously created with this method. If `mode`
            is 'r', the file must exist
        :param dbname: the name of the database table. It will be the name
            of the group (kind of sub-folder) of the underlying HDF file
        :param mode: string (default: 'r'). The mode ('r', 'w') whereby
            the underlying hdf file will be opened **when this object
            is used in a with statement**. Usually, 'w' should not be used
            outside a :meth:`GMTableParser.parse` method. Note however that
            'w' does not overwrite the whole file, but the table
            data only. More specifically:
            'r': opens file in 'r' mode, raises if the file or the table in
                the file content where not found
            'w': opens file in 'a' mode, creates the table if it does not
                exists, clears all table data if it exists.
        '''
        self.filepath = filepath
        self.dbname = dbname
        self.mode = mode
        self._root = '/%s' % dbname
        self._condition = None
        self.__h5file = None
        self._table = None
        self._w_mark = None

    @property
    def is_open(self):
        return self.__h5file is not None

    @raises_if_file_is_open
    def delete(self):
        '''Deletes the HDF data related to this table stored in the underlying
        HDF file. USE WITH CARE. Example:
            GroundMotionTable(filepath, dbname, 'w').delete()
        '''
        with self:
            self._h5file.remove_node(self._root, recursive=True)

    @raises_if_file_is_open
    def filter(self, condition):
        '''Returns a read-only copy of this database filtered according to
        the given condition (numexpr expression on the database scalar
        columns, see :class:`GMTableDescription`). Raises ValueError if this
        method is accessed while the underlying HDF file is open (e.g.,
        inside a with statement).

        See module's function `func`:`records_where` and :func:`read_where`

        Example
        ```
            condition = ("(pga < 0.14) | (pga > 1.1) & (pgv != nan) &
                          (event_time < '2006-01-01T00:00:00'")

            filtered_gmdb = GroundMotionTable(...).filter(condition)
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

            filtered_gmdb = GroundMotionTable(...).filter(condition)
        ```
        '''
        gmdb = GroundMotionTable(self.filepath, self.dbname, 'r')
        gmdb._condition = condition  # pylint: disable=protected-access
        return gmdb

    @raises_if_file_is_open
    def __enter__(self):
        '''Yields a pytable Group object representing a Gm database
        in the given hdf5 file `filepath`. If such a group does not exist
        and mode is 'w', creates the group. In any other case
        where such  a group does not exist, raises a :class:`NoSuchNodeError`

        Example:
        ```
            with GroundMotionTable(filepath, name, 'r') as dbase:
                # ... do your operation here
        ```

        :raises: :class:`tables.exceptions.NoSuchNodeError` if mode is 'r'
            and the table was not found in `filepath`, IOError if the
            file does not exist
        '''
        filepath, mode, name = self.filepath, self.mode, self.dbname
        h5file = self.__h5file = \
            tables.open_file(filepath, mode if mode == 'r' else 'a')
        if mode == 'w':
            h5file.enable_undo()
            self._w_mark = h5file.mark()

        grouppath = self._root
        try:
            group = h5file.get_node(grouppath, classname=Group.__name__)
            if mode == 'w':
                for node in group:  # make group empty
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
            if exc_val is not None and self._w_mark is not None:
                self.__h5file.undo(self._w_mark)
            if self.__h5file.is_undo_enabled():
                self.__h5file.disable_undo()
            self.__h5file.close()
        self.__h5file = None
        self._table = None
        self._w_mark = None

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

#     def create_table(self, sa_periods, has_imt_components=False):
#         '''
#             Creates the HDF table representing this Ground motion table
#             where to write records.
# 
#             NOTE: This method must be called when this
#             object is opened in 'w' mode and inside a `with` statement, PRIOR
#             to any call to `self.write_record`. If a table mapping this object
#             already exists on the underlying HDF file, it will be overwritten
# 
#             :param sa_periods: a **monotonically increasing** numeric array of
#                 the periods (x values) for the SA
#             :param has_imt_components: boolean telling if this table supports
#                 imts with components, i.e. for each IMT a 3-length vector
#                 is asslocated where to store the imt components (two horizontal
#                 and vertical, in this order). The SA will be stored as a
#                 [3 X len(sa_periods)] matrix, in case
#         '''
#         try:
#             table = self.table  # table exists, overwrite
#             self._h5file.remove_node(table, recursive=True)
#         except NoSuchNodeError:
#             pass
# 
#         comps = {}
#         sa_length = len(sa_periods)
#         comps['sa'] = Float64Col(shape=(sa_length,))
#         if has_imt_components:
#             comps = {comp + '_components': Float64Col(shape=(3,))
#                      for comp in ('pga', 'pgv', 'sa', 'pgd', 'duration_5_75',
#                                   'duration_5_95', 'arias_intensity', 'cav')}
#             comps['sa_components'] = Float64Col(shape=(3, sa_length))
# 
#         # add internal ids (populated automatically for each written record):
#         comps['record_id'] = UInt32Col()  # max id: 4,294,967,295
#         comps['station_id'] = StringCol(itemsize=20)
#         comps['event_id'] = StringCol(20)
# 
#         desc = dict(GMTableDescription, **comps)
#         self._table = table = self._h5file.create_table(self._root, "table",
#                                                         description=desc)
#         table.attrs.sa_periods = np.asarray(sa_periods, dtype=float)
#         table.attrs._current_row_id = 1
#         table.attrs.filename = os.path.basename(self.filepath)
#         table.attrs._has_imt_components = has_imt_components
#         return self

#     def write_record(self, csvrow, sa_periods):
#         '''writes the content of `csvrow` into tablerow  on the table mapped
#         to this object in the undelrying HDF file, which must be open
#         (i.e., the user must be inside a with statement) in write mode.
# 
#         NOTE: `self.create_table` must have been called ONCE prior to this
#             method call
# 
#         Returns the tuple:
#         ```written, missing_colnames, bad_colnames, outofbounds_colnames```
#         where the last three elements are lists of strings (the record
#         column names under the given categories) and the first element is a
#         boolean inicating if the record has been written. A record might not
#         been written if the sanity check did not pass
# 
#         :param csvrow: a dict representing a record, usually read froma  csv
#         file.
#         '''
#         # NOTE: if parsed from a csv reader (the usual case),
#         # values of `csvrow` the dict are all strings
#         missing_colnames, bad_colnames, outofbounds_colnames = [], [], []
#         if not self._sanity_check(csvrow):
#             return False, missing_colnames, bad_colnames, outofbounds_colnames
# 
#         table = self.table
# 
#         # build a record hashes as ids:
#         evid, staid, recid = self._get_ids(csvrow)
#         csvrow['event_id'] = evid
#         csvrow['station_id'] = staid
#         # do not use record id, rather an incremental integer:
#         csvrow['record_id'] = table.attrs._current_row_id
#         table.attrs._current_row_id += 1
# 
#         # write sa periods (if not already written):
#         try:
#             table.attrs.sa_periods
#         except AttributeError:
#             table.attrs.sa_periods = np.asarray(sa_periods, dtype=float)
# 
#         tablerow = table.row
# 
#         for col, colobj in tablerow.table.coldescrs.items():
#             if col not in csvrow:
#                 missing_colnames.append(col)
#                 continue
#             try:
#                 # remember: if val is a castable string -> ok
#                 #   (e.g. table column float, val is '5.5' or '5.5 ')
#                 # if val is out of bounds for the specific type, -> ok
#                 #   (casted to the closest value)
#                 # if val is scalar and the table column is a N length array,
#                 # val it is broadcasted
#                 #   (val= 5, then tablerow will have a np.array of N 5s)
#                 # TypeError is raised when there is a non castable element
#                 #   (e.g. 'abc' or '' for a Float column): in this case pass
#                 tablerow[col] = csvrow[col]
# 
#                 bound = getattr(colobj, 'min_value', None)
#                 if bound is not None and \
#                         (np.asarray(tablerow[col]) < bound).any():
#                     tablerow[col] = colobj.dflt
#                     outofbounds_colnames.append(col)
#                     continue
# 
#                 bound = getattr(colobj, 'max_value', None)
#                 if bound is not None and \
#                         (np.asarray(tablerow[col]) > bound).any():
#                     tablerow[col] = colobj.dflt
#                     outofbounds_colnames.append(col)
#                     continue  # actually useless, but if we add code below ...
# 
#             except (ValueError, TypeError):
#                 if isinstance(csvrow[col], (str, bytes)) and \
#                         csvrow[col] in ('', b''):
#                     missing_colnames.append(col)
#                 else:
#                     bad_colnames.append(col)
# 
#         tablerow.append()  # pylint: disable=no-member
#         table.flush()
# 
#         return True, missing_colnames, bad_colnames, outofbounds_colnames

#     @classmethod
#     def _sanity_check(cls, csvrow):
#         '''performs sanity checks on the dict `csvrow` before
#         writing it. Note that  pytables does not support roll backs,
#         and when closing the file pending data is automatically flushed.
#         Therefore, the data has to be checked before, on the csv row
# 
#         :param csvrow: a row of a parsed csv file representing a record to add
#         '''
#         # for the moment, just do a pga/sa[0] check for unit consistency
#         # other methods might be added in the future
#         return cls._pga_sa_unit_ok(csvrow)
# 
#     @classmethod
#     def _pga_sa_unit_ok(cls, csvrow):
#         '''Checks that pga unit and sa unit are in accordance
# 
#         :param csvrow: a row of a parsed csv file representing a record to add
#         '''
#         # if the PGA and the acceleration in the shortest period of the SA
#         # columns differ by more than an order of magnitude then certainly
#         # there is something wrong and the units of the PGA and SA are not
#         # in agreement and an error should be raised.
#         try:
#             pga, sa0 = float(csvrow['pga']), float(csvrow['sa'][0])
#             retol = abs(max(pga, sa0) / min(pga, sa0))
#             if not np.isnan(retol) and round(retol) >= 10:
#                 return False
#         except Exception as _:  # disable=broad-except
#             # it might seem weird to return true on exceptions, but this method
#             # should only check wheather there is certainly a unit
#             # mismatch between sa and pga, when they are given (i.e., not in
#             # this case)
#             pass
#         return True

    @property
    def table(self):
        '''Returns the underlying hdf file's table'''
        tab = self._table
        if self._table is None:
            tablepath = "%s/%s" % (self._root, "table")
            tab = self._table = self._h5file.get_node(tablepath,
                                                      classname=Table.__name__)
        return tab

    def attrnames(self, key='user'):
        '''Returns this object attribute names, i.e. the attribute
            names of the underlying pytables table attributes object:
            `self.table.attrs`.
            The table attributes are intended to be read only, modify them
            only if you know what you are doing

        :param key: string in ('sys', 'user', 'all'), default: 'user'.
            'user' returns the user-defined attributes set e.g. by this object
            during creation. 'sys' returns the system attributes, **which
            should never be modified**. 'all' returns all attributes
        '''
        return self.table.attrs._f_list(key)

    def write_array(self, relative_path, values, create_path=True):
        '''
        Writes the given array on the group mapped to this object in the
        undelrying HDF file, which must be open (i.e., the user must be
        inside a with statement) in write mode

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
        # TODO: Currently NOT USED. For future potential development
        _splitpath = relative_path.split('/')
        group = self.get_group("/".join(_splitpath[:-1]), create_path)
        self._h5file.create_array(group, _splitpath[-1],
                                  obj=np.asarray(values))

    @property
    def has_imt_components(self):
        return self.table.attrs._has_imt_components

    # ----- IO PRIVATE METHODS  ----- #

    @property
    def _h5file(self):
        h5file = self.__h5file
        if h5file is None:
            raise ValueError('The underlying HDF5 file is not open. '
                             'Are you inside a "with" statement?')
        return h5file

    def _fullpath(self, path):
        return "%s/%s" % (self._root, path)

#     def _get_ids(self, csvrow):
#         '''Returns the tuple record_id, event_id and station_id from
#         the given HDF5 row `csvrow`'''
#         # FIXME: record_id (recid) NOT USED: remove?
#         dbname = self.dbname
#         toint = self._toint
#         ids = (dbname,
#                toint(csvrow['pga'], 0),  # (first two decimals of pga in g)
#                toint(csvrow['event_longitude'], 5),
#                toint(csvrow['event_latitude'], 5),
#                toint(csvrow['hypocenter_depth'], 3),
#                csvrow['event_time'],
#                toint(csvrow['station_longitude'], 5),
#                toint(csvrow['station_latitude'], 5))
#         # return event_id, station_id, record_id:
#         evid, staid, recid = \
#             self._hash(*ids[2:6]), self._hash(*ids[6:]), self._hash(*ids)
#         return evid, staid, recid
# 
#     @classmethod
#     def _toint(cls, value, decimals):
#         '''returns an integer by multiplying value * 10^decimals
#         and rounding the result to int. Returns nan if value is nan'''
#         try:
#             value = float(value)
#         except ValueError:
#             value = float('nan')
#         return value if np.isnan(value) else \
#             int(round((10**decimals)*value))
# 
#     @classmethod
#     def _hash(cls, *values):
#         '''generates a 160bit (20bytes) hash bytestring which uniquely
#         identifies the given tuple of `values`.
#         The returned string is assured to be the same for equal `values`
#         tuples (note that the order of values matters).
#         Conversely, the probability of colliding hashes, i.e., returning
#         the same bytestring for two different tuples of values, is 1 in
#         100 billion for roughly 19000 hashes (roughly 10 flatfiles with
#         all different records), and apporaches 50% for for 1.42e24 hashes
#         generated (for info, see
#         https://preshing.com/20110504/hash-collision-probabilities/#small-collision-probabilities)
# 
#         :param values: a list of values, either bytes, str or numeric
#             (no support for other values sofar)
#         '''
#         hashalg = hashlib.sha1()
#         # use the slash as separator as it is unlikely to be in value(s):
#         hashalg.update(b'/'.join(cls._tobytestr(v) for v in values))
#         return hashalg.digest()
# 
#     @classmethod
#     def _tobytestr(cls, value):
#         '''converts a value to bytes. value can be bytes, str or numeric'''
#         if not isinstance(value, bytes):
#             value = str(value).encode('utf8')
#         return value

    ###########################################################
    # IMPLEMENTS ResidualsCompliantRecordSet ABSTRACT METHODS #
    ###########################################################

    @raises_if_file_is_open
    def get_contexts(self, nodal_plane_index=1, imts=None,
                     component="Geometric"):
        """
        Returns an iterable of dictionaries, each containing the site, distance
        and rupture contexts for individual records
        """
        with self:
            return super().get_contexts(nodal_plane_index, imts, component)

    @property
    def records(self):
        '''Yields an iterator of the records according to the specified filter
        `condition`. The underlying HDF file (including each yielded record)
        must not be modified while accessing this property, and thus must
        be opened in read mode.
        ```
        with GroundMotionTable(filepath, name, 'r').filter(condition) as dbase:
            # ... do your operation here
            for record in dbase.records:
                ...
        ```
        '''
        return records_where(self.table, self._condition)

    def get_record_eventid(self, record):
        '''Returns the record event id (usually int) of the given record'''
        return record['event_id']

    def update_sites_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.SitesContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g. `context.vs30.append(record.vs30)`
        '''
        isnan = np.isnan
        context.lons.append(record['station_longitude'])
        context.lats.append(record['station_latitude'])
        context.depths.append(0.0 if isnan(record['station_elevation'])
                              else record['station_elevation'] * -1.0E-3)
        vs30 = record['vs30']
        context.vs30.append(vs30)
        context.vs30measured.append(record['vs30_measured'])
        context.z1pt0.append(vs30_to_z1pt0_cy14(vs30)
                             if isnan(record['z1']) else record['z1'])
        context.z2pt5.append(vs30_to_z2pt5_cb14(vs30)
                             if isnan(record['z2pt5']) else record['z2pt5'])
        context.backarc.append(record['backarc'])

    def update_distances_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.DistancesContext`
        object. In the typical implementation it has the attributes defined in
        `self.distances_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g. `context.rjb.append(record.rjb)`
        '''
        isnan = np.isnan
        # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
        # is a hack! Need feedback on how to fix
        context.repi.append(record['repi'])
        context.rhypo.append(record['rhypo'])
        context.rjb.append(record['repi'] if isnan(record['rjb']) else
                           record['rjb'])
        context.rrup.append(record['rhypo'] if isnan(record['rrup']) else
                            record['rrup'])
        context.rx.append(-record['repi'] if isnan(record['rx']) else
                          record['rx'])
        context.ry0.append(record['repi'] if isnan(record['ry0']) else
                           record['ry0'])
        context.rcdpp.append(0.0)
        context.rvolc.append(0.0)
        context.azimuth.append(record['azimuth'])

    def update_rupture_context(self, record, context, nodal_plane_index=1):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.RuptureContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to NaN (numpy.nan).
        Here you should set those attributes with the relative record value,
        e.g. `context.mag = record.event.mag`
        '''
        # FIXME: nodal_plane_index not used??
        isnan = np.isnan

        strike, dip, rake = \
            record['strike_1'], record['dip_1'], record['rake_1']

        if np.isnan([strike, dip, rake]).any():
            strike, dip, rake = \
                record['strike_2'], record['dip_2'], record['rake_2']

        if np.isnan([strike, dip, rake]).any():
            strike = 0.0
            dip = 90.0
            try:
                sof = record['style_of_faulting']
                # might be bytes:
                if hasattr(sof, 'decode'):
                    sof = sof.decode('utf8')
                rake = MECHANISM_TYPE[sof]
                dip = DIP_TYPE[sof]
            except KeyError:
                rake = 0.0

        context.mag = record['magnitude']
        context.strike = strike
        context.dip = dip
        context.rake = rake
        context.hypo_depth = record['hypocenter_depth']
        _ = record['depth_top_of_rupture']
        context.ztor = context.hypo_depth if isnan(_) else _
        rec_width = record['rupture_width']
        if np.isnan(rec_width):
            rec_width = np.sqrt(DEFAULT_MSR.get_median_area(context.mag, 0))
        context.width = rec_width
        context.hypo_lat = record['event_latitude']
        context.hypo_lon = record['event_longitude']
        context.hypo_loc = (0.5, 0.5)

    def update_observations(self, record, observations, component="Geometric"):
        '''Updates the observed intensity measures types (imt) with the given
        `record` data. `observations` is a `dict` of imts (string) mapped to
        numeric lists. Here you should append to each list the imt value
        derived from `record`, ususally numeric or NaN (`numpy.nan`):
        ```
            for imt, values in observations.items():
                if imtx in self.SCALAR_IMTS:  # currently, 'PGA' or 'PGV'
                    val = ... get the imt scalar value from record ...
                elif "SA(" in imtx:
                    val = ... get the SA numeric array / list from record ...
                else:
                    raise ValueError("IMT %s is unsupported!" % imtx)
                values.append(val)
        ```
        '''
        has_imt_components = self.has_imt_components
        scalar_func = SCALAR_XY[component]
        sa_periods = self.table.attrs.sa_periods
        for imtx, values in observations.items():
            value = np.nan
            components = [np.nan, np.nan]
            if "SA(" in imtx:
                target_period = imt.from_string(imtx).period
                if has_imt_components:
                    spectrum = record['sa_components'][:2]
                    components[0] = get_interpolated_period(target_period,
                                                            sa_periods,
                                                            spectrum[0])
                    components[1] = get_interpolated_period(target_period,
                                                            sa_periods,
                                                            spectrum[1])
                else:
                    spectrum = record['sa']
                    value = get_interpolated_period(target_period, sa_periods,
                                                    spectrum)
            else:
                imtx_ = imtx.lower()
                if has_imt_components:
                    components = record['%s_components' % imtx_][:2]
                value = record[imtx_]

            if has_imt_components:
                value = scalar_func(*components)
            values.append(value)

    ###########################
    # END OF ABSTRACT METHODS #
    ###########################

