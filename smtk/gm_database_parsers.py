'''
NgaWest2 try (FIXME write doc)
'''
from collections import OrderedDict
import re

import numpy as np
from smtk.gm_database import GMDatabaseParser
from smtk import sm_utils

SCALAR_XY = {"Geometric": lambda x, y: np.sqrt(x * y),
             "Arithmetic": lambda x, y: (x + y) / 2.,
             "Larger": lambda x, y: np.max(np.array([x, y])),
             "Vectorial": lambda x, y: np.sqrt(x ** 2. + y ** 2.)}

class DefaultParser(GMDatabaseParser):
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
            # mismatch between sa and pga, when they are given (i.e., not in
            # this case)
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

        return cls.timestamp(dtime)



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



class NgaWest2(GMDatabaseParser):
    _mappings: {}

#     @classmethod
#     def process_flatfile_row(cls, rowdict):
#         '''do any further processing of the given `rowdict`, a dict
#         represenitng a parsed csv row. At this point, `rowdict` keys are
#         already mapped to the :class:`GMDatabaseTable` columns (see `_mappings`
#         class attribute), spectra values are already set in `rowdict['sa']`
#         (interpolating csv spectra columns, if needed).
#         This method should process `rowdict` in place, the returned value
#         is ignored. Any exception is wrapped in the caller method.
# 
#         :param rowdict: a row of the csv flatfile, as Python dict. Values
#             are strings and will be casted to the matching Table column type
#             after this method call
#         '''
#         # convert event time from cells into a datetime string:
#         evtime = cls.datetime(rowdict.pop('Year'),
#                               rowdict.pop('Month'),
#                               rowdict.pop('Day'),
#                               rowdict.pop('Hour', None) or 0,
#                               rowdict.pop('Minute', None) or 0,
#                               rowdict.pop('Second', None) or 0)
#         rowdict['event_time'] = evtime


class NgaEast2(GMDatabaseParser):
    _mappings: {}


class Esm(GMDatabaseParser):

    _mappings = {'ev_nation_code': 'event_country',
                 'ev_latitude': 'event_latitude',
                 'ev_longitude': 'event_longitude',
                 'ev_depth_km': 'event_depth',
                 'fm_type_code': 'style_of_faulting',
                 'tectonic_environment': 'tectonic_environment',
                 'st_nation_code': 'station_country',
                 'st_latitude': 'station_latitude',
                 'st_longitude': 'station)_longitude',
                 'st_elevation': 'station_elevation',
                 'epi_dist': 'repi', 'epi_az': 'azimuth',
                 'JB_dist': 'rjb', 'rup_dist': 'rrup',  'Rx_dist': 'rx',
                 'Ry0_dist': 'ry0', 'U_hp': 'hp_h1',
                 'V_hp': 'hp_h2', 'U_lp': 'lp_h1', 'V_lp': 'lp_h2'}
#                   U_T0_010
#                   U_T0_025    U_T0_040    U_T0_050    U_T0_070    U_T0_100    U_T0_150
#                   U_T0_200    U_T0_250    U_T0_300    U_T0_350    U_T0_400    U_T0_450
#                   U_T0_500    U_T0_600    U_T0_700    U_T0_750    U_T0_800    U_T0_900
#                   U_T1_000    U_T1_200    U_T1_400    U_T1_600    U_T1_800    U_T2_000
#                   U_T2_500    U_T3_000    U_T3_500    U_T4_000    U_T4_500    U_T5_000
#                   U_T6_000    U_T7_000    U_T8_000    U_T9_000    U_T10_000    V_T0_010
#                   V_T0_025    V_T0_040    V_T0_050    V_T0_070    V_T0_100    V_T0_150
#                   V_T0_200    V_T0_250    V_T0_300    V_T0_350    V_T0_400    V_T0_450
#                   V_T0_500    V_T0_600    V_T0_700    V_T0_750    V_T0_800    V_T0_900
#                   V_T1_000    V_T1_200    V_T1_400    V_T1_600    V_T1_800    V_T2_000
#                   V_T2_500    V_T3_000    V_T3_500    V_T4_000    V_T4_500    V_T5_000
#                   V_T6_000    V_T7_000    V_T8_000    V_T9_000    V_T10_000    W_T0_010
#                   W_T0_025    W_T0_040    W_T0_050    W_T0_070    W_T0_100    W_T0_150
#                   W_T0_200    W_T0_250    W_T0_300    W_T0_350    W_T0_400    W_T0_450
#                   W_T0_500    W_T0_600    W_T0_700    W_T0_750    W_T0_800    W_T0_900
#                   W_T1_000    W_T1_200    W_T1_400    W_T1_600    W_T1_800    W_T2_000
#                   W_T2_500    W_T3_000    W_T3_500    W_T4_000    W_T4_500    W_T5_000
#                   W_T6_000    W_T7_000    W_T8_000    W_T9_000    W_T10_000

    _sa_u = OrderedDict([(k, float(k[3:].replace('_', '.'))) for k in
                         ['U_T0_010', 'U_T0_025', 'U_T0_040', 'U_T0_050',
                          'U_T0_070', 'U_T0_100', 'U_T0_150', 'U_T0_200',
                          'U_T0_250', 'U_T0_300', 'U_T0_350', 'U_T0_400',
                          'U_T0_450', 'U_T0_500', 'U_T0_600', 'U_T0_700',
                          'U_T0_750', 'U_T0_800', 'U_T0_900', 'U_T1_000',
                          'U_T1_200', 'U_T1_400', 'U_T1_600', 'U_T1_800',
                          'U_T2_000', 'U_T2_500', 'U_T3_000', 'U_T3_500',
                          'U_T4_000', 'U_T4_500', 'U_T5_000', 'U_T6_000',
                          'U_T7_000', 'U_T8_000', 'U_T9_000', 'U_T10_000']])
    _sa_v = OrderedDict([(k, float(k[3:].replace('_', '.'))) for k in
                         ['V_T0_010', 'V_T0_025', 'V_T0_040', 'V_T0_050',
                          'V_T0_070', 'V_T0_100', 'V_T0_150', 'V_T0_200',
                          'V_T0_250', 'V_T0_300', 'V_T0_350', 'V_T0_400',
                          'V_T0_450', 'V_T0_500', 'V_T0_600', 'V_T0_700',
                          'V_T0_750', 'V_T0_800', 'V_T0_900', 'V_T1_000',
                          'V_T1_200', 'V_T1_400', 'V_T1_600', 'V_T1_800',
                          'V_T2_000', 'V_T2_500', 'V_T3_000', 'V_T3_500',
                          'V_T4_000', 'V_T4_500', 'V_T5_000', 'V_T6_000',
                          'V_T7_000', 'V_T8_000', 'V_T9_000', 'V_T10_000']])
    _sa_w = OrderedDict([(k, float(k[3:].replace('_', '.'))) for k in
                         ['W_T0_010', 'W_T0_025', 'W_T0_040', 'W_T0_050',
                          'W_T0_070', 'W_T0_100', 'W_T0_150', 'W_T0_200',
                          'W_T0_250', 'W_T0_300', 'W_T0_350', 'W_T0_400',
                          'W_T0_450', 'W_T0_500', 'W_T0_600', 'W_T0_700',
                          'W_T0_750', 'W_T0_800', 'W_T0_900', 'W_T1_000',
                          'W_T1_200', 'W_T1_400', 'W_T1_600', 'W_T1_800',
                          'W_T2_000', 'W_T2_500', 'W_T3_000', 'W_T3_500',
                          'W_T4_000', 'W_T4_500', 'W_T5_000', 'W_T6_000',
                          'W_T7_000', 'W_T8_000', 'W_T9_000', 'W_T10_000']])

    @classmethod
    def process_flatfile_row(cls, rowdict):
        '''do any further processing of the given `rowdict`, a dict
        represenitng a parsed csv row. At this point, `rowdict` keys are
        already mapped to the :class:`GMDatabaseTable` columns (see `_mappings`
        class attribute), spectra values are already set in `rowdict['sa']`
        (interpolating csv spectra columns, if needed).
        This method should process `rowdict` in place, the returned value
        is ignored. Any exception is wrapped in the caller method.
 
        :param rowdict: a row of the csv flatfile, as Python dict. Values
            are strings and will be casted to the matching Table column type
            after this method call
        '''
        tofloat = cls.float  # coerces to nan in case of errors

        # convert event time from cells into a timestamp:
        rowdict['event_time'] = cls.timestamp(rowdict.get('event_time', ''))

        # put in rowdict['station_name'] the usual NETWORK.STATION
        # from the fields network_code  and  station_code
        rowdict['station_name'] = "%s.%s" % (rowdict.get('network_code', ''),
                                             rowdict.get('station_code', ''))

        # the magnitude: see _parse_magnitude in esm_flatfile_parser and
        # implement it here. In this order (take first not nan): EMEC_Mw ->
        # Mw -> Ms -> Ml. FIXME: check with Graeme
        # Also store in magnitude type (Mw Ms Ml)
        for mag, magtype in [('EMEC_Mw', 'Mw'), ('Mw', 'Mw'), ('Ms', 'Ms'),
                             ('Ml', 'Ml')]:
            magnitude = rowdict.pop(mag, '').strip()
            if magnitude:
                rowdict['magnitude'] = magnitude
                rowdict['magnitude_type'] = magtype
                break

        # If all es_strike    es_dip    es_rake are not nan use those as
        # strike dip rake. 
        # Otherwise see below (all es_strike    es_dip    es_rake are nans:
        # do not set strike dip and rake and use style of faulting later during
        # residuals calc)
        rowdict['strike_1'], rowdict['dip_1'], rowdict['rake_1'] = \
            rowdict.pop('es_strike', ''), rowdict.pop('es_dip', ''), \
            rowdict.pop('es_rake', '')

        # if vs30_meas_type is not empty  then vs30_measured is True else False
        rowdict['vs30_measured'] = bool(rowdict.get('vs30_meas_type', ''))

        # if vs30_meas_sec has value, then vs30 is that value, vs30_measured is True
        # Otherwise if vs30_sec_WA has value, then vs30 is that value and
        # vs30_measure is False
        # Othersie, vs30 is obviously missing and vs30 measured is not given
        # (check needs to be False by default)
        if rowdict.get('vs30_meas_sec', ''):
            rowdict['vs30'] = rowdict['vs30_meas_sec']
            rowdict['vs30_measured'] = True
        elif rowdict.get('vs30_sec_WA', ''):
            rowdict['vs30'] = rowdict['vs30_sec_WA']
            rowdict['vs30_measured'] = False

        # rhyopo is sqrt of repi**2 + event_depth**2 (basic Pitagora)
        rowdict['rhypo'] = np.sqrt(tofloat(rowdict['repi']) ** 2 +
                                   tofloat(rowdict['event_depth']) ** 2)

        # if instrument_type_code is D then digital_recording is True otherwise
        # False
        rowdict['digital_recording'] = \
            rowdict.get('instrument_type_code', '') == 'D'

        # U_pga    V_pga    W_pga are the three components of pga
        # U_pgv    V_pgv    W_pgv are the three components of pgv
        # U_pgd    V_pgd    W_pgd are the three components of pgd
        # U_T90    V_T90    W_T90 are the three components of duration_5_95
        # U_CAV    V_CAV    W_CAV are the 3 comps for cav
        # U_ia    V_ia    W_ia  are the 3 comps for arias_intensity
        dflt, cfunc = np.nan, SCALAR_XY['Geometric']
        rowdict['_pga_components'] = [tofloat(rowdict.pop('U_pga', dflt)),
                                      tofloat(rowdict.pop('V_pga', dflt)),
                                      tofloat(rowdict.pop('W_pga', dflt))]
        rowdict['pga'] = cfunc(rowdict['_pga_components'][0],
                               rowdict['_pga_components'][1])

        rowdict['_pgv_components'] = [tofloat(rowdict.pop('U_pgv', dflt)),
                                      tofloat(rowdict.pop('V_pgv', dflt)),
                                      tofloat(rowdict.pop('W_pgv', dflt))]
        rowdict['pgv'] = cfunc(rowdict['_pgv_components'][0],
                               rowdict['_pgv_components'][1])

        rowdict['_pgd_components'] = [tofloat(rowdict.pop('U_pgd', dflt)),
                                      tofloat(rowdict.pop('V_pgd', dflt)),
                                      tofloat(rowdict.pop('W_pgd', dflt))]
        rowdict['pgd'] = cfunc(rowdict['_pgd_components'][0],
                               rowdict['_pgd_components'][1])

        rowdict['_duration_5_95_components'] = \
            [tofloat(rowdict.pop('U_T90', dflt)),
             tofloat(rowdict.pop('V_T90', dflt)),
             tofloat(rowdict.pop('W_T90', dflt))]
        rowdict['duration_5_95'] = \
            cfunc(rowdict['_duration_5_95_components'][0],
                  rowdict['_duration_5_95_components'][1])

        rowdict['_cav_components'] = [tofloat(rowdict.pop('U_CAV', dflt)),
                                      tofloat(rowdict.pop('V_CAV', dflt)),
                                      tofloat(rowdict.pop('W_CAV', dflt))]
        rowdict['cav'] = cfunc(rowdict['_cav_components'][0],
                               rowdict['_cav_components'][1])

        rowdict['_arias_intensity_components'] = \
            [tofloat(rowdict.pop('U_ia', dflt)),
             tofloat(rowdict.pop('V_ia', dflt)),
             tofloat(rowdict.pop('W_ia', dflt))]
        rowdict['arias_intensity'] = \
            cfunc(rowdict['_arias_intensity_components'][0],
                  rowdict['_arias_intensity_components'][1])

        sa_u = [tofloat(rowdict[_]) for _ in cls._sa_u.keys()]
        sa_v = [tofloat(rowdict[_]) for _ in cls._sa_v.keys()]
        sa_w = [tofloat(rowdict[_]) for _ in cls._sa_w.keys()]
        rowdict['_sa_components'] = np.array([sa_u, sa_v, sa_w])
        rowdict['sa'] = cfunc(rowdict['_sa_components'][0],
                              rowdict['_sa_components'][1])

        # these parameters:
        # depth_top_of_rupture = es_z_top if given else event_depth_km
        # rupture_length = es_length if given else nan
        # rupture_width = es_width if given else nan
        if 'es_z_top' in rowdict:
            rowdict['depth_top_of_rupture'] = rowdict['es_z_top']
        if 'es_length' in rowdict:
            rowdict['rupture_length'] = rowdict['es_length']
        if 'es_width' in rowdict:
            rowdict['rupture_width'] = rowdict['es_width']

        # All these should be nan:
        # magnitude_uncertainty strike_1 dip_1 rake_1,
        #  strike_2, dip_2 rake_2


class WeakMotion(GMDatabaseParser):
    _mappings: {}

