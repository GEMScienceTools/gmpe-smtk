'''
NgaWest2 try (FIXME write doc)
'''
from collections import OrderedDict
import re

import numpy as np
from scipy.constants import g
from smtk.gm_database import GMDatabaseParser
from smtk import sm_utils

SCALAR_XY = {"Geometric": lambda x, y: np.sqrt(x * y),
             "Arithmetic": lambda x, y: (x + y) / 2.,
             "Larger": lambda x, y: np.max(np.array([x, y])),
             "Vectorial": lambda x, y: np.sqrt(x ** 2. + y ** 2.)}


class UserDefinedParser(GMDatabaseParser):
    '''
    Implements a base class for parsing flatfiles in csv format into
    GmDatabase files in HDF5 format.
    This class should be user for any user defined flatfile. A flatfile
    (csv file) should have the columns defined as keys of
    :class:GmDatabaseTable (upper/lower case is ignored), with the following
    caveats:

    1. SA columns must be given in the form 'sa(period)', where 'period is a
        number
    2. pga column is supposed to be in g unit, but the user can also supply it
        as 'pga(cm/s^2)' or 'pga(m/s^2)'. The column header will
        be recognized as pga and the unit conversion will be done automatically
        A column 'pga' can be also given but then there must be a column
        'acceleration_unit' with one of the values 'g', 'cm/s^2', 'm/s^2'
    3. Event time can be given as datetime in ISO format under the column
        'event_time' , or as separate columns 'year' 'month' 'day' /
        'year', 'month', 'day', 'hour', 'minute' ,' second'.
    4. All columns will be converted to lower case before processing

    The parsing is done in the `parse` method. The typical workflow
    is to implement a new subclass for each new flatfile released.
    Subclasses should override the `mapping` dict where flatfile
    specific column names are mapped to :class:`GMDatabaseTable` column names
    and optionally the `parse_row` method where additional operation
    is performed in-place on each flatfile row. For more details, see
    the :method:`parse_row` method docstring
    '''
    csv_delimiter = ';'  # the csv delimiter (same as superclass)

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

    # the column denoting event time. Note that this attribute might be
    # modified during parsing (they could be e.g. ['year', 'month', 'day']
    _evtime_fieldnames = ['event_time']

    # pga column. Might be modified during parsing:
    _pga_col = 'pga'
    _pga_unit = 'g'

    # fields to be converted to lower case will be populated during parsing:
    _non_lcase_fieldnames = []

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
    def get_sa_columns(cls, csv_fieldnames):
        """This method is intended to be overridden by subclasses (by default
        it raises :class:`NotImplementedError`) to return a `dict` of SA
        column names (string), mapped to a numeric value representing the SA
        period.
        This class will then sort and save SA periods accordingly.

        :param csv_fieldnames: an iterable of strings representing the
            header of the persed csv file
        """

        # this method is run once per parse action, setup here the
        # needed class attributes we will need inside parse_row (maybe could be
        # better implemented):
        try:
            cls._evtime_fieldnames = \
                cls._get_event_time_columns(csv_fieldnames)
        except Exception as exc:
            raise ValueError('Unable to parse event '
                             'time column(s): %s' % str(exc))

        # get pga fieldname and units:
        try:
            cls._pga_col, cls._pga_unit = cls._get_pga_column(csv_fieldnames)
        except Exception as exc:
            raise ValueError('Unable to parse PGA column: %s' % str(exc))

        # set non-lowercase fields, so that we replace these later:
        cls._non_lcase_fieldnames = \
            [_ for _ in csv_fieldnames if _.lower() != _]

        # extract the sa columns:
        periods_names = []
        reg = cls._sa_periods_re
        for fname in csv_fieldnames:
            match = reg.match(fname)
            if match:
                periods_names.append((fname, float(match.group(1))))

        periods_names.sort(key=lambda item: item[1])
        return OrderedDict(periods_names)

    @classmethod
    def _get_event_time_columns(cls, csv_fieldnames):
        '''returns the event time column names'''

        fnames = set(_.lower() for _ in csv_fieldnames)

        for evtnames in [['event_time'],
                         ['year', 'month', 'day'],
                         ['year', 'month', 'day', 'hour', 'minute', 'second']]:
            if fnames & set(evtnames) == set(evtnames):
                return evtnames
        raise Exception('no event time related column found')

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
           lists or tuples to convert arrays and silenttly coerce unparsable
           values to nan (Note that nan represents a missing value for any
           numeric or timestamp column).
        4. the `rowdict` keys 'event_id', 'station_id' and 'record_id' are
           reserved and their values will be overridden anyway

        :param rowdict: a row of the csv flatfile, as Python dict

        :param sa_colnames: a list of strings of the column
            names denoting the SA values. The list is sorted ascending
            according to the relative numeric period defined in
            :method:`get_sa_periods`
        '''
        # assign values (sa, event time, pga):
        tofloat = cls.float
        rowdict['sa'] = tofloat([rowdict[p] for p in sa_colnames])

        # assign event time:
        evtime_fieldnames = cls._evtime_fieldnames
        dtime = ""
        if len(evtime_fieldnames) == 6:
            dtime = "{}-{}-{}T{}:{}:{}".format(*(rowdict[i]
                                                 for i in evtime_fieldnames))
        elif len(evtime_fieldnames) == 3:
            dtime = "{}-{}-{}".format(*(rowdict[i]
                                        for i in evtime_fieldnames))
        else:
            dtime = rowdict[evtime_fieldnames[0]]
        rowdict['event_time'] = cls.timestamp(dtime)

        # assign pga:
        pga_col, pga_unit = cls._pga_col, cls._pga_unit
        acc_unit = rowdict[pga_unit] \
            if pga_unit == 'acceleration_unit' else pga_unit
        rowdict['pga'] = \
            sm_utils.convert_accel_units(tofloat(rowdict[pga_col]), acc_unit)

        # replace non lower case keys with their lower case counterpart:
        for key in cls._non_lcase_fieldnames:
            rowdict[key.lower()] = rowdict.pop(key)


class EsmParser(GMDatabaseParser):

    mappings = {'ev_nation_code': 'event_country',
                'event_id': 'event_name',
                'ev_latitude': 'event_latitude',
                'ev_longitude': 'event_longitude',
                'ev_depth_km': 'hypocenter_depth',
                'fm_type_code': 'style_of_faulting',
                 # 'tectonic_environment': 'tectonic_environment',
                'st_nation_code': 'station_country',
                'st_latitude': 'station_latitude',
                'st_longitude': 'station_longitude',
                'st_elevation': 'station_elevation',
                'epi_dist': 'repi',
                'epi_az': 'azimuth',
                'JB_dist': 'rjb',
                'rup_dist': 'rrup',
                'Rx_dist': 'rx',
                'Ry0_dist': 'ry0',
                'U_hp': 'hp_h1',
                'V_hp': 'hp_h2',
                'U_lp': 'lp_h1',
                'V_lp': 'lp_h2'}

    @classmethod
    def get_sa_columns(cls, csv_fieldnames):
        """This method is intended to be overridden by subclasses (by default
        it raises :class:`NotImplementedError`) to return a `dict` of SA
        column names (string), mapped to a numeric value representing the SA
        period.
        This class will then sort and save SA periods accordingly.

        :param csv_fieldnames: an iterable of strings representing the
            header of the persed csv file
        """
        # return just the U component. V and W will be handled in `parse_row`:
        return OrderedDict([(k, float(k[3:].replace('_', '.'))) for k in
                            ['U_T0_010', 'U_T0_025', 'U_T0_040', 'U_T0_050',
                             'U_T0_070', 'U_T0_100', 'U_T0_150', 'U_T0_200',
                             'U_T0_250', 'U_T0_300', 'U_T0_350', 'U_T0_400',
                             'U_T0_450', 'U_T0_500', 'U_T0_600', 'U_T0_700',
                             'U_T0_750', 'U_T0_800', 'U_T0_900', 'U_T1_000',
                             'U_T1_200', 'U_T1_400', 'U_T1_600', 'U_T1_800',
                             'U_T2_000', 'U_T2_500', 'U_T3_000', 'U_T3_500',
                             'U_T4_000', 'U_T4_500', 'U_T5_000', 'U_T6_000',
                             'U_T7_000', 'U_T8_000', 'U_T9_000', 'U_T10_000']])

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
           lists or tuples to convert arrays and silenttly coerce unparsable
           values to nan (Note that nan represents a missing value for any
           numeric or timestamp column).
        4. the `rowdict` keys 'event_id', 'station_id' and 'record_id' are
           reserved and their values will be overridden anyway

        :param rowdict: a row of the csv flatfile, as Python dict

        :param sa_colnames: a list of strings of the column
            names denoting the SA values. The list is sorted ascending
            according to the relative numeric period defined in
            :method:`get_sa_periods`
        '''
        tofloat = cls.float  # coerces to nan in case of errors

        def tofloatabs(val):
            '''ESM not reporting correctly some values: PGA, PGV, PGD and SA
            should always be positive (absolute value)'''
            return np.abs(tofloat(val))

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
        else:
            sdf = 9

        # If all es_strike    es_dip    es_rake are not nan use those as
        # strike dip rake.
        # Otherwise see below (all es_strike    es_dip    es_rake are nans:
        # do not set strike dip and rake and use style of faulting later
        # during residuals calc)
        rowdict['strike_1'], rowdict['dip_1'], rowdict['rake_1'] = \
            rowdict.pop('es_strike', ''), rowdict.pop('es_dip', ''), \
            rowdict.pop('es_rake', '')

        # if vs30_meas_type is not empty  then vs30_measured is True else False
        rowdict['vs30_measured'] = bool(rowdict.get('vs30_meas_type', ''))

        # if vs30_meas_sec has value, then vs30 is that value, vs30_measured
        # is True
        # Otherwise if vs30_sec_WA has value, then vs30 is that value and
        # vs30_measure is False
        # Othersie, vs30 is obviously missing and vs30 measured is not given
        # (check needs to be False by default)
        if rowdict.get('vs30_m_sec', ''):
            rowdict['vs30'] = rowdict['vs30_m_sec']
            rowdict['vs30_measured'] = True
        elif rowdict.get('vs30_m_sec_WA', ''):
            rowdict['vs30'] = rowdict['vs30_m_sec_WA']
            rowdict['vs30_measured'] = False

        # rhyopo is sqrt of repi**2 + event_depth**2 (basic Pitagora)
        rowdict['rhypo'] = np.sqrt(tofloat(rowdict['repi']) ** 2 +
                                   tofloat(rowdict['hypocenter_depth']) ** 2)

        # if instrument_type_code is D then digital_recording is True otherwise
        # False
        rowdict['digital_recording'] = \
            rowdict.get('instrument_type_code', '') == 'D'

        # U_pga    V_pga    W_pga are the three components of pga
        dflt, cfunc, fromg = np.nan, SCALAR_XY['Geometric'],\
            lambda val: sm_utils.convert_accel_units(val, 'g')
        rowdict['_pga_components'] = \
            [(tofloatabs(rowdict.pop('U_pga', dflt))),
             (tofloatabs(rowdict.pop('V_pga', dflt))),
             (tofloatabs(rowdict.pop('W_pga', dflt)))]
        rowdict['pga'] = cfunc(rowdict['_pga_components'][0],
                               rowdict['_pga_components'][1])

        # U_pgv    V_pgv    W_pgv are the three components of pgv
        rowdict['_pgv_components'] = [tofloatabs(rowdict.pop('U_pgv', dflt)),
                                      tofloatabs(rowdict.pop('V_pgv', dflt)),
                                      tofloatabs(rowdict.pop('W_pgv', dflt))]
        rowdict['pgv'] = cfunc(rowdict['_pgv_components'][0],
                               rowdict['_pgv_components'][1])

        # U_pgd    V_pgd    W_pgd are the three components of pgd
        rowdict['_pgd_components'] = [tofloatabs(rowdict.pop('U_pgd', dflt)),
                                      tofloatabs(rowdict.pop('V_pgd', dflt)),
                                      tofloatabs(rowdict.pop('W_pgd', dflt))]
        rowdict['pgd'] = cfunc(rowdict['_pgd_components'][0],
                               rowdict['_pgd_components'][1])

        # U_T90    V_T90    W_T90 are the three components of duration_5_95
        rowdict['_duration_5_95_components'] = \
            [tofloat(rowdict.pop('U_T90', dflt)),
             tofloat(rowdict.pop('V_T90', dflt)),
             tofloat(rowdict.pop('W_T90', dflt))]
        rowdict['duration_5_95'] = \
            cfunc(rowdict['_duration_5_95_components'][0],
                  rowdict['_duration_5_95_components'][1])

        # U_CAV    V_CAV    W_CAV are the 3 comps for cav
        rowdict['_cav_components'] = [tofloat(rowdict.pop('U_CAV', dflt)),
                                      tofloat(rowdict.pop('V_CAV', dflt)),
                                      tofloat(rowdict.pop('W_CAV', dflt))]
        rowdict['cav'] = cfunc(rowdict['_cav_components'][0],
                               rowdict['_cav_components'][1])

        # U_ia    V_ia    W_ia  are the 3 comps for arias_intensity
        rowdict['_arias_intensity_components'] = \
            [tofloat(rowdict.pop('U_ia', dflt)),
             tofloat(rowdict.pop('V_ia', dflt)),
             tofloat(rowdict.pop('W_ia', dflt))]
        rowdict['arias_intensity'] = \
            cfunc(rowdict['_arias_intensity_components'][0],
                  rowdict['_arias_intensity_components'][1])

        # SA columns are defined in `get_sa_columns`
        sa_u = [tofloatabs(rowdict[_]) for _ in sa_colnames]
        sa_v = [tofloatabs(rowdict[_]) for _ in
                (_.replace('U_', 'V_') for _ in sa_colnames)]
        sa_w = [tofloatabs(rowdict[_]) for _ in
                (_.replace('U_', 'W_') for _ in sa_colnames)]
        rowdict['_sa_components'] = np.array([sa_u, sa_v, sa_w])
        rowdict['sa'] = cfunc(rowdict['_sa_components'][0],
                              rowdict['_sa_components'][1])

        # depth_top_of_rupture = es_z_top if given else event_depth_km
        if 'es_z_top' in rowdict:
            rowdict['depth_top_of_rupture'] = rowdict['es_z_top']
        # rupture_length = es_length if given else nan
        if 'es_length' in rowdict:
            rowdict['rupture_length'] = rowdict['es_length']
        # rupture_width = es_width if given else nan
        if 'es_width' in rowdict:
            rowdict['rupture_width'] = rowdict['es_width']

        # All these should be nan:
        # magnitude_uncertainty strike_1 dip_1 rake_1,
        #  strike_2, dip_2 rake_2

class NgaWest2Parser(GMDatabaseParser):
    mappings: {}

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


class NgaEast2Parser(GMDatabaseParser):
    mappings: {}


class EuropeanWeakMotionParser(GMDatabaseParser):
    _mappings: {}

