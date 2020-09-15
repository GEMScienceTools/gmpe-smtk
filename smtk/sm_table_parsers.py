'''
NgaWest2 try (FIXME write doc))
'''
from collections import OrderedDict
import re

import numpy as np
from scipy.constants import g
from smtk.sm_table import GMTableParser
from smtk import sm_utils
from smtk.sm_utils import convert_accel_units


class UserDefinedParser(GMTableParser):
    '''
    Implements a base class for parsing flatfiles in csv format into
    GmDatabase files in HDF5 format.
    This class should be user for any user defined flatfile. A flatfile
    (csv file) should have the columns defined as keys of
    :class:GmDatabaseTable (upper/lower case is ignored), with the following
    options/requirements:

    1. SA columns must be given in the form 'sa(period)', where 'period is a
       number. The unit of SA can be given in the column 'acceleration_unit'.
       If such a column is missing, SA is assumed to be in g as unit
    2. PGA column ('pga') can also be supplied as 'pga(<unit>)', e.g.
        'pga(cm/s^2)' or 'pga(g)'. In the former case (no unit specified),
        the unit of the PGA will be the value given in the column
        'acceleration_unit'. If such a column is missing, the PGA is assumed
        to be in g as unit
    3. PGV must is assumed to be given in cm/sec
    4. Event time can be given as datetime in ISO format in the column
        'event_time', or via six separate numeric columns:
        'year', 'month' (from 1 to 12), 'day' (1-31), 'hour' (0-24),
        'minute' (0-60),' second' (0-60).
    5. All columns will be converted to lower case before processing

    This class inherits from :class:`GMTableParser`: the parsing is done in
    the `parse` method.
    Subclasses should override the `mapping` dict where flatfile
    specific column names are mapped to :class:`GMDatabaseTable` column names
    and optionally the `parse_row` method where additional operation
    is performed in-place on each flatfile row. For more details, see
    the :method:`parse_row` docstring
    '''
    has_imt_components = False

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
    _acc_unit_col = None

    # fields to be converted to lower case will be populated during parsing:
    _non_lcase_fieldnames = []

    # The csv column names will be then converted according to the
    # `mappings` dict below, where a csv flatfile column is mapped to its
    # corresponding Gm database column name. The mapping is the first
    # operation performed on any row
    mappings = {}

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
        # convert to lower case:
        csv_fieldnames_lc = tuple(_.lower() for _ in csv_fieldnames)
        csv_fieldnames_lc_set = set(csv_fieldnames_lc)

        # this method is run once per parse action, setup here the
        # needed class attributes we will need inside parse_row (maybe could be
        # better implemented):
        try:
            cls._evtime_fieldnames = \
                cls._get_event_time_columns(csv_fieldnames_lc_set)
        except Exception as exc:
            raise ValueError('Unable to parse event '
                             'time column(s): %s' % str(exc))

        # get pga fieldname and units:
        try:
            cls._pga_col, cls._pga_unit, cls._acc_unit_col = \
                cls._get_pga_column_and_acc_units(csv_fieldnames_lc_set)
        except Exception as exc:
            raise ValueError('Unable to parse PGA column: %s' % str(exc))

        # set non-lowercase fields, so that we replace these later:
        cls._non_lcase_fieldnames = \
            set(csv_fieldnames) - csv_fieldnames_lc_set

        # extract the sa columns:
        periods_names = []
        reg = cls._sa_periods_re
        for fname in csv_fieldnames_lc:
            match = reg.match(fname)
            if match:
                periods_names.append((fname, float(match.group(1))))

        periods_names.sort(key=lambda item: item[1])
        return OrderedDict(periods_names)

    @classmethod
    def _get_event_time_columns(cls, csv_fieldnames_lc_set):
        '''returns the event time column names'''

        for evtnames in [['event_time'],
                         ['year', 'month', 'day', 'hour', 'minute', 'second']]:
            if csv_fieldnames_lc_set & set(evtnames) == set(evtnames):
                return evtnames
        raise Exception('no event time related column found')

    @classmethod
    def _get_pga_column_and_acc_units(cls, csv_fieldnames_lc_set):
        '''returns the column name denoting the PGA and the PGA unit.
        The latter is usually retrieved in the PGA column name. Otherwise,
        if a column 'PGA' *and* 'acceleration_unit' are found, returns
        the names of those columns'''
        pgacol, pgaunit, acc_unit_col = None, None, None
        if 'acceleration_unit' in csv_fieldnames_lc_set:
            acc_unit_col = 'acceleration_unit'
        if 'pga' in csv_fieldnames_lc_set:
            pgacol = 'pga'
        else:
            reg = cls._pga_unit_re
            # store fields 'pga' and 'acceleration_unit', if present:
            for fname in csv_fieldnames_lc_set:
                match = reg.match(fname)
                if match:
                    unit = match.group(1)
                    if unit not in cls._accel_units:
                        raise Exception('unit not in %s' %
                                        str(cls._accel_units))
                    pgacol, pgaunit = fname, unit
        if pgacol is None:
            raise ValueError("no PGA column found")
        return pgacol, pgaunit, acc_unit_col

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
        # replace non lower case keys with their lower case counterpart:
        for key in cls._non_lcase_fieldnames:
            rowdict[key.lower()] = rowdict.pop(key)

        # assign values (sa, event time, pga):
        tofloat = cls.float
        sa_ = tofloat([rowdict[p] for p in sa_colnames])
        sa_unit = rowdict[cls._acc_unit_col] if cls._acc_unit_col else 'g'
        sa_ = convert_accel_units(sa_, sa_unit)
        rowdict['sa'] = sa_

        # assign event time:
        evtime_fieldnames = cls._evtime_fieldnames
        dtime = ""
        if len(evtime_fieldnames) == 6:
            # use zfill to account for '934' formatted as '0934' for years,
            # and '5' formatted as '05' for all other fields:
            dtime = "{}-{}-{}T{}:{}:{}".\
                format(*(rowdict[c].zfill(4 if i == 0 else 2)
                       for i, c in enumerate(evtime_fieldnames)))
        else:
            dtime = rowdict[evtime_fieldnames[0]]
        rowdict['event_time'] = cls.timestamp(dtime)

        # assign pga:
        pga_col, pga_unit = cls._pga_col, cls._pga_unit
        if not pga_unit:
            pga_unit = cls._acc_unit_col if cls._acc_unit_col else 'g'
        rowdict['pga'] = \
            convert_accel_units(tofloat(rowdict[pga_col]), pga_unit)


class EsmParser(GMTableParser):

    has_imt_components = True

    mappings = {'ev_nation_code': 'event_country',
                'event_id': 'event_name',
                'ev_latitude': 'event_latitude',
                'ev_longitude': 'event_longitude',
                'ev_depth_km': 'hypocenter_depth',
                'fm_type_code': 'style_of_faulting',
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
        period. This class will then sort and save SA periods accordingly.

        You can also implement here operations which should be executed once
        at the beginning of the flatfile parsing, such as e.g.
        creating objects and storing them as class attributes later accessible
        in :method:`parse_row`

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
            magnitude = rowdict.get(mag, '').strip()
            if magnitude:
                rowdict['magnitude'] = magnitude
                rowdict['magnitude_type'] = magtype
                break

        # If all es_strike    es_dip    es_rake are not nan use those as
        # strike dip rake.
        # Otherwise see below (all es_strike    es_dip    es_rake are nans:
        # do not set strike dip and rake and use style of faulting later
        # during residuals calc)
        es_strike, es_dip, es_rake = tofloat([rowdict.get('es_strike', ""),
                                              rowdict.get('es_dip', ""),
                                              rowdict.get('es_rake', "")])
        if not np.isnan([es_strike, es_dip, es_rake]).any():
            rowdict['strike_1'], rowdict['dip_1'], rowdict['rake_1'] = \
                es_strike, es_dip, es_rake

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

        # IMTS: define functions:
        dflt = np.nan
        geom_mean = sm_utils.SCALAR_XY['Geometric']

        # Note: ESM not reporting correctly some values: PGA, PGV, PGD and SA
        # should always be positive (absolute value)

        # U_pga    V_pga    W_pga are the three components of pga
        # IT IS SUPPOSED TO BE ALREADY IN CM/S/S
        rowdict['pga_components'] = \
            np.abs([tofloat(rowdict.get('U_pga', dflt)),
                    tofloat(rowdict.get('V_pga', dflt)),
                    tofloat(rowdict.get('W_pga', dflt))])
        rowdict['pga'] = geom_mean(rowdict['pga_components'][0],
                                   rowdict['pga_components'][1])

        # U_pgv    V_pgv    W_pgv are the three components of pgv
        rowdict['pgv_components'] = \
            np.abs([tofloat(rowdict.get('U_pgv', dflt)),
                    tofloat(rowdict.get('V_pgv', dflt)),
                    tofloat(rowdict.get('W_pgv', dflt))])
        rowdict['pgv'] = geom_mean(rowdict['pgv_components'][0],
                                   rowdict['pgv_components'][1])

        # U_pgd    V_pgd    W_pgd are the three components of pgd
        rowdict['pgd_components'] = \
            np.abs([tofloat(rowdict.get('U_pgd', dflt)),
                    tofloat(rowdict.get('V_pgd', dflt)),
                    tofloat(rowdict.get('W_pgd', dflt))])
        rowdict['pgd'] = geom_mean(rowdict['pgd_components'][0],
                                   rowdict['pgd_components'][1])

        # U_T90    V_T90    W_T90 are the three components of duration_5_95
        rowdict['duration_5_95_components'] = \
            [tofloat(rowdict.get('U_T90', dflt)),
             tofloat(rowdict.get('V_T90', dflt)),
             tofloat(rowdict.get('W_T90', dflt))]
        rowdict['duration_5_95'] = \
            geom_mean(rowdict['duration_5_95_components'][0],
                      rowdict['duration_5_95_components'][1])

        # U_CAV    V_CAV    W_CAV are the 3 comps for cav
        rowdict['cav_components'] = [tofloat(rowdict.get('U_CAV', dflt)),
                                     tofloat(rowdict.get('V_CAV', dflt)),
                                     tofloat(rowdict.get('W_CAV', dflt))]
        rowdict['cav'] = geom_mean(rowdict['cav_components'][0],
                                   rowdict['cav_components'][1])

        # U_ia    V_ia    W_ia  are the 3 comps for arias_intensity
        rowdict['arias_intensity_components'] = \
            [tofloat(rowdict.get('U_ia', dflt)),
             tofloat(rowdict.get('V_ia', dflt)),
             tofloat(rowdict.get('W_ia', dflt))]
        rowdict['arias_intensity'] = \
            geom_mean(rowdict['arias_intensity_components'][0],
                      rowdict['arias_intensity_components'][1])

        # SA columns are defined in `get_sa_columns
        # THEY ARE SUPPOSED TO BE ALREADY IN CM/S/S
        sa_u = np.abs([tofloat(rowdict[_]) for _ in sa_colnames])
        sa_v = np.abs([tofloat(rowdict[_]) for _ in
                       (_.replace('U_', 'V_') for _ in sa_colnames)])
        sa_w = np.abs([tofloat(rowdict[_]) for _ in
                       (_.replace('U_', 'W_') for _ in sa_colnames)])
        rowdict['sa_components'] = np.array([sa_u, sa_v, sa_w])
        rowdict['sa'] = geom_mean(rowdict['sa_components'][0],
                                  rowdict['sa_components'][1])

        # depth_top_of_rupture = es_z_top if given else event_depth_km
        if 'es_z_top' in rowdict:
            rowdict['depth_top_of_rupture'] = rowdict['es_z_top']
        # rupture_length = es_length if given else nan
        if 'es_length' in rowdict:
            rowdict['rupture_length'] = rowdict['es_length']
        # rupture_width = es_width if given else nan
        if 'es_width' in rowdict:
            rowdict['rupture_width'] = rowdict['es_width']


class NgaWest2Parser(GMTableParser):
    mappings: {}


class NgaEast2Parser(GMTableParser):
    mappings: {}


class EuropeanWeakMotionParser(GMTableParser):
    mappings: {}

