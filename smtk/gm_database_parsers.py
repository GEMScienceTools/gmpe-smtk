'''
NgaWest2 try (FIXME write doc)
'''
from smtk.gm_database import GMDatabaseParser


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
    _mappings: { 'ev_nation_code': 'event_country',
                'ev_latitude' : 'event_latitude',
                'ev_longitude': 'event_longitude',
                'ev_depth_km': 'event_depth',
                'fm_type_code': 'style_of_faulting',
                'tectonic_environment': 'tectonic_environment'
                 'st_nation_code': 'station_country',
                 'st_latitude': 'station_latitude',
                 'st_longitude': 'station)_longitude',
                 'st_elevation': 'station_elevation',
                  'epi_dist': 'repi', 'epi_az': 'azimuth',
                  'JB_dist': 'rjb', 'rup_dist': 'rrup',  'Rx_dist': 'rx',
                  'Ry0_dist': 'ry0'
                  
                  'U_hp': 'hp_h1',
                  'V_hp': 'hp_h2'    'U_lp': 'lp_h1', 'V_lp': 'lp_h2'    
                  U_T0_010
                  U_T0_025    U_T0_040    U_T0_050    U_T0_070    U_T0_100    U_T0_150
                  U_T0_200    U_T0_250    U_T0_300    U_T0_350    U_T0_400    U_T0_450
                  U_T0_500    U_T0_600    U_T0_700    U_T0_750    U_T0_800    U_T0_900
                  U_T1_000    U_T1_200    U_T1_400    U_T1_600    U_T1_800    U_T2_000
                  U_T2_500    U_T3_000    U_T3_500    U_T4_000    U_T4_500    U_T5_000
                  U_T6_000    U_T7_000    U_T8_000    U_T9_000    U_T10_000    V_T0_010
                  V_T0_025    V_T0_040    V_T0_050    V_T0_070    V_T0_100    V_T0_150
                  V_T0_200    V_T0_250    V_T0_300    V_T0_350    V_T0_400    V_T0_450
                  V_T0_500    V_T0_600    V_T0_700    V_T0_750    V_T0_800    V_T0_900
                  V_T1_000    V_T1_200    V_T1_400    V_T1_600    V_T1_800    V_T2_000
                  V_T2_500    V_T3_000    V_T3_500    V_T4_000    V_T4_500    V_T5_000
                  V_T6_000    V_T7_000    V_T8_000    V_T9_000    V_T10_000    W_T0_010
                  W_T0_025    W_T0_040    W_T0_050    W_T0_070    W_T0_100    W_T0_150
                  W_T0_200    W_T0_250    W_T0_300    W_T0_350    W_T0_400    W_T0_450
                  W_T0_500    W_T0_600    W_T0_700    W_T0_750    W_T0_800    W_T0_900
                  W_T1_000    W_T1_200    W_T1_400    W_T1_600    W_T1_800    W_T2_000
                  W_T2_500    W_T3_000    W_T3_500    W_T4_000    W_T4_500    W_T5_000
                  W_T6_000    W_T7_000    W_T8_000    W_T9_000    W_T10_000

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
        # convert event time from cells into a datetime string:

        # put in rowdict['station_name'] the usual NETWORK.STATION
        # from the fields network_code  and  station_code
        
        # the magnitude: see _parse_magnitude in esm_flatfile_parser and
        # implement it here. In this order (take first not nan): EMEC_Mw ->
        # Mw -> Ms -> Ml
        
        # Also store in magnitude type (Mw Ms Ml)
        
        # If all es_strike    es_dip    es_rake are not nan use those as
        # strike dip rake. 
        # Otherwise see below (all es_strike    es_dip    es_rake are nans:
        # do not set strike dip and rake and use style of faulting later during residuals calc)
    
        # if vs30_meas_type ios not empty  then vs30_measures is True else False

        # if vs30_meas_sec has value, then vs30 is that value, vs30_measured is True
        # Otherwise if vs30_sec_WA has value, then vs30 is that value and vs30_measure is False
        # Othersie, vs30 is obviously missing and vs30 measured is not given (check needs to be False by default)
        
        # rhyopo is sqrt of repi**2 + event_depth**2 (basic Pitagora)

        # if instrument_type_code is D then digital_recording is True otherwise False

        # U_pga    V_pga    W_pga are the three components of pga
        # U_pgv    V_pgv    W_pgv are the three components of pgv
        # U_pgd    V_pgd    W_pgd are the three components of pgd
        # U_T90    V_T90    W_T90 are the three components of duration_5_95
        # U_CAV    V_CAV    W_CAV are the 3 comps for cav
        # U_ia    V_ia    W_ia  are the 3 comps for arias_intensity
                  
        # these parameters:
        # depth_top_of_rupture = es_z_top if given else event_depth_km
        # rupture_length = es_length if given else nan
        # rupture_width = es_width if given else nan

        
        # REMEMBER: All these are nan:
        # magnitude_uncertainty strike_1 dip_1 rake_1,
        #  strike_2, dip_2 rake_2
        
        

class WeakMotion(GMDatabaseParser):
    _mappings: {}

