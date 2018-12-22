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
Tests the GM database parsing and selection
"""
import os
# import shutil
import sys
from datetime import datetime
# import json
# import pprint
import unittest
import numpy as np
# import smtk.sm_table
from tables.file import open_file
from tables.description import IsDescription, Time64Col, EnumCol as _EnumCol

from smtk.sm_table_parsers import UserDefinedParser, EsmParser
from smtk.sm_table import GMTableParser, GMTableDescription, \
    records_where, read_where, get_dbnames, _normalize_condition, \
    GroundMotionTable, \
    DateTimeCol, Float64Col, Float32Col, StringCol

BASE_DATA_PATH = os.path.join(
    os.path.join(os.path.dirname(__file__), "file_samples")
    )

class ColTestCase(unittest.TestCase):

    def test_scol(self):
        scol = DateTimeCol()
        fcol = Float64Col()
        scol_bounds = DateTimeCol(min='2006', max='2007-01-01')
        sdf = 9


class DummyTable(IsDescription):
    '''dummy table class used for testing pytables'''
    floatcol = Float32Col()
    arraycol = Float32Col(shape=(10,))
    stringcol = StringCol(5)
    # first issue with EnumCol: dict values MUST be 
    ecol = _EnumCol({'a': -90, 'b': -5, 'c': 4}, 'a', 'int64')
    tcol = Time64Col(dflt=np.nan)
#    ballColor = EnumCol(['orange'], 'black', base='uint8')


class GroundMotionTableTestCase(unittest.TestCase):
    '''tests GroundMotionTable and selection'''
    @classmethod
    def setUpClass(cls):

        cls.input_file = os.path.join(BASE_DATA_PATH,
                                      "template_basic_flatfile.csv")
        cls.output_file = os.path.join(BASE_DATA_PATH,
                                       "template_basic_flatfile.hd5")

        cls.py3 = sys.version_info[0] >= 3

    def setUp(self):
        self.deletefile()

    def tearDown(self):
        self.deletefile()

    @classmethod
    def deletefile(cls):
        '''deletes tmp hdf5 file'''
        if os.path.isfile(cls.output_file):
            os.remove(cls.output_file)

    def test_timestamp(self):
        timestamp = GMTableParser.timestamp
        d = datetime(2006, 3, 31, 11, 12, 34)
        prevval = None
        for val in [d, d.isoformat(), [d, d], [d.isoformat(), d.isoformat()]]:
            _ = timestamp(val)
            if prevval is None:
                prevval = _
            else:
                if hasattr(_, '__len__'):
                    self.assertTrue((_ == prevval).all())
                else:
                    self.assertEqual(prevval, _)

        ddd = b''
        self.assertTrue(np.isnan(timestamp(ddd)))

        ddd = datetime(1970, 1, 1)
        self.assertEqual(0, timestamp(ddd))

        ddd = datetime(1969, 12, 31, 23, 59, 59)
        self.assertEqual(-1.0, timestamp(ddd))

        ddd = datetime(1969, 12, 31, 23, 59, 59, 500000)
        self.assertEqual(-.5, timestamp(ddd))

        dd = [d, b'', 'abc', d]
        _ = timestamp(dd)
        self.assertEqual(_[0], prevval)
        self.assertTrue(np.isnan(_[1]))
        self.assertTrue(np.isnan(_[2]))
        self.assertEqual(_[3], prevval)

        val1 = timestamp('2006')
        val2 = timestamp('2006-01-01')
        val3 = timestamp('2006-01-01 00:00:00')
        val4 = timestamp('2006-01-01T00:00:00')
        self.assertTrue(val1 == val2 == val3 ==val4)

    def test_float64(self):
        float64 = GMTableParser.float
        d = 0.357
        for val in [d, str(d), [d, d], [str(d), str(d)]]:
            _ = float64(val)
            if hasattr(_, '__len__'):
                self.assertTrue((_ == d).all())
            else:
                self.assertEqual(d, _)

        ddd = b''
        self.assertTrue(np.isnan(float64(ddd)))

        ddd = [d, 'abc', b'']
        _ = float64(ddd)
        self.assertEqual(d, ddd[0])
        self.assertTrue(np.isnan(_[1]))
        self.assertTrue(np.isnan(_[2]))


    def test_pytables_dummytable(self):
        '''Test some pytable casting stuff NOT clearly documented :( '''

        with open_file(self.output_file, 'a') as h5file:
            table = h5file.create_table("/",
                                        'table',
                                        description=DummyTable)
            row = table.row

            # define set and get to avoid pylint flase positives
            # everywhere:
            def set(field, val):  # @ReservedAssignment
                '''sets a value on the row'''
                row[field] = val  # pylint: disable=unsupported-assignment-operation

            def get(field):
                '''gets a value from the row'''
                return row[field]  # pylint: disable=unsubscriptable-object

            # assert the value is the default:
            self.assertTrue(np.isnan(get('floatcol')))
            # what if we supply a string? TypeError
            with self.assertRaises(TypeError):
                set('floatcol', 'a')
            # same if string empty: TypeError
            with self.assertRaises(TypeError):
                set('floatcol', '')
            # assert the value is still the default:
            self.assertTrue(np.isnan(get('floatcol')))
            # what if we supply a castable string instead? it is casted
            set('floatcol', '5.5')
            self.assertEqual(get('floatcol'), 5.5)
            # what if we supply a castable string instead WITH SPACES? casted
            set('floatcol', '1.0 ')
            self.assertEqual(get('floatcol'), 1.0)
            # what if we supply a scalr instead of an array?
            # the value is broadcasted:
            set('arraycol', 5)
            self.assertTrue(np.allclose([5] * 10, get('arraycol')))
            # what if arraycol string?
            set('arraycol', '5')
            self.assertTrue(np.allclose([5] * 10, get('arraycol')))
            # what if arraycol array of strings?
            set('arraycol', [str(_) for _ in [5] * 10])
            self.assertTrue(np.allclose([5] * 10, get('arraycol')))
            # what if arraycol array of strings with one nan?
            aaa = [str(_) for _ in [5] * 10]
            aaa[3] = 'asd'
            with self.assertRaises(ValueError):
                set('arraycol', aaa)
            # what if we supply a float out of bound? no error
            # but value is saved differently!
            maxfloat32 = 3.4028235e+38
            val = maxfloat32 * 10
            set('floatcol', val)
            val2 = get('floatcol')
            # assert they are not the same
            self.assertTrue(not np.isclose(val2, val))
            # now restore val to the max Float32, and assert they are the same:
            val = maxfloat32
            set('floatcol', val)
            val2 = get('floatcol')
            self.assertTrue(np.isclose(val2, val))
            # write to the table nan and see if we can select it later:
            set('floatcol', float('nan'))
            set('arraycol', [1, 2, 3, 4, float('nan'), 6.6, 7.7, 8.8, 9.9,
                             10.00045])

            # setting ascii str in stringcol is safe, we do not need to convert
            set('stringcol', "abc")
            # However, returned value is bytes
            self.assertEqual(get('stringcol'), b'abc')

            # test WHY enumcol IS USELESS:
            eval = get('ecol')
            # here is the point about enumcols: WE CANNOT SET the LABEL!
            # What's the point of having an enum if we actually need to
            # set/get the associated int?
            with self.assertRaises(Exception):
                set('ecol' , 'a')
            set('ecol' , -5)
            self.assertEqual(get('ecol'), -5)

            #test time64 col:
            tme = get('tcol')
            self.assertTrue(np.isnan(tme))
            dtime = datetime.utcnow()
            with self.assertRaises(TypeError):
                set('tcol', dtime)
            tme = get('tcol')
            self.assertTrue(np.isnan(tme))
            # now set a numpy datetim64. ERROR! WTF!
            tme = np.datetime64('2007-02-01T00:01:04')
            with self.assertRaises(TypeError):
                set('tcol', tme)
            # OK conclusion: pytables TimeCol is absolutely USELESS

            row.append()  # pylint: disable=no-member
            table.flush()

        # test selections:

        with open_file(self.output_file, 'a') as h5file:
            tbl = h5file.get_node("/", 'table')
            ###########################
            # TEST NAN SELECTION:
            ###########################
            # this does not work:
            with self.assertRaises(NameError):
                [r['floatcol'] for r in tbl.where('floatcol == nan')]
            # what if we provide the nan as string?
            # incompatible types (NotImplementedError)
            with self.assertRaises(NotImplementedError):
                [r['floatcol'] for r in tbl.where("floatcol == 'nan'")]
            # we should use the condvars dict:
            vals = [r['floatcol'] for r in
                    tbl.where('floatcol == nan',
                              condvars={'nan': float('nan')})]
            # does not raise, but we did not get what we wanted:
            self.assertTrue(not vals)
            # we should actually test nan equalisty with this weird
            # test: (https://stackoverflow.com/a/10821267)
            vals = [r['floatcol'] for r in tbl.where('floatcol != floatcol')]
            # now it works:
            self.assertEqual(len(vals), 1)

            ###########################
            # TEST ARRAY SELECTION:
            ###########################
            # this does not work. Array selection not yet supported:
            with self.assertRaises(NotImplementedError):
                vals = [r['arraycol'] for r in
                        tbl.where('arraycol == 2')]

            ###########################
            # TEST STRING SELECTION:
            ###########################
            # this does not work, needs quotes:
            with self.assertRaises(NameError):
                vals = [r['stringcol'] for r in
                        tbl.where("stringcol == %s" % 'abc')]
            # this SHOULD NOT WORK either (not binary strings), BUT IT DOES:
            vals = [r['stringcol'] for r in
                    tbl.where("stringcol == 'abc'")]
            self.assertTrue(len(vals) == 1)

    def test_template_basic_file(self):
        '''parses sample flatfile and perfomrs some tests'''
        # test a file not found
        with self.assertRaises(IOError):
            with GroundMotionTable(self.output_file + 'what',
                      dbname='whatever', mode='r') as gmdb:
                pass

        log = UserDefinedParser.parse(self.input_file,
                                      output_path=self.output_file,
                                      delimiter=',')
        dbname = os.path.splitext(os.path.basename(self.output_file))[0]
        # the flatfile parsed has:
        # 1. an event latitude out of bound (row 0)
        # 2. an event longitude out of bound (row 1)
        # 3. a pga with extremely high value (row 2)
        # 4. a sa[0] with extremely high value (row 3)

        total = log['total']
        written = total - 2  # row 2 and 3 not written
        self.assertEqual(log['total'], 99)
        self.assertEqual(log['written'], written)
        self.assertEqual(sorted(log['error']), [2, 3])
        self.assertEqual(len(log['outofbound_values']), 2)  # rows 0 and 1
        self.assertEqual(log['outofbound_values']['event_latitude'], 1)  # 0
        self.assertEqual(log['outofbound_values']['event_longitude'], 1)  # 1
        # self.assertEqual(log['missing_values']['pga'], 0)
        self.assertEqual(log['missing_values']['pgv'], log['written'])
        self.assertEqual(log['missing_values']['pgv'], log['written'])

        # assert auto generated ids are not missing:
        self.assertFalse('record_id' in log['missing_values'])
        self.assertFalse('event_id' in log['missing_values'])
        self.assertFalse('station_id' in log['missing_values'])

        # PYTABLES. IMPORTANT
        # seems that this is NOT possible:
        # list(table.iterrows())  # returns N times the LAST row
        # seems also that we should NOT break inside a iterrows or where loop
        # (see here: https://github.com/PyTables/PyTables/issues/8)

        # open HDF5 and check for incremental ids:
        test_col = 'event_name'
        test_col_oldval, test_col_newval = None, b'dummy'
        test_cols_found = 0
        with GroundMotionTable(self.output_file, dbname, 'a') as gmdb:
            tbl = gmdb.table
            ids = list(r['event_id'] for r in tbl.iterrows())
            # assert record ids are the number of rows
            self.assertTrue(len(ids) == written)
            # assert we have some event shared across records:
            self.assertTrue(len(set(ids)) < written)
            # modify one row
            for row in tbl.iterrows():
                if test_col_oldval is None:
                    test_col_oldval = row[test_col]
                if row[test_col] == test_col_oldval:
                    row[test_col] = test_col_newval
                    test_cols_found += 1
                    row.update()
            tbl.flush()
            # all written columns have the same value of row[test_col]:
            self.assertTrue(test_cols_found == 1)

        # assert that we modified the event name
        with GroundMotionTable(self.output_file, dbname, 'r') as gmdb:
            tbl = gmdb.table
            count = 0
            for row in tbl.where('%s == %s' % (test_col, test_col_oldval)):
                # we should never be here (no row with the old value):
                count += 1
            self.assertTrue(count == 0)
            count = 0
            for row in tbl.where('%s == %s' % (test_col, test_col_newval)):
                count += 1
            self.assertTrue(count == test_cols_found)

        # now re-write, with append mode
        log = UserDefinedParser.parse(self.input_file,
                                      output_path=self.output_file,
                                      delimiter=',')
        # open HDF5 with append='a' (the default)
        # and check that wewrote stuff twice
        with GroundMotionTable(self.output_file, dbname, 'r') as gmdb:
            tbl = gmdb.table
            self.assertTrue(tbl.nrows == written * 2)
            # assert the old rows are there
            oldrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_oldval)))
            self.assertTrue(len(oldrows) == test_cols_found)
            # assert the new rows are added:
            newrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_newval)))
            self.assertTrue(len(newrows) == test_cols_found)

        # now re-write, with no mode='w'
        log = UserDefinedParser.parse(self.input_file,
                                      output_path=self.output_file,
                                      mode='w', delimiter=',')
        with GroundMotionTable(self.output_file, dbname, 'r') as gmdb:
            tbl = gmdb.table
            self.assertTrue(tbl.nrows == written)
            # assert the old rows are not there anymore
            oldrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_oldval)))
            self.assertTrue(len(oldrows) == test_cols_found)
            # assert the new rows are added:
            newrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_newval)))
            self.assertTrue(not newrows)

        # get db names:
        dbnames = get_dbnames(self.output_file)
        self.assertTrue(len(dbnames) == 1)
        name = os.path.splitext(os.path.basename(self.output_file))[0]
        self.assertTrue(dbnames[0] == name)

    def test_template_basic_file_selection(self):
        '''parses a sample flatfile and tests some selection syntax on it'''
        # the test file has a comma delimiter. Test that we raise with
        # the default semicolon:
        with self.assertRaises(ValueError):
            log = UserDefinedParser.parse(self.input_file,
                                          output_path=self.output_file)
        # now should be ok:
        log = UserDefinedParser.parse(self.input_file,
                                      output_path=self.output_file,
                                      delimiter=',')

        dbname = os.path.splitext(os.path.basename(self.output_file))[0]
        with GroundMotionTable(self.output_file, dbname) as gmdb:
            table = gmdb.table
            total = table.nrows
            selection = 'pga <= %s' % 100.75
            ids = [r['record_id'] for r in records_where(table, selection)]
            ids_len = len(ids)
            # test that read where gets the same number of records:
            ids = [r['record_id'] for r in read_where(table, selection)]
            self.assertEqual(len(ids), ids_len)
            # test with limit given:
            ids = [r['record_id'] for r in records_where(table, selection,
                                                         ids_len-1)]
            self.assertEqual(len(ids), ids_len-1)
            # test by negating the selection condition and expect the remaining
            # records to be found:
            ids = [r['record_id'] for r in records_where(table,
                                                         "~(%s)" % selection)]
            self.assertEqual(len(ids), total - ids_len)
            # same should happend for read_where:
            ids = [r['record_id'] for r in read_where(table,
                                                      "~(%s)" % selection)]
            self.assertEqual(len(ids), total - ids_len)
            # test with limit 0 (expected: no record yielded):
            ids = [r['record_id'] for r in records_where(table,
                                                         "~(%s)" % selection,
                                                         0)]
            self.assertEqual(len(ids), 0)
            # restrict the search:
            # note that we must pass strings to event_time,
            # either 1935-01-01, 1935-01-01T00:00:00, or simply the year:
            selection2 = "(%s) & (%s)" % \
                (selection, '(event_time >= "1935") & '
                            '(event_time < \'1936-01-01\')')
            ids = [r['record_id'] for r in records_where(table, selection2)]
            ids_len2 = len(ids)
            # test that the search was restricted:
            self.assertTrue(ids_len2 < ids_len)
            # now negate the serarch on event_time and test that we get all
            # remaining records:
            selection2 = "(%s) & ~(%s)" % \
                (selection, '(event_time >= "1935") & '
                            '(event_time < "1936-01-01")')
            ids = [r['record_id'] for r in records_where(table, selection2)]
            self.assertEqual(len(ids) + ids_len2, ids_len)
            # test truthy condition (isaval on bool col returns True):
            selection = 'vs30_measured == vs30_measured'
            ids = read_where(table, selection)
            self.assertEqual(len(ids), total)
            # test with limit exceeding the available records (should get
            # all records as if limit was not given):
            ids = read_where(table, selection, total+1)
            self.assertEqual(len(ids), total)
            # records_where should get the same results as read_where:
            ids = [r['record_id'] for r in records_where(table, selection)]
            self.assertEqual(len(ids), total)
            # test falsy condition (isaval on bool col returns True):
            ids = read_where(table, "~(%s)" % selection)
            self.assertEqual(len(ids), 0)
            ids = read_where(table, selection, total+1)
            self.assertEqual(len(ids), total)
            ids = [r['record_id'] for r in records_where(table,
                                                         "~(%s)" % selection)]
            self.assertEqual(len(ids), 0)


    def test_normalize_condition(self):
        ''' test the _normalize_condition function'''
        # these are ok because logical operators relative position is not ok
        with self.assertRaises(ValueError):
            _normalize_condition("& (")
        with self.assertRaises(ValueError):
            _normalize_condition("& (")
        with self.assertRaises(ValueError):
            _normalize_condition(" & ")
        with self.assertRaises(ValueError):
            _normalize_condition("& (abc)")
        with self.assertRaises(ValueError):
            _normalize_condition(") & (abc) | cfd")
        with self.assertRaises(ValueError):
            _normalize_condition(") & (abc) | (cdf) ~")
        with self.assertRaises(ValueError):
            _normalize_condition(") & (abc) | (cdf) ~ ~ ~")

        # these are ok because operators relative position is ok.
        # Strings are not valid python expression, but
        # _normalize_condition is NOT a parser
        _normalize_condition(") & (abc)")
        _normalize_condition(") & (abc) | (cdf) ~ ~ ~ (")
        _normalize_condition(") & (\"a\\\"&c\") | (cdf) ~ ~ ~ (")
        # assert that we did not replace anything as the operator '='
        # is not recognized as valid:
        self.assertEqual(_normalize_condition("pga = nan"), "pga = nan")
        # same as above, but because pry is not recognized as column:
        self.assertEqual(_normalize_condition("pry = nan"), "pry = nan")
        # test minor stuff: leading spaces preserved, trailing not:
        self.assertEqual(_normalize_condition("(pkw != 0.5) "),
                         "(pkw != 0.5)")
        self.assertEqual(_normalize_condition(" (pkw != 0.5)"),
                         " (pkw != 0.5)")
        self.assertEqual(_normalize_condition(" (pkw != 0.5) "),
                         " (pkw != 0.5)")

        # set a series of types and the values you want to test:
        # for ints, supply also a float, as _normalize_condition
        # should not do this kind of conversion
        values = {str: ['2006-01-01 01:02:03',
                        # "b'2006-01-01 01:02:03'",
                        '2006-01-01 01:02:03',
                        '2006-01-01',
                        '2006'],
                  float: ["5", "0.5", "nan"],
                  int: ["5", "6.5"],
                  bool: ["True"]}

        for key, vals in values.items():
            for val in vals:
                for opr in ("==", "!=",  "<", ">", "<=", ">="):
                    if key == str:
                        cond = 'event_country %s "%s"' % (opr, val)
                        expected = cond
                        if self.py3:
                            expected = 'event_country %s b\'%s\'' % (opr, val)
                        self.assertEqual(_normalize_condition(cond), expected)
                        cond = 'event_time %s "%s"' % (opr, val)
                        expected_val = str(GMTableParser.timestamp(val))
                        expected = 'event_time %s %s' % (opr, expected_val)
                        self.assertEqual(_normalize_condition(cond), expected)
                        # now these cases should raise:
                        for col in ['pga', 'vs30_measured', 'npass']:
                            cond = '%s %s "%s"' % (col, opr, val)
                            with self.assertRaises(ValueError):
                                _normalize_condition(cond)
                    elif cond == float:
                        cond = 'pga %s %s' % (opr, val)
                        self.assertEqual(_normalize_condition(cond), cond)
                        # now these cases should raise:
                        for col in ['event_country', 'event_time',
                                    'vs30_measured', 'npass']:
                            cond = '%s %s "%s"' % (col, opr, val)
                            with self.assertRaises(ValueError):
                                _normalize_condition(cond)
                    elif cond == bool:
                        cond = 'vs30_measured %s %s' % (opr, val)
                        self.assertEqual(_normalize_condition(cond), cond)
                        # now these cases should raise:
                        for col in ['event_country', 'event_time', 'pga',
                                    'npass']:
                            cond = '%s %s "%s"' % (col, opr, val)
                            with self.assertRaises(ValueError):
                                _normalize_condition(cond)
                    elif cond == int:
                        cond = 'npass %s %s' % (opr, val)
                        self.assertEqual(_normalize_condition(cond), cond)
                        # now these cases should raise:
                        for col in ['event_country', 'event_time', 'pga',
                                    'vs30_measured']:
                            cond = '%s %s "%s"' % (col, opr, val)
                            with self.assertRaises(ValueError):
                                _normalize_condition(cond)

        # check nan conversion (not string). Note trailing spaces stripped
        cond = "(((pga == 'nan')) | ( pga != nan)  "
        #  'pga' values must be floats or nan:
        with self.assertRaises(ValueError):
            _normalize_condition(cond)
        # Parse both nans Note trailing spaces stripped
        cond = "(((pga == nan)) | ( pga != nan)  "
        self.assertEqual("(((pga != pga)) | ( pga == pga)",
                         _normalize_condition(cond))
        # test bytes stuff
        # for datetimes:
        for test in ['2006', '2006-12-31', '2016-12-31 00:01:02',
                     '2016-12-31T01:02:03']:
            cond = 'event_time == "%s"' % test
            expected_val = str(GMTableParser.timestamp(test))
            expected = 'event_time == %s' % expected_val
            self.assertEqual(expected, _normalize_condition(cond))
            cond = 'event_time == b"%s"' % test
            self.assertEqual(expected, _normalize_condition(cond))
        # for strings:
#         test = "å"
#         if self.py3:
#             byte, text = test.encode('utf8'), test
#         else:
#             byte, text = test, test.decode('utf8')
#         for test in [byte, text]:
#             cond = 'event_time == "%s"' % str(test)
#             expected_val = GMTableParser.normalize_dtime(test)
#             expected = 'event_time == b\'%s\'' % expected_val if self.py3\
#                  else 'event_time == \'%s\'' % expected_val
#             self.assertEqual(expected, _normalize_condition(cond))
#             cond = 'event_time == b"%s"' % test
#             self.assertEqual(expected, _normalize_condition(cond))

    def test_esm_flatfile(self):
        input_file = os.path.join(os.path.dirname(self.input_file),
                                  'esm_sa_flatfile_2018.csv')
        log = EsmParser.parse(input_file,
                              output_path=self.output_file)
        self.assertEqual(log['total'], 98)
        self.assertEqual(log['written'], 98)
        missingvals = log['missing_values']
        self.assertTrue(missingvals['rjb'] == missingvals['rrup'] ==
                        missingvals['rupture_length'] ==
                        missingvals['ry0'] == missingvals['rx'] ==
                        missingvals['strike_1'] == missingvals['dip_1'] ==
                        missingvals['rupture_width'] == 97)
        self.assertEqual(missingvals['_duration_5_75_components'], 98)
        self.assertEqual(missingvals['duration_5_75'], 98)
        self.assertTrue(missingvals['magnitude'] == 
                        missingvals['magnitude_type'] == 13)

        gmdb = GroundMotionTable(self.output_file, 'esm_sa_flatfile_2018')

        gmdb2 = gmdb.filter('magnitude <= 4')
        # underlying HDF5 file not open (ValueError):
        with self.assertRaises(ValueError):
            for rec in gmdb2.records:
                rec
        # check that we correctly wrote default attrs:
        with gmdb2:
            tbl = gmdb2.table.attrs
            self.assertTrue(isinstance(tbl.stats, dict))
            self.assertEqual(tbl.filename, 'template_basic_flatfile.hd5')
            self.assertEqual(len(gmdb2.attrnames()), 4)

        # now it works:
        with gmdb2:
            mag_le_4 = 0
            for rec in gmdb2.records:
                self.assertTrue(rec['magnitude'] <= 4)
                mag_le_4 += 1

        gmdb2 = gmdb.filter('magnitude > 4')
        with gmdb2:
            mag_gt_4 = 0
            for rec in gmdb2.records:
                self.assertTrue(rec['magnitude'] > 4)
                mag_gt_4 += 1

        self.assertTrue(mag_le_4 + mag_gt_4 == 98 - 13)

        # just open and set some selections to check it
        with GroundMotionTable(self.output_file, 'esm_sa_flatfile_2018') as gmdb:
            table = gmdb.table
            total = table.nrows
            gmdb.filter

# def get_interpolated_periods(target_periods, periods, values, sort=True,
#                              check_bounds=True):
#     """
#     Returns the spectra interpolated in loglog space
#     :param float target_period:
#         Period required for interpolation
#     :param np.ndarray periods:
#         Spectral Periods
#     :param np.ndarray values:
#         Ground motion values
#     """
#     target_periods, periods, values = np.asarray(target_periods),\
#         np.asarray(periods), np.asarray(values)
# 
#     if check_bounds:
#         pmin, pmax = (target_periods, target_periods) \
#             if target_periods.ndim == 0 \
#             else (target_periods[0], target_periods[-1])
#         if pmin < periods[0] or pmax > periods[-1]:
#             raise ValueError("Period not within calculated range")
# 
#     return np.interp(target_periods, periods, values)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
