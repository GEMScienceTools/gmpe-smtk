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
Tests the IO of a database between pickle and json
"""
import os
# import shutil
import sys
from datetime import datetime
# import json
# import pprint
import unittest
import numpy as np
# import smtk.gm_database
from tables.file import open_file
from tables.description import Float32Col, Col, IsDescription, Float64Col, \
    StringCol, EnumCol

from smtk.gm_database import expr, eq, ne, lt, gt, le, ge, \
    GMDatabaseParser, between, isaval, GMDatabaseTable, records_where, get_table,\
    read_where

BASE_DATA_PATH = os.path.join(
    os.path.join(os.path.dirname(__file__), "file_samples")
    )


class DummyTable(IsDescription):
    '''dummy table class used for testing pytables'''
    floatcol = Float32Col()
    arraycol = Float32Col(shape=(10,))
    stringcol = StringCol(5)
#    ballColor = EnumCol(['orange'], 'black', base='uint8')


class GmDatabaseTestCase(unittest.TestCase):
    '''tests Gm database parser and selection'''
    @classmethod
    def setUpClass(cls):

        cls.input_file = os.path.join(BASE_DATA_PATH,
                                      "template_basic_flatfile.csv")
        cls.output_file = os.path.join(BASE_DATA_PATH,
                                       "template_basic_flatfile.hd5")

    def setUp(self):
        self.deletefile()

    def tearDown(self):
        self.deletefile()

    @classmethod
    def deletefile(cls):
        '''deletes tmp hdf5 file'''
        if os.path.isfile(cls.output_file):
            os.remove(cls.output_file)

    def test_pytables(self):
        '''Test some pytable casting stuff NOT clearly documented :( '''

        with open_file(self.output_file, 'a') as h5file:
            table = h5file.create_table("/",
                                        'table',
                                        description=DummyTable)
            row = table.row

            # define set and get to avoid typing pylint flase positives
            # everywhere:
            def set(field, value):  # @ReservedAssignment
                '''sets a value on the row'''
                row[field] = value  # pylint: disable=unsupported-assignment-operation

            def get(field):
                '''gets a value from the row'''
                return row[field]  # pylint: disable=unsubscriptable-object

            # assert the value is the default:
            self.assertEqual(get('floatcol'), 0)
            # what if we supply a string? TypeError
            with self.assertRaises(TypeError):
                set('floatcol', 'a')
            # assert the value is still the default:
            self.assertEqual(get('floatcol'), 0)
            # what if we supply a castable string instead? it is casted
            set('floatcol', '5.5')
            self.assertEqual(get('floatcol'), 5.5)
            # what if we supply a scalr instead of an array?
            # the value is broadcasted:
            set('arraycol', 5)
            self.assertTrue(np.allclose([5] * 10, get('arraycol')))
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
            set('stringcol', "abc")
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
            with GMDatabaseParser.get_table(self.output_file + 'what',
                                            name='whatever', mode='r') as tbl:
                pass

        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)
        dbname = os.path.splitext(os.path.basename(self.output_file))[0]
        # the flatfile parsed has:
        # 1. an event latitude out of bound (row 0)
        # 2. an event longitude out of bound (row 1)
        # 3. a pga with extremely high value (row 2)
        # 4. a sa[0] with extremely high value (row 3)

        total = log['total']
        written = total - 2  # row 2 and 3 not written
        self.assertTrue(log['total'] == 99)
        self.assertTrue(log['written'] == written)
        self.assertTrue(sorted(log['error']) == [2, 3])  # pga sa[0] mismatch, skipped
        self.assertTrue(len(log['outofbound_values']) == 2)  # rows 0 and 1
        self.assertTrue(log['outofbound_values']['event_latitude'] == 1)  # row 0
        self.assertTrue(log['outofbound_values']['event_longitude'] == 1)  # row 1
        self.assertTrue(log['missing_values']['pga'] == 0)
        self.assertTrue(log['missing_values']['pgv'] == log['written'])
        self.assertTrue(log['missing_values']['pgv'] == log['written'])

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
        with GMDatabaseParser.get_table(self.output_file, dbname, 'a') as tbl:
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
        with GMDatabaseParser.get_table(self.output_file, dbname, 'r') as tbl:
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
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)
        # open HDF5 with append='a' (the default)
        # and check that wewrote stuff twice
        with GMDatabaseParser.get_table(self.output_file, dbname, 'r') as tbl:
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
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     mode='w')
        with GMDatabaseParser.get_table(self.output_file, dbname, 'r') as tbl:
            self.assertTrue(tbl.nrows == written)
            # assert the old rows are not there anymore
            oldrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_oldval)))
            self.assertTrue(len(oldrows) == test_cols_found)
            # assert the new rows are added:
            newrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_newval)))
            self.assertTrue(not newrows)

    def test_expr(self):
        '''tests gm database `expr` class and subclasses'''
        exp = expr('pga', '!=', np.nan) & expr('pgv', '==', 6.5)
        self.assertTrue(str(exp) == "(pga == pga) & (pgv == 6.5)")
        for dtime in [datetime(2016, 1, 31), '2016-01-31']:
            exp = expr('event_time', '<=', dtime)
            self.assertTrue(str(exp) == "event_time <= b'2016-01-31T00:00:00'")
        # try with string and bytes str, supplying quotes and special chars
        for ecn in ['itå"\'ly'.encode('utf8'), 'itå"\'ly']:
            exp = expr('event_country', '>', ecn)
            self.assertTrue(str(exp) ==
                            "event_country > b'it\\xc3\\xa5\"\\'ly'")
        # test boolean and combined expression
        exp = expr('vs30_measured', '==', 'True') & \
            expr('event_time', '<=', '2016-01-01')
        self.assertTrue(str(exp) == str(~~exp) ==
                        ("(vs30_measured == True) & "
                         "(event_time <= b'2016-01-01T00:00:00')"))
#         exp = ~~exp
#         self.assertTrue(str(exp) ==
#                         ("(vs30_measured == True) & "
#                          "(event_time <= b'2016-01-01T00:00:00')"))
        exp |= eq('pga', 9.85)
        self.assertTrue(str(exp) ==
                        ("((vs30_measured == True) & "
                         "(event_time <= b'2016-01-01T00:00:00')) | "
                         "(pga == 9.85)"))
        self.assertTrue(str(~exp) ==
                        ("((vs30_measured != True) | "
                         "(event_time > b'2016-01-01T00:00:00')) & "
                         "(pga != 9.85)"))
        # test simple operator expressions:
        for func, symbol in zip([eq, ne, lt, gt, le, ge],
                                ['==', '!=', '<', '>', '<=', '>=']):
            # assert that with event_time we cannot cast to datetime:
            with self.assertRaises(TypeError):
                exp = func('event_time', 0.75)
                self.assertTrue(str(exp) == 'event_time %s 0.75' % symbol)
            # with string columns it does not raise, and converts to str:
            exp = func('event_country', 0.75)
            self.assertTrue(str(exp) == "event_country %s b'0.75'" % symbol)
            # it works, obviously, with float columns
            exp = func('pgv', 0.75)
            self.assertTrue(str(exp) == 'pgv %s 0.75' % symbol)
            # what if with numeric we provide a castable string?: it casts:
            exp = func('pgv', "0.75")
            self.assertTrue(str(exp) == 'pgv %s 0.75' % symbol)
            # what if with numeric we provide a non castable string?: raises:
            with self.assertRaises(ValueError):
                exp = func('pgv', "abc")
        # what if with numeric we provide 'nan'? it casts:
        for nan in ['nan', np.nan, float('nan')]:
            exp = eq('pgv', nan)
            self.assertTrue(str(exp) == 'pgv != pgv')
            exp = ne('pgv', nan)
            self.assertTrue(str(exp) == 'pgv == pgv')
            for func in [lt, le, gt, ge]:
                with self.assertRaises(ValueError):
                    exp = func('pgv', "nan")
        # test edge cases:
        self.assertTrue(expr(True) == expr('True') == expr('true') ==
                        ~expr(False))
        self.assertTrue(expr(False) == expr('False') == expr('false') ==
                        ~expr(True))
        with self.assertRaises(ValueError):
            expr('')
        with self.assertRaises(ValueError):
            expr(None)
        # test builtin expressions:
        # test isin:
        exp1 = eq('pga', 0.5, 0.75)
        self.assertTrue(str(exp1) == '(pga == 0.5) | (pga == 0.75)')
        exp2 = ~eq('pga', 0.5, 0.75)
        self.assertTrue(str(exp2) == '(pga != 0.5) & (pga != 0.75)')
        exp2 = ne('pga', 0.5, 0.75)
        self.assertTrue(str(exp1) == str(~exp2))
        self.assertTrue(str(exp2) == str(~exp1))
        with self.assertRaises(TypeError):
            exp = eq('pga')
        # test with more args:
        exp1 = eq('pga', 0.5, 0.75, 1)
        self.assertTrue(str(exp1) == ('(pga == 0.5) | (pga == 0.75) | '
                                      '(pga == 1.0)'))
        exp2 = ne('pga', 0.5, 0.75, 1)
        self.assertTrue(str(exp2) == ('(pga != 0.5) & (pga != 0.75) & '
                                      '(pga != 1.0)'))
        self.assertEqual(~exp1, exp2)
        self.assertEqual(exp1, ~exp2)
        # test between:
        exp = between('pga', 0.5, 0.75)
        self.assertTrue(str(exp) == '(pga >= 0.5) & (pga <= 0.75)')
        exp = ~between('pga', 0.5, 0.75)
        self.assertTrue(str(exp) == '(pga < 0.5) | (pga > 0.75)')
        # between with no args
        with self.assertRaises(TypeError):
            exp = between('pga')  # pylint: disable=no-value-for-parameter
        # test is aval:
        exp = isaval('pga')
        self.assertTrue(str(exp) == 'pga == pga')
        exp = ~isaval('pga')
        self.assertTrue(str(exp) == 'pga != pga')
        exp = isaval('vs30_measured')
        # with booleans, defaults do not mean "missing", so it depends on the
        # default provided. For bool cols, e.g. vs30_measured, the return
        # expression is either True or False:
        exp = isaval('vs30_measured')
        self.assertTrue(str(exp) == 'True')
        exp = ~isaval('vs30_measured')
        self.assertTrue(str(exp) == 'False')
        exp = isaval('event_country')
        self.assertTrue(str(exp) == "event_country != b''")
        exp = ~isaval('event_country')
        self.assertTrue(str(exp) == "event_country == b''")

    def test_template_basic_file_selection(self):
        '''parses a sample flatfile and tests some selection syntax on it'''
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)
        dbname = os.path.splitext(os.path.basename(self.output_file))[0]
        with get_table(self.output_file, dbname) as table:
            total = table.nrows
            selection = le('pga', 100.75)
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
            ids = [r['record_id'] for r in records_where(table, ~selection)]
            self.assertEqual(len(ids), total - ids_len)
            # same should happend for read_where:
            ids = [r['record_id'] for r in read_where(table, ~selection)]
            self.assertEqual(len(ids), total - ids_len)
            # test with limit 0 (expected: no record yielded):
            ids = [r['record_id'] for r in records_where(table, ~selection, 0)]
            self.assertEqual(len(ids), 0)
            # restrict the search:
            # note that we must pass strings to event_time,
            # either 1935-01-01, 1935-01-01T00:00:00, or simply the year:
            selection2 = selection & between('event_time', '1935', '1936')
            ids = [r['record_id'] for r in records_where(table, selection2)]
            ids_len2 = len(ids)
            # test that the search was restricted:
            self.assertTrue(ids_len2 < ids_len)
            # now negate the serarch on event_time and test that we get all
            # remaining records:
            selection2 = selection & ~between('event_time', '1935', '1936')
            ids = [r['record_id'] for r in records_where(table, selection2)]
            self.assertEqual(len(ids) + ids_len2, ids_len)
            # test truthy condition (isaval on bool col returns True):
            selection = isaval('vs30_measured')
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
            ids = read_where(table, ~selection)
            self.assertEqual(len(ids), 0)
            ids = read_where(table, selection, total+1)
            self.assertEqual(len(ids), total)
            ids = [r['record_id'] for r in records_where(table, ~selection)]
            self.assertEqual(len(ids), 0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
