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
from tables.file import open_file
from tables.description import Float32Col, Col, IsDescription, Float64Col, StringCol,\
    EnumCol
"""
Tests the IO of a database between pickle and json
"""
import os
import shutil
import sys
import json
import pprint
import unittest
from smtk import load_database
import numpy as np
from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser
import smtk.gm_database

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle


BASE_DATA_PATH = os.path.join(
    os.path.join(os.path.dirname(__file__), "file_samples")
    )

class GMDatabaseParser(smtk.gm_database.GMDatabaseParser):
    '''Tests the GmDatabaseParser, allowing to override the parserow method
    if needed ij the future'''

    @classmethod
    def parse_row(cls, rowdict):
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
        pass

class DummyTable(IsDescription):
    floatcol = Float32Col()
    floatarray = Float32Col(shape=(10,))
#    ballColor = EnumCol(['orange'], 'black', base='uint8')

class GmDatabaseTestCase(unittest.TestCase):

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
        if os.path.isfile(cls.output_file):
            os.remove(cls.output_file)

    def testPyTable(self):

        
        
        '''Test some pytable casting stuff NOT clearly documented :( '''
        with open_file(self.output_file, 'a') as h5file:
            table = h5file.create_table("/",
                                        'table',
                                        description=DummyTable)
            row = table.row
            # assert the value is the default:
            assert row['floatcol'] == 0
            # what if we supply a string? TypeError
            with self.assertRaises(TypeError):
                row['floatcol'] = 'a'
            # assert the value is still the default:
            assert row['floatcol'] == 0
            # what if we supply a castable string instead? it is casted
            row['floatcol'] = '5.5'
            assert row['floatcol'] == 5.5
            # what if we supply a scalr instead of an array?
            # the value is broadcasted:
            row['floatarray'] = 5
            assert np.allclose([5] * 10, row['floatarray'])
            # what if we supply a float out of bound? no error
            # but value is saved differently!
            maxfloat32 = 3.4028235e+38
            val = maxfloat32 * 10
            row['floatcol'] = val
            h = row['floatcol']
            # assert they are not the same
            assert not np.isclose(h, val)
            # now restore val to the max Float32, and assert they are the same:
            val = maxfloat32
            row['floatcol'] = val
            h = row['floatcol']
            assert np.isclose(h, val)
            
            # write to the table nan and see if we can select it later:
            row['floatcol'] = float('nan')
            row.append()
            
        with open_file(self.output_file, 'a') as h5file:
            table = h5file.get_node("/", 'table')
            vals = [r['floatcol'] for r in 
                    table.where('floatcol != floatcol', condvars={'nan': float('nan')})]
            h = 9

    def test_template_basic_file(self):

        # test a file not found
        with self.assertRaises(IOError):
            with GMDatabaseParser.get_table(self.output_file + 'what',
                                            name='whatever', mode='r') as t:
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
        assert log['total'] == 99
        assert log['written'] == written
        assert sorted(log['error']) == [2, 3]  # pga sa[0] mismatch, skipped
        assert len(log['outofbound_values']) == 2  # rows 0 and 1
        assert log['outofbound_values']['event_latitude'] == 1  # row 0
        assert log['outofbound_values']['event_longitude'] == 1  # row 1
        assert log['missing_values']['pga'] == 0
        assert log['missing_values']['pgv'] == log['written']
        assert log['missing_values']['pgv'] == log['written']

        # assert auto generated ids are not missing:
        assert 'record_id' not in log['missing_values']
        assert 'event_id' not in log['missing_values']
        assert 'station_id' not in log['missing_values']

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
            assert len(ids) == written
            # assert we have some event shared across records:
            assert len(set(ids)) < written
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
            assert test_cols_found == 1

        # assert that we modified the event name
        with GMDatabaseParser.get_table(self.output_file, dbname, 'r') as tbl:
            for row in tbl.where('%s == %s' % (test_col, test_col_oldval)):
                # we should never be here (no row with the old value):
                assert False

            count = 0
            for row in tbl.where('%s == %s' % (test_col, test_col_newval)):
                count += 1
            assert count == test_cols_found

        # now re-write, with append mode
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)
        # open HDF5 with append='a' (the default)
        # and check that wewrote stuff twice
        with GMDatabaseParser.get_table(self.output_file, dbname, 'r') as tbl:
            assert tbl.nrows == written * 2
            # assert the old rows are there
            oldrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_oldval)))
            assert len(oldrows) == test_cols_found
            # assert the new rows are added:
            newrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_newval)))
            assert len(newrows) == test_cols_found

        # now re-write, with no mode='w'
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     mode='w')
        with GMDatabaseParser.get_table(self.output_file, dbname, 'r') as tbl:
            assert tbl.nrows == written
            # assert the old rows are not there anymore
            oldrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_oldval)))
            assert len(oldrows) == test_cols_found
            # assert the new rows are added:
            newrows = list(row[test_col] for row in
                           tbl.where('%s == %s' % (test_col, test_col_newval)))
            assert not newrows


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
