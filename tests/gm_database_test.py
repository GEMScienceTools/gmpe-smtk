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
from tables.description import Float32Col, Col, IsDescription, Float64Col
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
        evtime = cls.datetime(rowdict.pop('year'),
                              rowdict.pop('month'),
                              rowdict.pop('day'),
                              rowdict.pop('hour', None) or 0,
                              rowdict.pop('minute', None) or 0,
                              rowdict.pop('second', None) or 0)
        rowdict['event_time'] = evtime

class DummyTable(IsDescription):
    floatcol = Float32Col()
    floatarray = Float32Col(shape=(10,))

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
            

    def test_template_basic_file(self):

        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)
        groupname = os.path.splitext(os.path.basename(self.output_file))[0]
        assert log['total'] == log['written'] == 99
        assert log['missing_values']['pga'] == 99
        assert log['missing_values']['pgv'] == 99
        assert 'record_id' not in log['missing_values']

        # PYTABLES. IMPORTANT
        # seems that this is NOT possible:
        # list(table.iterrows())  # returns N times the LAST row
        # seems also that we should NOT break inside a iterrows or where loop

        # open HDF5 and check for incremental ids:
        row_id = b'1'
        with open_file(self.output_file, 'a') as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            ids = list(r['record_id'] for r in table.iterrows())
            assert len(ids) == len(set(ids))
            # assert max(ids) == 99

            # modify one row
            count = 0
            for row in table.where('record_id == %s' % row_id):
                count += 1
                modified_row_event_name = row['event_name']
                row['event_name'] = 'dummy'
                row.update()
            table.flush()
            assert count == 1

        # assert that we modified the event name
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            for row in table.where('record_id == %s' % row_id):
                assert row['event_name'] == b'dummy'

        # now re-write, with updated mode:
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)

        # open HDF5 and check that we updated the value:
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            assert table.nrows == 99
            row = list(row['event_name'] for row in
                       table.where('record_id == %s' % row_id))
            assert len(row) == 1
            assert row[0] == modified_row_event_name

        # now re-write, with append='a':
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     mode='a')
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            assert table.nrows == 99*2

        # now re-write, with no append='w'
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     mode='w')
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            assert table.nrows == 99



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
