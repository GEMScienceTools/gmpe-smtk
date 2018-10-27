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
from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser

from smtk.gm_database import GMDatabaseParser

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle


BASE_DATA_PATH = os.path.join(
    os.path.join(os.path.dirname(__file__), "file_samples")
    )

class GmDatabaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.input_file = os.path.join(BASE_DATA_PATH,
                                      "template_basic_flatfile.csv")
        cls.output_file = os.path.join(BASE_DATA_PATH,
                                      "template_basic_flatfile.hd5")
        if os.path.isfile(cls.output_file):
            os.remove(cls.output_file)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.output_file):
            os.remove(cls.output_file)

    def test_json_io_roundtrip(self):

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
        row_id = 1
        with open_file(self.output_file, 'a') as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            ids = list(r['id'] for r in table.iterrows())
            assert len(ids) == len(set(ids))
            assert max(ids) == 99

            # modify one row
            count = 0
            for row in table.where('id == %d' % row_id):
                count += 1
                modified_row_event_name = row['event_name']
                row['event_name'] = 'dummy'
                row.update()
            table.flush()
            assert count == 1

        # assert that we modified the event name
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            for row in table.where('id == %d' % row_id):
                assert row['event_name'] == b'dummy'

        # now re-write, with append mode:
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file)

        # open HDF5 and check that we updated the value:
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            assert table.nrows == 99
            row = list(row['event_name'] for row in
                       table.where('id == %d' % row_id))
            assert len(row) == 1
            assert row[0] == modified_row_event_name

        # now re-write, with no col_id:
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     col_id=None)
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            assert table.nrows == 99*2

        # now re-write, with no append=False:
        log = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     append=False)
        with open_file(self.output_file) as h5file:
            table = h5file.get_node('/%s/table' % groupname)
            assert table.nrows == 99



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
