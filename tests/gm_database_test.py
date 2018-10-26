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

        g = GMDatabaseParser.parse(self.input_file,
                                     output_path=self.output_file,
                                     return_log_dict=True)
        h = 9



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
