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
from smtk.sm_database import load_database
from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

def compare_two_json_files(file1, file2):
    """
    Compares two json files by parsing them into strings
    """
    # Load both jsons
    with open(file1, "r") as f1:
        data1 = json.load(f1)
    with open(file2, "r") as f2:
        data2 = json.load(f2)
    # Print them as strings
    d1 = pprint.pformat(data1)
    d2 = pprint.pformat(data2)
    # cleanup whitespace
    d1 = d1.replace(" ", "")
    d2 = d2.replace(" ", "")
    n1, n2 = len(d1), len(d2)
    for i in range(max(n1, n2)):
        if not d1[i] == d2[i]:
            # Is a mismatch - print the surrounding info
            print(i, d1[(i - 20):(i+20)], d2[(i - 20):(i + 20)])
            return False
    return True


BASE_DATA_PATH = os.path.join(
    os.path.join(os.path.dirname(__file__), "parsers"), "data"
    )

class DatabaseLoadingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pkl_dir = "esm_as_pickle"
        #os.mkdir(cls.pkl_dir)
        cls.json_dir = "esm_as_json"
        os.mkdir(cls.json_dir)
        cls.input_file = os.path.join(BASE_DATA_PATH,
                                      "esm_flatfile_sample_file.csv")
        # Build the database
        _ = ESMFlatfileParser.autobuild("000", "DUMMY",
                                        cls.pkl_dir,
                                        cls.input_file)

    def test_json_io_roundtrip(self):
        # Load the db from pickle
        db = load_database(self.pkl_dir)
        # Export to json
        file1 = os.path.join(self.json_dir, "metadatafile.json")
        with open(file1, "w") as fi1:
            fi1.write(db.to_json())
        # Load from json
        db1 = load_database(self.json_dir)
        # Dump back to a json file
        file2 = os.path.join(self.json_dir, "test_dump.json")
        with open(file2, "w") as fi2:
            fi2.write(db1.to_json())
        # Now compare files
        self.assertTrue(compare_two_json_files(file1, file2))

    @classmethod
    def tearDownClass(cls):
        # Remove the directories
        shutil.rmtree(cls.pkl_dir)
        shutil.rmtree(cls.json_dir)
