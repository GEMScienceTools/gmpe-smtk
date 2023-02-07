#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation and G. Weatherill
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
Tests parsing of the ESM23 flatfile format (i.e. flatfile downloaded from custom header HTML) in SMTK

This parser assumes you have selected all available headers in your URL search when downloading the flatfile
"""
import os
import sys
import shutil
import unittest
from smtk.parsers.esm23_flatfile_parser import ESM23FlatfileParser

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

# Defines the record IDs for the target data set
TARGET_IDS = [
"EMSC_20110702_0000061_GE_IMMV_",
"EMSC_20110728_0000038_GE_KARP_",
"EMSC_20110801_0000064_GE_KTHA_",
"EMSC_20110910_0000003_HI_CH01_",
"EMSC_20110910_0000021_HI_KSS1_",
"EMSC_20110915_0000035_HI_MYT1_",
"EMSC_20111025_0000141_HI_NAX1_",
"EMSC_20111105_0000125_HI_RDI1_",
"EMSC_20111221_0000032_HI_RGE1_"]

#Specify base directory
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

class ESM23FlatfileParserTestCase(unittest.TestCase):
    """
    Tests the parsing of the reformatted ESM23 flatfile
    """
    @classmethod
    def setUpClass(cls):
        cls.ESM23_flatfile_directory = os.path.join(BASE_DATA_PATH,"ESM23_Greece_test.csv")
        cls.db_file = os.path.join(BASE_DATA_PATH, "ESM23_conversion_test_metadata")       

    def test_esm23_flatfile_parser(self):
        """
        Tests the parsing of the reformatted ESM23 flatfile
        """
        parser = ESM23FlatfileParser.autobuild("000", "ESM23_conversion_test",
                                             self.db_file, self.ESM23_flatfile_directory)
        with open(os.path.join(self.db_file, "metadatafile.pkl"), "rb") as f:
            db = pickle.load(f)
        # Should contain 9 records
        self.assertEqual(len(db), 9)
        # Record IDs should be equal to the specified target IDs
        for rec in db:
            print(rec.id)
        self.assertListEqual([rec.id for rec in db], TARGET_IDS)
        del parser

    @classmethod
    def tearDownClass(cls):
        """
        Remove the database
        """
        shutil.rmtree(cls.db_file)