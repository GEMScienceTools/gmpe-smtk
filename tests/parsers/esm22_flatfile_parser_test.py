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
Tests parsing of the ESM22 flatfile format (i.e. flatfile downloaded from web service) in SMTK
"""
import os
import sys
import shutil
import unittest
from smtk.parsers.esm22_flatfile_parser import ESM22FlatfileParser

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

# Defines the record IDs for the target data set

TARGET_IDS = [
"EMSC_20161029_0000147_AC_DURR_0.0",
"EMSC_20160831_0000112_AC_DURR_0.0",
"EMSC_20191128_0000157_AC_DURR_0.0",
"EMSC_20150207_0000001_AC_DURR_0.0",
"EMSC_20170127_0000087_AC_DURR_0.0",
"EMSC_20191128_0000068_AC_DURR_0.0",
"EMSC_20151101_0000023_AC_DURR_0.0",
"EMSC_20190921_0000063_AC_DURR_0.0",
"EMSC_20151031_0000024_AC_DURR_0.0",
"EMSC_20161104_0000026_AC_DURR_0.0",
"EMSC_20170918_0000091_HI_VAS2_",
"EMSC_20161029_0000147_HI_VAS2_",
"EMSC_20140512_0000004_HI_VAS2_",
"EMSC_20140519_0000003_HI_VAS2_",
"EMSC_20210112_0000070_HI_VAS2_",
"EMSC_20150403_0000004_AC_TIR1_0.0",
"EMSC_20191128_0000157_AC_TIR1_0.0",
"EMSC_20140520_0000006_AC_TIR1_0.0",
"EMSC_20150207_0000001_AC_TIR1_0.0"]

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

class ESM22FlatfileParserTestCase(unittest.TestCase):
    """
    Tests the parsing of the reformatted ESM22 flatfile
    """
    @classmethod
    def setUpClass(cls):
        #Specify base directory
        cls.ESM22_flatfile_directory = os.path.join(BASE_DATA_PATH,"ESM22_Albania_filtered_test.csv")
        cls.db_file = os.path.join(BASE_DATA_PATH, "ESM22_conversion_test_metadata")    
        
    def test_esm22_flatfile_parser(self):
        """
        Tests the parsing of the reformatted ESM22 flatfile
        """
        parser = ESM22FlatfileParser.autobuild("000", "ESM22_conversion_test",
                                             self.db_file, self.ESM22_flatfile_directory)
        with open(os.path.join(self.db_file, "metadatafile.pkl"), "rb") as f:
            db = pickle.load(f)
        # Should contain 19 records
        self.assertEqual(len(db), 19)
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