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
Tests for execution of ESM Flatfile Parser
"""
import os
import sys
import shutil
import unittest
from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

# Defines the record IDs for the target data set
TARGET_IDS = ["AM_1988_0001_A_GUK_0", "AM_1988_0002_A_GUK_0",
"AM_1988_0004_A_LEN_0", "AM_1988_0004_A_NAB_0", "AM_1988_0004_A_STRS_0",
"AM_1988_0005_A_DZHN_0", "AM_1988_0005_A_LEN_0", "AM_1988_0005_A_MET_0",
"AM_1988_0005_A_NAB_0", "AM_1988_0005_A_STPV_0", "AM_1988_0005_A_STRS_0",
"AM_1989_0006_A_MET_0", "AM_1989_0006_A_NAB_0", "AM_1989_0006_A_STPV_0",
"AM_1989_0006_A_STRS_0", "AM_1989_0008_A_STRS_0", "AM_1989_0009_A_NAB_0",
"AM_1990_0013_A_SAKH_0", "AM_1990_0013_A_SBGD_0", "AM_1990_0013_A_SBKR_0",
"AM_1990_0013_A_SBVR_0", "AM_1990_0013_A_SPIK_0", "AM_1990_0013_A_STPV_0",
"AM_1990_0013_A_STRS_0", "AM_1990_0013_A_SVNZ_0", "AT_1996_0001_OE_RWNA_0",
"AT_1996_0001_OE_VIE1_0", "AT_1996_0001_OE_VIE3_0", "AT_1996_0001_OE_VIE5_0",
"AT_1996_0001_OE_WRN2_0", "DE_1992_0010_LE_BAW_0", "DE_1992_0010_LE_BFO_0",
"DE_1992_0010_LE_DOS_0", "DE_1992_0010_LE_EFR_0", "DE_1992_0010_LE_END_0",
"DE_1992_0010_LE_GLO_0", "DE_1992_0010_LE_HEX_0", "DE_1992_0010_LE_KIR_0",
"DE_1992_0010_LE_KRE_0", "DE_1992_0010_LE_SLB_0", "DE_1992_0010_LE_SOL_0",
"DE_1992_0010_LE_STA_0", "DE_1992_0010_LE_WYH_0", "DZ_1980_0016_EU_BRS_0",
"DZ_1989_0023_FC_ALG_0", "EMSC_19980224_0000009_HI_ROD1_0",
"EMSC_19980224_0000009_HI_ROD2_0", "EMSC_19980224_0000009_HI_ROD3_0",
"EMSC_19980224_0000009_HI_ROD4_0", "EMSC_19980423_0000011_HL_LXRA_0",
"EMSC_19980501_0000001_HI_GTH1_0", "EMSC_19980716_0000001_HI_LEF1_0",
"EMSC_19980716_0000001_HL_LEFA_0", "EMSC_19981006_0000006_HI_KYP1_0",
"EMSC_19981008_0000001_HI_ZAK1_0", "EMSC_19981008_0000001_HL_ZAKA_0",
"EMSC_19981122_0000005_HL_ARGA_0", "EMSC_19981122_0000005_HL_LXRA_0",
"EMSC_19990202_0000009_HI_PAT1_0", "EMSC_19990202_0000009_HI_PAT2_0",
"EMSC_19990202_0000009_HL_PATB_0", "EMSC_19990314_0000005_HL_ZAKA_0",
"EMSC_19990406_0000004_HL_ZAKA_0", "EMSC_19990605_0000004_HL_AIGA_0",
"EMSC_19990611_0000011_HI_ZAK1_0", "EMSC_19990611_0000011_HL_ARGA_0",
"EMSC_19990611_0000011_HL_LEFA_0", "EMSC_19990611_0000011_HL_LXRA_0",
"EMSC_19990611_0000011_HL_ZAKA_0", "EMSC_19990629_0000011_HI_AIG1_0",
"EMSC_19990629_0000011_HL_AIGA_0", "EMSC_19990629_0000011_HL_PATB_0",
"EMSC_19990907_0000020_HI_THV1_0", "EMSC_19990907_0000020_HL_THVC_0",
"EMSC_19990907_0000055_HL_ATHA_0", "EMSC_19990908_0000027_HI_THV1_0",
"EMSC_19990908_0000027_HL_DEKA_0", "EMSC_19990908_0000027_HL_SPLB_0",
"EMSC_19991021_0000008_HI_AIG1_0", "EMSC_19991021_0000008_HI_PAT1_0",
"EMSC_19991021_0000008_HI_PAT2_0", "EMSC_19991021_0000008_HI_PAT3_0",
"EMSC_19991021_0000008_HL_PATB_0", "EMSC_19991104_0000001_HI_PYR1_0",
"EMSC_19991212_0000012_HI_IER1_0", "EMSC_19991219_0000012_HL_ARGA_0",
"EMSC_19991219_0000012_HL_LXRA_0", "EMSC_19991226_0000012_HI_ZAK1_0",
"EMSC_19991226_0000012_HL_ZAKA_0", "EMSC_20000627_0000002_FR_SMPL_0",
"EMSC_20010206_0000009_RA_STET_0", "EMSC_20010225_0000008_FR_ARBF_0",
"EMSC_20010225_0000008_FR_CALF_0", "EMSC_20010225_0000008_FR_SAOF_0",
"EMSC_20010225_0000008_RA_RUSF_0", "EMSC_20010225_0000008_RA_STET_0",
"EMSC_20010718_0000012_RA_STET_0", "EMSC_20030124_0000011_IU_PAB_10",
"EMSC_20030222_0000013_CH_MMK_0", "EMSC_20030222_0000013_CH_SENIN_0"]

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class ESMFlatfileParserTestCase(unittest.TestCase):
    """
    Tests the parsing of the ESM flatfile
    """
    @classmethod
    def setUpClass(cls):
        cls.datafile = os.path.join(BASE_DATA_PATH,
                                    "esm_flatfile_sample_file.csv")
        cls.db_file = os.path.join(BASE_DATA_PATH, "esm_flatfile_test")

    def test_esm_flatfile_parser(self):
        """
        Tests the parsing of the ESM flatfile
        """
        parser = ESMFlatfileParser.autobuild("000", "ESM Test",
                                             self.db_file, self.datafile)
        with open(os.path.join(self.db_file, "metadatafile.pkl"), "rb") as f:
            db = pickle.load(f)
        # Should contain 100 records
        self.assertEqual(len(db), 100)
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
