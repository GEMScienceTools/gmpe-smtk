#!/usr/bin/env python
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
Tests for execution of Conditional Simulation tools
"""
import unittest
import os
import shutil
import smtk.hazard.conditional_simulation as csim
import smtk.sm_database_builder as sdb
from smtk.sm_utils import load_pickle
from smtk.residuals.gmpe_residuals import Residuals
from smtk.parsers.sigma_database_parser import (SigmaDatabaseMetadataReader,
                                                SigmaRecordParser,
                                                SigmaSpectraParser)

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class ConditionalSimulationTestCase(unittest.TestCase):
    """
    Test suite for the conditional simulation

    Currently only testing execution - not correctness of values
    """
    @classmethod
    def setUpClass(cls):
        """
        Import the database and the rupture before tests
        """
        input_db = os.path.join(BASE_DATA_PATH, "LAquila_Database")
        input_dir = os.path.join(BASE_DATA_PATH, "LAquila_Good_Records")
        # Build the record database using the L'Aquila subset
        builder = sdb.SMDatabaseBuilder(SigmaDatabaseMetadataReader,
                                        input_db)
        builder.build_database("001", "LAquila Mainshock", input_dir)
        builder.parse_records(SigmaRecordParser, SigmaSpectraParser)
        sdb.add_horizontal_im(builder.database, ["PGA", "PGV", "Geometric"])

        # Load in the data from the database
        cls.db = load_pickle(os.path.join(input_db, "metadatafile.pkl"))
        input_rupture_file = os.path.join(BASE_DATA_PATH,
                                          "laquila_rupture.xml")
        cls.rupture = csim.build_rupture_from_file(input_rupture_file)
        cls.gsims = ["AkkarEtAlRjb2014"]
        cls.imts = ["PGA", "SA(1.0)"]
        for rec in cls.db:
            rec.datafile = os.path.join(os.path.dirname(__file__),
                                        rec.datafile)
            print(rec.datafile)
        # Generate the residuals
        cls.residuals = Residuals(cls.gsims, cls.imts)
        cls.residuals.get_residuals(cls.db)

    def test_generation_residual_fields(self):
        # Executes the site collection
        observed_sites = self.db.get_site_collection()
        self.assertEqual(len(observed_sites), 13)
        # Get the target sites
        limits = [12.5, 15.0, 0.05, 40.5, 43.0, 0.05]
        vs30 = 800.0
        unknown_sites = csim.get_regular_site_collection(limits, vs30)
        # Generate a conditional field of residuals
        pga_residuals = (self.residuals.residuals["AkkarEtAlRjb2014"]
                         ["PGA"]["Intra event"])
        sa1_residuals = (self.residuals.residuals["AkkarEtAlRjb2014"]
                         ["SA(1.0)"]["Intra event"])
        csim.conditional_simulation(observed_sites,
                                    pga_residuals,
                                    unknown_sites, "PGA", 1)
        csim.conditional_simulation(observed_sites,
                                    sa1_residuals,
                                    unknown_sites, "SA(1.0)", 1)

    def tests_generation_gmfs(self):
        # Tests the generation of the full ground motion fields
        limits = [12.5, 15.0, 0.05, 40.5, 43.0, 0.05]
        vs30 = 800.0
        unknown_sites = csim.get_regular_site_collection(limits, vs30)
        gmfs = csim.get_conditional_gmfs(self.db,
                                         self.rupture,
                                         sites=unknown_sites,
                                         gsims=self.gsims,
                                         imts=self.imts,
                                         number_simulations=5,
                                         truncation_level=3.0)
        self.assertEqual(gmfs["AkkarEtAlRjb2014"]["PGA"].shape[1], 5)
        self.assertEqual(gmfs["AkkarEtAlRjb2014"]["SA(1.0)"].shape[1], 5)

    @classmethod
    def tearDownClass(cls):
        """
        Delete the temporary record database
        """
        shutil.rmtree(os.path.join(BASE_DATA_PATH, "LAquila_Database"))
        cls.db = None
