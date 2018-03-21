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
Tests for generation of data for trellis plots
"""
import unittest
import os
import json
import numpy as np
import smtk.trellis.trellis_plots as trpl
import smtk.trellis.configure as rcfg


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class BaseTrellisTest(unittest.TestCase):
    """
    This core test is designed to run a series of trellis plot calculations
    and ensure compatibility with previously generated results
    """

    TEST_FILE = None

    def setUp(self):
        self.imts = ["PGA", "SA(0.2)", "SA(2.0)", "SA(3.0)"]
        self.periods = [0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                        0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
                        0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
                        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                        3.0, 4.001, 5.0, 7.5, 10.0]

        self.gsims = ["AkkarBommer2010", "CauzziFaccioli2008",
                      "ChiouYoungs2008", "ZhaoEtAl2006Asc", "AkkarEtAlRjb2014",
                      "BindiEtAl2014Rjb", "CauzziEtAl2014", "DerrasEtAl2014",
                      "AbrahamsonEtAl2014", "BooreEtAl2014", "ChiouYoungs2014",
                      "CampbellBozorgnia2014", "KothaEtAl2016Italy",
                      "KothaEtAl2016Other", "KothaEtAl2016Turkey",
                      "ZhaoEtAl2016Asc", "BindiEtAl2017Rjb"]

    def compare_jsons(self, old, new):
        """
        Compares the json data from file with the new data from trellis plot

        This version works with the magnitude IMT and distance IMT trellis
        plots. Magnitude-distance Spectra has a slightly different means of
        comparison, so this will be over-ridden in that test
        """
        # Check x-labels are the same
        self.assertEqual(old["xlabel"], new["xlabel"])
        # Check x-values are the same
        np.testing.assert_array_almost_equal(old["xvalues"], new["xvalues"], 7)
        for i in range(len(old["figures"])):
            self.assertEqual(old["figures"][i]["ylabel"],
                             new["figures"][i]["ylabel"])
            self.assertEqual(old["figures"][i]["column"],
                             new["figures"][i]["column"])
            self.assertEqual(old["figures"][i]["row"],
                             new["figures"][i]["row"])

            for gsim in old["figures"][i]["yvalues"]:
                np.testing.assert_array_almost_equal(
                    old["figures"][i]["yvalues"][gsim],
                    new["figures"][i]["yvalues"][gsim], 7)


class DistanceTrellisTest(BaseTrellisTest):
    TEST_FILE = "test_distance_imt_trellis.json"

    def _run_trellis(self, rupture):
        """
        Executes the trellis plotting - for mean
        """
        return trpl.DistanceIMTTrellis.from_rupture_model(rupture,
                                                          self.gsims,
                                                          self.imts,
                                                          distance_type="rrup")

    def test_distance_imt_trellis(self):
        """
        Tests the DistanceIMT trellis data generation
        """
        reference = json.load(open(
            os.path.join(BASE_DATA_PATH, self.TEST_FILE), "r"))
        # Setup rupture
        rupture = rcfg.GSIMRupture(6.5, 60., 1.5,
                                   hypocentre_location=(0.5, 0.5))
        rupture.get_target_sites_line(250.0, 1.0, 800.0)
        # Get trellis calculations
        trl = self._run_trellis(rupture)
        # Parse the json formatted string to a dictionary string
        results = json.loads(trl.to_json())
        # Compare the two dictionaries
        self.compare_jsons(reference, results)


class DistanceSigmaTrellisTest(DistanceTrellisTest):
    TEST_FILE = "test_distance_sigma_imt_trellis.json"

    def _run_trellis(self, rupture):
        """
        Executes the trellis plotting - for standard deviation
        """
        return trpl.DistanceSigmaIMTTrellis.from_rupture_model(
            rupture,
            self.gsims,
            self.imts,
            distance_type="rrup")


class MagnitudeTrellisTest(BaseTrellisTest):
    TEST_FILE = "test_magnitude_imt_trellis.json"

    def _run_trellis(self, magnitudes, distance, properties):
        """
        Executes the trellis plotting - for mean
        """
        return trpl.MagnitudeIMTTrellis.from_rupture_model(properties,
                                                           magnitudes,
                                                           distance,
                                                           self.gsims,
                                                           self.imts)

    def test_magnitude_imt_trellis(self):
        """
        Tests the MagnitudeIMT trellis data generation
        """
        reference = json.load(open(
            os.path.join(BASE_DATA_PATH, self.TEST_FILE), "r"))
        magnitudes = np.arange(4., 8.1, 0.1)
        distance = 20.
        properties = {"dip": 60.0, "rake": -90.0, "aspect": 1.5, "ztor": 0.0,
                      "vs30": 800.0, "backarc": False, "z1pt0": 50.0,
                      "z2pt5": 1.0, "line_azimuth": 90.0}
        trl = self._run_trellis(magnitudes, distance, properties)
        results = json.loads(trl.to_json())
        self.compare_jsons(reference, results)


class MagnitudeSigmaTrellisTest(MagnitudeTrellisTest):
    TEST_FILE = "test_magnitude_sigma_imt_trellis.json"

    def _run_trellis(self, magnitudes, distance, properties):
        """
        Executes the trellis plotting - for standard deviation
        """
        return trpl.MagnitudeSigmaIMTTrellis.from_rupture_model(properties,
                                                                magnitudes,
                                                                distance,
                                                                self.gsims,
                                                                self.imts)


class MagnitudeDistanceSpectraTrellisTest(BaseTrellisTest):
    TEST_FILE = "test_magnitude_distance_spectra_trellis.json"

    def compare_jsons(self, old, new):
        """
        Compares the MagnitudeDistanceSpectra jsons
        """
        self.assertEqual(old["xlabel"], new["xlabel"])
        np.testing.assert_array_almost_equal(old["xvalues"], new["xvalues"], 7)
        for i in range(len(old["figures"])):
            self.assertAlmostEqual(old["figures"][i]["magnitude"],
                                   new["figures"][i]["magnitude"], 7)
            self.assertAlmostEqual(old["figures"][i]["distance"],
                                   new["figures"][i]["distance"], 7)
            self.assertEqual(old["figures"][i]["row"],
                             new["figures"][i]["row"])
            self.assertEqual(old["figures"][i]["column"],
                             new["figures"][i]["column"])
            for gsim in old["figures"][i]["yvalues"]:
                old_vals = np.array(old["figures"][i]["yvalues"][gsim])
                new_vals = np.array(new["figures"][i]["yvalues"][gsim])
                if old_vals.dtype == "O":
                    # Has None Values - compare element by element
                    for old_val, new_val in zip(old_vals, new_vals):
                        if old_val and new_val:
                            self.assertAlmostEqual(old_val, new_val, 7)
                        else:
                            self.assertEqual(old_val, new_val)
                else:
                    np.testing.assert_array_almost_equal(old_vals, new_vals, 7)

    def _run_trellis(self, magnitudes, distances, properties):
        """
        Executes the trellis plotting - for mean
        """
        return trpl.MagnitudeDistanceSpectraTrellis.from_rupture_model(
            properties, magnitudes, distances, self.gsims, self.periods,
            distance_type="rrup")

    def test_magnitude_distance_spectra_trellis(self):
        """
        Tests the MagnitudeDistanceSpectra Trellis data generation
        """
        reference = json.load(open(
            os.path.join(BASE_DATA_PATH, self.TEST_FILE), "r"))
        properties = {"dip": 60.0, "rake": -90.0, "aspect": 1.5, "ztor": 0.0,
                      "vs30": 800.0, "backarc": False, "z1pt0": 50.0,
                      "z2pt5": 1.0}
        magnitudes = [4.0, 5.0, 6.0, 7.0]
        distances = [5., 20., 50., 150.0]
        trl = self._run_trellis(magnitudes, distances, properties)
        results = json.loads(trl.to_json())
        self.compare_jsons(reference, results)


class MagnitudeDistanceSpectraSigmaTrellisTest(
        MagnitudeDistanceSpectraTrellisTest):
    TEST_FILE = "test_magnitude_distance_spectra_sigma_trellis.json"

    def _run_trellis(self, magnitudes, distances, properties):
        """
        Executes the trellis plotting - for standard deviation
        """
        return trpl.MagnitudeDistanceSpectraSigmaTrellis.from_rupture_model(
            properties, magnitudes, distances, self.gsims, self.periods,
            distance_type="rrup")
