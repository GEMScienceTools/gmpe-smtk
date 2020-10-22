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
Tests for bugfixes in to_dict Trellis methods:

1. the to_dict method of Distance Trellis plots
   failed to "dictionarize" empty yvalues arrays:
   The method(s) `get_ground_motion_values*`
   were returning different types: numpy arrays or empty
   lists. The fix was to return empty numpy arrays instead.
2. The MagnitudeDistance* trellis classes where returning NaNs
   instead of Nones. Whereas NaN are correctly parsable by
   Javascript, they are not JSON standard and need to
   be encoded as None
"""
import unittest
import os
import json
import numpy as np
from openquake.hazardlib.geo import Point
from openquake.hazardlib.scalerel import get_available_magnitude_scalerel
import smtk.trellis.trellis_plots as trpl
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14


class BaseTrellisTest(unittest.TestCase):
    """
    Base test class
    """

    # TEST_FILE = None

    def setUp(self):
        self.imts = ["PGA", "PGV"]
        self.periods = [0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                        0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
                        0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
                        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                        3.0, 4.001, 5.0, 7.5, 10.0]

        self.gsims = ["AbrahamsonEtAl2014",
                      # "AbrahamsonEtAl2014NSHMPLower",
                      # "AbrahamsonEtAl2014NSHMPMean",
                      # "AbrahamsonEtAl2014NSHMPUpper",
                      "AbrahamsonEtAl2014RegCHN",
                      "AbrahamsonEtAl2014RegJPN",
                      "AbrahamsonEtAl2014RegTWN",
                      "AkkarBommer2010SWISS01",
                      "AkkarBommer2010SWISS04",
                      "AkkarBommer2010SWISS08",
                      "AkkarEtAl2013", "AkkarEtAlRepi2014"]

        # this set of parameters raised exceptions, as e.g.
        # AkkarBommer2010SWISS01's PGV eas empty
        vs30 = 760.0
        self.params = {"magnitude": np.array([3, 4]),
                       "distance": np.array([10, 11, 12]),
                       "dip": 60, "aspect": 1.5, "rake": 0.0, "ztor": 0.0,
                       "strike": 0.0,
                       "msr": get_available_magnitude_scalerel()["WC1994"],
                       "initial_point": Point(0, 0), "hypocentre_location": [0.5, 0.5],
                       "vs30": vs30, "vs30_measured": True,
                       "line_azimuth": 0.0, "backarc": False,
                       "z1pt0": vs30_to_z1pt0_cy14(vs30),
                       "z2pt5": vs30_to_z2pt5_cb14(vs30)}


class DistanceTrellisTest(BaseTrellisTest):

    def _run_trellis(self, magnitude, distances, properties):
        """
        Executes the trellis plotting - for mean
        """
        return trpl.DistanceIMTTrellis.from_rupture_properties(
            properties,
            magnitude,
            distances,
            self.gsims,
            self.imts,
            distance_type="rrup")

    def test_distance_imt_trellis(self):
        """
        Tests the DistanceIMT trellis data generation
        """
        # Setup rupture
        properties = {k: self.params[k]
                      for k in ['dip', 'aspect',
                                'hypocentre_location', 'vs30']}
        distances = self.params['distance']
        magnitude = self.params['magnitude'][0]
        # Get trellis calculations
        trl = self._run_trellis(magnitude, distances, properties)
        # simply run the to_dict to assure it does not raise:
        figures = trl.to_dict()['figures']
        self.assertEqual(figures[1]['yvalues']['AkkarBommer2010SWISS01'],
                         [])


class DistanceSigmaTrellisTest(DistanceTrellisTest):

    def _run_trellis(self, magnitude, distances, properties):
        """
        Executes the trellis plotting - for standard deviation
        """
        return trpl.DistanceSigmaIMTTrellis.from_rupture_properties(
            properties,
            magnitude,
            distances,
            self.gsims,
            self.imts,
            distance_type="rrup")

# We did not spot a set of parameters for which we have empty
# yvalues in case of MagnitudeTrellis...
# thus the set of classes below is commented (for the moment?)

# class MagnitudeTrellisTest(BaseTrellisTest):
# 
#     def _run_trellis(self, magnitudes, distance, properties):
#         """
#         Executes the trellis plotting - for mean
#         """
#         return trpl.MagnitudeIMTTrellis.from_rupture_properties(
#             properties,
#             magnitudes,
#             distance,
#             self.gsims,
#             self.imts)
# 
#     def test_magnitude_imt_trellis(self):
#         """
#         Tests the MagnitudeIMT trellis data generation
#         """
#         magnitudes = self.params['magnitude']
#         distance = self.params['distance'][0]
#         properties = {k: self.params[k] for k in
#                       ["dip", "rake", "aspect", "ztor",
#                        "vs30", "backarc", "z1pt0",
#                        "z2pt5", "line_azimuth"]}
#         trl = self._run_trellis(magnitudes, distance, properties)
#         # simply run the to_dict to assure it does not raise:
#         trl.to_dict()
# 
# 
# class MagnitudeSigmaTrellisTest(MagnitudeTrellisTest):
# 
#     def _run_trellis(self, magnitudes, distance, properties):
#         """
#         Executes the trellis plotting - for standard deviation
#         """
#         return trpl.MagnitudeSigmaIMTTrellis.from_rupture_properties(
#             properties,
#             magnitudes,
#             distance,
#             self.gsims,
#             self.imts)
# 
#

# MagnitudeDistanceSpectraTrellisTest tests a different thing, i.e.
# that nan's are not in the returned yvalues. For performance reasons,
# we test a single GSIM (which contained nans before the fix)
# and assert it has Nones now:

class MagnitudeDistanceSpectraTrellisTest(BaseTrellisTest):

    def _run_trellis(self, magnitudes, distances, properties):
        """
        Executes the trellis plotting - for mean
        """
        return trpl.MagnitudeDistanceSpectraTrellis.from_rupture_properties(
            properties, magnitudes, distances, self.gsims, self.periods,
            distance_type="rrup")
 
    def test_magnitude_distance_spectra_trellis(self):
        """
        Tests the MagnitudeDistanceSpectra Trellis data generation
        """
        properties = {k: self.params[k] for k in
                      ["dip", "rake", "aspect", "ztor",
                       "vs30", "backarc", "z1pt0",
                       "z2pt5"]}
        magnitudes = self.params['magnitude']
        distances = self.params['distance']
        trl = self._run_trellis(magnitudes, distances, properties)
        dic = trl.to_dict()
        # test that a Gsim having nan's has nones now:
        yvalues = dic['figures'][0]['yvalues']["AkkarBommer2010SWISS01"]
        self.assertTrue(any(_ is None for _ in yvalues))
 
 
class MagnitudeDistanceSpectraSigmaTrellisTest(
        MagnitudeDistanceSpectraTrellisTest):
 
    def _run_trellis(self, magnitudes, distances, properties):
        """
        Executes the trellis plotting - for standard deviation
        """
        return trpl.MagnitudeDistanceSpectraSigmaTrellis.from_rupture_properties(
            properties, magnitudes, distances, self.gsims, self.periods,
            distance_type="rrup")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()