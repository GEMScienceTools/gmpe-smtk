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
import h5py
import numpy as np
import smtk.response_spectrum as rsp
import smtk.intensity_measures as ims


BASE_DATA_PATH = os.path.dirname(__file__)


class BaseIMSTestCase(unittest.TestCase):
    """
    Base test case for Response Spectra and Intensity Measure functions
    """

    @staticmethod
    def arr_diff(x, y, percent):
        """
        Retrieving data from hdf5 leads to precision differences use relative
        error (i.e. < X % difference)
        """
        idx = np.logical_and(x > 0.0, y > 0.0)
        diff = np.zeros_like(x)
        diff[idx] = ((x[idx] / y[idx]) - 1.0) * 100
        if np.all(np.fabs(diff) < percent):
            return True
        else:
            iloc = np.argmax(diff)
            print(x, y, diff, x[iloc], y[iloc], diff[iloc])
            return False

    def _compare_sa_sets(self, sax, fle_loc, disc=1.0):
        """
        When data is stored in a dictionary of arrays, compare by keys
        """
        for key in sax:
            if not isinstance(sax[key], np.ndarray) or len(sax[key]) == 1:
                continue
            reference_data = self.fle[fle_loc + "/{:s}".format(key)][:]
            self.assertTrue(self.arr_diff(sax[key], reference_data, disc))

    def setUp(self):
        """
        Connect to hdf5 data store
        """
        self.fle = h5py.File(os.path.join(BASE_DATA_PATH,
                                          "smtk_ims_test_data.hdf5"), "r")
        self.periods = self.fle["INPUTS/periods"][:]

    def tearDown(self):
        """
        Close hdf5 connection
        """
        self.fle.close()


class ResponseSpectrumTestCase(BaseIMSTestCase):
    """
    Tests the response spectrum methods
    """

    def test_response_spectrum(self):
        # Tests the Nigam & Jennings Response Spectrum
        x_record = self.fle["INPUTS/RECORD1/XRECORD"][:]
        x_time_step = self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"]
        nigam_jennings = rsp.NigamJennings(x_record, x_time_step, self.periods,
                                           damping=0.05, units="cm/s/s")
        sax, timeseries, acc, vel, dis = nigam_jennings()
        self._compare_sa_sets(sax, "TEST1/X/spectra")
        for key in ["Acceleration", "Velocity", "Displacement"]:
            if not isinstance(timeseries[key], np.ndarray):
                continue
            self.assertTrue(
                self.arr_diff(
                    timeseries[key],
                    self.fle["TEST1/X/timeseries/{:s}".format(key)][:],
                    1.0))

    def test_get_response_spectrum_pair(self):
        # Tests the call to the response spectrum via ims
        sax, say = ims.get_response_spectrum_pair(
            self.fle["INPUTS/RECORD1/XRECORD"][:],
            self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"],
            self.fle["INPUTS/RECORD1/YRECORD"][:],
            self.fle["INPUTS/RECORD1/YRECORD"].attrs["timestep"],
            self.periods, damping=0.05, units="cm/s/s",
            method="Nigam-Jennings")
        self._compare_sa_sets(sax, "TEST1/X/spectra")
        self._compare_sa_sets(say, "TEST1/Y/spectra")

    def test_get_geometric_mean_spectrum(self):
        # Tests the geometric mean spectrum
        sax, say = ims.get_response_spectrum_pair(
            self.fle["INPUTS/RECORD1/XRECORD"][:],
            self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"],
            self.fle["INPUTS/RECORD1/YRECORD"][:],
            self.fle["INPUTS/RECORD1/YRECORD"].attrs["timestep"],
            self.periods, damping=0.05, units="cm/s/s",
            method="Nigam-Jennings")
        sa_gm = ims.geometric_mean_spectrum(sax, say)
        self._compare_sa_sets(sa_gm, "TEST1/GM/spectra")

    def test_envelope_spectrum(self):
        # Tests the envelope spectrum
        sax, say = ims.get_response_spectrum_pair(
            self.fle["INPUTS/RECORD1/XRECORD"][:],
            self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"],
            self.fle["INPUTS/RECORD1/YRECORD"][:],
            self.fle["INPUTS/RECORD1/YRECORD"].attrs["timestep"],
            self.periods, damping=0.05, units="cm/s/s",
            method="Nigam-Jennings")
        sa_env = ims.envelope_spectrum(sax, say)
        self._compare_sa_sets(sa_env, "TEST1/ENV/spectra")

    def test_gmrotd50(self):
        # Tests the function to get GMRotD50
        gmrotd50 = ims.gmrotdpp(
            self.fle["INPUTS/RECORD1/XRECORD"][:],
            self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"],
            self.fle["INPUTS/RECORD1/YRECORD"][:],
            self.fle["INPUTS/RECORD1/YRECORD"].attrs["timestep"],
            self.periods, percentile=50.0, damping=0.05, units="cm/s/s",
            method="Nigam-Jennings")
        self._compare_sa_sets(gmrotd50, "TEST1/GMRotD50/spectra")

    def test_gmroti50(self):
        # Tests the function to get GMRotI50
        gmroti50 = ims.gmrotipp(
            self.fle["INPUTS/RECORD1/XRECORD"][:],
            self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"],
            self.fle["INPUTS/RECORD1/YRECORD"][:],
            self.fle["INPUTS/RECORD1/YRECORD"].attrs["timestep"],
            self.periods, percentile=50.0, damping=0.05, units="cm/s/s",
            method="Nigam-Jennings")
        self._compare_sa_sets(gmroti50, "TEST1/GMRotI50/spectra")


class ScalarIntensityMeasureTestCase(BaseIMSTestCase):
    """
    Tests the functions returning scalar intensity measures
    """
    def test_get_peak_measures(self):
        # Tests the PGA, PGV, PGD functions
        pga_x, pgv_x, pgd_x, _, _ = ims.get_peak_measures(
            self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"],
            self.fle["INPUTS/RECORD1/XRECORD"][:],
            True,
            True)
        self.assertAlmostEqual(pga_x, 523.6900024, 3)
        self.assertAlmostEqual(pgv_x, 46.7632261, 3)
        self.assertAlmostEqual(pgd_x, 13.6729804, 3)

    def test_get_durations(self):
        # Tests the bracketed, uniform and significant duration
        x_record = self.fle["INPUTS/RECORD1/XRECORD"][:]
        x_timestep = self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"]
        self.assertAlmostEqual(
            ims.get_bracketed_duration(x_record, x_timestep, 5.0),
            19.7360000, 3)

        self.assertAlmostEqual(
            ims.get_uniform_duration(x_record, x_timestep, 5.0),
            14.6820000, 3)

        self.assertAlmostEqual(
            ims.get_significant_duration(x_record, x_timestep, 0.05, 0.95),
            4.0320000, 3)

    def test_arias_cav_arms(self):
        # Tests the functions for Ia, CAV, CAV5 and Arms
        x_record = self.fle["INPUTS/RECORD1/XRECORD"][:]
        x_timestep = self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"]
        # Arias intensity
        self.assertAlmostEqual(
            ims.get_arias_intensity(x_record, x_timestep),
            111.1540091, 3)
        # 5 - 95 % Arias Intensity
        self.assertAlmostEqual(
            ims.get_arias_intensity(x_record, x_timestep, 0.05, 0.95),
            99.9621952, 3)
        # CAV
        self.assertAlmostEqual(
            ims.get_cav(x_record, x_timestep),
            509.9941624, 3)
        # CAV5
        self.assertAlmostEqual(
            ims.get_cav(x_record, x_timestep, threshold=5.0),
            496.7741956, 3)
        # Arms
        self.assertAlmostEqual(
            ims.get_arms(x_record, x_timestep),
            56.8495087, 3)

    def test_spectrum_intensities(self):
        # Tests Housner Intensity and Acceleration Spectrum Intensity
        x_record = self.fle["INPUTS/RECORD1/XRECORD"][:]
        x_timestep = self.fle["INPUTS/RECORD1/XRECORD"].attrs["timestep"]
        sax = ims.get_response_spectrum(x_record, x_timestep, self.periods)[0]
        housner = ims.get_response_spectrum_intensity(sax)
        self.assertAlmostEqual(housner, 121.3103787, 3)
        asi = ims.get_acceleration_spectrum_intensity(sax)
        self.assertAlmostEqual(asi, 432.5134666, 3)
