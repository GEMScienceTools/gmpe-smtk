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
Tests the GM database parsing and selection
"""
import os
# import shutil
import sys
from datetime import datetime
# import json
# import pprint
import unittest
import numpy as np
from scipy.constants import g

from smtk.sm_utils import convert_accel_units


# OLD IMPLEMENTATION OF CONVERT ACCELERATION UNITS. USED HERE
# TO COMPARE RESULTS
def convert_accel_units_old(acceleration, units):
    """
    Converts acceleration from different units into cm/s^2

    :param units: string in "g", "m/s/s", "m/s**2", "m/s^2",
        "cm/s/s", "cm/s**2" or "cm/s^2" (in the last three cases, this
        function simply returns `acceleration`)

    :return: acceleration converted to the given units
    """
    if units == "g":
        return (100*g) * acceleration
    if units in ("m/s/s", "m/s**2", "m/s^2"):
        return 100. * acceleration
    if units in ("cm/s/s", "cm/s**2", "cm/s^2"):
        return acceleration

    raise ValueError("Unrecognised time history units. "
                     "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")

class SmUtilsTestCase(unittest.TestCase):
    '''tests GroundMotionTable and selection'''

    def assertNEqual(self, first, second, rtol=1e-6, atol=1e-9,
                     equal_nan=True):
        self.assertTrue(np.allclose(first, second, rtol=rtol, atol=atol,
                                    equal_nan=equal_nan))

    def test_accel_units(self):
        func = convert_accel_units
        for acc in [np.nan, 0, 100, -g*5, g*6.5,
                    np.array([np.nan, 0, 100, g*5, g*6.5])]:

            # check that cm_sec and m_sec produce the same result:
            _1, _2 = func(acc, 'g', 'cm/s/s'), func(acc, 'cm/s/s', 'g')
            for cmsec in ('cm/s^2', 'cm/s**2'):
                self.assertNEqual(_1, func(acc, 'g', cmsec))
                self.assertNEqual(_2, func(acc, cmsec, 'g'))

            _1, _2 = func(acc, 'g', 'm/s/s'), func(acc, 'm/s/s', 'g')
            for msec in ('m/s^2', 'm/s**2'):
                self.assertNEqual(_1, func(acc, 'g', msec))
                self.assertNEqual(_2, func(acc, msec, 'g'))
            # assert same label is no-op:
            self.assertNEqual(func(acc, 'g', 'g'), acc)
            self.assertNEqual(func(acc, 'cm/s/s', 'cm/s/s'), acc)
            self.assertNEqual(func(acc, 'm/s/s', 'm/s/s'), acc)

            # assume input in g:
            # to cm/s/s
            expected = acc * (100 * g)
            self.assertNEqual(func(acc, 'g', 'cm/s/s'), expected)
            self.assertNEqual(func(acc, 'g', 'cm/s/s'),
                              convert_accel_units_old(acc, 'g'))
            # to m/s/s:
            expected /= 100
            self.assertNEqual(func(acc, 'g', 'm/s/s'), expected)
            self.assertNEqual(func(acc, 'g', 'm/s/s'),
                              convert_accel_units_old(acc, 'g')/100)

            # check that the old calls to convert_accel_units:
            # are the same as the actual:
            for unit in ['g', 'cm/s/s', 'm/s/s']:
                self.assertNEqual(convert_accel_units_old(acc, unit),
                                  func(acc, unit))




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
