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
IMS
IMS/V
IMS/V/Scalar
IMS/V/Scalar/PGA
IMS/V/Scalar/PGV
IMS/V/Spectra
IMS/V/Spectra/Fourier
IMS/V/Spectra/Response
IMS/V/Spectra/Response/Acceleration
IMS/V/Spectra/Response/Acceleration/damping_02
IMS/V/Spectra/Response/Acceleration/damping_05
IMS/V/Spectra/Response/Acceleration/damping_07
IMS/V/Spectra/Response/Acceleration/damping_10
IMS/V/Spectra/Response/Acceleration/damping_20
IMS/V/Spectra/Response/Acceleration/damping_30
IMS/V/Spectra/Response/Periods
IMS/X
IMS/X/Scalar
IMS/X/Scalar/PGA
IMS/X/Scalar/PGV
IMS/X/Spectra
IMS/X/Spectra/Fourier
IMS/X/Spectra/Response
IMS/X/Spectra/Response/Acceleration
IMS/X/Spectra/Response/Acceleration/damping_02
IMS/X/Spectra/Response/Acceleration/damping_05
IMS/X/Spectra/Response/Acceleration/damping_07
IMS/X/Spectra/Response/Acceleration/damping_10
IMS/X/Spectra/Response/Acceleration/damping_20
IMS/X/Spectra/Response/Acceleration/damping_30
IMS/X/Spectra/Response/Periods
IMS/Y
IMS/Y/Scalar
IMS/Y/Scalar/PGA
IMS/Y/Scalar/PGV
IMS/Y/Spectra
IMS/Y/Spectra/Fourier
IMS/Y/Spectra/Response
IMS/Y/Spectra/Response/Acceleration
IMS/Y/Spectra/Response/Acceleration/damping_02
IMS/Y/Spectra/Response/Acceleration/damping_05
IMS/Y/Spectra/Response/Acceleration/damping_07
IMS/Y/Spectra/Response/Acceleration/damping_10
IMS/Y/Spectra/Response/Acceleration/damping_20
IMS/Y/Spectra/Response/Acceleration/damping_30
IMS/Y/Spectra/Response/Periods
Time Series
Time Series/V
Time Series/V/Original Record
Time Series/V/Original Record/Acceleration
Time Series/X
Time Series/X/Original Record
Time Series/X/Original Record/Acceleration
Time Series/Y
Time Series/Y/Original Recor
Time Series/Y/Original Record/Acceleration
"""


DATA_DICT = {
    "IMS": {
    "X": {"Scalar": {"PGA": None,
                     "PGV": None,
                     "PGD": None,
                     "CAV": None,
                     "Ia": None},
          "Spectra":{"Fourier": {},
                     "Response": {"Acceleration": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "Velocity": {"damping_02": None,
                                               "damping_05": None,
                                               "damping_07": None,
                                               "damping_10": None,
                                               "damping_20": None,
                                               "damping_30": None},
                                  "Displacement": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "PSA": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None},
                                  "PSV": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None}}}},
    "Y": {"Scalar": {"PGA": None,
                     "PGV": None,
                     "PGD": None,
                     "CAV": None,
                     "Ia": None},
          "Spectra":{"Fourier": {},
                     "Response": {"Acceleration": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "Velocity": {"damping_02": None,
                                               "damping_05": None,
                                               "damping_07": None,
                                               "damping_10": None,
                                               "damping_20": None,
                                               "damping_30": None},
                                  "Displacement": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "PSA": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None},
                                  "PSV": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None}}}},
    "Z": {"Scalar": {"PGA": None,
                     "PGV": None,
                     "PGD": None,
                     "CAV": None,
                     "Ia": None},
          "Spectra":{"Fourier": {},
                     "Response": {"Acceleration": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "Velocity": {"damping_02": None,
                                               "damping_05": None,
                                               "damping_07": None,
                                               "damping_10": None,
                                               "damping_20": None,
                                               "damping_30": None},
                                  "Displacement": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "PSA": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None},
                                  "PSV": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None}}}},
    "H": {"Scalar": {"PGA": None,
                     "PGV": None,
                     "PGD": None,
                     "CAV": None,
                     "Ia": None},
          "Spectra":{"Fourier": {},
                     "Response": {"Acceleration": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "Velocity": {"damping_02": None,
                                               "damping_05": None,
                                               "damping_07": None,
                                               "damping_10": None,
                                               "damping_20": None,
                                               "damping_30": None},
                                  "Displacement": {"damping_02": None,
                                                   "damping_05": None,
                                                   "damping_07": None,
                                                   "damping_10": None,
                                                   "damping_20": None,
                                                   "damping_30": None},
                                  "PSA": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None},
                                  "PSV": {"damping_02": None,
                                          "damping_05": None,
                                          "damping_07": None,
                                          "damping_10": None,
                                          "damping_20": None,
                                          "damping_30": None}}}}},
    "Time Series": {
        "X": {"Original Record": {"Acceleration": None,
                                  "Velocity": None,
                                  "Displacement": None}},
        "Y": {"Original Record": {"Acceleration": None,
                                  "Velocity": None,
                                  "Displacement": None}},

        "Z": {"Original Record": {"Acceleration": None,
                                  "Velocity": None,
                                  "Displacement": None}}}
}


