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
Abstract base class for applying smoothing to a spectrum
"""
import os
import abc


class BaseSpectralSmoother(object):
    """
    Abstract base class for method to apply smoothing to a spectrum
    :param dict params:
        Smoothing model parameters
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, params):
        """
        Instantiate with dictionary of parameters
        """
        self.params = self._check_params(params)

    def _check_params(self, params):
        """
        In the simple case the parameters are valid
        """
        return params
    
    @abc.abstractmethod
    def apply_smoothing(self, spectra, frequencies):
        """
        Applies the smoothing to a given spectrum
        """
