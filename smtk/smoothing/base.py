#!/usr/bin/env python

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
