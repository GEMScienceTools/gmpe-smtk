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
Applies spectral smoothing via the Konno & Ohmachi (1998) smoothing algorithm

The algorithm itself is taken directly from the Obspy implementation by
Lion Krischer
"""
import numpy as np
import warnings
from smtk.smoothing.base import BaseSpectralSmoother

def konnoOhmachiSmoothingWindow(frequencies, center_frequency, bandwidth=40.0,
                                normalize=False):
    """
    Returns the Konno & Ohmachi Smoothing window for every frequency in
    frequencies.

    Returns the smoothing window around the center frequency with one value per
    input frequency defined as follows (see [Konno1998]_):

    [sin(b * log_10(f/f_c)) / (b * log_10(f/f_c)]^4
        b   = bandwidth
        f   = frequency
        f_c = center frequency

    The bandwidth of the smoothing function is constant on a logarithmic scale.
    A small value will lead to a strong smoothing, while a large value of will
    lead to a low smoothing of the Fourier spectra.
    The default (and generally used) value for the bandwidth is 40. (From the
    Geopsy documentation - www.geopsy.org)

    All parameters need to be positive. This is not checked due to performance
    reasons and therefore any negative parameters might have unexpected
    results.

    This function might raise some numpy warnings due to divisions by zero and
    logarithms of zero. This is intentional and faster than prefiltering the
    special cases. You can disable numpy warnings (they usually do not show up
    anyways) with:

    temp = np.geterr()
    np.seterr(all='ignore')
    ...code that raises numpy warning due to division by zero...
    np.seterr(**temp)

    :param frequencies: numpy.ndarray (float32 or float64)
        All frequencies for which the smoothing window will be returned.
    :param center_frequency: float >= 0.0
        The frequency around which the smoothing is performed.
    :param bandwidth: float > 0.0
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Defaults to 40.
    :param normalize: boolean, optional
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
        msg = 'frequencies needs to have a dtype of float32/64.'
        raise ValueError(msg)
    # If the center_frequency is 0 return an array with zero everywhere except
    # at zero.
    if center_frequency == 0:
        smoothing_window = np.zeros(len(frequencies), dtype=frequencies.dtype)
        smoothing_window[frequencies == 0.0] = 1.0
        return smoothing_window
    # Calculate the bandwidth*log10(f/f_c)
    smoothing_window = bandwidth * np.log10(frequencies / center_frequency)
    # Just the Konno-Ohmachi formulae.
    smoothing_window[...] = (np.sin(smoothing_window) / smoothing_window) ** 4
    # Check if the center frequency is exactly part of the provided
    # frequencies. This will result in a division by 0. The limit of f->f_c is
    # one.
    smoothing_window[frequencies == center_frequency] = 1.0
    # Also a frequency of zero will result in a logarithm of -inf. The limit of
    # f->0 with f_c!=0 is zero.
    smoothing_window[frequencies == 0.0] = 0.0
    # Normalize to one if wished.
    if normalize:
        smoothing_window /= smoothing_window.sum()
    return smoothing_window


def calculateSmoothingMatrix(frequencies, bandwidth=40.0, normalize=False):
    """
    Calculates a len(frequencies) x len(frequencies) matrix with the Konno &
    Ohmachi window for each frequency as the center frequency.

    Any spectrum with the same frequency bins as this matrix can later be
    smoothed by a simple matrix multiplication with this matrix:
        smoothed_spectrum = np.dot(spectrum, smoothing_matrix)

    This also works for many spectra stored in one large matrix and is even
    more efficient.

    This makes it very efficient for smoothing the same spectra again and again
    but it comes with a high memory consumption for larger frequency arrays!

    :param frequencies: numpy.ndarray (float32 or float64)
        The input frequencies.
    :param bandwidth: float > 0.0
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Defaults to 40.
    :param normalize: boolean, optional
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    # Create matrix to be filled with smoothing entries.
    sm_matrix = np.empty((len(frequencies), len(frequencies)),
                         frequencies.dtype)
    for _i, freq in enumerate(frequencies):
        sm_matrix[_i, :] = konnoOhmachiSmoothingWindow(frequencies, freq,
                                              bandwidth, normalize=normalize)
    return sm_matrix


def konnoOhmachiSmoothing(spectra, frequencies, bandwidth=40, count=1,
                  enforce_no_matrix=False, max_memory_usage=512,
                  normalize=False):
    """
    Smoothes a matrix containing one spectra per row with the Konno-Ohmachi
    smoothing window.

    All spectra need to have frequency bins corresponding to the same
    frequencies.

    This method first will estimate the memory usage and then either use a fast
    and memory intensive method or a slow one with a better memory usage.

    :param spectra: numpy.ndarray (float32 or float64)
        One or more spectra per row. If more than one the first spectrum has to
        be accessible via spectra[0], the next via spectra[1], ...
    :param frequencies: numpy.ndarray (float32 or float64)
        Contains the frequencies for the spectra.
    :param bandwidth: float > 0.0
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Defaults to 40.
    :param count: integer, optional
        How often the apply the filter. For very noisy spectra it is useful to
        apply is more than once. Defaults to 1.
    :param enforce_no_matrix: boolean, optional
        An efficient but memory intensive matrix-multiplication algorithm is
        used in case more than one spectra is to be smoothed or one spectrum is
        to be smoothed more than once if enough memory is available. This flag
        disables the matrix algorithm altogether. Defaults to False
    :param max_memory_usage: integer, optional
        Set the maximum amount of extra memory in MB for this method. Decides
        whether or not the matrix multiplication method is used. Defaults to
        512 MB.
    :param normalize: boolean, optional
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    if (frequencies.dtype != np.float32 and frequencies.dtype != np.float64) \
       or (spectra.dtype != np.float32 and spectra.dtype != np.float64):
        msg = 'frequencies and spectra need to have a dtype of float32/64.'
        raise ValueError(msg)
    # Spectra and frequencies should have the same dtype.
    if frequencies.dtype != spectra.dtype:
        frequencies = np.require(frequencies, np.float64)
        spectra = np.require(spectra, np.float64)
        msg = 'frequencies and spectra should have the same dtype. It ' + \
              'will be changed to np.float64 for both.'
        warnings.warn(msg)
    # Check the dtype to get the correct size.
    if frequencies.dtype == np.float32:
        size = 4.0
    elif frequencies.dtype == np.float64:
        size = 8.0
    # Calculate the approximate usage needs for the smoothing matrix algorithm.
    length = len(frequencies)
    approx_mem_usage = (length * length + 2 * len(spectra) + length) * \
            size / 1048576.0
    # If smaller than the allowed maximum memory consumption build a smoothing
    # matrix and apply to each spectrum. Also only use when more then one
    # spectrum is to be smoothed.
    if enforce_no_matrix is False and (len(spectra.shape) > 1 or count > 1) \
       and approx_mem_usage < max_memory_usage:
        # Disable numpy warnings due to possible divisions by zero/logarithms
        # of zero.
        temp = np.geterr()
        np.seterr(all='ignore')
        smoothing_matrix = calculateSmoothingMatrix(frequencies, bandwidth,
                                             normalize=normalize)
        np.seterr(**temp)
        new_spec = np.dot(spectra, smoothing_matrix)
        # Eventually apply more than once.
        for _i in xrange(count - 1):
            new_spec = np.dot(new_spec, smoothing_matrix)
        return new_spec
    # Otherwise just calculate the smoothing window every time and apply it.
    else:
        new_spec = np.empty(spectra.shape, spectra.dtype)
        # Separate case for just one spectrum.
        if len(new_spec.shape) == 1:
            # Disable numpy warnings due to possible divisions by
            # zero/logarithms of zero.
            temp = np.geterr()
            np.seterr(all='ignore')
            for _i in xrange(len(frequencies)):
                window = konnoOhmachiSmoothingWindow(frequencies,
                        frequencies[_i], bandwidth, normalize=normalize)
                new_spec[_i] = (window * spectra).sum()
            np.seterr(**temp)
        # Reuse smoothing window if more than one spectrum.
        else:
            # Disable numpy warnings due to possible divisions by
            # zero/logarithms of zero.
            temp = np.geterr()
            np.seterr(all='ignore')
            for _i in xrange(len(frequencies)):
                window = konnoOhmachiSmoothingWindow(frequencies,
                        frequencies[_i], bandwidth, normalize=normalize)
                for _j, spec in enumerate(spectra):
                    new_spec[_j, _i] = (window * spec).sum()
            np.seterr(**temp)
        # Eventually apply more than once.
        while count > 1:
            new_spec = konnoOhmachiSmoothing(new_spec, frequencies, bandwidth,
                                enforce_no_matrix=True, normalize=normalize)
            count -= 1
        return new_spec


class KonnoOhmachi(BaseSpectralSmoother):
    """
    
    """
    def _check_params(self, params):
        """
        Verify that "bandwidth" and "count" are both present and real positive.
        Adds the other defaults if not present
        """
        params_keys = params.keys()
        assert "bandwidth" in params_keys
        assert "count" in params_keys
        assert params["bandwidth"] > 0.0
        assert params["count"] > 0
        if not "enforce_no_matrix" in params_keys:
            params["enforce_no_matrix"] = False
        if not "max_memory_usage" in params_keys:
            params["max_memory_usage"] = 512
        if not "normalize" in params_keys:
            params["normalize"] = False
        return params

    def apply_smoothing(self, spectra, frequencies):
        """
        Applies the Konno & Ohmachi (1998) smoothing
        """
        return konnoOhmachiSmoothing(spectra,
                                     frequencies,
                                     self.params["bandwidth"],
                                     self.params["count"],
                                     self.params["enforce_no_matrix"],
                                     self.params["max_memory_usage"],
                                     self.params["normalize"])
