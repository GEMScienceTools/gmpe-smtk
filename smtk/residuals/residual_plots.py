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
Module managing GMPE+IMT residual plot data.
This module avoids the use of classes and inhertances as simple functions
accomplish the task without unnecessary overhead.
All non-private functions should return the same dicts (see docstrings
for details)
"""

import numpy as np
from scipy.stats import linregress


def _tojson(*numpy_objs):
    '''Utility function which returns a list where each element of numpy_objs
    is converted to its python equivalent (float or list)'''
    ret = []
    # problem: browsers might not be happy with JSON 'NAN', so convert
    # NaNs to None. Unfortunately, the conversion must be done element wise
    # in numpy (seems not to exist a pandas na filter):
    for obj in numpy_objs:
        isscalar = np.isscalar(obj)
        nan_indices = None if isscalar else \
            np.argwhere(np.isnan(obj)).flatten()
        # note: numpy.float64(N).tolist() returns a python float, so:
        obj = None if isscalar and np.isnan(obj) else obj.tolist()
        if nan_indices is not None:
            for idx in nan_indices:
                obj[idx] = None
        ret.append(obj)

    return ret  # tuple(_.tolist() for _ in numpy_objs)


def residuals_density_distribution(residuals, gmpe, imt, bin_width=0.5,
                                   as_json=False):
    '''Returns the density distribution of the given gmpe and imt

    :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type
    :param as_json: when True, converts all numpy numeric values (scalar
    and arrays) to their python equivalent. False by default

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'mean' (float) and 'Std Dev' (float) representing
    the mean and standard deviation of the data
    '''
    statistics = residuals.get_residual_statistics_for(gmpe, imt)
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    for res_type in data.keys():

        vals, bins = _get_histogram_data(data[res_type], bin_width=bin_width)

        mean = statistics[res_type]["Mean"]
        stddev = statistics[res_type]["Std Dev"]
        x = bins[:-1]
        y = vals

        if as_json:
            mean, stddev, x, y = _tojson(mean, stddev, x, y)

        plot_data[res_type] = \
            {'x': x, 'y': y, 'mean': mean, 'stddev': stddev,
             'xlabel': "Z (%s)" % imt, 'ylabel': "Frequency"}

    return plot_data


def _get_histogram_data(data, bin_width=0.5):
    """
    Retreives the histogram of the residuals
    """
    # ignore nans otherwise max and min raise:
    bins = np.arange(np.floor(np.nanmin(data)),
                     np.ceil(np.nanmax(data)) + bin_width,
                     bin_width)
    # work on finite numbers to prevent np.histogram raising:
    vals = np.histogram(data[np.isfinite(data)], bins, density=True)[0]
    return vals.astype(float), bins


def likelihood(residuals, gmpe, imt, bin_width=0.1, as_json=False):
    '''Returns the likelihood of the given gmpe and imt

    :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Likelihood
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type
    :param as_json: when True, converts all numpy numeric values (scalar
    and arrays) to their python equivalent. False by default

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'median' (float) representing
    the median of the data
    '''
    plot_data = {}
    data = residuals._get_likelihood_values_for(gmpe, imt)

    for res_type in data.keys():
        lh_vals, median_lh = data[res_type]
        vals, bins = _get_lh_histogram_data(lh_vals, bin_width=bin_width)

        x = bins[:-1]
        y = vals

        if as_json:
            median_lh, x, y = _tojson(median_lh, x, y)

        plot_data[res_type] = \
            {'x': x, 'y': y, 'median': median_lh,
             'xlabel': "LH (%s)" % imt, 'ylabel': "Frequency"}

    return plot_data


def _get_lh_histogram_data(lh_values, bin_width=0.1):
    """
    Retreives the histogram of the likelihoods
    """
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    # work on finite numbers to prevent np.histogram raising:
    vals = np.histogram(lh_values[np.isfinite(lh_values)],
                        bins, density=True)[0]
    return vals.astype(float), bins


def residuals_with_magnitude(residuals, gmpe, imt, as_json=False):
    '''Returns the residuals of the given gmpe and imt vs. magnitude

    :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type
    :param as_json: when True, converts all numpy numeric values (scalar
    and arrays) to their python equivalent. False by default

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    '''
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    for res_type in data.keys():

        x = _get_magnitudes(residuals, gmpe, imt, res_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        if as_json:
            x, y, slope, intercept, pval = \
                _tojson(x, y, slope, intercept, pval)

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "Magnitude", 'ylabel': "Z (%s)" % imt}

    return plot_data


def _get_magnitudes(residuals, gmpe, imt, res_type):
    """
    Returns an array of magnitudes equal in length to the number of
    residuals
    """
    magnitudes = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        if res_type == "Inter event":

            nval = np.ones(
                len(residuals.unique_indices[gmpe][imt][i])
                )
        else:
            nval = np.ones(len(ctxt["Ctx"].repi))

        magnitudes = np.hstack([magnitudes, ctxt["Ctx"].mag * nval])

    return magnitudes


def residuals_with_vs30(residuals, gmpe, imt, as_json=False):
    '''Returns the residuals of the given gmpe and imt vs. vs30

    :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type
    :param as_json: when True, converts all numpy numeric values (scalar
    and arrays) to their python equivalent. False by default

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    '''

    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    for res_type in data.keys():

        x = _get_vs30(residuals, gmpe, imt, res_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        if as_json:
            x, y, slope, intercept, pval = \
                _tojson(x, y, slope, intercept, pval)

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "Vs30 (m/s)", 'ylabel': "Z (%s)" % imt}

    return plot_data


def _get_vs30(residuals, gmpe, imt, res_type):
    vs30 = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        if res_type == "Inter event":
            vs30 = np.hstack([vs30, ctxt["Ctx"].vs30[
                residuals.unique_indices[gmpe][imt][i]]])
        else:
            vs30 = np.hstack([vs30, ctxt["Ctx"].vs30])
    return vs30


def residuals_with_distance(residuals, gmpe, imt, distance_type="rjb",
                            as_json=False):
    '''Returns the residuals of the given gmpe and imt vs. distance

    :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type
    :param as_json: when True, converts all numpy numeric values (scalar
    and arrays) to their python equivalent. False by default

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    '''
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    for res_type in data.keys():

        x = _get_distances(residuals, gmpe, imt, res_type, distance_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        if as_json:
            x, y, slope, intercept, pval = \
                _tojson(x, y, slope, intercept, pval)

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "%s Distance (km)" % distance_type,
             'ylabel': "Z (%s)" % imt}

    return plot_data


def _get_distances(residuals, gmpe, imt, res_type, distance_type):
    """

    """
    distances = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        # Get the distances
        if res_type == "Inter event":
            ctxt_dist = getattr(ctxt["Ctx"], distance_type)[
                residuals.unique_indices[gmpe][imt][i]]
            distances = np.hstack([distances, ctxt_dist])
        else:
            distances = np.hstack([
                distances, getattr(ctxt["Ctx"], distance_type)])
    return distances


def residuals_with_depth(residuals, gmpe, imt, as_json=False):
    '''Returns the residuals of the given gmpe and imt vs. depth

    :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type
    :param as_json: when True, converts all numpy numeric values (scalar
    and arrays) to their python equivalent. False by default

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    '''
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    for res_type in data.keys():

        x = _get_depths(residuals, gmpe, imt, res_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        if as_json:
            x, y, slope, intercept, pval = \
                _tojson(x, y, slope, intercept, pval)

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "Hypocentral Depth (km)",
             'ylabel': "Z (%s)" % imt}

    return plot_data


def _get_depths(residuals, gmpe, imt, res_type):
    """
    Returns an array of magnitudes equal in length to the number of
    residuals
    """
    depths = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        if res_type == "Inter event":
            nvals = np.ones(len(residuals.unique_indices[gmpe][imt][i]))
        else:
            nvals = np.ones(len(ctxt["Ctx"].repi))
        # TODO This hack needs to be fixed!!!
        if not ctxt["Ctx"].hypo_depth:
            depths = np.hstack([depths, 10.0 * nvals])
        else:
            depths = np.hstack([depths, ctxt["Ctx"].hypo_depth * nvals])
    return depths


def _nanlinregress(x, y):
    """
    Calls scipy linregress only on finite numbers of x and y
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        # empty arrays passed to linreg raise ValueError:
        # force returning an object with nans:
        return linregress([np.nan], [np.nan])
    return linregress(x[finite], y[finite])

