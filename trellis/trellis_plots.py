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
Sets up a simple rupture-site configuration to allow for physical comparison
of GMPEs
"""
import sys
import json
try:  # https://stackoverflow.com/q/53978542
    from collections.abc import Iterable  # noqa
except ImportError:
    from collections import Iterable  # noqa    
from math import floor, ceil
import matplotlib
from cycler import cycler
from copy import deepcopy
import matplotlib.pyplot as plt

import numpy as np
from openquake.hazardlib import imt
from openquake.hazardlib.gsim.base import (RuptureContext, DistancesContext,
                                           SitesContext)
from openquake.hazardlib.scalerel.wc1994 import WC1994
from smtk.sm_utils import _save_image, check_gsim_list
import smtk.trellis.trellis_utils as utils
from smtk.trellis.configure import GSIMRupture, DEFAULT_POINT


# Default - defines a 21 color and line-type cycle
matplotlib.rcParams["axes.prop_cycle"] = \
    cycler(u'color', ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                      'b', 'g', 'r', 'c', 'm', 'y', 'k',
                      'b', 'g', 'r', 'c', 'm', 'y', 'k',
                      'b', 'g', 'r', 'c', 'm', 'y', 'k']) +\
    cycler(u'linestyle', ["-", "-", "-", "-", "-", "-", "-",
                          "--", "--", "--", "--", "--", "--", "--",
                          "-.", "-.", "-.", "-.", "-.", "-.", "-.",
                          ":", ":", ":", ":", ":", ":", ":"])

# Generic dictionary of parameters needed for a trellis calculation
PARAM_DICT = {
    'magnitudes': [],
    'distances': [],
    'distance_type': 'rjb',
    'vs30': [],
    'strike': None,
    'dip': None,
    'rake': None,
    'ztor': None,
    'hypocentre_location': (0.5, 0.5),
    'hypo_loc': (0.5, 0.5),
    'msr': WC1994()
}

# Defines the plotting units for given intensitiy measure type
PLOT_UNITS = {
    'PGA': 'g',
    'PGV': 'cm/s',
    'SA': 'g',
    'SD': 'cm',
    'IA': 'm/s',
    'CSV': 'g-sec',
    'RSD': 's',
    'MMI': ''
}

# Verbose label for each given distance type
DISTANCE_LABEL_MAP = {
    'repi': 'Epicentral Dist.',
    'rhypo': 'Hypocentral Dist.',
    'rjb': 'Joyner-Boore Dist.',
    'rrup': 'Rupture Dist.',
    'rx': 'Rx Dist.'
}

# Default figure size
FIG_SIZE = (7, 5)

# RESET Axes tick labels
matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)


def _get_imts(imts):
    """
    Reads a list of IMT strings and returns the corresponding
    openquake.hazardlib.imt class
    :param list imts:
        List of IMTs(str)
    """
    out_imts = []
    for imtl in imts:
        out_imts.append(imt.from_string(imtl))
    return out_imts


class BaseTrellis(object):
    """
    Base class for holding functions related to the trellis plotting
    :param list or np.ndarray magnitudes:
        List of rupture magnitudes
    :param dict distances:
        Dictionary of distance measures as a set of np.ndarrays -
        {'repi', np.ndarray,
         'rjb': np.ndarray,
         'rrup': np.ndarray,
         'rhypo': np.ndarray}
        The number of elements in all arrays must be equal
    :param list gsims:
        List of strings or instance of the openquake.hazardlib.gsim classes
        to representing GMPE names or GMPEs
    :param list imts:
        List of intensity measures
    :param int nsites:
        Number of sites
    :param str stddev:
        Standard deviation type
    :param str filename:
        Name of output file for exporting the figure
    :param str filetype:
        String to indicate file type for exporting the figure
    :param int dpi:
        Dots per inch for export figure
    :param str plot_type:
        Type of plot (only used in distance Trellis)
    :param str distance_type:
        Type of source-site distance to be used in distances trellis
    :param tuple figure_size:
        Size of figure (passed to Matplotlib pyplot.figure() function)
    :param tuple xlim:
        Limits on the x-axis (will apply to all subplot axes)
    :param tuple ylim:
        Limits on the y-axis (will apply to all subplot axes)
    :param float legend_fontsize:
        Controls the fontsize of the legend (default 14)
    :param int ncol:
        Number of columns for the legend (default 1)
    """
    magdist = False

    def __init__(self, magnitudes, distances, gsims, imts, params,
                 stddev="Total", rupture=None, **kwargs):
        # Set default keyword arguments
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('figure_size', FIG_SIZE)
        kwargs.setdefault('xlim', None)
        kwargs.setdefault('ylim', None)
        kwargs.setdefault("legend_fontsize", 14)
        kwargs.setdefault("ncol", 1)
        self.rupture = rupture
        self.magnitudes = magnitudes
        self.distances = distances
        self.gsims = check_gsim_list(gsims)
        self.params = params
        self.imts = imts
        self.dctx = []
        self.rctx = []
        self.sctx = None
        self.nsites = 0
        self._build_ctxs()
        self.stddev = stddev
        self.filename = kwargs['filename']
        self.filetype = kwargs['filetype']
        self.dpi = kwargs['dpi']
        self.plot_type = kwargs['plot_type']
        self.distance_type = kwargs['distance_type']
        self.figure_size = kwargs["figure_size"]
        self.xlim = kwargs["xlim"]
        self.ylim = kwargs["ylim"]
        self.legend_fontsize = kwargs["legend_fontsize"]
        self.ncol = kwargs["ncol"]

    def _build_ctxs(self):

        if not self.magdist:
            dctx = DistancesContext()
            required_dists = []
            for gmpe_name, gmpe in self.gsims.items():
                gsim_distances = [dist for dist in gmpe.REQUIRES_DISTANCES]
                for dist in gsim_distances:
                    if dist not in self.distances:
                        raise ValueError('GMPE %s requires distance type %s'
                                         % (gmpe_name, dist))
                    if dist not in required_dists:
                        required_dists.append(dist)
            dist_check = False
            for dist in required_dists:
                if dist_check and not (
                        len(self.distances[dist]) == self.nsites):
                    raise ValueError("Distances arrays not equal length!")
                else:
                    self.nsites = len(self.distances[dist])
                    dist_check = True
                setattr(dctx, dist, self.distances[dist])
            self.dctx = [dctx for mag in self.magnitudes]
        else:
            # magdist case: there is a distance dictionary for each magnitude
            if isinstance(self.distances, dict):
                # Copy the same distances across
                self.distances = [deepcopy(self.distances)
                                  for mag in self.magnitudes]
            assert (len(self.distances) == len(self.magnitudes))
            # Distances should be a list of dictionaries
            self.dctx = []
            required_distances = []
            for gmpe_name, gmpe in self.gsims.items():
                gsim_distances = [dist for dist in gmpe.REQUIRES_DISTANCES]
                for mag_distances in self.distances:
                    for dist in gsim_distances:
                        if dist not in mag_distances:
                            raise ValueError(
                                'GMPE %s requires distance type %s'
                                % (gmpe_name, dist))

                        if dist not in required_distances:
                            required_distances.append(dist)

            for distance in self.distances:
                dctx = DistancesContext()
                dist_check = False
                for dist in required_distances:
                    if dist_check and not (len(distance[dist]) == self.nsites):
                        raise ValueError("Distances arrays not equal length!")
                    else:
                        self.nsites = len(distance[dist])
                        dist_check = True
                    setattr(dctx, dist, distance[dist])
                self.dctx.append(dctx)

        # If magnitudes was provided with a list of RuptureContexts
        if all([isinstance(mag, RuptureContext)
                for mag in self.magnitudes]):
            # Get all required rupture attributes
            self.rctx = [mag for mag in self.magnitudes]
            for gmpe_name, gmpe in self.gsims.items():
                rup_params = [param
                              for param in gmpe.REQUIRES_RUPTURE_PARAMETERS]
                for rctx in self.rctx:
                    for param in rup_params:
                        if param not in rctx.__dict__:
                            raise ValueError(
                                "GMPE %s requires rupture parameter %s"
                                % (gmpe_name, param))
        else:
            self.rctx = []
            if (not isinstance(self.magnitudes, list) and not
                    isinstance(self.magnitudes, np.ndarray)):
                self.magnitudes = np.array(self.magnitudes)
            # Get all required rupture attributes
            required_attributes = []
            for gmpe_name, gmpe in self.gsims.items():
                rup_params = [param for param in
                              gmpe.REQUIRES_RUPTURE_PARAMETERS]
                for param in rup_params:
                    if param == 'mag':
                        continue
                    elif param not in self.params:
                        raise ValueError(
                            "GMPE %s requires rupture parameter %s"
                            % (gmpe_name, param))
                    elif param not in required_attributes:
                        required_attributes.append(param)
                    else:
                        pass
            for mag in self.magnitudes:
                rup = RuptureContext()
                setattr(rup, 'mag', mag)
                for attr in required_attributes:
                    setattr(rup, attr, self.params[attr])
                self.rctx.append(rup)

        # sites
        slots = set()
        for gmpe in self.gsims.values():
            slots.update(gmpe.REQUIRES_SITES_PARAMETERS)
        self.sctx = SitesContext(slots=slots)
        self.sctx.sids = np.arange(self.nsites)
        required_attributes = []
        for gmpe_name, gmpe in self.gsims.items():
            site_params = [param for param in gmpe.REQUIRES_SITES_PARAMETERS]
            for param in site_params:
                if param not in self.params:
                    raise ValueError("GMPE %s requires site parameter %s"
                                     % (gmpe_name, param))
                elif param not in required_attributes:
                    required_attributes.append(param)
                else:
                    pass
        for param in required_attributes:
            if isinstance(self.params[param], float):
                setattr(self.sctx, param,
                        self.params[param] * np.ones(self.nsites, dtype=float))

            if isinstance(self.params[param], bool):
                if self.params[param]:
                    setattr(self.sctx, param, self.params[param] *
                            np.ones(self.nsites, dtype=bool))
                else:
                    setattr(self.sctx, param, self.params[param] *
                            np.zeros(self.nsites, dtype=bool))
            elif isinstance(self.params[param], Iterable):
                if not len(self.params[param]) == self.nsites:
                    raise ValueError("Length of sites value %s not equal to "
                                     "number of sites %s" %
                                     (param, self.nsites))
                setattr(self.sctx, param, self.params[param])
            else:
                pass

    @classmethod
    def from_rupture_model(cls, rupture, gsims, imts, stddev='Total',
                           **kwargs):
        """
        Constructs the Base Trellis Class from a rupture model
        :param rupture:
            Rupture as instance of the :class:
            smtk.trellis.configure.GSIMRupture
        """
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('xlim', None)
        kwargs.setdefault('ylim', None)
        assert isinstance(rupture, GSIMRupture)
        magnitudes = [rupture.magnitude]
        sctx, rctx, dctx = rupture.get_gsim_contexts()
        # Create distances dictionary
        distances = {}
        for key in dctx._slots_:
            distances[key] = getattr(dctx, key)
        # Add all other parameters to the dictionary
        params = {}
        for key in rctx._slots_:
            params[key] = getattr(rctx, key)
        for key in sctx._slots_:
            params[key] = getattr(sctx, key)
        return cls(magnitudes, distances, gsims, imts, params, stddev,
                   rupture=rupture, **kwargs)

    def plot(self):
        """
        Creates the plot!
        """
        raise NotImplementedError("Cannot create plot of base class!")

    def _get_ylabel(self, imt):
        """
        Returns the label for plotting on a y axis
        """
        raise NotImplementedError


class MagnitudeIMTTrellis(BaseTrellis):
    """
    Class to generate plots showing the scaling of a set of IMTs with
    magnitude
    """
    def __init__(self, magnitudes, distances, gsims, imts, params,
                 stddev="Total", **kwargs):
        """
        Instantiate with list of magnitude and the corresponding distances
        given in a dictionary
        """
        for key in distances:
            if isinstance(distances[key], float):
                distances[key] = np.array([distances[key]])
        super().__init__(
            magnitudes, distances, gsims, imts, params, stddev, **kwargs)

    @classmethod
    def from_rupture_properties(cls, properties, magnitudes, distance,
                                gsims, imts, stddev='Total', **kwargs):
        '''Constructs the Base Trellis Class from a dictionary of
        properties. In this class, this method is simply an alias of
        `from_rupture_model`
        '''
        return cls.from_rupture_model(properties, magnitudes, distance,
                                      gsims, imts, stddev=stddev,
                                      **kwargs)

    @classmethod
    def from_rupture_model(cls, properties, magnitudes, distance, gsims, imts,
                           stddev='Total', **kwargs):
        """
        Implements the magnitude trellis from a dictionary of properties,
        magnitudes and distance
        """
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('xlim', None)
        kwargs.setdefault('ylim', None)
        # Properties
        properties.setdefault("tectonic_region", "Active Shallow Crust")
        properties.setdefault("rake", 0.)
        properties.setdefault("ztor", 0.)
        properties.setdefault("strike", 0.)
        properties.setdefault("msr", WC1994())
        properties.setdefault("initial_point", DEFAULT_POINT)
        properties.setdefault("hypocentre_location", None)
        properties.setdefault("line_azimuth", 90.)
        properties.setdefault("origin_point", (0.5, 0.5))
        properties.setdefault("vs30measured", True)
        properties.setdefault("z1pt0", None)
        properties.setdefault("z2pt5", None)
        properties.setdefault("backarc", False)
        properties.setdefault("distance_type", "rrup")
        # Define a basic rupture configuration
        rup = GSIMRupture(magnitudes[0], properties["dip"],
                          properties["aspect"], properties["tectonic_region"],
                          properties["rake"], properties["ztor"],
                          properties["strike"], properties["msr"],
                          properties["initial_point"],
                          properties["hypocentre_location"])
        # Add the target sites
        _ = rup.get_target_sites_point(distance, properties['distance_type'],
                                       properties["vs30"],
                                       properties["line_azimuth"],
                                       properties["origin_point"],
                                       properties["vs30measured"],
                                       properties["z1pt0"],
                                       properties["z2pt5"],
                                       properties["backarc"])
        # Get the contexts
        sctx, rctx, dctx = rup.get_gsim_contexts()
        # Create an equivalent 'params' dictionary by merging the site and
        # rupture properties
        sctx.__dict__.update(rctx.__dict__)
        for val in dctx.__dict__:
            if getattr(dctx, val) is not None:
                setattr(dctx, val, getattr(dctx, val)[0])
        return cls(magnitudes, dctx.__dict__, gsims, imts, sctx.__dict__,
                   **kwargs)

    def plot(self):
        fig = self.get_fig()
        fig.show()

    def get_fig(self):
        """
        Creates the trellis plot!
        """
        # Determine the optimum number of rows and columns
        nrow, ncol = utils.best_subplot_dimensions(len(self.imts))
        # Get means and standard deviations
        gmvs = self.get_ground_motion_values()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        row_loc = 0
        col_loc = 0
        for i_m in self.imts:
            if col_loc == ncol:
                row_loc += 1
                col_loc = 0
            # Construct the plot
            self._build_plot(
                plt.subplot2grid((nrow, ncol), (row_loc, col_loc)), i_m, gmvs)
            col_loc += 1
        # Add legend
        lgd = plt.legend(self.lines,
                         self.labels,
                         loc=3,
                         bbox_to_anchor=(1.1, 0.),
                         fontsize=self.legend_fontsize,
                         ncol=self.ncol)
        _save_image(self.filename, fig, self.filetype, self.dpi,
                    bbox_extra_artists=(lgd,), bbox_inches="tight")
        return fig

    def _build_plot(self, ax, i_m, gmvs):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        for gmpe_name in self.gsims:
            self.labels.append(gmpe_name)
            line, = ax.semilogy(self.magnitudes, gmvs[gmpe_name][i_m][:, 0],
                                linewidth=2.0, label=gmpe_name)
            self.lines.append(line)
            ax.grid(True)
            if isinstance(self.xlim, tuple):
                ax.set_xlim(self.xlim[0], self.xlim[1])
            else:
                ax.set_xlim(floor(self.magnitudes[0]),
                            ceil(self.magnitudes[-1]))
            if isinstance(self.ylim, tuple):
                ax.set_ylim(self.ylim[0], self.ylim[1])
            self._set_labels(i_m, ax)

    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("Magnitude", fontsize=16)
        ax.set_ylabel(self._get_ylabel(i_m), fontsize=16)

    def _get_ylabel(self, i_m):
        """
        Return the y-label for the magnitude IMT trellis
        """
        if 'SA(' in i_m:
            units = PLOT_UNITS['SA']
        else:
            units = PLOT_UNITS[i_m]
        return "Median {:s} ({:s})".format(i_m, units)

    def to_dict(self):
        """
        Parse the ground motion values to a dictionary
        """
        gmvs = self.get_ground_motion_values()
        nrow, ncol = utils.best_subplot_dimensions(len(self.imts))
        gmv_dict = dict([
            ("xvalues", self.magnitudes.tolist()),
            ("xlabel", "Magnitude")])
        nvals = len(self.magnitudes)
        gmv_dict["figures"] = []
        row_loc = 0
        col_loc = 0
        for im in self.imts:
            if col_loc == ncol:
                row_loc += 1
                col_loc = 0
            # Set the dictionary of y-values
            ydict = {"ylabel": self._get_ylabel(im),
                     "imt": im,
                     "row": row_loc,
                     "column": col_loc,
                     "yvalues": {}}
            for gsim in gmvs:
                if not len(gmvs[gsim][im]):
                    # GSIM missing, set None
                    ydict["yvalues"][gsim] = [None] * nvals
                    continue
                iml_to_list = []
                for val in gmvs[gsim][im].flatten().tolist():
                    if np.isnan(val) or (val < 0.0):
                        iml_to_list.append(None)
                    else:
                        iml_to_list.append(val)
                    ydict["yvalues"][gsim] = iml_to_list
            gmv_dict["figures"].append(ydict)
            col_loc += 1
        return gmv_dict

    def to_json(self):
        """
        Serializes the ground motion values to json
        """
        return json.dumps(self.to_dict())

    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retrieve ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = {}
        for gmpe_name, gmpe in self.gsims.items():
            gmvs[gmpe_name] = {}
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, (rct, dct) in enumerate(zip(self.rctx, self.dctx)):
                    try:
                        means, _ = gmpe.get_mean_and_stddevs(
                            self.sctx,
                            rct,
                            dct,
                            imt.from_string(i_m),
                            [self.stddev])

                        gmvs[gmpe_name][i_m][iloc, :] = np.exp(means)
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = np.array([])
                        break
        return gmvs

    def pretty_print(self, filename=None, sep=","):
        """
        Format the ground motion for printing to file or to screen
        :param str filename:
            Path to file
        :param str sep:
            Separator character
        """
        if filename:
            fid = open(filename, "w")
        else:
            fid = sys.stdout
        # Print Meta information
        self._write_pprint_header_line(fid, sep)
        # Print Distances
        distance_str = sep.join(["{:s}{:s}{:s}".format(key, sep, str(val[0]))
                                 for (key, val) in self.dctx[0].items()])
        fid.write("Distances%s%s\n" % (sep, distance_str))
        # Loop over IMTs
        gmvs = self.get_ground_motion_values()
        for imt in self.imts:
            fid.write("%s\n" % imt)
            header_str = "Magnitude" + sep + sep.join(self.gsims)
            fid.write("%s\n" % header_str)
            for i, mag in enumerate(self.magnitudes):
                data_string = sep.join(["{:.8f}".format(
                     gmvs[gmpe_name][imt][i, 0]) for gmpe_name in self.gsims])
                fid.write("{:s}{:s}{:s}\n".format(str(mag), sep, data_string))
            fid.write("====================================================\n")
        if filename:
            fid.close()

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude IMT Trellis\n")
        fid.write("%s\n" % sep.join([
            "{:s}{:s}{:s}".format(key, sep, str(val))
            for (key, val) in self.params.items()]))


class MagnitudeSigmaIMTTrellis(MagnitudeIMTTrellis):
    """
    Creates the Trellis plot for the standard deviations
    """

    def _build_plot(self, ax, i_m, gmvs):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        for gmpe_name in self.gsims:
            self.labels.append(gmpe_name)
            line, = ax.plot(self.magnitudes,
                            gmvs[gmpe_name][i_m][:, 0],
                            linewidth=2.0,
                            label=gmpe_name)
            self.lines.append(line)
            ax.grid(True)
            if isinstance(self.xlim, tuple):
                ax.set_xlim(self.xlim[0], self.xlim[1])
            else:
                ax.set_xlim(floor(self.magnitudes[0]),
                            ceil(self.magnitudes[-1]))
            if isinstance(self.ylim, tuple):
                ax.set_ylim(self.ylim[0], self.ylim[1])
            self._set_labels(i_m, ax)

    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = {}
        for gmpe_name, gmpe in self.gsims.items():
            gmvs[gmpe_name] = {}
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros([len(self.rctx),
                                                 self.nsites],
                                                dtype=float)
                for iloc, (rct, dct) in enumerate(zip(self.rctx, self.dctx)):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx,
                             rct,
                             dct,
                             imt.from_string(i_m),
                             [self.stddev])
                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except KeyError:
                        gmvs[gmpe_name][i_m] = np.array([], dtype=float)
                        break

        return gmvs

    def _get_ylabel(self, i_m):
        """
        """
        return self.stddev + " Std. Dev. ({:s})".format(str(i_m))

    def _set_labels(self, i_m, ax):
        """
        Sets the axes labels
        """
        ax.set_xlabel("Magnitude", fontsize=16)
        ax.set_ylabel(self._get_ylabel(i_m), fontsize=16)

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude IMT %s Standard Deviations Trellis\n" %
                  self.stddev)
        fid.write("%s\n" % sep.join([
            "{:s}{:s}{:s}".format(key, sep, str(val))
            for (key, val) in self.params.items()]))


class DistanceIMTTrellis(MagnitudeIMTTrellis):
    """
    Trellis class to generate a plot of the GMPE attenuation with distance
    """
    XLABEL = "%s (km)"
    YLABEL = "Median %s (%s)"

    def __init__(self, magnitudes, distances, gsims, imts, params,
                 stddev="Total", **kwargs):
        """
        Instantiation
        """
        if isinstance(magnitudes, float):
            magnitudes = [magnitudes]

        super(DistanceIMTTrellis, self).__init__(magnitudes, distances, gsims,
                                                 imts, params, stddev,
                                                 **kwargs)

    @classmethod
    def from_rupture_properties(cls, properties, magnitude, distances,
                                gsims, imts, stddev='Total', **kwargs):
        """Constructs the Base Trellis Class from a rupture properties.
        It internally creates a Rupture object and calls
        `from_rupture_model`. When not listed, arguments take the same
        values as `from_rupture_model`

        :param distances: a numeric array of chosen distances
        """
        params = {k: properties[k] for k in ['rake', 'initial_point', 'ztor',
                                             'hypocentre_location', 'strike',
                                             'msr', 'tectonic_region']
                  if k in properties}
        rupture = GSIMRupture(magnitude, properties['dip'],
                              properties['aspect'], **params)

        params = {k: properties[k] for k in ['line_azimuth', 'as_log',
                                             'vs30measured', 'z1pt0', 'z2pt5',
                                             'origin_point', 'backarc']
                  if k in properties}
        rupture.get_target_sites_line_from_given_distances(distances,
                                                           properties['vs30'],
                                                           **params)

        return cls.from_rupture_model(rupture, gsims, imts,
                                      stddev=stddev, **kwargs)

    @classmethod
    def from_rupture_model(cls, rupture, gsims, imts, stddev='Total',
                           **kwargs):
        """
        Constructs the Base Trellis Class from a rupture model
        :param rupture:
            Rupture as instance of the :class:
            smtk.trellis.configure.GSIMRupture
        """
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('xlim', None)
        kwargs.setdefault('ylim', None)
        assert isinstance(rupture, GSIMRupture)
        magnitudes = [rupture.magnitude]
        sctx, rctx, dctx = rupture.get_gsim_contexts()
        # Create distances dictionary
        distances = {}
        for key in dctx._slots_:
            distances[key] = getattr(dctx, key)
        # Add all other parameters to the dictionary
        params = {}
        for key in rctx._slots_:
            params[key] = getattr(rctx, key)
        for key in sctx._slots_:
            params[key] = getattr(sctx, key)
        return cls(magnitudes, distances, gsims, imts, params, stddev,
                   **kwargs)

    def _build_plot(self, ax, i_m, gmvs):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        distance_vals = getattr(self.dctx, self.distance_type)

        assert (self.plot_type == "loglog") or (self.plot_type == "semilogy")

        for gmpe_name in self.gsims:
            self.labels.append(gmpe_name)
            if self.plot_type == "semilogy":
                line, = ax.semilogy(distance_vals,
                                    gmvs[gmpe_name][i_m][0, :],
                                    linewidth=2.0,
                                    label=gmpe_name)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]
            else:
                line, = ax.loglog(distance_vals,
                                  gmvs[gmpe_name][i_m][0, :],
                                  linewidth=2.0,
                                  label=gmpe_name)
                min_x = 0.5
                max_x = distance_vals[-1]

            self.lines.append(line)
            ax.grid(True)
            if isinstance(self.xlim, tuple):
                ax.set_xlim(self.xlim[0], self.xlim[1])
            else:
                ax.set_xlim(min_x, max_x)
            if isinstance(self.ylim, tuple):
                ax.set_ylim(self.ylim[0], self.ylim[1])
            self._set_labels(i_m, ax)

    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("%s (km)" % DISTANCE_LABEL_MAP[self.distance_type],
                      fontsize=16)
        ax.set_ylabel(self._get_ylabel(i_m), fontsize=16)

    def _get_ylabel(self, i_m):
        """
        Returns the y-label for the given IMT
        """
        if 'SA(' in i_m:
            units = PLOT_UNITS['SA']
        else:
            units = PLOT_UNITS[i_m]
        return "Median {:s} ({:s})".format(i_m, units)

    def to_dict(self):
        """
        Parses the ground motion values to a dictionary
        """
        gmvs = self.get_ground_motion_values()
        nrow, ncol = utils.best_subplot_dimensions(len(self.imts))
        dist_label = "{:s} (km)".format(DISTANCE_LABEL_MAP[self.distance_type])
        gmv_dict = dict([
            ("xvalues", self.distances[self.distance_type].tolist()),
            ("xlabel", dist_label)])
        gmv_dict["figures"] = []
        row_loc = 0
        col_loc = 0
        for im in self.imts:
            if col_loc == ncol:
                row_loc += 1
                col_loc = 0
            # Set the dictionary of y-values
            ydict = {"ylabel": self._get_ylabel(im),
                     "imt": im,
                     "row": row_loc,
                     "column": col_loc,
                     "yvalues": {}}
            for gsim in gmvs:
                data = [None if np.isnan(val) else val
                        for val in gmvs[gsim][im].flatten()]
                ydict["yvalues"][gsim] = data
            gmv_dict["figures"].append(ydict)
            col_loc += 1
        return gmv_dict

    def to_json(self):
        """
        Exports ground motion values to json
        """
        return json.dumps(self.to_dict())

    def pretty_print(self, filename=None, sep=","):
        """
        Format the ground motion for printing to file or to screen
        :param str filename:
            Path to file
        :param str sep:
            Separator character
        """
        if filename:
            fid = open(filename, "w")
        else:
            fid = sys.stdout
        # Print Meta information
        self._write_pprint_header_line(fid, sep)
        fid.write("Magnitude%s%.2f\n" % (sep, self.magnitudes[0]))
        # Loop over IMTs
        gmvs = self.get_ground_motion_values()
        for im in self.imts:
            fid.write("%s\n" % im)
            header_str = sep.join([key for key in self.distances])
            header_str = "{:s}{:s}{:s}".format(
                header_str,
                sep,
                sep.join(self.gsims))
            fid.write("%s\n" % header_str)
            for i in range(self.nsites):
                dist_string = sep.join(["{:.4f}".format(self.distances[key][i])
                                        for key in self.distances])
                data_string = sep.join(["{:.8f}".format(
                     gmvs[gmpe_name][im][0, i]) for gmpe_name in self.gsims])
                fid.write("{:s}{:s}{:s}\n".format(dist_string,
                                                  sep,
                                                  data_string))
            fid.write("====================================================\n")
        if filename:
            fid.close()

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Distance (km) IMT Trellis\n")
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))


class DistanceSigmaIMTTrellis(DistanceIMTTrellis):
    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = {}
        for gmpe_name, gmpe in self.gsims.items():
            gmvs[gmpe_name] = {}
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros([len(self.rctx),
                                                 self.nsites])
                for iloc, (rct, dct) in enumerate(zip(self.rctx, self.dctx)):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx,
                             rct,
                             dct,
                             imt.from_string(i_m),
                             [self.stddev])
                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = np.array([], dtype=float)
                        break
        return gmvs

    def _build_plot(self, ax, i_m, gmvs):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        distance_vals = getattr(self.dctx, self.distance_type)

        assert (self.plot_type == "loglog") or (self.plot_type == "semilogy")

        for gmpe_name in self.gsims:
            self.labels.append(gmpe_name)
            if self.plot_type == "loglog":
                line, = ax.semilogx(distance_vals,
                                    gmvs[gmpe_name][i_m][0, :],
                                    linewidth=2.0,
                                    label=gmpe_name)
                min_x = 0.5
                max_x = distance_vals[-1]
            else:
                line, = ax.plot(distance_vals,
                                gmvs[gmpe_name][i_m][0, :],
                                linewidth=2.0,
                                label=gmpe_name)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]

            self.lines.append(line)
            ax.grid(True)

            if isinstance(self.xlim, tuple):
                ax.set_xlim(self.xlim[0], self.xlim[1])
            else:
                ax.set_xlim(min_x, max_x)
            if isinstance(self.ylim, tuple):
                ax.set_ylim(self.ylim[0], self.ylim[1])

            self._set_labels(i_m, ax)

    def _get_ylabel(self, i_m):
        """
        """
        return self.stddev + " Std. Dev. ({:s})".format(str(i_m))

    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("%s (km)" % DISTANCE_LABEL_MAP[self.distance_type],
                      fontsize=16)
        ax.set_ylabel(self._get_ylabel(i_m), fontsize=16)

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Distance (km) %s Standard Deviations Trellis\n" %
                  self.stddev)
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))


class MagnitudeDistanceSpectraTrellis(BaseTrellis):
    # In this case the preprocessor needs to be removed
    magdist = True

    def __init__(self, magnitudes, distances, gsims, imts, params,
                 stddev="Total", **kwargs):
        """
        Builds the trellis plots for variation in response spectra with
        magnitude and distance.

        In this case the class is instantiated with a set of magnitudes
        and a dictionary indicating the different distance types.

        :param imts: (numeric list or numpy array)
            the Spectral Acceleration's
            natural period(s) to be used
        """
        imts = ["SA(%s)" % i_m for i_m in imts]
        super().__init__(magnitudes, distances, gsims, imts, params,
                         stddev, **kwargs)

    @classmethod
    def from_rupture_properties(cls, properties, magnitudes, distance,
                                gsims, periods, stddev='Total', **kwargs):
        """Constructs the Base Trellis Class from a dictionary of
        properties. In this class, this method is simply an alias of
        `from_rupture_model`

        :param periods: (numeric list or numpy array)
            the Spectral Acceleration's
            natural period(s) to be used. Note that this parameter
            is called `imt` in `from_rupture_model` where the name
            `imt` has been kept for legacy code compatibility
        """
        return cls.from_rupture_model(properties, magnitudes, distance,
                                      gsims, periods, stddev=stddev,
                                      **kwargs)

    @classmethod
    def from_rupture_model(cls, properties, magnitudes, distances,
                           gsims, imts, stddev='Total', **kwargs):
        """
        Constructs the Base Trellis Class from a rupture model
        :param dict properties:
            Properties of the rupture and sites, including (* indicates
            required): *dip, *aspect, tectonic_region, rake, ztor, strike,
                       msr, initial_point, hypocentre_location, distance_type,
                       vs30, line_azimuth, origin_point, vs30measured, z1pt0,
                       z2pt5, backarc
        :param list magnitudes:
            List of magnitudes
        :param list distances:
            List of distances (the distance type should be specified in the
            properties dict - rrup, by default)
        """
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('xlim', None)
        kwargs.setdefault('ylim', None)
        # Defaults for the properties of the rupture and site configuration
        properties.setdefault("tectonic_region", "Active Shallow Crust")
        properties.setdefault("rake", 0.)
        properties.setdefault("ztor", 0.)
        properties.setdefault("strike", 0.)
        properties.setdefault("msr", WC1994())
        properties.setdefault("initial_point", DEFAULT_POINT)
        properties.setdefault("hypocentre_location", None)
        properties.setdefault("line_azimuth", 90.)
        properties.setdefault("origin_point", (0.5, 0.5))
        properties.setdefault("vs30measured", True)
        properties.setdefault("z1pt0", None)
        properties.setdefault("z2pt5", None)
        properties.setdefault("backarc", False)
        properties.setdefault("distance_type", "rrup")
        distance_dicts = []
        rupture_dicts = []
        for magnitude in magnitudes:
            # Generate the rupture for the specific magnitude
            rup = GSIMRupture(magnitude, properties["dip"],
                              properties["aspect"],
                              properties["tectonic_region"],
                              properties["rake"], properties["ztor"],
                              properties["strike"], properties["msr"],
                              properties["initial_point"],
                              properties["hypocentre_location"])
            distance_dict = None
            for distance in distances:
                # Define the target sites with respect to the rupture
                _ = rup.get_target_sites_point(distance,
                                               properties["distance_type"],
                                               properties["vs30"],
                                               properties["line_azimuth"],
                                               properties["origin_point"],
                                               properties["vs30measured"],
                                               properties["z1pt0"],
                                               properties["z2pt5"],
                                               properties["backarc"])
                sctx, rctx, dctx = rup.get_gsim_contexts()
                if not distance_dict:
                    distance_dict = []
                    for (key, val) in dctx.__dict__.items():
                        distance_dict.append((key, val))
                    distance_dict = dict(distance_dict)
                else:
                    for (key, val) in dctx.__dict__.items():
                        distance_dict[key] = np.hstack([
                                distance_dict[key], val])
            distance_dicts.append(distance_dict)
            rupture_dicts.append(rctx)
        return cls(rupture_dicts, distance_dicts, gsims, imts, properties,
                   stddev, **kwargs)

    def plot(self):
        """
        Create plot!
        """
        nrow = len(self.magnitudes)
        # Get means and standard deviations
        gmvs = self.get_ground_motion_values()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout({"pad": 0.5})
        for rowloc in range(nrow):
            for colloc in range(self.nsites):
                self._build_plot(
                    plt.subplot2grid((nrow, self.nsites), (rowloc, colloc)),
                    gmvs,
                    rowloc,
                    colloc)
        # Add legend
        lgd = plt.legend(self.lines,
                         self.labels,
                         loc=3.,
                         bbox_to_anchor=(1.1, 0.0),
                         fontsize=self.legend_fontsize)

        _save_image(self.filename, fig, self.filetype, self.dpi,
                    bbox_extra_artists=(lgd,), bbox_inches="tight")
        plt.show()

    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retrieve ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = {}
        for gmpe_name, gmpe in self.gsims.items():
            gmvs[gmpe_name] = {}
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, (rct, dct) in enumerate(zip(self.rctx, self.dctx)):
                    try:
                        means, _ = gmpe.get_mean_and_stddevs(
                            self.sctx, rct, dct,
                            imt.from_string(i_m),
                            [self.stddev])

                        gmvs[gmpe_name][i_m][iloc, :] = \
                            np.exp(means)
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = np.array([], dtype=float)
                        break
        return gmvs

    def _build_plot(self, ax, gmvs, rloc, cloc):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        max_period = 0.0
        min_period = np.inf
        for gmpe_name in self.gsims:
            periods = []
            spec = []
            for i_m in self.imts:
                if len(gmvs[gmpe_name][i_m]):
                    periods.append(imt.from_string(i_m).period)
                    spec.append(gmvs[gmpe_name][i_m][rloc, cloc])
            periods = np.array(periods)
            spec = np.array(spec)
            max_period = np.max(periods) if np.max(periods) > max_period else \
                max_period
            min_period = np.min(periods) if np.min(periods) < min_period else \
                min_period

            self.labels.append(gmpe_name)
            # Get spectrum from gmvs
            if self.plot_type == "loglog":
                line, = ax.loglog(periods,
                                  spec,
                                  linewidth=2.0,
                                  label=gmpe_name)
            else:
                line, = ax.semilogy(periods,
                                    spec,
                                    linewidth=2.0,
                                    label=gmpe_name)
            # On the top row, add the distance as a title
            if rloc == 0:
                ax.set_title("%s = %9.1f (km)" %
                             (self.distance_type,
                              self.distances[rloc][self.distance_type][cloc]),
                             fontsize=14)
            # On the last column add a vertical label with magnitude
            if cloc == (self.nsites - 1):
                ax.annotate("M = %s" % self.rctx[rloc].mag,
                            (1.05, 0.5),
                            xycoords="axes fraction",
                            fontsize=14,
                            rotation="vertical")

            self.lines.append(line)
            ax.set_xlim(min_period, max_period)
            if isinstance(self.ylim, tuple):
                ax.set_ylim(self.ylim[0], self.ylim[1])
            ax.grid(True)
            self._set_labels(i_m, ax)

    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("Period (s)", fontsize=14)
        ax.set_ylabel(self._get_ylabel(None), fontsize=14)

    def to_dict(self):
        """
        Export ground motion values to a dictionary
        """
        gmvs = self.get_ground_motion_values()
        periods = [float(val.split("SA(")[1].rstrip(")"))
                   for val in self.imts]

        gmv_dict = dict([
            ("xlabel", "Period (s)"),
            ("xvalues", periods),
            ("figures", [])
            ])

        mags = [rup.mag for rup in self.magnitudes]
        dists = self.distances[0][self.distance_type]
        for i, mag in enumerate(mags):
            for j, dist in enumerate(dists):
                ydict = dict([
                    ("ylabel", self._get_ylabel(None)),  # arg 'None' not used
                    ("magnitude", mag),
                    ("distance", np.around(dist, 3)),
                    ("imt", 'SA'),
                    ("row", i),
                    ("column", j),
                    ("yvalues", dict([(gsim, []) for gsim in gmvs]))
                ])
                for gsim in gmvs:
                    for im in self.imts:
                        if len(gmvs[gsim][im]):
                            value = gmvs[gsim][im][i, j]
                            if np.isnan(value):
                                value = None
                            ydict["yvalues"][gsim].append(value)
                        else:
                            ydict["yvalues"][gsim].append(None)
                gmv_dict["figures"].append(ydict)
        return gmv_dict

    def to_json(self):
        """
        Exports the ground motion values to json
        """
        return json.dumps(self.to_dict())

    def _get_ylabel(self, i_m):
        """
        In this case only the spectra are being shown, so return only the
        Sa (g) label
        """
        return "Sa (g)"

    def pretty_print(self, filename=None, sep=","):
        """
        Format the ground motion for printing to file or to screen
        :param str filename:
            Path to file
        :param str sep:
            Separator character
        """
        if filename:
            fid = open(filename, "w")
        else:
            fid = sys.stdout
        # Print Meta information
        self._write_pprint_header_line(fid, sep)
        # Loop over IMTs
        gmvs = self.get_ground_motion_values()
        # Get the GMPE list header string
        gsim_str = "IMT{:s}{:s}".format(
            sep,
            sep.join(self.gsims))
        for i, mag in enumerate(self.magnitudes):
            for j in range(self.nsites):
                dist_string = sep.join([
                    "{:s}{:s}{:s}".format(dist_type, sep, str(val[j]))
                    for (dist_type, val) in self.distances.items()])
                # Get M-R header string
                mr_header = "Magnitude{:s}{:s}{:s}{:s}".format(sep, str(mag),
                                                               sep,
                                                               dist_string)
                fid.write("%s\n" % mr_header)
                fid.write("%s\n" % gsim_str)
                for imt in self.imts:
                    iml_str = []
                    for gmpe_name in self.gsims:
                        # Need to deal with case that GSIMs don't define
                        # values for the period
                        if len(gmvs[gmpe_name][imt]):
                            iml_str.append("{:.8f}".format(
                                gmvs[gmpe_name][imt][i, j]))
                        else:
                            iml_str.append("-999.000")
                    # Retreived IMT string
                    imt_str = imt.split("(")[1].rstrip(")")
                    iml_str = sep.join(iml_str)
                    fid.write("{:s}{:s}{:s}\n".format(imt_str, sep, iml_str))
                fid.write("================================================\n")
        if filename:
            fid.close()

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude - Distance Spectra (km) IMT Trellis\n")
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))


class MagnitudeDistanceSpectraSigmaTrellis(MagnitudeDistanceSpectraTrellis):

    def _build_plot(self, ax, gmvs, rloc, cloc):
        """
        Plots the lines for a given axis
        :param ax:
            Axes object
        :param str i_m:
            Intensity Measure
        :param dict gmvs:
            Ground Motion Values Dictionary
        """
        self.labels = []
        self.lines = []
        max_period = 0.0
        min_period = np.inf

        for gmpe_name in self.gsims:
            periods = []
            spec = []
            # Get spectrum from gmvs
            for i_m in self.imts:
                if len(gmvs[gmpe_name][i_m]):
                    periods.append(imt.from_string(i_m).period)
                    spec.append(gmvs[gmpe_name][i_m][rloc, cloc])
            periods = np.array(periods)
            spec = np.array(spec)
            self.labels.append(gmpe_name)

            max_period = np.max(periods) if np.max(periods) > max_period else \
                max_period
            min_period = np.min(periods) if np.min(periods) < min_period else \
                min_period

            if self.plot_type == "loglog":
                line, = ax.semilogx(periods,
                                    spec,
                                    linewidth=2.0,
                                    label=gmpe_name)
            else:
                line, = ax.plot(periods,
                                spec,
                                linewidth=2.0,
                                label=gmpe_name)
            # On the top row, add the distance as a title
            if rloc == 0:
                ax.set_title("%s = %9.3f (km)" %
                             (self.distance_type,
                              self.distances[0][self.distance_type][cloc]),
                             fontsize=14)
            # On the last column add a vertical label with magnitude
            if cloc == (self.nsites - 1):
                ax.annotate("M = %s" % self.rctx[rloc].mag,
                            (1.05, 0.5),
                            xycoords="axes fraction",
                            fontsize=14,
                            rotation="vertical")

            self.lines.append(line)
            ax.set_xlim(min_period, max_period)
            if isinstance(self.ylim, tuple):
                ax.set_ylim(self.ylim[0], self.ylim[1])
            ax.grid(True)
            self._set_labels(i_m, ax)

    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = {}
        for gmpe_name, gmpe in self.gsims.items():
            gmvs[gmpe_name] = {}
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros([len(self.rctx), self.nsites],
                                                dtype=float)
                for iloc, (rct, dct) in enumerate(zip(self.rctx, self.dctx)):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx, rct, dct,
                             imt.from_string(i_m),
                             [self.stddev])
                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = np.array([], dtype=float)
                        break
        return gmvs

    def _get_ylabel(self, i_m):
        """
        Returns the standard deviation term (specific to the standard deviation
        type specified for the class)
        """
        return "{:s} Std. Dev.".format(self.stddev)

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude - Distance (km) Spectra %s Standard Deviations "
                  "Trellis\n" % self.stddev)
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))
