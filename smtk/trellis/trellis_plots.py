#!/usr/bin/env python
# LICENSE
#
# Copyright (c) 2010-2014, GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>

'''
Sets up a simple rupture-site configuration to allow for physical comparison
of GMPEs 
'''
import sys, re, os
import numpy as np
from collections import Iterable, OrderedDict
from itertools import cycle
from cycler import cycler
from math import floor, ceil
from sets import Set
import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt
from openquake.hazardlib import gsim, imt
from openquake.hazardlib.gsim.gsim_table import GMPETable
from openquake.hazardlib.scalerel.wc1994 import WC1994
from smtk.sm_utils import _save_image, _save_image_tight
import smtk.trellis.trellis_utils as utils
from smtk.trellis.configure import GSIMRupture

matplotlib.rcParams["axes.prop_cycle"] = \
    cycler(u'color', [u'b', u'g', u'r', u'c', u'm', u'y', u'k',
                      u'b', u'g', u'r', u'c', u'm', u'y', u'k',
                      u'b', u'g', u'r', u'c', u'm', u'y', u'k']) +\
    cycler(u'linestyle', ["-", "-", "-", "-", "-", "-", "-",
                          "--", "--", "--", "--", "--", "--", "--",
                          "-.", "-.", "-.", "-.", "-.", "-.", "-."])
AVAILABLE_GSIMS = gsim.get_available_gsims()

PARAM_DICT = {'magnitudes': [],
              'distances': [],
              'distance_type': 'rjb',
              'vs30': [],
              'strike': None,
              'dip': None,
              'rake': None,
              'ztor': None,
              'hypocentre_location': (0.5, 0.5),
              'hypo_loc': (0.5, 0.5),
              'msr': WC1994()}

PLOT_UNITS = {'PGA': 'g',
              'PGV': 'cm/s',
              'SA': 'g',
              'IA': 'm/s',
              'CSV': 'g-sec',
              'RSD': 's',
              'MMI': ''}

DISTANCE_LABEL_MAP = {'repi': 'Epicentral Dist.',
                      'rhypo': 'Hypocentral Dist.',
                      'rjb': 'Joyner-Boore Dist.',
                      'rrup': 'Rupture Dist.',
                      'rx': 'Rx Dist.'}

FIG_SIZE = (7, 5)

# RESET Axes tick labels
matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)

def simplify_contexts(rupture):
    """
    """
    sctx, rctx, dctx = rupture.get_gsim_contexts()
    sctx.__dict__.update(rctx.__dict__)
    #print dctx.__dict__
    for val in dctx.__dict__:
        if getattr(dctx, val) is not None:
            setattr(dctx, val, getattr(dctx, val)[0])
    return sctx.__dict__, rctx.__dict__, dctx.__dict__


def _get_gmpe_name(gsim):
    """
    Returns the name of the GMPE given an instance of the class
    """
    if gsim.__class__.__name__.startswith("GMPETable"):
        match = re.match(r'^GMPETable\(([^)]+?)\)$', str(gsim))
        filepath = match.group(1).split("=")[1][1:-1]
        return os.path.splitext(filepath.split("/")[-1])[0]
    else:
        return gsim.__class__.__name__

def _check_gsim_list(gsim_list):
    """
    Checks the list of GSIM models and returns an instance of the 
    openquake.hazardlib.gsim class. Raises error if GSIM is not supported in
    OpenQuake
    :param list gsim_list:
        List of GSIM names (str)
    """
    output_gsims = []
    for gsim in gsim_list:
        if gsim.startswith("GMPETable"):
            # Get filename
            match = re.match(r'^GMPETable\(([^)]+?)\)$', gsim)
            filepath = match.group(1).split("=")[1]
            output_gsims.append(GMPETable(gmpe_table=filepath))
        elif not gsim in AVAILABLE_GSIMS.keys():
            raise ValueError('%s Not supported by OpenQuake' % gsim)
        else:
            output_gsims.append(AVAILABLE_GSIMS[gsim]())
    return output_gsims

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
        List of instance of the openquake.hazardlib.gsim classes to represent
        GMPEs
    :param list imts:
        List of intensity measures
    :param dctx:
        Distance context as instance of :class:
            openquake.hazardlib.gsim.base.DistancesContext
    :param rctx:
        Rupture context as instance of :class:
            openquake.hazardlib.gsim.base.RuptureContext
    :param sctx:
        Rupture context as instance of :class:
            openquake.hazardlib.gsim.base.SitesContext
    :param int nsites:
        Number of sites
    :param str stddevs:
        Standard deviation types
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
    """

    def __init__(self, magnitudes, distances, gsims, imts, params,
            stddevs="Total", rupture=None, **kwargs):
        """
        """
        # Set default keyword arguments
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('figure_size', FIG_SIZE)
        self.rupture = rupture
        self.magnitudes = magnitudes
        self.distances = distances
        self.gsims = _check_gsim_list(gsims)
        self.params = params
        self.imts = imts
        self.dctx = None
        self.rctx = None
        self.sctx = None
        self.nsites = 0
        self._preprocess_distances()
        self._preprocess_ruptures()
        self._preprocess_sites()
        self.stddevs = stddevs
        self.filename = kwargs['filename']
        self.filetype = kwargs['filetype']
        self.dpi = kwargs['dpi']
        self.plot_type = kwargs['plot_type']
        self.distance_type = kwargs['distance_type']
        self.figure_size = kwargs["figure_size"]
        self.create_plot()


    def _preprocess_distances(self):
        """
        Preprocesses the input distances to check that all the necessary
        distance types required by the GSIMS are found in the
        DistancesContext()
        """
        self.dctx = gsim.base.DistancesContext()
        required_dists = []
        for gmpe in self.gsims:
            gsim_distances = [dist for dist in gmpe.REQUIRES_DISTANCES]
            for dist in gsim_distances:
                if not dist in self.distances.keys():
                    raise ValueError('GMPE %s requires distance type %s'
                                     % (_get_gmpe_name(gmpe), dist))
                                    # % (gmpe.__class.__.__name__, dist))
                if not dist in required_dists:
                    required_dists.append(dist)
        dist_check = False
        for dist in required_dists:
            if dist_check and not (len(self.distances[dist]) == self.nsites):
                raise ValueError("Distances arrays not equal length!")
            else:
                self.nsites = len(self.distances[dist])
                dist_check = True
            setattr(self.dctx, dist, self.distances[dist])
            
        
    def _preprocess_ruptures(self):
        """
        Preprocesses rupture parameters to ensure all the necessary rupture
        information for the GSIMS is found in the input parameters
        """
        self.rctx = []
        if not isinstance(self.magnitudes, list) and not\
            isinstance(self.magnitudes, np.ndarray):
            self.magnitudes = np.array(self.magnitudes)
        # Get all required rupture attributes
        required_attributes = []
        for gmpe in self.gsims:
            rup_params = [param for param in gmpe.REQUIRES_RUPTURE_PARAMETERS]
            for param in rup_params:
                if param == 'mag':
                    continue
                elif not param in self.params.keys():
                    raise ValueError("GMPE %s requires rupture parameter %s"
                                     % (_get_gmpe_name(gmpe), param))
                                     #% (gmpe.__class__.__name__, param))
                elif not param in required_attributes:
                    required_attributes.append(param)
                else:
                    pass
        for mag in self.magnitudes:
            rup = gsim.base.RuptureContext()
            setattr(rup, 'mag', mag)
            for attr in required_attributes:
                setattr(rup, attr, self.params[attr])
            self.rctx.append(rup)

    def _preprocess_sites(self):
        """
        Preprocesses site parameters to ensure all the necessary rupture
        information for the GSIMS is found in the input parameters
        """
        self.sctx = gsim.base.SitesContext()
        required_attributes = []
        for gmpe in self.gsims:
            site_params = [param for param in gmpe.REQUIRES_SITES_PARAMETERS]
            for param in site_params:
                if not param in self.params.keys():
                    raise ValueError("GMPE %s requires site parameter %s"
                                     % (_get_gmpe_name(gmpe), param))
                                     #% (gmpe.__class__.__name__, param))
                elif not param in required_attributes:
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
                    raise ValueError("Length of sites value %s not equal to"
                                     " number of sites %" % (param, 
                                     self.nsites))
                setattr(self.sctx, param, self.params[param])
            else:
                pass
    
    @classmethod
    def from_rupture_model(cls, rupture, gsims, imts, stddevs='Total',
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
        #for key in sctx.__slots__:
        for key in sctx._slots_:
        #for key in ['vs30', 'vs30measured', 'z1pt0', 'z2pt5']:
            params[key] = getattr(sctx, key)
        return cls(magnitudes, distances, gsims, imts, params, stddevs,
                   rupture=rupture, **kwargs)

    def create_plot(self):
        """
        Creates the plot!
        """
        raise NotImplementedError("Cannot create plot of base class!")


class MagnitudeIMTTrellis(BaseTrellis):
    """
    Class to generate a plots showing the scaling of a set of IMTs with
    magnitude
    """
    def __init__(self, magnitudes, distances, gsims, imts, params,
            stddevs="Total", **kwargs):
        """ 
        """
        for key in distances.keys():
            if isinstance(distances[key], float):
                distances[key] = np.array([distances[key]])
        super(MagnitudeIMTTrellis, self).__init__(magnitudes, distances, gsims,
            imts, params, stddevs, **kwargs)

    def create_plot(self):
        """
        Creates the trellis plot!
        """
        # Determine the optimum number of rows and columns
        nrow, ncol = utils.best_subplot_dimensions(len(self.imts))
        # Get means and standard deviations
        if self.rupture:
            gmvs = self.get_ground_motion_values_from_rupture()
        else:
            gmvs = self.get_ground_motion_values()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        row_loc = 0
        col_loc = 0
        first_plot = True
        for i_m in self.imts:
            if col_loc == ncol:
                row_loc += 1
                col_loc = 0
            # Construct the plot
            self._build_plot(
                plt.subplot2grid((nrow, ncol), (row_loc, col_loc)), 
                i_m, 
                gmvs)
            col_loc += 1
        # Add legend
        lgd = plt.legend(self.lines,
                         self.labels,
                         loc=2,
                         bbox_to_anchor=(1.05, 1.))
        _save_image_tight(fig, lgd, self.filename, self.filetype, self.dpi)
        plt.show()

       
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
        for gmpe in self.gsims:
            #self.labels.append(gmpe.__class__.__name__)
            gmpe_name = _get_gmpe_name(gmpe)
            self.labels.append(gmpe_name)
            line, = ax.semilogy(self.magnitudes,
                        gmvs[gmpe_name][i_m][:, 0],
                        #'-',
                        linewidth=2.0,
                        label=gmpe_name)
            self.lines.append(line)
            ax.grid(True)
            #ax.set_title(i_m, fontsize=12)
            ax.set_xlim(floor(self.magnitudes[0]), ceil(self.magnitudes[-1]))
            self._set_labels(i_m, ax)
     
    def _set_labels(self, i_m, ax):
            """
            Sets the labels on the specified axes
            """
            ax.set_xlabel("Magnitude", fontsize=16)
            if 'SA(' in i_m:
                units = PLOT_UNITS['SA']
            else:
                units = PLOT_UNITS[i_m]
            ax.set_ylabel("Median %s (%s)" % (i_m, units), fontsize=16)
        

    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = OrderedDict()
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    try:
                        means, _ = gmpe.get_mean_and_stddevs(
                            self.sctx,
                            rct,
                            self.dctx,
                            imt.from_string(i_m),
                            [self.stddevs])
                       
                        gmvs[gmpe_name][i_m][iloc, :] = \
                            np.exp(means)
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = []
                        break
        return gmvs

    def get_ground_motion_values_from_rupture(self):
        """
        """
        gmvs = OrderedDict()
        rctx, dctx, sctx = self._get_context_sets()                           
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, (rct, dct, sct) in enumerate(zip(rctx, dctx, sctx)):
                    try:
                        means, _ = gmpe.get_mean_and_stddevs(
                            sct,
                            rct,
                            dct,
                            imt.from_string(i_m),
                            [self.stddevs])

                        gmvs[gmpe_name][i_m][iloc, :] = \
                            np.exp(means)
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = []
                        break
        return gmvs


    def _get_context_sets(self):
        """
        When building from the rupture it is possible that it may be preferable
        to re-build the contexts (e.g. for magnitude scaling).
        """
        # Build context sets 
        rct = []
        dct = []
        sct = []
        for rctx in self.rctx:
            temp_rup = deepcopy(self.rupture)
            # Update mag
            temp_rup.mag = deepcopy(rctx.mag)
            temp_rup.rupture = temp_rup.get_rupture()
            temp_rup.target_sites = None
            # Update target sites
            if temp_rup.target_sites_config["TYPE"] == "Mesh":
                _ = temp_rup.get_target_sites_mesh(
                    *[temp_rup.target_sites_config[key] for key in
                      ["RMAX", "SPACING", "VS30", "VS30MEASURED",
                      "Z1.0", "Z2.5", "BACKARC"]])
            elif temp_rup.target_sites_config["TYPE"] == "Line":
                _ = temp_rup.get_target_sites_line(
                    *[temp_rup.target_sites_config[key] for key in
                    ["RMAX", "SPACING", "VS30", "AZIMUTH", "ORIGIN",
                     "AS_LOG", "VS30MEASURED", "Z1.0", "Z2.5", "BACKARC"]])
            else:
                _ = temp_rup.get_target_sites_point(
                    *[temp_rup.target_sites_config[key] for key in
                     ["R", "RTYPE", "VS30", "AZIMUTH", "ORIGIN", 
                     "VS30MEASURED", "Z1.0", "Z2.5", "BACKARC"]]) 
            s_c, r_c, d_c = temp_rup.get_gsim_contexts()
            rct.append(r_c)
            dct.append(d_c)
            sct.append(s_c)
        return rct, dct, sct
 


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
                                 for (key, val) in self.dctx.__dict__.items()])
        fid.write("Distances%s%s\n" % (sep, distance_str))
        # Loop over IMTs
        if self.rupture:
            gmvs = self.get_ground_motion_values_from_rupture()
        else:
            gmvs = self.get_ground_motion_values()
        for imt in self.imts:
            fid.write("%s\n" % imt)
            header_str = "Magnitude" + sep + sep.join([_get_gmpe_name(gsim)
                                                       for gsim in self.gsims])
            fid.write("%s\n" % header_str)
            for i, mag in enumerate(self.magnitudes):
                data_string = sep.join(["{:.8f}".format(
                     gmvs[_get_gmpe_name(gsim)][imt][i, 0])
                     for gsim in self.gsims])
                fid.write("{:s}{:s}{:s}\n".format(str(mag), sep, data_string))
            fid.write("====================================================\n")
        if filename:
            fid.close()

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude IMT Trellis\n")
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
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
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            self.labels.append(gmpe_name)
            line, = ax.plot(self.magnitudes,
                            gmvs[gmpe_name][i_m][:, 0],
                            #'-',
                            linewidth=2.0,
                            label=gmpe_name)
            self.lines.append(line)
            ax.grid(True)
            #ax.set_title(i_m, fontsize=12)
            ax.set_xlim(floor(self.magnitudes[0]), ceil(self.magnitudes[-1]))
            self._set_labels(i_m, ax)
    
    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = OrderedDict()
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros([len(self.rctx),
                                                 self.nsites],
                                                 dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx,
                             rct,
                             self.dctx,
                             imt.from_string(i_m),
                             [self.stddevs])
                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except KeyError:
                        gmvs[gmpe_name][i_m] = []
                        break

        return gmvs
    
    def get_ground_motion_values_from_rupture(self):
        """
        """
        gmvs = OrderedDict()
        rctx, dctx, sctx = self._get_context_sets()                           
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, (rct, dct, sct) in enumerate(zip(rctx, dctx, sctx)):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                            sct,
                            rct,
                            dct,
                            imt.from_string(i_m),
                            [self.stddevs])

                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = []
                        break
        return gmvs

    def _set_labels(self, i_m, ax):
        """
        Sets the axes labels
        """
        ax.set_xlabel("Magnitude", fontsize=16)
        ax.set_ylabel(self.stddevs + " Std. Dev. ({:s})".format(str(i_m)),
                      fontsize=16)
    
    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude IMT %s Standard Deviations Trellis\n" %
                  self.stddevs)
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))
        
    
class DistanceIMTTrellis(MagnitudeIMTTrellis):
    """
    Trellis class to generate a plot of the GMPE attenuation with distance
    """
    XLABEL = "%s (km)"
    YLABEL = "Median %s (%s)"
    def __init__(self, magnitudes, distances, gsims, imts, params, 
            stddevs="Total", **kwargs):
        """
       
        """
        if isinstance(magnitudes, float):
            magnitudes = [magnitudes]

        super(DistanceIMTTrellis, self).__init__(magnitudes, distances, gsims,
            imts, params, stddevs, **kwargs)
    
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

        assert (self.plot_type=="loglog") or (self.plot_type=="semilogy")
        
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            self.labels.append(gmpe_name)
            if self.plot_type == "semilogy":
                line, = ax.semilogy(distance_vals,
                                  gmvs[gmpe_name][i_m][0, :],
                                  #'-',
                                  linewidth=2.0,
                                  label=gmpe_name)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]
            else:
                line, = ax.loglog(distance_vals,
                                  gmvs[gmpe_name][i_m][0, :],
                                  #'-',
                                  linewidth=2.0,
                                  label=gmpe_name)
                #min_x = distance_vals[0]
                min_x = 0.5
                max_x = distance_vals[-1]

            self.lines.append(line)
            ax.grid(True)
            #ax.set_title(i_m, fontsize=12)
            ax.set_xlim(min_x, max_x)
            self._set_labels(i_m, ax)

        
    def _set_labels(self, i_m, ax):
            """
            Sets the labels on the specified axes
            """
            ax.set_xlabel("%s (km)" % DISTANCE_LABEL_MAP[self.distance_type],
                          fontsize=16)
            if 'SA(' in i_m:
                units = PLOT_UNITS['SA']
            else:
                units = PLOT_UNITS[i_m]
            ax.set_ylabel("Median %s (%s)" % (i_m, units), fontsize=16)

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
        for imt in self.imts:
            fid.write("%s\n" % imt)
            header_str = sep.join([key for key in self.distances])
            header_str = "{:s}{:s}{:s}".format(
                header_str,
                sep,
                sep.join([_get_gmpe_name(gsim) for gsim in self.gsims]))
            fid.write("%s\n" % header_str)
            for i in range(self.nsites):
                dist_string = sep.join(["{:.4f}".format(self.distances[key][i])
                                        for key in self.distances])
                data_string = sep.join(["{:.8f}".format(
                     gmvs[_get_gmpe_name(gsim)][imt][0, i])
                     for gsim in self.gsims])
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
    """

    """
    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of values
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = OrderedDict()
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros([len(self.rctx),
                                                 self.nsites],
                                                 dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx,
                             rct,
                             self.dctx,
                             imt.from_string(i_m),
                             [self.stddevs])
                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = []
                        break
                        
                        
        return gmvs

    def get_ground_motion_values_from_rupture(self):
        """
        """
        gmvs = OrderedDict()
        rctx, dctx, sctx = self._get_context_sets()                           
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, (rct, dct, sct) in enumerate(zip(rctx, dctx, sctx)):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                            sct,
                            rct,
                            dct,
                            imt.from_string(i_m),
                            [self.stddevs])

                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except KeyError:
                        gmvs[gmpe_name][i_m] = []
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

        assert (self.plot_type=="loglog") or (self.plot_type=="semilogy")
        
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            self.labels.append(gmpe_name)
            if self.plot_type == "loglog":
                line, = ax.semilogx(distance_vals,
                                  gmvs[gmpe_name][i_m][0, :],
                                  #'-',
                                  linewidth=2.0,
                                  label=gmpe_name)
                #min_x = distance_vals[0]
                min_x = 0.5
                max_x = distance_vals[-1]
            else:
                line, = ax.plot(distance_vals,
                                gmvs[gmpe_name][i_m][0, :],
                                #'-',
                                linewidth=2.0,
                                label=gmpe_name)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]


            self.lines.append(line)
            ax.grid(True)
            #ax.set_title(i_m, fontsize=12)
            
            ax.set_xlim(min_x, max_x)
            self._set_labels(i_m, ax)

        
    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("%s (km)" % DISTANCE_LABEL_MAP[self.distance_type],
                      fontsize=16)
        ax.set_ylabel(self.stddevs + " Std. Dev. ({:s})".format(str(i_m)),
                      fontsize=16)

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Distance (km) %s Standard Deviations Trellis\n" %
                  self.stddevs)
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))

class MagnitudeDistanceSpectraTrellis(MagnitudeIMTTrellis):
    """

    """
    def __init__(self, magnitudes, distances, gsims, imts, params,
            stddevs="Total", **kwargs):
        """ 
        """
        imts = ["SA(%s)" % i_m for i_m in imts]

        super(MagnitudeDistanceSpectraTrellis, self).__init__(magnitudes, 
            distances, gsims, imts, params, stddevs, **kwargs)
    
#    @classmethod
#    def from_rupture_model(cls, rupture, magnitudes, distances,
#                           gsims, imts, stddevs='Total', **kwargs):
#        """
#        Constructs the Base Trellis Class from a rupture model
#        :param rupture:
#            Rupture as instance of the :class:
#            smtk.trellis.configure.GSIMRupture
#        """
#        kwargs.setdefault('filename', None)
#        kwargs.setdefault('filetype', "png")
#        kwargs.setdefault('dpi', 300)
#        kwargs.setdefault('plot_type', "loglog")
#        kwargs.setdefault('distance_type', "rjb")
#        assert isinstance(rupture, GSIMRupture)
#        #magnitudes = [rupture.magnitude]
#        sctx, rctx, dctx = rupture.get_gsim_contexts()
#        # Create distances dictionary
#        distances = {}
#        for key in dctx._slots_:
#            distances[key] = getattr(dctx, key)
#        # Add all other parameters to the dictionary
#        params = {}
#        for key in rctx._slots_:
#            params[key] = getattr(rctx, key)
#        #for key in sctx.__slots__:
#        for key in sctx._slots_:
#        #for key in ['vs30', 'vs30measured', 'z1pt0', 'z2pt5']:
#            params[key] = getattr(sctx, key)
#        return cls(magnitudes, distances, gsims, imts, params, stddevs,
#                   rupture=rupture)

    def create_plot(self):
        """
        Create plot!
        """
        nrow = len(self.magnitudes)
        # Get means and standard deviations
        #if self.rupture:
        #    gmvs = self.get_ground_motion_values_from_rupture()
        #else:
        gmvs = self.get_ground_motion_values()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout({"pad":0.5})
        for rowloc in xrange(nrow):
            for colloc in xrange(self.nsites):
                self._build_plot(
                    plt.subplot2grid((nrow, self.nsites), (rowloc, colloc)), 
                    gmvs,
                    rowloc,
                    colloc)
        # Add legend
        lgd = plt.legend(self.lines,
                         self.labels,
                         loc=2,
                         bbox_to_anchor=(1.1, 1.))
        _save_image_tight(fig, lgd, self.filename, self.filetype, self.dpi)
        plt.show()



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
        #periods = np.array([imt.from_string(i_m).period
        #                    for i_m in self.imts])
        max_period = 0.0
        min_period = np.inf
        for gmpe in self.gsims:
            periods = []
            spec = []
            gmpe_name = _get_gmpe_name(gmpe)
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
                                  #"-",
                                  linewidth=2.0,
                                  label=gmpe_name)
            else:
                line, = ax.semilogy(periods,
                                    spec,
                                    #"-",
                                    linewidth=2.0,
                                    label=gmpe_name)
            # On the top row, add the distance as a title
            if rloc == 0:
                ax.set_title("%s = %9.3f (km)" %(
                    self.distance_type,
                    self.distances[self.distance_type][cloc]),
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
            ax.grid(True)
            self._set_labels(i_m, ax)


    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("Period (s)", fontsize=14)
        ax.set_ylabel("Sa (g)", fontsize=14)

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
        #if self.rupture:
        #    gmvs = self.get_ground_motion_values_from_rupture()
        #else:
        gmvs = self.get_ground_motion_values()
        # Get the GMPE list header string
        gsim_str = "IMT{:s}{:s}".format(
            sep,
            sep.join([_get_gmpe_name(gsim) for gsim in self.gsims]))
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
                    for gsim in self.gsims:
                        gmpe_name = _get_gmpe_name(gsim)
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
                fid.write("====================================================\n")
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
    """

    """
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

        for gmpe in self.gsims:
            periods = []
            spec = []
            # Get spectrum from gmvs
            gmpe_name = _get_gmpe_name(gmpe)
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

            #spec = np.array([gmvs[gmpe.__class__.__name__][i_m][rloc, cloc]
            #                 for i_m in self.imts])
            if self.plot_type == "loglog":
                line, = ax.semilogx(periods,
                                  spec,
                                  #"-",
                                  linewidth=2.0,
                                  label=gmpe_name)
            else:
                line, = ax.plot(periods,
                                    spec,
                                    #"-",
                                    linewidth=2.0,
                                    label=gmpe_name)
            # On the top row, add the distance as a title
            if rloc == 0:
                ax.set_title("%s = %9.3f (km)" %(
                    self.distance_type,
                    self.distances[self.distance_type][cloc]),
                    fontsize=12)
            # On the last column add a vertical label with magnitude
            if cloc == (self.nsites - 1):
                ax.annotate("M = %s" % self.rctx[rloc].mag,
                            (1.02, 0.5),
                            xycoords="axes fraction",
                            fontsize=12,
                            rotation="vertical")

            self.lines.append(line)
            ax.set_xlim(min_period, max_period)
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
        gmvs = OrderedDict()
        for gmpe in self.gsims:
            gmpe_name = _get_gmpe_name(gmpe)
            gmvs.update([(gmpe_name, {})])
            for i_m in self.imts:
                gmvs[gmpe_name][i_m] = np.zeros([len(self.rctx),
                                                               self.nsites],
                                                               dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    try:
                        _, sigmas = gmpe.get_mean_and_stddevs(
                             self.sctx,
                             rct,
                             self.dctx,
                             imt.from_string(i_m),
                             [self.stddevs])
                        gmvs[gmpe_name][i_m][iloc, :] = sigmas[0]
                    except (KeyError, ValueError):
                        gmvs[gmpe_name][i_m] = []
                        break
        return gmvs
    
    
    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("Period (s)", fontsize=14)
        ax.set_ylabel("%s Std. Dev." % self.stddevs, fontsize=14)

    def _write_pprint_header_line(self, fid, sep=","):
        """
        Write the header lines of the pretty print function
        """
        fid.write("Magnitude - Distance (km) Spectra %s Standard Deviations Trellis\n" %
                  self.stddevs)
        fid.write("%s\n" % sep.join(["{:s}{:s}{:s}".format(key, sep, str(val))
                                     for (key, val) in self.params.items()]))

