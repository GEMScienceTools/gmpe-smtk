#!/usr/bin/env/python
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
import numpy as np
from collections import Iterable, OrderedDict
from math import floor, ceil
from sets import Set
import matplotlib
import matplotlib.pyplot as plt
from openquake.hazardlib import gsim, imt
from openquake.hazardlib.scalerel.wc1994 import WC1994
from smtk.sm_utils import _save_image, _save_image_tight
import smtk.trellis.trellis_utils as utils
from smtk.trellis.configure import GSIMRupture

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
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)


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
        if not gsim in AVAILABLE_GSIMS.keys():
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
            stddevs="Total", **kwargs):
        """
        """
        # Set default keyword arguments
        kwargs.setdefault('filename', None)
        kwargs.setdefault('filetype', "png")
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('plot_type', "loglog")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault('figure_size', FIG_SIZE)

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
                                     % (gmpe.__class.__.__name__, dist))
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
                                     % (gmpe.__class__.__name__, param))
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
                                     % (gmpe.__class__.__name__, param))
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
        for key in dctx.__slots__:
            distances[key] = getattr(dctx, key)
        # Add all other parameters to the dictionary
        params = {}
        for key in rctx.__slots__:
            params[key] = getattr(rctx, key)
        #for key in sctx.__slots__:
        for key in ['vs30', 'vs30measured', 'z1pt0', 'z2pt5']:
            params[key] = getattr(sctx, key)
        return cls(magnitudes, distances, gsims, imts, params, stddevs, 
                **kwargs)

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
                         bbox_to_anchor=(1.1, 1.))
        #fig.savefig(self.filename, bbox_extra_artists=(lgd,),
        #            bbox_inches="tight",
        #            dpi=self.dpi, format=self.filetype)
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
            self.labels.append(gmpe.__class__.__name__)
            line, = ax.semilogy(self.magnitudes,
                        gmvs[gmpe.__class__.__name__][i_m][:, 0],
                        '-',
                        linewidth=2.0,
                        label=gmpe.__class__.__name__)
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
            
            gmvs.update([(gmpe.__class__.__name__, {})])
            for i_m in self.imts:
                gmvs[gmpe.__class__.__name__][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    means, _ = gmpe.get_mean_and_stddevs(self.sctx,
                                                         rct,
                                                         self.dctx,
                                                         imt.from_string(i_m),
                                                         [self.stddevs])
                    gmvs[gmpe.__class__.__name__][i_m][iloc, :] = np.exp(means)
                   
        return gmvs


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
            self.labels.append(gmpe.__class__.__name__)
            line, = ax.plot(self.magnitudes,
                            gmvs[gmpe.__class__.__name__][i_m][:, 0],
                            '-',
                            linewidth=2.0,
                            label=gmpe.__class__.__name__)
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
            gmvs.update([(gmpe.__class__.__name__, {})])
            for i_m in self.imts:
                gmvs[gmpe.__class__.__name__][i_m] = np.zeros([len(self.rctx),
                                                               self.nsites],
                                                               dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    _, sigmas = gmpe.get_mean_and_stddevs(
                         self.sctx,
                         rct,
                         self.dctx,
                         imt.from_string(i_m),
                         [self.stddevs])
                    gmvs[gmpe.__class__.__name__][i_m][iloc, :] = sigmas[0]
        return gmvs

    def _set_labels(self, i_m, ax):
        """
        Sets the axes labels
        """
        ax.set_xlabel("Magnitude", fontsize=16)
        ax.set_ylabel(self.stddevs + " Std. Dev.", fontsize=16)
        
    
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
            self.labels.append(gmpe.__class__.__name__)
            if self.plot_type == "semilogy":
                line, = ax.semilogy(distance_vals,
                                  gmvs[gmpe.__class__.__name__][i_m][0, :],
                                  '-',
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]
            else:
                line, = ax.loglog(distance_vals,
                                  gmvs[gmpe.__class__.__name__][i_m][0, :],
                                  '-',
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
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
            if 'SA(' in i_m:
                units = PLOT_UNITS['SA']
            else:
                units = PLOT_UNITS[i_m]
            ax.set_ylabel("Median %s (%s)" % (i_m, units), fontsize=16)


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
            gmvs.update([(gmpe.__class__.__name__, {})])
            for i_m in self.imts:
                gmvs[gmpe.__class__.__name__][i_m] = np.zeros([len(self.rctx),
                                                               self.nsites],
                                                               dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    _, sigmas = gmpe.get_mean_and_stddevs(
                         self.sctx,
                         rct,
                         self.dctx,
                         imt.from_string(i_m),
                         [self.stddevs])
                    gmvs[gmpe.__class__.__name__][i_m][iloc, :] = sigmas[0]
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
            self.labels.append(gmpe.__class__.__name__)
            if self.plot_type == "loglog":
                line, = ax.semilogx(distance_vals,
                                  gmvs[gmpe.__class__.__name__][i_m][0, :],
                                  '-',
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
                min_x = distance_vals[0]
                max_x = distance_vals[-1]
            else:
                line, = ax.plot(distance_vals,
                                gmvs[gmpe.__class__.__name__][i_m][0, :],
                                '-',
                                linewidth=2.0,
                                label=gmpe.__class__.__name__)
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
        ax.set_ylabel(self.stddevs + " Std. Dev.", fontsize=16)


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

    def create_plot(self):
        """
        Create plot!
        """
        nrow = len(self.magnitudes)
        # Get means and standard deviations
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
        #fig.text(0.5, 0.0, "Period", fontsize=18, ha='center', va='center')
        #fig.text(0.0, 0.5, "Sa (g)", fontsize=18, ha='center', va='center',
        #         rotation='vertical')
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
        periods = np.array([imt.from_string(i_m).period
                            for i_m in self.imts])

        for gmpe in self.gsims:
            self.labels.append(gmpe.__class__.__name__)
            # Get spectrum from gmvs
            spec = np.array([gmvs[gmpe.__class__.__name__][i_m][rloc, cloc]
                             for i_m in self.imts])
            if self.plot_type == "loglog":
                line, = ax.loglog(periods,
                                  spec,
                                  "-",
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
            else:
                line, = ax.semilogy(periods,
                                    spec,
                                    "-",
                                    linewidth=2.0,
                                    label=gmpe.__class__.__name__)
            # On the top row, add the distance as a title
            if rloc == 0:
                ax.set_title("%s = %9.3f (km)" %(
                    self.distance_type,
                    self.distances[self.distance_type][cloc]),
                    fontsize=12)
            # On the last column add a vertical label with magnitude
            if cloc == (self.nsites - 1):
                ax.annotate("M = %s" % self.rctx[rloc].mag,
                            (1.05, 0.5),
                            xycoords="axes fraction",
                            fontsize=12,
                            rotation="vertical")

            self.lines.append(line)
            #if self.plot_type == "loglog":
            #    ax.set_xlim(10.0 ** floor(np.log10(periods[0])), 
            #                10.0 ** ceil(np.log10(periods[-1])))
            #else:
            ax.set_xlim(periods[0], periods[-1])
            ax.grid(True)
            self._set_labels(i_m, ax)


    def _set_labels(self, i_m, ax):
            """
            Sets the labels on the specified axes
            """
            ax.set_xlabel("Period (s)", fontsize=12)
            ax.set_ylabel("Sa (g)", fontsize=12)


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
        periods = np.array([imt.from_string(i_m).period
                            for i_m in self.imts])

        for gmpe in self.gsims:
            self.labels.append(gmpe.__class__.__name__)
            # Get spectrum from gmvs
            spec = np.array([gmvs[gmpe.__class__.__name__][i_m][rloc, cloc]
                             for i_m in self.imts])
            if self.plot_type == "loglog":
                line, = ax.semilogx(periods,
                                  spec,
                                  "-",
                                  linewidth=2.0,
                                  label=gmpe.__class__.__name__)
            else:
                line, = ax.plot(periods,
                                    spec,
                                    "-",
                                    linewidth=2.0,
                                    label=gmpe.__class__.__name__)
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
            #if self.plot_type == "loglog":
            #    ax.set_xlim(10.0 ** floor(np.log10(periods[0])), 
            #                10.0 ** ceil(np.log10(periods[-1])))
            #else:
            ax.set_xlim(periods[0], periods[-1])
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
            gmvs.update([(gmpe.__class__.__name__, {})])
            for i_m in self.imts:
                gmvs[gmpe.__class__.__name__][i_m] = np.zeros([len(self.rctx),
                                                               self.nsites],
                                                               dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    _, sigmas = gmpe.get_mean_and_stddevs(
                         self.sctx,
                         rct,
                         self.dctx,
                         imt.from_string(i_m),
                         [self.stddevs])
                    gmvs[gmpe.__class__.__name__][i_m][iloc, :] = sigmas[0]
        return gmvs
    
    
    def _set_labels(self, i_m, ax):
        """
        Sets the labels on the specified axes
        """
        ax.set_xlabel("Period (s)", fontsize=16)
        ax.set_ylabel("%s Std. Dev." % self.stddevs, fontsize=16)
