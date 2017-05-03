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
Class to hold GMPE residual plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
from math import floor, ceil
from scipy.stats import norm, linregress
from smtk.sm_utils import _save_image
from smtk.residuals.gmpe_residuals import (Residuals,
                                           Likelihood,
                                           SingleStationAnalysis)


class ResidualPlot(object):
    """
    Class to create a simple histrogram of strong ground motion residuals 
    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
        dpi=300, **kwargs):
        """
        :param residuals:
            Residuals as instance of :class: smtk.gmpe_residuals.Residuals
        :param str gmpe:
            Choice of GMPE
        :param str imt:
            Choice of IMT
        """
        kwargs.setdefault('plot_type', "log")
        kwargs.setdefault('distance_type', "rjb")
        kwargs.setdefault("figure_size", (7, 5))
        kwargs.setdefault("show", True)
        self._assertion_check(residuals)
        self.residuals = residuals
        if not gmpe in residuals.gmpe_list:
            raise ValueError("No residual data found for GMPE %s" % gmpe)
        if not imt in residuals.imts:
            raise ValueError("No residual data found for IMT %s" % imt)
        self.gmpe = gmpe
        self.imt = imt
        self.filename = filename
        self.filetype = filetype
        self.dpi = dpi
        self.num_plots = len(residuals.types[gmpe][imt])
        self.distance_type = kwargs["distance_type"]
        self.plot_type = kwargs["plot_type"]
        self.figure_size = kwargs["figure_size"]
        self.show = kwargs["show"]
        self.create_plot()

    def _assertion_check(self, residuals):
        """
        Checks that residuals is an instance of the residuals class
        """
        assert isinstance(residuals, Residuals)

    def create_plot(self, bin_width=0.5):
        """
        Creates a histogram plot
        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        if not data:
            print("No residuals found for %s (%s)" % (self.gmpe, self.imt))
            return
        statistics = self.residuals.get_residual_statistics()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            self._density_plot(
                plt.subplot(nrow, ncol, tloc),
                data,
                res_type,
                statistics,
                bin_width)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()
           
    def _density_plot(self, ax, data, res_type, statistics, bin_width=0.5):
        """
        Plots the density distribution on the subplot axis
        """
        vals, bins = self.get_histogram_data(data[res_type], bin_width)
        ax.bar(bins[:-1], vals, width=0.95 * bin_width, color="LightSteelBlue",
               edgecolor="k")
        # Get equivalent normal distribution
        mean = statistics[self.gmpe][self.imt][res_type]["Mean"]
        stddev = statistics[self.gmpe][self.imt][res_type]["Std Dev"]
        #print mean, stddev
        xdata = np.arange(bins[0], bins[-1] + 0.01, 0.01)
        ax.plot(xdata, norm.pdf(xdata, mean, stddev), '-',
                color="LightSlateGrey", linewidth=2.0)
        ax.plot(xdata, norm.pdf(xdata, 0.0, 1.0), 'k-', linewidth=2.0)
        ax.set_xlabel("Z (%s)" % self.imt, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        title_string = "%s - %s\n Mean = %7.3f, Std Dev = %7.3f" %(self.gmpe,
                                                                   res_type,
                                                                   mean,
                                                                   stddev)
        ax.set_title(title_string, fontsize=12)
        
    def get_histogram_data(self, data, bin_width=0.5):
        """
        Retreives the histogram of the residuals
        """
        bins = np.arange(np.floor(np.min(data)),
                         np.ceil(np.max(data)) + bin_width,
                         bin_width)
        vals = np.histogram(data, bins, density=True)[0]
        return vals.astype(float), bins
        

class LikelihoodPlot(ResidualPlot):
    """

    """

    def _assertion_check(self, residuals):
        """

        """
        assert isinstance(residuals, Likelihood)

    def create_plot(self, bin_width=0.1):
        """
        Creates a histogram plot
        """
        #data = self.residuals.residuals[self.gmpe][self.imt]
        lh_vals, statistics = self.residuals.get_likelihood_values()
        lh_vals = lh_vals[self.gmpe][self.imt]
        statistics = statistics[self.gmpe][self.imt]
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in lh_vals.keys():
            self._density_plot(
                plt.subplot(nrow, ncol, tloc),
                lh_vals[res_type],
                res_type,
                statistics[res_type],
                bin_width)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()
        

    def _density_plot(self, ax, lh_values, res_type, statistics, 
            bin_width=0.1):
        """
        """
        vals, bins = self.get_histogram_data(lh_values, bin_width)
        ax.bar(bins[:-1], vals, width=0.95 * bin_width, color="LightSteelBlue",
               edgecolor="k")
        # Get equivalent normal distribution
        median_lh = statistics["Median LH"]
        ax.set_xlabel("LH (%s)" % self.imt, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlim(0., 1.0)
        title_string = "%s - %s\n Median LH = %7.3f" %(self.gmpe,
                                                       res_type,
                                                       median_lh)
        ax.set_title(title_string, fontsize=12)
    
    
    def get_histogram_data(self, lh_values, bin_width=0.1):
        """

        """
        bins = np.arange(0.0, 1.0 + bin_width, bin_width)
        vals = np.histogram(lh_values, bins, density=True)[0]
        return vals.astype(float), bins


class ResidualWithDistance(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithDistance, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """

        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        if not data:
            print("No residuals found for %s (%s)" % (self.gmpe, self.imt))
            return

        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            distances = self._get_distances(self.gmpe, self.imt, res_type)
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                distances,
                data,
                res_type)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, ax, distances, data, res_type):
        """

        """
        slope, intercept, _, pval, _ = linregress(distances, data[res_type])
        print("Distance (%s): a = %.5f  b = %.5f  p = %.5f" % (res_type,
            intercept, slope, pval))
        model_x = np.arange(np.min(distances),
                            np.max(distances) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        if self.plot_type == "log":
            ax.semilogx(distances,
                        data[res_type],
                        'o',
                        markeredgecolor='Gray',
                        markerfacecolor='LightSteelBlue')
            ax.semilogx(model_x, model_y, 'r-', linewidth=2.0)
            ax.set_xlim(0.1, 10.0 ** (ceil(np.log10(np.max(distances)))))
        else:
            ax.plot(distances,
                    data[res_type],
                    'o',
                    markeredgecolor='Gray',
                    markerfacecolor='LightSteelBlue')
            ax.plot(model_x, model_y, 'r-', linewidth=2.0)
            if self.distance_type == "rcdpp":
                ax.set_xlim(np.min(distances), np.max(distances))
            else:
                ax.set_xlim(0, np.max(distances))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("%s Distance (km)" % self.distance_type, fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        #title_string = "%s - %s (p = %.5e)" %(self.gmpe, res_type, pval)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_distances(self, gmpe, imt, res_type):
        """

        """
        distances = np.array([])
        for i, ctxt in enumerate(self.residuals.contexts):
            # Get the distances
            if res_type == "Inter event":
                ctxt_dist = getattr(ctxt["Distances"], self.distance_type)[
                    self.residuals.unique_indices[gmpe][imt][i]]
                distances = np.hstack([distances, ctxt_dist])
            else:
                distances = np.hstack([
                    distances,
                    getattr(ctxt["Distances"], self.distance_type)
                    ])
        return distances


class ResidualWithMagnitude(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithMagnitude, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """
        Creates the plot
        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        if not data:
            print("No residuals found for %s (%s)" % (self.gmpe, self.imt))
            return

        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            magnitudes = self._get_magnitudes(self.gmpe, self.imt, res_type)
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                magnitudes,
                data,
                res_type)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, ax, magnitudes, data, res_type):
        """
        Plots the residuals with magnitude
        """
        slope, intercept, _, pval, _ = linregress(magnitudes, data[res_type])
        print("Magnitude (%s): a = %.5f  b = %.5f  p = %.5f" % (res_type,
            intercept, slope, pval))
        model_x = np.arange(np.min(magnitudes),
                            np.max(magnitudes) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        ax.plot(magnitudes,
                data[res_type],
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue')
        ax.plot(model_x, model_y, 'r-', linewidth=2.0)
        ax.set_xlim(floor(np.min(magnitudes)), ceil(np.max(magnitudes)))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("Magnitude", fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        #title_string = "%s - %s (p = %.5e)" %(self.gmpe, res_type, pval)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_magnitudes(self, gmpe, imt, res_type):
        """
        Returns an array of magnitudes equal in length to the number of
        residuals
        """
        magnitudes = np.array([])
        for i, ctxt in enumerate(self.residuals.contexts):
            if res_type == "Inter event":

                nval = np.ones(
                    len(self.residuals.unique_indices[gmpe][imt][i])
                    )
            else:
                nval = np.ones(len(ctxt["Distances"].repi))

            magnitudes = np.hstack([magnitudes, ctxt["Rupture"].mag * nval])
                #magnitudes,
                #ctxt["Rupture"].mag * np.ones(len(ctxt["Distances"].repi))])
        return magnitudes
    


class ResidualWithDepth(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithDepth, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """
        Creates the plot
        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        if not data:
            print("No residuals found for %s (%s)" % (self.gmpe, self.imt))
            return

        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            depths = self._get_depths(self.gmpe, self.imt, res_type)
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                depths,
                data,
                res_type)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, ax, depths, data, res_type):
        """
        Plots the residuals with magnitude
        """
        slope, intercept, _, pval, _ = linregress(depths, data[res_type])
        print("Depth (%s): a = %.5f  b = %.5f  p = %.5f" % (res_type,
            intercept, slope, pval))
        model_x = np.arange(np.min(depths),
                            np.max(depths) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        ax.plot(depths,
                data[res_type],
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue')
        ax.plot(model_x, model_y, 'r-', linewidth=2.0)
        ax.set_xlim(floor(np.min(depths)), ceil(np.max(depths)))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("Hypocentral Depth (km)", fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_depths(self, gmpe, imt, res_type):
        """
        Returns an array of magnitudes equal in length to the number of
        residuals
        """
        depths = np.array([])
        for i, ctxt in enumerate(self.residuals.contexts):
            if res_type == "Inter event":
                nvals = np.ones(
                    len(self.residuals.unique_indices[gmpe][imt][i]))
            else:
                nvals = np.ones(len(ctxt["Distances"].repi))
            # TODO This hack needs to be fixed!!!
            if not ctxt["Rupture"].hypo_depth:
                depths = np.hstack([depths, 10.0 * nvals])
            else:
                depths = np.hstack([depths,
                                    ctxt["Rupture"].hypo_depth * nvals])
#                    depths, 
#                    10.0 * np.ones(len(ctxt["Distances"].repi))])
#            else:
#                depths = np.hstack([
#                    depths,
#                    ctxt["Rupture"].hypo_depth *
#                    np.ones(len(ctxt["Distances"].repi))])
        return depths



class ResidualWithVs30(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithVs30, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)

    def create_plot(self):
        """

        """
        data = self.residuals.residuals[self.gmpe][self.imt]
        if not data:
            print("No residuals found for %s (%s)" % (self.gmpe, self.imt))
            return

        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in data.keys():
            vs30 = self._get_vs30(self.gmpe, self.imt, res_type)
            self._residual_plot(
                plt.subplot(nrow, ncol, tloc),
                vs30,
                data,
                res_type)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, ax, vs30, data, res_type):
        """

        """
        slope, intercept, _, pval, _ = linregress(vs30, data[res_type])
        print("Site (%s): a = %.5f  b = %.5f  p = %.5f" % (res_type,
            intercept, slope, pval))
        model_x = np.arange(np.min(vs30),
                            np.max(vs30) + 1.0,
                            1.0)
        model_y = intercept + slope * model_x
        ax.plot(vs30,
                data[res_type],
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue')
        ax.plot(model_x, model_y, 'r-', linewidth=2.0)
        ax.set_xlim(0.1, np.max(vs30))
        max_lim = ceil(np.max(np.fabs(data[res_type])))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_xlabel("Vs30 (m/s)", fontsize=12)
        ax.set_ylabel("Z (%s)" % self.imt, fontsize=12)
        title_string = "%s - %s\n Slope = %.4e, Intercept = %7.3f"\
                       " p = %.6e " % (self.gmpe, res_type, slope, intercept,
                                       pval)
        ax.set_title(title_string, fontsize=12)

    def _get_vs30(self, gmpe, imt, res_type):
        """

        """
        vs30 = np.array([])
        for i, ctxt in enumerate(self.residuals.contexts):
            if res_type == "Inter event":
                vs30 = np.hstack([vs30, ctxt["Sites"].vs30[
                    self.residuals.unique_indices[gmpe][imt][i]]])
            else:
                vs30 = np.hstack([vs30, ctxt["Sites"].vs30])
        return vs30


class ResidualWithSite(ResidualPlot):
    """
    Class uses Single-Station residuals to plot residual for specific sites
    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(ResidualWithSite, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)
    
    def _assertion_check(self, residuals):
        """
        Checks that residuals is en instance of the residuals class
        """
        assert isinstance(residuals, SingleStationAnalysis)
    
    def create_plot(self):
        """

        """
        phi_ss, phi_s2ss = self.residuals.residual_statistics()
        data = self._get_site_data()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in self.residuals.types[self.gmpe][self.imt]:
            self._residual_plot(
                fig.add_subplot(nrow, ncol, tloc),
                data,
                res_type)
            tloc += 1
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, ax, data, res_type):
        """

        """
        xmean = np.array([data[site_id]["x-val"][0]
                          for site_id in self.residuals.site_ids])

        yvals = np.array([])
        xvals = np.array([])
        for site_id in self.residuals.site_ids:
            xvals = np.hstack([xvals, data[site_id]["x-val"]])
            yvals = np.hstack([yvals, data[site_id][res_type]])
        ax.plot(xvals,
                yvals,
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue',
                zorder=-32)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")

        max_lim = ceil(np.max(np.fabs(yvals)))
        ax.set_ylim(-max_lim, max_lim)
        ax.set_ylabel("%s" % res_type, fontsize=12)
        ax.grid()
        title_string = "%s - %s - %s Residual" % (self.gmpe,
                                                  self.imt,
                                                  res_type)
        ax.set_title(title_string, fontsize=12)

    def _get_site_data(self):
        """

        """
        data = OrderedDict([(site_id, {}) 
                            for site_id in self.residuals.site_ids])
        for iloc, site_resid in enumerate(self.residuals.site_residuals):
            resid = deepcopy(site_resid)

            site_id = self.residuals.site_ids[iloc]
            n_events = resid.site_analysis[self.gmpe][self.imt]["events"]
            data[site_id]["Total"] = (
                resid.site_analysis[self.gmpe][self.imt]["Total"] /
                resid.site_analysis[self.gmpe][self.imt]["Expected Total"])
            if "Intra event" in\
                resid.site_analysis[self.gmpe][self.imt].keys():
                data[site_id]["Inter event"] = (
                    resid.site_analysis[self.gmpe][self.imt]["Inter event"] /
                    resid.site_analysis[self.gmpe][self.imt]["Expected Inter"])
                data[site_id]["Intra event"] = (
                    resid.site_analysis[self.gmpe][self.imt]["Intra event"] /
                    resid.site_analysis[self.gmpe][self.imt]["Expected Intra"])

            data[site_id]["ID"] = self.residuals.site_ids[iloc]
            data[site_id]["N"] = n_events
            data[site_id]["x-val"] =(float(iloc) + 0.5) *\
                np.ones_like(data[site_id]["Intra event"])
        return data

class IntraEventResidualWithSite(ResidualPlot):
    """

    """
    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png", 
         dpi=300, **kwargs):
         """

         """
         super(IntraEventResidualWithSite, self).__init__(residuals, gmpe, imt,
             filename, filetype, dpi, **kwargs)
    
    def _assertion_check(self, residuals):
        """
        Checks that residuals is an instance of the residuals class
        """
        assert isinstance(residuals, SingleStationAnalysis)
    
    def create_plot(self):
        """
        Creates the plot
        """
        phi_ss, phi_s2ss = self.residuals.residual_statistics()
        data = self._get_site_data()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        
        self._residual_plot(fig, data,
                            phi_ss[self.gmpe][self.imt],
                            phi_s2ss[self.gmpe][self.imt])
        _save_image(self.filename, self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, fig, data, phi_ss, phi_s2ss):
        """
        Creates three plots:
        1) Plot of the intra-event residual for each station
        2) Plot of the site term
        3) Plot of the remainder-residual
        """
        dwess = np.array([])
        dwoess = np.array([])
        ds2ss = []
        xvals = np.array([])
        for site_id in self.residuals.site_ids:
            xvals = np.hstack([xvals, data[site_id]["x-val"]])
            dwess = np.hstack([dwess, data[site_id]["Intra event"]])
            dwoess = np.hstack([dwoess, data[site_id]["dWo,es"]])
            ds2ss.append(data[site_id]["dS2ss"])
        ds2ss = np.array(ds2ss)
        ax = fig.add_subplot(311)
        # Show intra-event residuals
        mean = np.array([np.mean(data[site_id]["Intra event"])
                         for site_id in self.residuals.site_ids])
        stddevs = np.array([np.std(data[site_id]["Intra event"])
                            for site_id in self.residuals.site_ids])
        xmean = np.array([data[site_id]["x-val"][0]
                          for site_id in self.residuals.site_ids])

        ax.plot(xvals,
                dwess,
                'x',
                markeredgecolor='k',
                markerfacecolor='k',
                markersize=8,
                zorder=-32)
        ax.errorbar(xmean, mean, yerr=stddevs,
                    ecolor="r", elinewidth=3.0, barsabove=True,
                    fmt="s", mfc="r", mec="k", ms=6)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(dwess)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{es}$ (%s)' % self.imt, fontsize=12)
        phi = np.std(dwess)
        ax.plot(xvals, phi * np.ones(len(xvals)), 'k--', linewidth=2.)
        ax.plot(xvals, -phi * np.ones(len(xvals)), 'k--', linewidth=2.)
        title_string = "%s - %s (Std Dev = %8.5f)" % (self.gmpe, self.imt, phi)
        ax.set_title(title_string, fontsize=16)
        # Show delta s2ss
        ax = fig.add_subplot(312)
        ax.plot(xmean,
                ds2ss,
                's',
                markeredgecolor='k',
                markerfacecolor='LightSteelBlue',
                markersize=8,
                zorder=-32)
        ax.plot(xmean, 
                (phi_s2ss["Mean"] - phi_s2ss["StdDev"]) * np.ones(len(xmean)),
                "k--", linewidth=1.5)
        ax.plot(xmean, 
                (phi_s2ss["Mean"] + phi_s2ss["StdDev"]) * np.ones(len(xmean)),
                "k--", linewidth=1.5)
        ax.plot(xmean, 
                (phi_s2ss["Mean"]) * np.ones(len(xmean)),
                "k-", linewidth=2)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(ds2ss)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta S2S_S$ (%s)' % self.imt, fontsize=12)
        title_string = r'%s - %s ($\phi_{S2S}$ = %8.5f)' % (self.gmpe,
            self.imt, phi_s2ss["StdDev"])
        ax.set_title(title_string, fontsize=16)
        # Show dwoes
        ax = fig.add_subplot(313)
        ax.plot(xvals,
                dwoess,
                'x',
                markeredgecolor='k',
                markerfacecolor='k',
                markersize=8,
                zorder=-32)
        ax.plot(xmean, -phi_ss * np.ones(len(xmean)), "k--", linewidth=1.5)
        ax.plot(xmean, phi_ss * np.ones(len(xmean)), "k--", linewidth=1.5)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(dwoess)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{o,es} = \delta W_{es} - \delta S2S_S$ (%s)'
                      % self.imt, 
                      fontsize=12)
        title_string = r'%s - %s ($\phi_{SS}$ = %8.5f)' % (self.gmpe,
                                                           self.imt,
                                                           phi_ss)
        ax.set_title(title_string, fontsize=16)

    def _get_site_data(self):
        """

        """
        data = OrderedDict([(site_id, {}) 
                            for site_id in self.residuals.site_ids])
        for iloc, site_resid in enumerate(self.residuals.site_residuals):
            resid = deepcopy(site_resid)

            site_id = self.residuals.site_ids[iloc]
            n_events = resid.site_analysis[self.gmpe][self.imt]["events"]
            data[site_id] = resid.site_analysis[self.gmpe][self.imt]
            data[site_id]["ID"] = self.residuals.site_ids[iloc]
            data[site_id]["N"] = n_events
            data[site_id]["Intra event"] =\
                resid.site_analysis[self.gmpe][self.imt]["Intra event"]
            data[site_id]["dS2ss"] =\
                resid.site_analysis[self.gmpe][self.imt]["dS2ss"]
            data[site_id]["dWo,es"] =\
                resid.site_analysis[self.gmpe][self.imt]["dWo,es"]
            data[site_id]["x-val"] =(float(iloc) + 0.5) *\
                np.ones_like(data[site_id]["Intra event"])
        return data
