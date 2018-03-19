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
Module to get GMPE residuals - total, inter and intra
{'GMPE': {'IMT1': {'Total': [], 'Inter event': [], 'Intra event': []},
          'IMT2': { ... }}}
          


"""
from __future__ import print_function
import sys
import re, os
import h5py
import numpy as np
from datetime import datetime
from math import sqrt, ceil
from scipy.special import erf
from scipy.stats import scoreatpercentile, norm, multivariate_normal
from scipy.linalg import solve
from copy import deepcopy
from collections import OrderedDict
from openquake.baselib import parallel
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.gsim.gmpe_table import GMPETable
import smtk.intensity_measures as ims
from openquake.hazardlib import imt
from smtk.strong_motion_selector import SMRecordSelector
from smtk.trellis.trellis_plots import _get_gmpe_name, _check_gsim_list


GSIM_LIST = get_available_gsims()
GSIM_KEYS = set(GSIM_LIST.keys())

#SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
SCALAR_IMTS = ["PGA", "PGV"]
STDDEV_KEYS = ["Mean", "Total", "Inter event", "Intra event"]


def _check_gsim_list(gsim_list):
    """
    Checks the list of GSIM models and returns an instance of the 
    openquake.hazardlib.gsim class. Raises error if GSIM is not supported in
    OpenQuake
    :param list gsim_list:
        List of GSIM names (str)
    :returns:
        Ordered dictionary of GMPE names and instances
    """
    output_gsims = []
    for gsim in gsim_list:
        if gsim.startswith("GMPETable"):
            # Get filename
            match = re.match(r'^GMPETable\(([^)]+?)\)$', gsim)
            filepath = match.group(1).split("=")[1]
            gmpe_table = GMPETable(gmpe_table=filepath[1:-1])
            output_gsims.append((_get_gmpe_name(gmpe_table), gmpe_table))
        elif not gsim in GSIM_LIST.keys():
            raise ValueError('%s Not supported by OpenQuake' % gsim)
        else:
            output_gsims.append((gsim, GSIM_LIST[gsim]()))
    return OrderedDict(output_gsims)


def get_interpolated_period(target_period, periods, values):
    """
    Returns the spectra interpolated in loglog space
    :param float target_period:
        Period required for interpolation
    :param np.ndarray periods:
        Spectral Periods
    :param np.ndarray values:
        Ground motion values
    """
    if (target_period < np.min(periods)) or (target_period > np.max(periods)):
        return None, "Period not within calculated range %s"
    lval = np.where(periods <= target_period)[0][-1]
    uval = np.where(periods >= target_period)[0][0]
    if (uval - lval) == 0:
        return values[lval]
    else:
        dy = np.log10(values[uval]) - np.log10(values[lval])
        dx = np.log10(periods[uval]) - np.log10(periods[lval])
        
        return 10.0 ** (
            np.log10(values[lval]) + 
            (np.log10(target_period) - np.log10(periods[lval])) * dy / dx
            )


def get_geometric_mean(fle):
    """
    Retreive geometric mean of the ground motions from the file - or calculate
    if not in file
    :param fle:
        Instance of :class: h5py.File
    """
    #periods = fle["IMS/X/Spectra/Response/Periods"].value
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_spc = fle["IMS/X/Spectra/Response/Acceleration/damping_05"].values
        y_spc = fle["IMS/Y/Spectra/Response/Acceleration/damping_05"].values
        periods = fle["IMS/X/Spectra/Response/Periods"].values
        sa_geom = np.sqrt(x_spc * y_spc)
    else:
        if "Geometric" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_geom =fle[
                "IMS/H/Spectra/Response/Acceleration/Geometric/damping_05"
                ].value
            periods = fle["IMS/X/Spectra/Periods"].values
            idx = periods > 0
            periods = periods[idx]
            sa_geom = sa_geom[idx]
        else:
            # Horizontal spectra not in record
            x_spc = fle[
                "IMS/X/Spectra/Response/Acceleration/damping_05"].values
            y_spc = fle[
                "IMS/Y/Spectra/Response/Acceleration/damping_05"].values
            sa_geom = np.sqrt(x_spc * y_spc)
    return sa_geom

def get_gmrotd50(fle):
    """
    Retrieve GMRotD50 from file (or calculate if not present)
    :param fle:
        Instance of :class: h5py.File
    """
    periods = fle["IMS/X/Spectra/Response/Periods"].value
    periods = periods[periods > 0.]
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_acc = ["Time Series/X/Original Record/Acceleration"]
        y_acc = ["Time Series/Y/Original Record/Acceleration"]
        sa_gmrotd50 = ims.gmrotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                   y_acc.value, y_acc.attrs["Time-step"],
                                   periods, 50.0)[0]
        
    else:
        if "GMRotD50" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_gmrotd50 =fle[
                "IMS/H/Spectra/Response/Acceleration/GMRotD50/damping_05"
                ].value
        else:
            # Horizontal spectra not in record - calculate from time series
            x_acc = ["Time Series/X/Original Record/Acceleration"]
            y_acc = ["Time Series/Y/Original Record/Acceleration"]
            sa_gmrotd50 = ims.gmrotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                       y_acc.value, y_acc.attrs["Time-step"],
                                       periods, 50.0)[0]
    return sa_gmrotd50

def get_gmroti50(fle):
    """   
    Retreive GMRotI50 from file (or calculate if not present)
    :param fle:
        Instance of :class: h5py.File
    """
    periods = fle["IMS/X/Spectra/Response/Periods"].value
    periods = periods[periods > 0.]
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_acc = ["Time Series/X/Original Record/Acceleration"]
        y_acc = ["Time Series/Y/Original Record/Acceleration"]
        sa_gmroti50 = ims.gmrotipp(x_acc.value, x_acc.attrs["Time-step"],
                                   y_acc.value, y_acc.attrs["Time-step"],
                                   periods, 50.0)[0]
        
    else:
        if "GMRotI50" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_gmroti50 =fle[
                "IMS/H/Spectra/Response/Acceleration/GMRotI50/damping_05"
                ].value
        else:
            # Horizontal spectra not in record - calculate from time series
            x_acc = ["Time Series/X/Original Record/Acceleration"]
            y_acc = ["Time Series/Y/Original Record/Acceleration"]
            sa_gmroti50 = ims.gmrotipp(x_acc.value, x_acc.attrs["Time-step"],
                                       y_acc.value, y_acc.attrs["Time-step"],
                                       periods, 50.0)
            # Assumes Psuedo-spectral acceleration
            sa_gmroti50 = sa_gmroti50["PSA"]
    return sa_gmroti50


def get_rotd50(fle):
    """
    Retrieve RotD50 from file (or calculate if not present)
    :param fle:
        Instance of :class: h5py.File
    """
    periods = fle["IMS/H/Spectra/Response/Periods"].value
    periods = periods[periods > 0.]
    if not "H" in fle["IMS"].keys():
        # Horizontal spectra not in record
        x_acc = ["Time Series/X/Original Record/Acceleration"]
        y_acc = ["Time Series/Y/Original Record/Acceleration"]
        sa_rotd50 = ims.rotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                   y_acc.value, y_acc.attrs["Time-step"],
                                   periods, 50.0)[0]

        
    else:
        if "RotD50" in fle["IMS/H/Spectra/Response/Acceleration"].keys():
            sa_rotd50 =fle[
                "IMS/H/Spectra/Response/Acceleration/RotD50/damping_05"
                ].value
        else:
            # Horizontal spectra not in record - calculate from time series
            x_acc = ["Time Series/X/Original Record/Acceleration"]
            y_acc = ["Time Series/Y/Original Record/Acceleration"]
            sa_rotd50 = ims.rotdpp(x_acc.value, x_acc.attrs["Time-step"],
                                       y_acc.value, y_acc.attrs["Time-step"],
                                       periods, 50.0)[0]
    return sa_rotd50



SPECTRA_FROM_FILE = {"Geometric": get_geometric_mean,
                     "GMRotI50": get_gmroti50,
                     "GMRotD50": get_gmrotd50,
                     "RotD50": get_rotd50}


SCALAR_XY = {"Geometric": lambda x, y : np.sqrt(x * y),
             "Arithmetic": lambda x, y : (x + y) / 2.,
             "Larger": lambda x, y: np.max(np.array([x, y])),
             "Vectorial": lambda x, y : np.sqrt(x ** 2. + y ** 2.)}

def get_scalar(fle, i_m, component="Geometric"):
    """
    Retrieves the scalar IM from the database
    :param fle:
        Instance of :class: h5py.File
    :param str i_m:
        Intensity measure
    :param str component:
        Horizontal component of IM
    """
     
    if not "H" in fle["IMS"].keys():
        x_im = fle["IMS/X/Scalar/" + i_m].value[0]
        y_im = fle["IMS/Y/Scalar/" + i_m].value[0]
        return SCALAR_XY[component](x_im, y_im)
    else:
        if i_m in fle["IMS/H/Scalar"].keys():
            return fle["IMS/H/Scalar/" + i_m].value[0]
        else:
            raise ValueError("Scalar IM %s not in record database" % i_m)



class Residuals(object):
    """
    Class to derive sets of residuals for a list of ground motion residuals
    according to the GMPEs
    """
    def __init__(self, gmpe_list, imts):
        """
        :param list gmpe_list:
            List of GMPE names (using the standard openquake strings)
        :param list imts:
            List of Intensity Measures
        """
        self.gmpe_list = _check_gsim_list(gmpe_list)
        self.number_gmpes = len(self.gmpe_list)
        self.types = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        self.residuals = []
        self.modelled = []
        self.imts = imts
        self.unique_indices = {}
        self.gmpe_sa_limits = {}
        self.gmpe_scalars = {}
        for gmpe in self.gmpe_list:
            gmpe_dict_1 = OrderedDict([])
            gmpe_dict_2 = OrderedDict([])
            self.unique_indices[gmpe] = {}
            # Get the period range and the coefficient types
            gmpe_i = GSIM_LIST[gmpe]()
            for c in dir(gmpe_i):
                if 'COEFFS' in c:
                    pers = [sa.period for sa in getattr(gmpe_i,c).sa_coeffs]
            min_per, max_per = (min(pers), max(pers))
            self.gmpe_sa_limits[gmpe] = (min_per, max_per)
            for c in dir(gmpe_i):
                if 'COEFFS' in c:
                    self.gmpe_scalars[gmpe] = getattr(gmpe_i,c).non_sa_coeffs.keys()
            for imtx in self.imts:
                if "SA(" in imtx:
                    period = imt.from_string(imtx).period
                    if period < min_per or period > max_per:
                        print("IMT %s outside period range for GMPE %s"
                              % (imtx, gmpe))
                        gmpe_dict_1[imtx] = None
                        gmpe_dict_2[imtx] = None
                        continue
                gmpe_dict_1[imtx] = {}
                gmpe_dict_2[imtx] = {}
                self.unique_indices[gmpe][imtx] = []
                self.types[gmpe][imtx] = []
                for res_type in \
                    self.gmpe_list[gmpe].DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                    gmpe_dict_1[imtx][res_type] = []
                    gmpe_dict_2[imtx][res_type] = []
                    self.types[gmpe][imtx].append(res_type)
                gmpe_dict_2[imtx]["Mean"] = []
            self.residuals.append([gmpe, gmpe_dict_1])
            self.modelled.append([gmpe, gmpe_dict_2])
        self.residuals = OrderedDict(self.residuals)
        self.modelled = OrderedDict(self.modelled)
        self.database = None
        self.number_records = None
        self.contexts = None
    

    def get_residuals(self, database, nodal_plane_index=1,
            component="Geometric", normalise=True):
        """
        Calculate the residuals for a set of ground motion records
        """
        # Contexts is a list of dictionaries 
        contexts = database.get_contexts(nodal_plane_index)
        self.database = SMRecordSelector(database)
        self.contexts = []
        for context in contexts:
            # Get the observed strong ground motions
            context = self.get_observations(context, component)
            # Get the expected ground motions
            context = self.get_expected_motions(context)
            context = self.calculate_residuals(context, normalise)
            for gmpe in self.residuals.keys():
                for imtx in self.residuals[gmpe].keys():
                    if not context["Residual"][gmpe][imtx]:
                        continue
                    for res_type in self.residuals[gmpe][imtx].keys():
                        if res_type == "Inter event":
                            inter_ev = \
                                context["Residual"][gmpe][imtx][res_type]
                            if np.all(
                                np.fabs(inter_ev - inter_ev[0]) < 1.0E-12):
                                # Single inter-event residual
                                self.residuals[gmpe][imtx][res_type].append(
                                    inter_ev[0])
                                # Append indices
                                self.unique_indices[gmpe][imtx].append(
                                    np.array([0]))
                            else:
                                # Inter event residuals per-site e.g. Chiou
                                # & Youngs (2008; 2014) case
                                self.residuals[gmpe][imtx][res_type].extend(
                                    inter_ev.tolist())
                                self.unique_indices[gmpe][imtx].append(
                                    np.arange(len(inter_ev)))
                        else:
                            self.residuals[gmpe][imtx][res_type].extend(
                                context["Residual"][gmpe][imtx][res_type].tolist())
                        self.modelled[gmpe][imtx][res_type].extend(
                            context["Expected"][gmpe][imtx][res_type].tolist())

                    self.modelled[gmpe][imtx]["Mean"].extend(
                        context["Expected"][gmpe][imtx]["Mean"].tolist())

            self.contexts.append(context)
       
        for gmpe in self.residuals.keys():
            for imtx in self.residuals[gmpe].keys():
                if not self.residuals[gmpe][imtx]:
                    continue
                for res_type in self.residuals[gmpe][imtx].keys():
                    self.residuals[gmpe][imtx][res_type] = np.array(
                        self.residuals[gmpe][imtx][res_type])
                    self.modelled[gmpe][imtx][res_type] = np.array(
                        self.modelled[gmpe][imtx][res_type])
                self.modelled[gmpe][imtx]["Mean"] = np.array(
                    self.modelled[gmpe][imtx]["Mean"])

    def get_observations(self, context, component="Geometric"):
        """
        Get the obsered ground motions from the database
        """
        select_records = self.database.select_from_event_id(context["EventID"])
        observations = OrderedDict([(imtx, []) for imtx in self.imts])
        selection_string = "IMS/H/Spectra/Response/Acceleration/"
        for record in select_records:
            fle = h5py.File(record.datafile, "r")
            for imtx in self.imts:
                if imtx in SCALAR_IMTS:
                    if imtx == "PGA":
                        observations[imtx].append(
                            get_scalar(fle, imtx, component) / 981.0)
                    else:
                        observations[imtx].append(
                            get_scalar(fle, imtx, component))

                elif "SA(" in imtx:
                    target_period = imt.from_string(imtx).period
                    spectrum = fle[selection_string + component 
                                   + "/damping_05"].value
                    periods = fle["IMS/H/Spectra/Response/Periods"].value
                    observations[imtx].append(get_interpolated_period(
                        target_period, periods, spectrum) / 981.0)
                else:
                    raise "IMT %s is unsupported!" % imtx
            fle.close()
        for imtx in self.imts:
            observations[imtx] = np.array(observations[imtx])
        context["Observations"] = observations
        context["Num. Sites"] = len(select_records)
        return context

    def get_expected_motions(self, context):
        """
        Calculate the expected ground motions from the context
        """
        # TODO Rake hack will be removed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not context["Rupture"].rake:
            context["Rupture"].rake = 0.0
        expected = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        # Period range for GSIM
        for gmpe in self.gmpe_list:
            expected[gmpe] = OrderedDict([(imtx, {}) for imtx in self.imts])
            for imtx in self.imts:
                gsim = self.gmpe_list[gmpe]
                if "SA(" in imtx:
                    period = imt.from_string(imtx).period
                    if period < self.gmpe_sa_limits[gmpe][0] or\
                        period > self.gmpe_sa_limits[gmpe][1]:
                        expected[gmpe][imtx] = None
                        continue
                mean, stddev = gsim.get_mean_and_stddevs(
                    context["Sites"],
                    context["Rupture"],
                    context["Distances"],
                    imt.from_string(imtx),
                    self.types[gmpe][imtx])
                expected[gmpe][imtx]["Mean"] = mean
                for i, res_type in enumerate(self.types[gmpe][imtx]):
                    expected[gmpe][imtx][res_type] = stddev[i]

        context["Expected"] = expected
        return context
                    
    def calculate_residuals(self, context, normalise=True):
        """
        Calculate the residual terms
        """
        # Calculate residual
        residual = {}
        for gmpe in self.gmpe_list:
            residual[gmpe] = OrderedDict([])
            for imtx in self.imts:
                residual[gmpe][imtx] = {}
                obs = np.log(context["Observations"][imtx])
                if not context["Expected"][gmpe][imtx]:
                    residual[gmpe][imtx] = None
                    continue
                mean = context["Expected"][gmpe][imtx]["Mean"]
                total_stddev = context["Expected"][gmpe][imtx]["Total"]
                residual[gmpe][imtx]["Total"] = (obs - mean) / total_stddev
                if "Inter event" in self.residuals[gmpe][imtx].keys():
                    inter, intra = self._get_random_effects_residuals(
                        obs,
                        mean,
                        context["Expected"][gmpe][imtx]["Inter event"],
                        context["Expected"][gmpe][imtx]["Intra event"],
                        normalise)
                    residual[gmpe][imtx]["Inter event"] = inter
                    residual[gmpe][imtx]["Intra event"] = intra
        context["Residual"] = residual
        return context

    def _get_random_effects_residuals(self, obs, mean, inter, intra,
            normalise=True):
        """
        Calculates the random effects residuals using the inter-event
        residual formula described in Abrahamson & Youngs (1992) Eq. 10
        """
        nvals = float(len(mean))
        inter_res = ((inter ** 2.) * sum(obs - mean)) /\
                     (nvals * (inter ** 2.) + (intra ** 2.))
        intra_res = obs - (mean + inter_res)
        if normalise:
            return inter_res / inter, intra_res / intra
        else:
            return inter_res, intra_res

    def get_residual_statistics(self):
        """
        Retreives the mean and standard deviation values of the residuals
        """
        statistics = OrderedDict([(gmpe, OrderedDict([]))
                                  for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                if not self.residuals[gmpe][imtx]:
                    continue
                statistics[gmpe][imtx] = {}
                for res_type in self.types[gmpe][imtx]:
                    data = {
                        "Mean": np.mean(
                            self.residuals[gmpe][imtx][res_type]),
                        "Std Dev": np.std(
                            self.residuals[gmpe][imtx][res_type])}
                    statistics[gmpe][imtx][res_type] = data
        return statistics

    def pretty_print(self, filename=None, sep=","):
        """
        Print the information to screen or to file
        """
        if filename:
            fid = open(filename, "w")
        else:
            fid = sys.stdout
        fid.write("Ground Motion Residuals\n")
        # Print headers
        event = self.contexts[0]
        header_set = []
        header_set.extend([key for key in event["Distances"].__dict__])
        header_set.extend([key for key in event["Sites"].__dict__])
        header_set.extend(["{:s}-Obs.".format(imt) for imt in self.imts])
        for imt in self.imts:
            for gmpe in self.gmpe_list:
                if not event["Expected"][gmpe][imt]:
                    continue
                for key in event["Expected"][gmpe][imt].keys():
                    header_set.append(
                        "{:s}-{:s}-{:s}-Exp.".format(imt, gmpe, key))
        for imt in self.imts:
            for gmpe in self.gmpe_list:
                if not event["Residual"][gmpe][imt]:
                    continue
                for key in event["Residual"][gmpe][imt].keys():
                    header_set.append(
                        "{:s}-{:s}-{:s}-Res.".format(imt, gmpe, key))
        header_set = self._extend_header_set(header_set)
        fid.write("%s\n" % sep.join(header_set))
        for event in self.contexts:
            self._pprint_event(fid, event, sep)
        if filename:
            fid.close()

    def _pprint_event(self, fid, event, sep):
        """
        Pretty print the information for each event
        """
        # Print rupture info
        rupture_str = sep.join([
            "{:s}{:s}{:s}".format(key, sep, str(val))
            for key, val in event["Rupture"].__dict__.items()])
        fid.write("Rupture: %s %s %s\n" % (str(event["EventID"]), sep,
                                           rupture_str))
        # For each record
        for i in range(event["Num. Sites"]):
            data = []
            # Distances
            for key in event["Distances"].__dict__:
                data.append("{:.4f}".format(
                    getattr(event["Distances"], key)[i]))
            # Sites
            for key in event["Sites"].__dict__:
                data.append("{:.4f}".format(getattr(event["Sites"], key)[i]))
            # Observations
            for imt in self.imts:
                data.append("{:.8e}".format(event["Observations"][imt][i]))
            # Expected
            for imt in self.imts:
                for gmpe in self.gmpe_list:
                    if not event["Expected"][gmpe][imt]:
                        continue
                    for key in event["Expected"][gmpe][imt].keys():
                        data.append("{:.8e}".format(
                            event["Expected"][gmpe][imt][key][i]))
            # Residuals
            for imt in self.imts:
                for gmpe in self.gmpe_list:
                    if not event["Expected"][gmpe][imt]:
                        continue
                    for key in event["Residual"][gmpe][imt].keys():
                        data.append("{:.8e}".format(
                            event["Residual"][gmpe][imt][key][i]))
            self._extend_data_print(data, event, i)
            fid.write("%s\n" % sep.join(data))

    def _extend_header_set(self, header_set):
        """
        Additional headers to add to the pretty print - does nothing here but
        overwritten in subclasses
        """
        return header_set

    def _extend_data_print(self, data, event, i):
        """
        Additional data to add to the pretty print - also does nothing here
        but overwritten in subclasses
        """
        return data

    def _get_magnitudes(self):
        """
        Returns an array of magnitudes equal in length to the number of
        residuals
        """
        magnitudes = np.array([])
        for ctxt in self.contexts:
            magnitudes = np.hstack([
                magnitudes,
                ctxt["Rupture"].mag * np.ones(len(ctxt["Distances"].repi))])
        return magnitudes


class Likelihood(Residuals):
    """
    Implements the likelihood function of Scherbaum et al. (2004)
    """
        
    def get_likelihood_values(self):
        """
        Returns the likelihood values for Total, plus inter- and intra-event
        residuals according to Equation 9 of Scherbaum et al (2004)
        """
        statistics = self.get_residual_statistics()
        lh_values = OrderedDict([(gmpe, OrderedDict([]))
                                 for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                if not self.residuals[gmpe][imtx]:
                    print("IMT %s not found in Residuals for %s"
                          % (imtx, gmpe))
                    continue
                lh_values[gmpe][imtx] = {}
                for res_type in self.types[gmpe][imtx]:
                    zvals = np.fabs(self.residuals[gmpe][imtx][res_type])
                    l_h = 1.0 - erf(zvals / sqrt(2.))
                    lh_values[gmpe][imtx][res_type] = l_h
                    statistics[gmpe][imtx][res_type]["Median LH"] =\
                        scoreatpercentile(l_h, 50.0)
        return lh_values, statistics


class LLH(Residuals):
    """
    Implements of average sample log-likelihood estimator from
    Scherbaum et al (2009)
    """
    def get_loglikelihood_values(self, imts):
        log_residuals = OrderedDict([(gmpe, np.array([]))
                                      for gmpe in self.gmpe_list])
        imt_list = [(imtx, None) for imtx in imts]
        imt_list.append(("All", None))
        llh = OrderedDict([(gmpe, OrderedDict(imt_list))
                           for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in imts:
                if not imtx in self.imts or not self.residuals[gmpe][imtx]:
                    print("IMT %s not found in Residuals for %s"
                          % (imtx, gmpe))
                    continue
                # Get log-likelihood distance for IMT
                asll = np.log2(norm.pdf(self.residuals[gmpe][imtx]["Total"],
                               0., 
                               1.0))
                log_residuals[gmpe] = np.hstack([
                    log_residuals[gmpe],
                    asll])
                llh[gmpe][imtx] = -(1.0 / float(len(asll))) * np.sum(asll)

            llh[gmpe]["All"] = -(1. / float(len(log_residuals[gmpe]))) *\
                np.sum(log_residuals[gmpe])
        # Get weights
        weights = np.array([2.0 ** -llh[gmpe]["All"]
                            for gmpe in self.gmpe_list])
        weights = weights / np.sum(weights)
        model_weights = OrderedDict([
            (gmpe, weights[iloc]) for iloc, gmpe in enumerate(self.gmpe_list)]
            )
        return llh, model_weights


def _build_matrices(contexts, gmpe, imt):
    """
    Constructs the R and Z_G matrices (based on the implementation
    in the supplement to Mak et al (2017)
    """
    neqs = len(contexts)
    nrecs = sum([ctxt["Num. Sites"] for ctxt in contexts])

    r_mat = np.zeros(nrecs, dtype=float)
    z_g_mat = np.zeros([nrecs, neqs], dtype=float)
    expected_mat = np.zeros(nrecs, dtype=float)
    # Get observations
    observations = np.zeros(nrecs)
    i = 0
    # Determine the total number of records and pass the log of the
    # obserations to the observations dictionary
    for ctxt in contexts:
        n_s = ctxt["Num. Sites"]
        observations[i:(i + n_s)] = np.log(ctxt["Observations"][imt])
        i += n_s

    i = 0 
    for j, ctxt in enumerate(contexts):
        if not("Intra event" in ctxt["Expected"][gmpe][imt]) and\
            not("Inter event" in ctxt["Expected"][gmpe][imt]):
            # Only the total sigma exists
            # Total sigma is used as intra-event sigma (from S. Mak)
            n_r = len(ctxt["Expected"][gmpe][imt]["Total"])
            r_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imt]["Total"]
            expected_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imt]["Mean"]
            # Inter-event sigma is set to 0
            i += n_r
            continue
        n_r = len(ctxt["Expected"][gmpe][imt]["Intra event"])
        r_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imt]["Intra event"]
        # Get expected mean
        expected_mat[i:(i + n_r)] = ctxt["Expected"][gmpe][imt]["Mean"]
        if len(ctxt["Expected"][gmpe][imt]["Inter event"]) == 1:
            # Single inter event residual
            z_g_mat[i:(i + n_r), j] =\
                ctxt["Expected"][gmpe][imt]["Inter event"][0]
        else:
            # inter-event residual given at a vector
            z_g_mat[i:(i + n_r), j] =\
                ctxt["Expected"][gmpe][imt]["Inter event"]
        i += n_r
    
    v_mat = np.diag(r_mat ** 2.) + z_g_mat.dot(z_g_mat.T)
    return observations, v_mat, expected_mat, neqs, nrecs


def get_multivariate_ll(contexts, gmpe, imt):
    """
    Returns the multivariate loglikelihood, as described om equation 7 of
    Mak et al. (2017)
    """ 
    observations, v_mat, expected_mat, neqs, nrecs = _build_matrices(
        contexts, gmpe, imt)
    sign, logdetv = np.linalg.slogdet(v_mat)
    b_mat = observations - expected_mat
    return (float(nrecs) * np.log(2.0 * np.pi) + logdetv +
            (b_mat.T.dot(solve(v_mat, b_mat)))) / 2.


def bootstrap_llh(ij, contexts, gmpes, imts):
    """
    Applyies the cluster bootstrap. A set of events, equal in length to that
    of the original data, is sampled randomly from the list of contexts. All of
    the sigmas for that specific event are transfered to the sample
    """
    # Sample contexts
    timer_on = datetime.now()
    neqs = len(contexts)
    isamp = np.random.randint(0, neqs, neqs)
    new_contexts = [contexts[i] for i in isamp]
    outputs = np.zeros([len(gmpes), len(imts)])
    for i, gmpe in enumerate(gmpes):
        for j, imt in enumerate(imts):
            outputs[i, j] = get_multivariate_ll(new_contexts, gmpe, imt)
    print("Bootstrap completed in {:.2f} seconds".format(
        (datetime.now() - timer_on).total_seconds()))
    return outputs


class MultivariateLLH(Residuals):
    """
    Multivariate formulation of the LLH function as proposed by Mak et al.
    (2017)
    """
    def get_likelihood_values(self, sum_imts=False):
        """
        Calculates the multivariate LLH to return a 
        """
        multi_llh_values = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        # Get number of events and records
        for gmpe in self.gmpe_list:
            print("GMPE = {:s}".format(gmpe))
            for j, imt in enumerate(self.imts):
                if self.residuals[gmpe][imt] is None:
                    # IMT missing for this GMPE
                    multi_llh_values[gmpe][imt] = 0.0
                else:
                    multi_llh_values[gmpe][imt] = get_multivariate_ll(
                        self.contexts, gmpe, imt)
            if sum_imts:
                total_llh = 0.0
                for imt in self.imts:
                    if np.isnan(multi_llh_values[gmpe][imt]):
                        continue
                    total_llh += multi_llh_values[gmpe][imt]
                multi_llh_values[gmpe] = total_llh
        return multi_llh_values

    def bootstrap_likelihood_values(self, number_bootstraps, sum_imts=False,
                                    parallelize=False, concurrent_tasks=8):
        """
        Bootstrap the analysis using cluster sampling. OpenQuake's :class:
        `openquake.baselib.parallel.Starmap` utility is invoked to parallelise
        the calculations by bootstrap
        """
        # Setup multivariate log-likelihood dict
        multi_llh_values = []
        nmods = []
        for i, gmpe in enumerate(self.gmpe_list):
            for j, imt in enumerate(self.imts):
                nmods.append((i, j))
                multi_llh_values.append((gmpe, imt))
                #nmods += 1
        outputs = np.zeros([len(self.gmpe_list), len(self.imts),
                            number_bootstraps])
        if parallelize:
            raise NotImplementedError("Parellelisation not turned on yet!")
#            results = parallel.Starmap.apply(
#                bootstrap_llh,
#                (np.arange(0, number_bootstraps),
#                 self.contexts, self.gmpe_list, self.imts),
#                 concurrent_tasks=concurrent_tasks).reduce()
#            print results
#            for j, result in enumerate(results):
#                print(result)
#                outputs[:, :, j] = result
        else:
            for j in range(number_bootstraps):
                print("Bootstrap {:g} of {:g}".format(j + 1,
                      number_bootstraps))
                outputs[:, :, j] = bootstrap_llh(j,
                                                 self.contexts,
                                                 self.gmpe_list,
                                                 self.imts)
                    
        distinctiveness = self.get_distinctiveness(outputs,
                                                   number_bootstraps,
                                                   sum_imts)
        return distinctiveness, outputs
    
    def get_distinctiveness(self, outputs, number_bootstraps, sum_imts):
        """
        Return the distinctiveness index as described in equation 9 of Mak
        et al. (2017)
        """
        ngmpes = len(self.gmpe_list)
        nbs = float(number_bootstraps)
        nimts = float(len(self.imts))
        if sum_imts:
            distinctiveness = np.zeros([ngmpes, ngmpes])
            # Get only one index for each GMPE
            for i, gmpe in enumerate(self.gmpe_list):
                for j, gmpe in enumerate(self.gmpe_list):
                    if i == j:
                        continue
                    data_i = outputs[i, :, :]
                    data_j = outputs[j, :, :]
                    distinctiveness[i, j] = float(np.sum(data_i < data_j) -
                        np.sum(data_j < data_i)) / (nbs * nimts)
            return distinctiveness
        else:
            distinctiveness = np.zeros([ngmpes, ngmpes, len(self.imts)])
            for i, gmpe in enumerate(self.gmpe_list):
                for j, gmpe in enumerate(self.gmpe_list):
                    if i == j:
                        continue
                    for k in range(len(self.imts)):
                        data_i = outputs[i, k, :]
                        data_j = outputs[j, k, :]
                        distinctiveness[i, j, k] =\
                            float(np.sum(data_i < data_j) -
                                  np.sum(data_j < data_i)) / nbs
        return distinctiveness


class EDR(Residuals):
    """
    Implements the Euclidean Distance-Based Ranking Method for GMPE selection
    by Kale & Akkar (2013)
    Kale, O., and Akkar, S. (2013) A New Procedure for Selecting and Ranking
    Ground Motion Predicion Equations (GMPEs): The Euclidean Distance-Based
    Ranking Method
    """
    def get_edr_values(self, bandwidth=0.01, multiplier=3.0):
        """
        Calculates the EDR values for each GMPE
        :param float bandwidth:
            Discretisation width
        :param float multiplier:
            "Multiplier of standard deviation (equation 8 of Kale and Akkar)
        """
        edr_values = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            obs, expected, stddev = self._get_gmpe_information(gmpe)
            results = self._get_edr(obs,
                                    expected,
                                    stddev,
                                    bandwidth,
                                    multiplier)
            edr_values[gmpe]["MDE Norm"] = results[0]
            edr_values[gmpe]["sqrt Kappa"] = results[1]
            edr_values[gmpe]["EDR"] = results[2]
        return edr_values

    def _get_gmpe_information(self, gmpe):
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE (aggregating over all IMS)
        """
        obs = np.array([], dtype=float)
        expected = np.array([], dtype=float)
        stddev = np.array([], dtype=float)
        for imtx in self.imts:
            for context in self.contexts:
                obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                expected = np.hstack([expected,
                                      context["Expected"][gmpe][imtx]["Mean"]])
                stddev = np.hstack([stddev,
                                    context["Expected"][gmpe][imtx]["Total"]])
        return obs, expected, stddev

    def _get_edr(self, obs, expected, stddev, bandwidth=0.01, multiplier=3.0):
        """
        Calculated the Euclidean Distanced-Based Rank for a set of
        observed and expected values from a particular GMPE
        """
        nvals = len(obs)
        min_d = bandwidth / 2.
        kappa = self._get_kappa(obs, expected)
        mu_d = obs - expected
        d1c = np.fabs(obs - (expected - (multiplier * stddev)))
        d2c = np.fabs(obs - (expected + (multiplier * stddev)))
        dc_max = ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
        num_d = len(np.arange(min_d, dc_max, bandwidth))
        mde = np.zeros(nvals)
        for iloc in range(0, num_d):
            d_val = (min_d + (float(iloc) * bandwidth)) * np.ones(nvals)
            d_1 = d_val - min_d
            d_2 = d_val + min_d
            p_1 = norm.cdf((d_1 - mu_d) / stddev) -\
                norm.cdf((-d_1 - mu_d) / stddev)
            p_2 = norm.cdf((d_2 - mu_d) / stddev) -\
                norm.cdf((-d_2 - mu_d) / stddev)
            mde += (p_2 - p_1) * d_val
        inv_n = 1.0 / float(nvals)
        mde_norm = np.sqrt(inv_n * np.sum(mde ** 2.))
        edr = np.sqrt(kappa * inv_n * np.sum(mde ** 2.))
        return mde_norm, np.sqrt(kappa), edr


    def _get_kappa(self, obs, expected):
        """
        Returns the correction factor kappa
        """
        mu_a = np.mean(obs)
        mu_y = np.mean(expected)
        b_1 = np.sum((obs - mu_a) * (expected - mu_y)) /\
            np.sum((obs - mu_a) ** 2.)
        b_0 = mu_y - b_1 * mu_a
        y_c =  expected - ((b_0 + b_1 * obs) - obs)
        de_orig = np.sum((obs - expected) ** 2.)
        de_corr = np.sum((obs - y_c) ** 2.)
        return de_orig / de_corr
        
      
class SingleStationAnalysis(object):
    """
    Class to analyse residual sets recorded at specific stations
    """
    def __init__(self, site_id_list, gmpe_list, imts):
        """

        """
        self.site_ids = site_id_list
        self.input_gmpe_list = deepcopy(gmpe_list)
        self.gmpe_list = _check_gsim_list(gmpe_list)
        self.imts = imts
        self.site_residuals = []
        self.types = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            #if not gmpe in GSIM_LIST:
            #    raise ValueError("%s not supported in OpenQuake" % gmpe) 
            for imtx in self.imts:
                self.types[gmpe][imtx] = []
                for res_type in \
                    self.gmpe_list[gmpe].DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                    self.types[gmpe][imtx].append(res_type)

    def get_site_residuals(self, database, component="Geometric"):
        """
        Calculates the total, inter-event and within-event residuals for
        each site
        """
        imt_dict = dict([(imtx, {}) for imtx in self.imts])
        for site_id in self.site_ids:
            print(site_id)
            selector = SMRecordSelector(database)
            site_db = selector.select_from_site_id(site_id, as_db=True)
            resid = Residuals(self.input_gmpe_list, self.imts)
            resid.get_residuals(site_db, normalise=False, component=component)
            setattr(
                resid,
                "site_analysis",
                self._set_empty_dict())
            setattr(
                resid,
                "site_expected",
                self._set_empty_dict())
            self.site_residuals.append(resid)

    def _set_empty_dict(self):
        """
        Sets an empty set of nested dictionaries for each GMPE and each IMT
        """
        return OrderedDict([
            (gmpe, dict([(imtx, {}) for imtx in self.imts]))
            for gmpe in self.gmpe_list])


    def residual_statistics(self, pretty_print=False, filename=None):
        """
        Get single-station residual statistics for each site
        """
        output_resid = []
        
        for t_resid in self.site_residuals:
            resid = deepcopy(t_resid)

            for gmpe in self.gmpe_list:
                for imtx in self.imts:
                    if not resid.residuals[gmpe][imtx]:
                        continue
                    n_events = len(resid.residuals[gmpe][imtx]["Total"])
                    resid.site_analysis[gmpe][imtx]["events"] = n_events
                    resid.site_analysis[gmpe][imtx]["Total"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Total"])
                    resid.site_analysis[gmpe][imtx]["Expected Total"] = \
                        np.copy(t_resid.modelled[gmpe][imtx]["Total"])
                    if not "Intra event" in t_resid.residuals[gmpe][imtx]:
                        # GMPE has no within-event term - skip
                        continue

                    resid.site_analysis[gmpe][imtx]["Intra event"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Intra event"])
                    resid.site_analysis[gmpe][imtx]["Inter event"] = np.copy(
                        t_resid.residuals[gmpe][imtx]["Inter event"])

                    delta_s2ss = self._get_delta_s2ss(
                        resid.residuals[gmpe][imtx]["Intra event"],
                        n_events)
                    delta_woes = \
                        resid.site_analysis[gmpe][imtx]["Intra event"] - \
                        delta_s2ss
                    resid.site_analysis[gmpe][imtx]["dS2ss"] = delta_s2ss
                    resid.site_analysis[gmpe][imtx]["dWo,es"] = delta_woes

                    resid.site_analysis[gmpe][imtx]["phi_ss,s"] = \
                        self._get_single_station_phi(
                            resid.residuals[gmpe][imtx]["Intra event"],
                            delta_s2ss,
                            n_events)
                    # Get expected values too

                    resid.site_analysis[gmpe][imtx]["Expected Inter"] =\
                        np.copy(t_resid.modelled[gmpe][imtx]["Inter event"])
                    resid.site_analysis[gmpe][imtx]["Expected Intra"] =\
                        np.copy(t_resid.modelled[gmpe][imtx]["Intra event"])
            output_resid.append(resid)
        self.site_residuals = output_resid
        return self.get_total_phi_ss(pretty_print, filename) 

    def _get_delta_s2ss(self, intra_event, n_events):
        """
        Returns the average within-event residual for the site from
        Rodriguez-Marek et al. (2011) Equation 8
        """
        return (1. / float(n_events)) * np.sum(intra_event)

    def _get_single_station_phi(self, intra_event, delta_s2ss, n_events):
        """
        Returns the single-station phi for the specific station
        Rodriguez-Marek et al. (2011) Equation 11
        """
        phiss = np.sum((intra_event - delta_s2ss) ** 2.) / float(n_events - 1)
        return np.sqrt(phiss)

    def get_total_phi_ss(self, pretty_print=None, filename=None):
        """
        Returns the station averaged single-station phi
        Rodriguez-Marek et al. (2011) Equation 10
        """
        if pretty_print:
            if filename:
                fid = open(filename, "w")
            else:
                fid = sys.stdout
        phi_ss = self._set_empty_dict()
        phi_s2ss = self._set_empty_dict()
        n_sites = float(len(self.site_residuals))
        for gmpe in self.gmpe_list:
            if pretty_print:
                print("%s" % gmpe, file=fid) 
                
            for imtx in self.imts:
                if pretty_print:
                    print("%s" % imtx, file=fid)
                if not "Intra event" in self.site_residuals[0].site_analysis[
                    gmpe][imtx]:
                    print("GMPE %s and IMT %s do not have defined "\
                        "random effects residuals" % (str(gmpe), str(imtx)))
                    continue
                n_events = []
                numerator_sum = 0.0
                d2ss = []
                for iloc, resid in enumerate(self.site_residuals):
                    d2ss.append(resid.site_analysis[gmpe][imtx]["dS2ss"])
                    n_events.append(resid.site_analysis[gmpe][imtx]["events"])
                    numerator_sum += np.sum((
                        resid.site_analysis[gmpe][imtx]["Intra event"] -
                        resid.site_analysis[gmpe][imtx]["dS2ss"]) ** 2.)
                    if pretty_print:
                        print("Site ID, %s, dS2Ss, %12.8f, "\
                            "phiss_s, %12.8f, Num Records, %s" % (
                            self.site_ids[iloc],
                            resid.site_analysis[gmpe][imtx]["dS2ss"],
                            resid.site_analysis[gmpe][imtx]["phi_ss,s"],
                            resid.site_analysis[gmpe][imtx]["events"]),
                            file=fid)
                d2ss = np.array(d2ss)
                phi_s2ss[gmpe][imtx] = {"Mean": np.mean(d2ss),
                                        "StdDev": np.std(d2ss)}
                phi_ss[gmpe][imtx] = np.sqrt(
                    numerator_sum / 
                    float(np.sum(np.array(n_events)) - 1))
        if pretty_print:
            print("TOTAL RESULTS FOR GMPE", file=fid)
            for gmpe in self.gmpe_list:
                print("%s" % gmpe, file=fid) 
                
                for imtx in self.imts:
                    print("%s, phi_ss, %12.8f, phi_s2ss(Mean),"\
                        " %12.8f, phi_s2ss(Std. Dev), %12.8f" % (imtx,
                        phi_ss[gmpe][imtx], phi_s2ss[gmpe][imtx]["Mean"],
                        phi_s2ss[gmpe][imtx]["StdDev"]), file=fid)
            if filename:
                fid.close()
        return phi_ss, phi_s2ss
       
