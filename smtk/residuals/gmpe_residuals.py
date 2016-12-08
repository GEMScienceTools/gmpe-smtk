#!/usr/bin/env/python

"""
Module to get GMPE residuals - total, inter and intra
{'GMPE': {'IMT1': {'Total': [], 'Inter event': [], 'Intra event': []},
          'IMT2': { ... }}}
          


"""
import sys
import re, os
import h5py
import numpy as np
from math import sqrt, ceil
from scipy.special import erf
from scipy.stats import scoreatpercentile, norm
from copy import deepcopy
from collections import OrderedDict
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.gsim.gsim_table import GMPETable
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
        for gmpe in self.gmpe_list:
            gmpe_dict_1 = {}
            gmpe_dict_2 = {}
            self.unique_indices[gmpe] = {}
            for imtx in self.imts:
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
                    for res_type in self.residuals[gmpe][imtx].keys():
                        if res_type == "Inter event":
                            inter_ev = \
                                context["Residual"][gmpe][imtx][res_type]
                            inter_ev, inter_idx = np.unique(
                                inter_ev,
                                return_index=True)
                            self.residuals[gmpe][imtx][res_type].extend(
                                inter_ev.tolist())
                            #inter_mags = (context["Rupture"].mag *
                            #              np.ones(len(inter_ev))).tolist()
                            self.unique_indices[gmpe][imtx].append(
                                inter_idx)
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
                for res_type in self.residuals[gmpe][imtx].keys():
                    self.residuals[gmpe][imtx][res_type] = np.array(
                        self.residuals[gmpe][imtx][res_type])
                    self.modelled[gmpe][imtx][res_type] = np.array(
                        self.modelled[gmpe][imtx][res_type])
                self.modelled[gmpe][imtx]["Mean"] = np.array(
                    self.modelled[gmpe][imtx]["Mean"])
                #self.unique_magnitudes[gmpe][imtx] = np.array(
                #    self.unique_magnitudes[gmpe][imtx])

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
        for gmpe in self.gmpe_list:
            expected[gmpe] = OrderedDict([(imtx, {}) for imtx in self.imts])
            for imtx in self.imts:
                gsim = self.gmpe_list[gmpe]
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
            residual[gmpe] = {}
            for imtx in self.imts:
                residual[gmpe][imtx] = {}
                obs = np.log(context["Observations"][imtx])
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
        statistics = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                statistics[gmpe][imtx] = {}
                for res_type in self.types[gmpe][imtx]:
#                    if res_type == "Inter event":
#                        # As the inter-event term may be vectorial with
#                        # repeated columns, get take only one value of inter-
#                        # event residual if unique
#                        delta_e = np.array([], dtype="float")
#                        for ctxt in self.contexts:
#                            iev_res = np.unique(
#                                ctxt["Residual"][gmpe][imtx]["Inter event"])
#                            delta_e = np.hstack([delta_e, iev_res])
#                        #print delta_e
#                        data = {
#                            "Mean": np.mean(delta_e),
#                                #self.residuals[gmpe][imtx][res_type]),
#                            "Std Dev": np.std(delta_e)}
#                                #self.residuals[gmpe][imtx][res_type])}
#                    else:
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
        # Prin headers
        event = self.contexts[0]
        header_set = []
        header_set.extend([key for key in event["Distances"].__dict__])
        header_set.extend([key for key in event["Sites"].__dict__])
        header_set.extend(["{:s}-Obs.".format(imt) for imt in self.imts])
        for imt in self.imts:
            for gmpe in self.gmpe_list:
                for key in event["Expected"][gmpe][imt].keys():
                    header_set.append(
                        "{:s}-{:s}-{:s}-Exp.".format(imt, gmpe, key))
        for imt in self.imts:
            for gmpe in self.gmpe_list:
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
                    for key in event["Expected"][gmpe][imt].keys():
                        data.append("{:.8e}".format(
                            event["Expected"][gmpe][imt][key][i]))
            # Residuals
            for imt in self.imts:
                for gmpe in self.gmpe_list:
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
        lh_values = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
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
    def get_loglikelihood_values(self):
        log_residuals = OrderedDict([(gmpe, np.array([]))
                                      for gmpe in self.gmpe_list])
        llh = OrderedDict([(gmpe, None) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                asll = np.log2(norm.pdf(self.residuals[gmpe][imtx]["Total"], 
                               0., 
                               1.0))
                log_residuals[gmpe] = np.hstack([log_residuals[gmpe], asll])
            llh[gmpe] = -(1. / float(len(log_residuals[gmpe]))) *\
                np.sum(log_residuals[gmpe])
        # Get weights
        weights = np.array([2.0 ** -llh[gmpe] for gmpe in self.gmpe_list])
        weights = weights / np.sum(weights)
        model_weights = OrderedDict([
            (gmpe, weights[iloc]) for iloc, gmpe in enumerate(self.gmpe_list)]
            )
        return llh, model_weights

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

    def get_site_residuals(self, database):
        """
        Calculates the total, inter-event and within-event residuals for
        each site
        """
        imt_dict = dict([(imtx, {}) for imtx in self.imts])
        for site_id in self.site_ids:
            print site_id
            selector = SMRecordSelector(database)
            site_db = selector.select_from_site_id(site_id, as_db=True)
            resid = Residuals(self.input_gmpe_list, self.imts)
            resid.get_residuals(site_db, normalise=False)
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
                print >> fid, "%s" % gmpe 
                
            for imtx in self.imts:
                if pretty_print:
                    print >> fid, "%s" % imtx
                if not "Intra event" in self.site_residuals[0].site_analysis[
                    gmpe][imtx]:
                    print "GMPE %s and IMT %s do not have defined "\
                        "random effects residuals" % (str(gmpe), str(imtx))
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
                        print >> fid, "Site ID, %s, dS2Ss, %12.8f, "\
                            "phiss_s, %12.8f, Num Records, %s" % (
                            self.site_ids[iloc],
                            resid.site_analysis[gmpe][imtx]["dS2ss"],
                            resid.site_analysis[gmpe][imtx]["phi_ss,s"],
                            resid.site_analysis[gmpe][imtx]["events"])
                d2ss = np.array(d2ss)
                phi_s2ss[gmpe][imtx] = {"Mean": np.mean(d2ss),
                                        "StdDev": np.std(d2ss)}
                phi_ss[gmpe][imtx] = np.sqrt(
                    numerator_sum / 
                    float(np.sum(np.array(n_events)) - 1))
        if pretty_print:
            print >> fid, "TOTAL RESULTS FOR GMPE"
            for gmpe in self.gmpe_list:
                print >> fid, "%s" % gmpe 
                
                for imtx in self.imts:
                    print >> fid, "%s, phi_ss, %12.8f, phi_s2ss(Mean),"\
                        " %12.8f, phi_s2ss(Std. Dev), %12.8f" % (imtx,
                        phi_ss[gmpe][imtx], phi_s2ss[gmpe][imtx]["Mean"],
                        phi_s2ss[gmpe][imtx]["StdDev"])
            if filename:
                fid.close()
        return phi_ss, phi_s2ss
       
