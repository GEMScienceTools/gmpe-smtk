#!/usr/bin/env/python

'''
General Class for extracting Ground Motion Intensity Measures (IMs) from a
set of acceleration time series
'''

import numpy as np
from math import pi
from scipy.integrate import cumtrapz
from scipy.stats import scoreatpercentile
from scipy import constants
import matplotlib.pyplot as plt
import smtk.response_spectrum as rsp
from smtk.sm_utils import (get_velocity_displacement,
                           get_time_vector,
                           convert_accel_units,
                           _save_image)

RESP_METHOD = {'Newmark-Beta': rsp.NewmarkBeta,
               'Nigam-Jennings': rsp.NigamJennings}


def get_peak_measures(time_step, acceleration, get_vel=False, 
    get_disp=False):
    '''

    '''
    pga = np.max(np.fabs(acceleration))
    velocity = None
    displacement = None
    if get_disp:
        get_vel = True
    if get_vel:
        velocity = time_step * cumtrapz(acceleration, initial=0.)
        pgv = np.max(np.fabs(velocity))
    else:
        pgv = None
    if get_disp:
        displacement = time_step * cumtrapz(velocity, initial=0.)
        pgd = np.max(np.fabs(displacement))
    else:
        pgd = None
    return pga, pgv, pgd, velocity, displacement

def get_response_spectrum(acceleration, time_step, periods, damping=0.05, 
        units="cm/s/s", method="Nigam-Jennings"):
    '''
    Returns the response spectrum
    '''
    response_spec = RESP_METHOD[method](acceleration,
                                        time_step,
                                        periods, 
                                        damping,
                                        units)
    spectrum, time_series, accel, vel, disp = response_spec.evaluate()
    spectrum["PGA"] = time_series["PGA"]
    spectrum["PGV"] = time_series["PGV"]
    spectrum["PGD"] = time_series["PGD"]
    return spectrum, time_series, accel, vel, disp


def get_response_spectrum_pair(acceleration_x, time_step_x, acceleration_y,
        time_step_y, periods, damping=0.05, units="cm/s/s",
        method="Nigam-Jennings"):
    '''
    Returns the response spectrum
    '''

    sax = get_response_spectrum(acceleration_x,
                                time_step_x,
                                periods,
                                damping, 
                                units, 
                                method)[0]
    say = get_response_spectrum(acceleration_y,
                                time_step_y,
                                periods, 
                                damping, 
                                units, 
                                method)[0]
    return sax, say

def geometric_mean_spectrum(sax, say):
    """
    Returns the geometric mean of the response spectrum
    """
    sa_gm = {}
    for key in sax.keys():
        if key == "Period":
            sa_gm[key] = sax[key]
        else:
            sa_gm[key] = np.sqrt(sax[key] * say[key])
    return sa_gm

def arithmetic_mean_spectrum(sax, say):
    """
    Returns the arithmetic mean of the response spectrum
    """
    sa_am = {}
    for key in sax.keys():
        if key == "Period":
            sa_am[key] = sax[key]
        else:
            sa_am[key] = (sax[key] + say[key]) / 2.0
    return sa_am

def envelope_spectrum(sax, say):
    """
    Returns the envelope of the response spectrum
    """
    sa_env = {}
    for key in sax.keys():
        if key == "Period":
            sa_env[key] = sax[key]
        else:
            sa_env[key] = np.max(np.column_stack([sax[key], say[key]]),
                                 axis=1)
    return sa_env

def larger_pga(sax, say):
    """
    Returns the spectral acceleration from the component with the larger PGA
    """
    if sax["PGA"] >= say["PGA"]:
        return sax
    else:
        return say

def rotate_horizontal(series_x, series_y, angle):
    """
    Rotates two time-series according to the angle
    """
    angle = angle * (pi / 180.0)
    rot_hist_x = (np.cos(angle) * series_x) + (np.sin(angle) * series_y)
    rot_hist_y = (-np.sin(angle) * series_x) + (np.cos(angle) * series_y)
    return rot_hist_x, rot_hist_y

def equalise_series(series_x, series_y):
    """
    """
    n_x = len(series_x)
    n_y = len(series_y)
    if n_x > n_y:
        return series_x[:n_y], series_y
    elif n_y > n_x:
        return series_x, series_y[:n_x]
    else:
        return series_x, series_y

def gmrotdpp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
        percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean
    """
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(acceleration_x,
                                              time_step_x,
                                              periods, damping,
                                              units, method)
    say, _, y_a, _, _ = get_response_spectrum(acceleration_y,
                                              time_step_y,
                                              periods, damping,
                                              units, method)
    x_a, y_a = equalise_series(x_a, y_a)
    angles = np.arange(0., 90., 1.)
    max_a_theta = np.zeros([len(angles), len(periods)], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                np.max(np.fabs(y_a), axis=0))
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                           np.max(np.fabs(y_a), axis=0))
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(rot_x), axis=0) *
                                           np.max(np.fabs(rot_y), axis=0))

    gmrotd = np.percentile(max_a_theta, percentile, axis=0)
    return gmrotd, max_a_theta, angles

KEY_LIST = ["PGA", "PGV", "PGD", "Acceleration", "Velocity", 
            "Displacement", "Pseudo-Acceleration", "Pseudo-Velocity"]

def gmrotdpp_slow(acceleration_x, time_step_x, acceleration_y, time_step_y,
        periods, percentile, damping=0.05, units="cm/s/s",
        method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean. This "slow" version
    will rotate the original time-series and calculate the response spectrum
    at each angle. This is a slower process, but it means that GMRotDpp values
    can be calculated for othe time-series parameters (i.e. PGA, PGV and PGD) 
    """
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    accel_x, accel_y = equalise_series(acceleration_x, acceleration_y)
    angles = np.arange(0., 90., 1.)
    #max_a_theta = np.zeros([len(angles), len(periods) + 3], dtype=float)
    #max_a_theta[0, :] = np.sqrt(np.max(np.fabs(accel_x), axis=0) *
    #                            np.max(np.fabs(accel_y), axis=0))
    gmrotdpp = {
        "Period": periods,
        "PGA": np.zeros(len(angles), dtype=float),
        "PGV": np.zeros(len(angles), dtype=float),
        "PGD": np.zeros(len(angles), dtype=float),
        "Acceleration": np.zeros([len(angles), len(periods)], dtype=float),
        "Velocity": np.zeros([len(angles), len(periods)], dtype=float),
        "Displacement": np.zeros([len(angles), len(periods)], dtype=float),
        "Pseudo-Acceleration": np.zeros([len(angles), len(periods)], 
                                        dtype=float),
        "Pseudo-Velocity": np.zeros([len(angles), len(periods)], dtype=float)}
    # Get the response spectra for each angle
    for iloc, theta in enumerate(angles):
        if np.fabs(theta) < 1E-9:
            rot_x, rot_y = (accel_x, accel_y)
        else:
            rot_x, rot_y = rotate_horizontal(accel_x, accel_y, theta)
        sax, say = get_response_spectrum_pair(rot_x, time_step_x,
                                              rot_y, time_step_y,
                                              periods, damping,
                                              units, method)

        sa_gm = geometric_mean_spectrum(sax, say)
        print iloc, theta, sa_gm["Pseudo-Acceleration"]
        for key in KEY_LIST:
            if key in ["PGA", "PGV", "PGD"]:
                 gmrotdpp[key][iloc] = sa_gm[key]
            else:
                 gmrotdpp[key][iloc, :] = sa_gm[key]
              
    # Get the desired fractile
    for key in KEY_LIST:
        gmrotdpp[key] = np.percentile(gmrotdpp[key], percentile, axis=0)
    return gmrotdpp

def _get_gmrotd_penalty(gmrotd, gmtheta):
    """
    
    """
    n_angles, n_per = np.shape(gmtheta)
    penalty = np.zeros(n_angles, dtype=float)
    coeff = 1. / float(n_per)
    for iloc in range(0, n_angles):
        penalty[iloc] = coeff * np.sum(((gmtheta[iloc] / gmrotd) - 1.) ** 2.)

    locn = np.argmin(penalty)
    return locn, penalty


def gmrotipp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
        percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-independent geometric mean (GMRotIpp)
    """
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    gmrotd, gmtheta, angle = gmrotdpp(acceleration_x, time_step_x,
                                      acceleration_y, time_step_y, 
                                      periods, percentile, damping, units, 
                                      method)
    
    min_loc, penalty = _get_gmrotd_penalty(gmrotd, gmtheta)
    target_angle = angle[min_loc]

    rot_hist_x, rot_hist_y = rotate_horizontal(acceleration_x,
                                               acceleration_y,
                                               target_angle)
    sax, say = get_response_spectrum_pair(rot_hist_x, time_step_x,
                                          rot_hist_y, time_step_y,
                                          periods, damping, units, method)

    return geometric_mean_spectrum(sax, say)

ARIAS_FACTOR = pi / (2.0 * (constants.g * 100.))

def get_husid(acceleration, time_step):
    """
    Returns the Husid vector, defined as \int{acceleration ** 2.}
    """
    time_vector = get_time_vector(time_step, len(acceleration))
    husid = np.hstack([0., cumtrapz(acceleration ** 2., time_vector)])
    return husid, time_vector

def get_arias_intensity(acceleration, time_step, start_level=0., end_level=1.):
    """
    Returns the Arias intensity of the recor
    """
    assert end_level >= start_level
    husid, time_vector = get_husid(acceleration, time_step)
    husid_norm = husid / husid[-1]
    idx = np.where(np.logical_and(husid_norm >= start_level,
                                  husid_norm <= end_level))[0]
    if len(idx) < len(acceleration):
        husid, time_vector = get_husid(acceleration[idx], time_step)
    return ARIAS_FACTOR * husid[-1]


def plot_husid(acceleration, time_step, start_level=0., end_level=1.0,
        figure_size=(7, 5), filename=None, filetype="png", dpi=300):
    """
    Creates a Husid plot for the record
    """
    plt.figure(figsize=figure_size)
    husid, time_vector = get_husid(acceleration, time_step)
    husid_norm = husid / husid[-1]
    idx = np.where(np.logical_and(husid_norm >= start_level,
                                  husid_norm <= end_level))[0]
    plt.plot(time_vector, husid_norm, "b-", linewidth=2.0,
             label="Original Record")
    plt.plot(time_vector[idx], husid_norm[idx], "r-", linewidth=2.0,
             label="%5.3f - %5.3f Arias" % (start_level, end_level))
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Fraction of Arias Intensity", fontsize=14)
    plt.title("Husid Plot")
    plt.legend(loc=4, fontsize=14)
    _save_image(filename, filetype, dpi)
    plt.show()

def get_bracketed_duration(acceleration, time_step, threshold):
    """
    Returns the bracketed duration, defined as the time between the first and
    last excrusions above a particular level of acceleration
    """
    idx = np.where(np.fabs(acceleration) >= threshold)[0]
    if len(idx) == 0:
        # Record does not exced threshold at any point
        return 0.
    else:
        time_vector = get_time_vector(time_step, len(acceleration))
        return time_vector[idx[-1]] - time_vector[idx[0]] + time_step

def get_uniform_duration(acceleration, time_step, threshold):
    """
    Returns the total duration for which the record exceeds the threshold
    """ 
    idx = np.where(np.fabs(acceleration) >= threshold)[0]
    return time_step * float(len(idx))

def get_significant_duration(acceleration, time_step, start_level=0.,
        end_level=1.0):
    """
    Returns the significant duration of the record
    """
    assert end_level >= start_level
    husid, time_vector = get_husid(acceleration, time_step)
    idx = np.where(np.logical_and(husid >= (start_level * husid[-1]),
                                  husid <= (end_level * husid[-1])))[0]
    return time_vector[idx[-1]] - time_vector[idx[0]] + time_step


def get_cav(acceleration, time_step, threshold=0.0):
    """
    Returns the cumulative absolute velocity above a given threshold of
    acceleration
    """
    acceleration = np.fabs(acceleration)
    idx = np.where(acceleration >= threshold)
    if len(idx) > 0:
        return np.trapz(acceleration[idx], dx=time_step)
    else:
        return 0.0

def get_arms(acceleration, time_step):
    """
    Returns the root mean square acceleration, defined as
    sqrt{(1 / duration) * \int{acc ^ 2} dt}
    """
    dur = time_step * float(len(acceleration) - 1)
    return np.sqrt((1. / dur) * np.trapz(acceleration  ** 2., dx=time_step))

def get_response_spectrum_intensity(spec):
    """
    Returns the response spectrum intensity (Housner intensity), defined
    as the integral of the pseudo-velocity spectrum between the periods of
    0.1 s and 2.5 s
    """
    idx = np.where(np.logical_and(spec["Period"] >= 0.1,
                                  spec["Period"] <= 2.5))[0]
    return np.trapz(spec["Pseudo-Velocity"][idx],
                    spec["Period"][idx])


def get_acceleration_spectrum_intensity(spec):
    """
    Returns the acceleration spectrum intensity, defined as the integral
    of the psuedo-acceleration spectrum between the periods of 0.1 and 0.5 s
    """
    idx = np.where(np.logical_and(spec["Period"] >= 0.1,
                                  spec["Period"] <= 0.5))[0]
    return np.trapz(spec["Pseudo-Acceleration"][idx],
                    spec["Period"][idx])


def get_quadratic_intensity(acc_x, acc_y, time_step):
    """
    Returns the quadratic intensity of a pair of records, define as:
    (1. / duration) * \int_0^{duration} a_1(t) a_2(t) dt
    """
    assert len(acc_x) == len(acc_y)
    dur = time_step * float(len(acc_x) - 1)
    return (1. /  dur) * np.trapz(acc_x * acc_y, dx=time_step)

def get_principal_axes(time_step, acc_x, acc_y, acc_z=None):
    """
    Returns the principal axes of a set of ground motion records
    """
    # If time-series are not of equal length then equalise
    acc_x, acc_y = equalise_series(acc_x, acc_y)
    if acc_z is not None:
        nhist = 3
        if len(acc_z) > len(acc_x):
            acc_x, acc_z = equalise_series(acc_x, acc_z)
        else:
            acc_x, acc_z = equalise_series(acc_x, acc_z)
            acc_x, acc_y = equalise_series(acc_x, acc_y)
        acc = np.column_stack([acc_x, acc_y, acc_z])
    else:
        nhist = 2
        acc = np.column_stack([acc_x, acc_y])
    # Calculate quadratic intensity matrix
    sigma = np.zeros([nhist, nhist])
    rho = np.zeros([nhist, nhist])
    for iloc in range(0, nhist):
        for jloc in range(0, nhist):
            sigma[iloc, jloc] = get_quadratic_intensity(acc[:, iloc],
                                                        acc[:, jloc],
                                                        time_step)
    # Calculate correlation matrix
    for iloc in range(0, nhist):
        for jloc in range(0, nhist):
            rho[iloc, jloc] = sigma[iloc, jloc] / np.sqrt(sigma[iloc, iloc] *
                                                          sigma[jloc, jloc])
    # Get transformation matrix
    phi = np.matrix(np.linalg.eig(sigma)[1])
    # Transform the time-series
    acc_trans = phi.T * np.matrix(acc.T)
    acc_1 = (acc_trans[0, :].A).flatten()
    acc_2 = (acc_trans[1, :].A).flatten()
    if nhist == 3:
        acc_3 = (acc_trans[2, :].A).flatten()
    else:
        acc_3 = None
    return acc_1, acc_2, acc_3
