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

'''
Simple Python Script to integrate a strong motion record using 
the Newmark-Beta method
'''

import numpy as np
from math import sqrt
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from smtk.sm_utils import (_save_image,
                      get_time_vector,
                      convert_accel_units,
                      get_velocity_displacement)
                     

class ResponseSpectrum(object):
    '''
    Base Class to implement a response spectrum calculation
    '''
    def __init__(self, acceleration, time_step, periods, damping=0.05,
            units="cm/s/s"):
        '''
        Setup the response spectrum calculator
        :param numpy.ndarray time_hist:
            Acceleration time history [Time, Acceleration]
        :param numpy.ndarray periods:
            Spectral periods (s) for calculation
        :param float damping:
            Fractional coefficient of damping
        :param str units:
            Units of the acceleration time history {"g", "m/s", "cm/s/s"}

        '''
        self.periods = periods
        self.num_per = len(periods)
        self.acceleration = convert_accel_units(acceleration, units)
        self.damping = damping
        self.d_t = time_step
        self.velocity, self.displacement = get_velocity_displacement(
            self.d_t, self.acceleration)
        self.num_steps = len(self.acceleration)
        self.omega = (2. * np.pi) / self.periods
        self.response_spectrum = None


    def __call__(self):
        '''
        Evaluates the response spectrum
        :returns:
            Response Spectrum - Dictionary containing all response spectrum
                                data
                'Time' - Time (s)
                'Acceleration' - Acceleration Response Spectrum (cm/s/s)
                'Velocity' - Velocity Response Spectrum (cm/s)
                'Displacement' - Displacement Response Spectrum (cm)
                'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum (cm/s)
                'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum 
                                       (cm/s/s)

            Time Series - Dictionary containing all time-series data
                'Time' - Time (s)
                'Acceleration' - Acceleration time series (cm/s/s)
                'Velocity' - Velocity time series (cm/s)
                'Displacement' - Displacement time series (cm)
                'PGA' - Peak ground acceleration (cm/s/s)
                'PGV' - Peak ground velocity (cm/s)
                'PGD' - Peak ground displacement (cm)
                
            accel - Acceleration response of Single Degree of Freedom Oscillator 
            vel - Velocity response of Single Degree of Freedom Oscillator 
            disp - Displacement response of Single Degree of Freedom Oscillator 
        '''
        raise NotImplementedError("Cannot call Base Response Spectrum")


class NewmarkBeta(ResponseSpectrum):
    '''
    Evaluates the response spectrum using the Newmark-Beta methodology
    '''

    def __call__(self):
        '''
        Evaluates the response spectrum
        :returns:
            Response Spectrum - Dictionary containing all response spectrum
                                data
                'Time' - Time (s)
                'Acceleration' - Acceleration Response Spectrum (cm/s/s)
                'Velocity' - Velocity Response Spectrum (cm/s)
                'Displacement' - Displacement Response Spectrum (cm)
                'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum (cm/s)
                'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum 
                                       (cm/s/s)

            Time Series - Dictionary containing all time-series data
                'Time' - Time (s)
                'Acceleration' - Acceleration time series (cm/s/s)
                'Velocity' - Velocity time series (cm/s)
                'Displacement' - Displacement time series (cm)
                'PGA' - Peak ground acceleration (cm/s/s)
                'PGV' - Peak ground velocity (cm/s)
                'PGD' - Peak ground displacement (cm)
                
            accel - Acceleration response of Single Degree of Freedom Oscillator 
            vel - Velocity response of Single Degree of Freedom Oscillator 
            disp - Displacement response of Single Degree of Freedom Oscillator 
        '''
        omega = (2. * np.pi) / self.periods
        cval = self.damping * 2. * omega
        kval = ((2. * np.pi) / self.periods) ** 2.
        # Perform Newmark - Beta integration
        accel, vel, disp, a_t = self._newmark_beta(omega, cval, kval)
        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': np.max(np.fabs(a_t), axis=0),
            'Velocity': np.max(np.fabs(vel), axis=0),
            'Displacement': np.max(np.fabs(disp), axis=0)}
        self.response_spectrum['Pseudo-Velocity'] =  omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] =  (omega ** 2.) * \
            self.response_spectrum['Displacement']
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)),
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}
        return self.response_spectrum, time_series, accel, vel, disp
        

    def _newmark_beta(self, omega, cval, kval):
        '''
        Newmark-beta integral
        :param numpy.ndarray omega:
            Angular period - (2 * pi) / T
        :param numpy.ndarray cval:
            Damping * 2 * omega
        :param numpy.ndarray kval:
            ((2. * pi) / T) ** 2.

        :returns:
            accel - Acceleration time series
            vel - Velocity response of a SDOF oscillator
            disp - Displacement response of a SDOF oscillator
            a_t - Acceleration response of a SDOF oscillator

        '''
        # Pre-allocate arrays
        accel = np.zeros([self.num_steps, self.num_per], dtype=float)
        vel = np.zeros([self.num_steps, self.num_per], dtype=float)
        disp = np.zeros([self.num_steps, self.num_per], dtype=float)
        a_t = np.zeros([self.num_steps, self.num_per], dtype=float)
        # Initial line
        accel[0, :] =(-self.acceleration[0] - (cval * vel[0, :])) - \
                      (kval * disp[0, :])
        a_t[0, :] = accel[0, :] + accel[0, :]
        for j in range(1, self.num_steps):
            disp[j, :] = disp[j-1, :] + (self.d_t * vel[j-1, :]) + \
                (((self.d_t ** 2.) / 2.) * accel[j-1, :])
                         
            accel[j, :] = (1./ (1. + self.d_t * 0.5 * cval)) * \
                (-self.acceleration[j] - kval * disp[j, :] - cval *
                (vel[j-1, :] + (self.d_t * 0.5) * accel[j-1, :]));
            vel[j, :] = vel[j - 1, :] + self.d_t * (0.5 * accel[j - 1, :] +
                0.5 * accel[j, :])
            a_t[j, :] = self.acceleration[j] + accel[j, :]
        return accel, vel, disp, a_t


class NigamJennings(ResponseSpectrum):
    """
    Evaluate the response spectrum using the algorithm of Nigam & Jennings
    (1969)
    In general this is faster than the classical Newmark-Beta method, and
    can provide estimates of the spectra at frequencies higher than that
    of the sampling frequency.
    """

    def __call__(self):
        """
        Define the response spectrum
        """
        omega = (2. * np.pi) / self.periods
        omega2 = omega ** 2.
        omega3 = omega ** 3.
        omega_d = omega * sqrt(1.0 - (self.damping ** 2.))
        const = {'f1': (2.0 * self.damping) / (omega3 * self.d_t),
                'f2': 1.0 / omega2,
                'f3': self.damping * omega,
                'f4': 1.0 / omega_d}
        const['f5'] = const['f3'] * const['f4']
        const['f6'] = 2.0 * const['f3']
        const['e'] = np.exp(-const['f3'] * self.d_t)
        const['s'] = np.sin(omega_d * self.d_t)
        const['c'] = np.cos(omega_d * self.d_t)
        const['g1'] = const['e'] * const['s']
        const['g2'] = const['e'] * const['c']
        const['h1'] = (omega_d * const['g2']) - (const['f3'] * const['g1'])
        const['h2'] = (omega_d * const['g1']) + (const['f3'] * const['g2'])
        x_a, x_v, x_d = self._get_time_series(const, omega2)

        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': np.max(np.fabs(x_a), axis=0),
            'Velocity': np.max(np.fabs(x_v), axis=0),
            'Displacement': np.max(np.fabs(x_d), axis=0)}
        self.response_spectrum['Pseudo-Velocity'] =  omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] =  (omega ** 2.) * \
            self.response_spectrum['Displacement']
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)),
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}

        return self.response_spectrum, time_series, x_a, x_v, x_d
        
    def _get_time_series(self, const, omega2):
        """
        Calculates the acceleration, velocity and displacement time series for
        the SDOF oscillator
        :param dict const:
            Constants of the algorithm
        :param np.ndarray omega2:
            Square of the oscillator period
        :returns:
            x_a = Acceleration time series
            x_v = Velocity time series
            x_d = Displacement time series
        """
        x_d = np.zeros([self.num_steps - 1, self.num_per], dtype=float)
        x_v = np.zeros_like(x_d)
        x_a = np.zeros_like(x_d)
        
        for k in range(0, self.num_steps - 1):
            yval = k - 1
            dug = self.acceleration[k + 1] - self.acceleration[k]
            z_1 = const['f2'] * dug
            z_2 = const['f2'] * self.acceleration[k]
            z_3 = const['f1'] * dug
            z_4 = z_1 / self.d_t
            if k == 0:
                b_val = z_2 - z_3
                a_val = (const['f5'] * b_val) + (const['f4'] * z_4)
            else:    
                b_val = x_d[k - 1, :] + z_2 - z_3
                a_val = (const['f4'] * x_v[k - 1, :]) +\
                    (const['f5'] * b_val) + (const['f4'] * z_4)

            x_d[k, :] = (a_val * const['g1']) + (b_val * const['g2']) +\
                z_3 - z_2 - z_1
            x_v[k, :] = (a_val * const['h1']) - (b_val * const['h2']) - z_4
            x_a[k, :] = (-const['f6'] * x_v[k, :]) - (omega2 * x_d[k, :])
        return x_a, x_v, x_d


PLOT_TYPE = {"loglog": lambda ax, x, y : ax.loglog(x, y),
             "semilogx": lambda ax, x, y : ax.semilogx(x, y),
             "semilogy": lambda ax, x, y : ax.semilogy(x, y),
             "linear": lambda ax, x, y : ax.plot(x, y)}


def plot_response_spectra(spectra, axis_type="loglog", figure_size=(8, 6),
        filename=None, filetype="png", dpi=300):
    """
    Creates a plot of the suite of response spectra (Acceleration,
    Velocity, Displacement, Pseudo-Acceleration, Pseudo-Velocity) derived
    from a particular ground motion record
    """
    fig = plt.figure(figsize=figure_size)
    fig.set_tight_layout(True)
    ax = plt.subplot(2, 2, 1)
    # Acceleration
    PLOT_TYPE[axis_type](ax, spectra["Period"], spectra["Acceleration"])
    PLOT_TYPE[axis_type](ax, spectra["Period"], spectra["Pseudo-Acceleration"])
    ax.set_xlabel("Periods (s)", fontsize=12)
    ax.set_ylabel("Acceleration (cm/s/s)", fontsize=12)
    ax.set_xlim(np.min(spectra["Period"]), np.max(spectra["Period"]))
    ax.grid()
    ax.legend(("Acceleration", "PSA"), loc=0) 
    ax = plt.subplot(2, 2, 2)
    # Velocity
    PLOT_TYPE[axis_type](ax, spectra["Period"], spectra["Velocity"])
    PLOT_TYPE[axis_type](ax, spectra["Period"], spectra["Pseudo-Velocity"])
    ax.set_xlabel("Periods (s)", fontsize=12)
    ax.set_ylabel("Velocity (cm/s)", fontsize=12)
    ax.set_xlim(np.min(spectra["Period"]), np.max(spectra["Period"]))
    ax.grid()
    ax.legend(("Velocity", "PSV"), loc=0) 
    ax = plt.subplot(2, 2, 3)
    # Displacement
    PLOT_TYPE[axis_type](ax, spectra["Period"], spectra["Displacement"])
    ax.set_xlabel("Periods (s)", fontsize=12)
    ax.set_ylabel("Displacement (cm)", fontsize=12)
    ax.set_xlim(np.min(spectra["Period"]), np.max(spectra["Period"]))
    ax.grid()
    _save_image(filename, filetype, dpi)
    plt.show()
   
def plot_time_series(acceleration, time_step, velocity=[], displacement=[],
        units="cm/s/s", figure_size=(8, 6), filename=None, filetype="png",
        dpi=300, linewidth=1.5):
    """
    Creates a plot of acceleration, velocity and displacement for a specific
    ground motion record
    """
    acceleration = convert_accel_units(acceleration, units)
    accel_time = get_time_vector(time_step, len(acceleration))
    if not len(velocity):
        velocity, dspl = get_velocity_displacement(time_step, acceleration)
    vel_time = get_time_vector(time_step, len(velocity))
    if not len(displacement):
        displacement = dspl
    disp_time = get_time_vector(time_step, len(displacement))
    fig = plt.figure(figsize=figure_size)
    fig.set_tight_layout(True)
    ax = plt.subplot(3, 1, 1)
    # Accleration
    ax.plot(accel_time, acceleration, 'k-', linewidth=linewidth)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Acceleration (cm/s/s)", fontsize=12)
    end_time = np.max(np.array([accel_time[-1], vel_time[-1], disp_time[-1]]))
    pga = np.max(np.fabs(acceleration))
    ax.set_xlim(0, end_time)
    ax.set_ylim(-1.1 * pga, 1.1 * pga)
    ax.grid()
    # Velocity
    ax = plt.subplot(3, 1, 2)
    ax.plot(vel_time, velocity, 'b-', linewidth=linewidth)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Velocity (cm/s)", fontsize=12)
    pgv = np.max(np.fabs(velocity))
    ax.set_xlim(0, end_time)
    ax.set_ylim(-1.1 * pgv, 1.1 * pgv)
    ax.grid()
    # Displacement
    ax = plt.subplot(3, 1, 3)
    ax.plot(disp_time, displacement, 'r-', linewidth=linewidth)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement (cm)", fontsize=12)
    pgd = np.max(np.fabs(displacement))
    ax.set_xlim(0, end_time)
    ax.set_ylim(-1.1 * pgd, 1.1 * pgd)
    ax.grid()
    _save_image(filename, filetype, dpi)
    plt.show()
