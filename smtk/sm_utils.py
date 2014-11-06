#!/usr/bin/env/python

"""
Strong motion utilities
"""
import os
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

def get_time_vector(time_step, number_steps):
    """
    General SMTK utils
    """
    return np.cumsum(time_step * np.ones(number_steps, dtype=float)) -\
        time_step

def nextpow2(nval):
    m_f = np.log2(nval)
    m_i = np.ceil(m_f)
    return int(2.0 ** m_i)

def convert_accel_units(acceleration, units):
    """
    Converts acceleration to different units
    """
    if units=="g":
        return 981. * acceleration
    elif (units=="m/s/s") or (units=="m/s**2"):
        return 100. * acceleration
    elif (units=="cm/s/s") or (units=="cm/s**2"):
        return acceleration
    else:
        raise ValueError("Unrecognised time history units. "
                         "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")

def get_velocity_displacement(time_step, acceleration, units="cm/s/s"):
    '''
    Returns the velocity and displacment time series using simple integration
    :param float time_step:
        Time-series time-step (s)
    :param numpy.ndarray acceleration:
        Acceleration time-history
    :returns:
        velocity - Velocity Time series (cm/s)
        displacement - Displacement Time series (cm)
    '''
    acceleration = convert_accel_units(acceleration, units)
    velocity = time_step * cumtrapz(acceleration, initial=0.)
    displacement = time_step * cumtrapz(velocity, initial=0.)
    return velocity, displacement


def build_filename(filename, filetype='png', resolution=300):
    """
    Uses the input properties to create the string of the filename
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    filevals = os.path.splitext(filename)
    if filevals[1]:
        filetype = filevals[1][1:]
    if not filetype:
        filetype = 'png'

    filename = filevals[0] + '.' + filetype

    if not resolution:
        resolution = 300
    return filename, filetype, resolution

def _save_image(filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        plt.savefig(filename, dpi=resolution, format=filetype)
    else:
        pass
    return

def _save_image_tight(fig, lgd, filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        fig.savefig(filename, bbox_extra_artists=(lgd,),
                    bbox_inches="tight", dpi=resolution, format=filetype)
    else:
        pass
    return


