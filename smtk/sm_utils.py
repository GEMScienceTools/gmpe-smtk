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
Strong motion utilities
"""
import os
import sys
import numpy as np
from scipy.integrate import cumtrapz
from collections import OrderedDict
from openquake.hazardlib.geo import (PlanarSurface, SimpleFaultSurface,
                                     ComplexFaultSurface, MultiSurface,
                                     Point, Line)
import matplotlib.pyplot as plt

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

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
    Converts acceleration from different units into cm/s^2

    :param units: string in "g", "m/s/s", "m/s**2", "m/s^2",
        "cm/s/s", "cm/s**2" or "cm/s^2" (in the last three cases, this
        function simply returns `acceleration`)

    :return: acceleration converted to the given units
    """
    if units == "g":
        return 981. * acceleration
    if units in ("m/s/s", "m/s**2", "m/s^2"):
        return 100. * acceleration
    if units in ("cm/s/s", "cm/s**2", "cm/s^2"):
        return acceleration

    raise ValueError("Unrecognised time history units. "
                     "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")


def get_velocity_displacement(time_step, acceleration, units="cm/s/s",
                              velocity=None, displacement=None):
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
    if velocity is None:
        velocity = time_step * cumtrapz(acceleration, initial=0.)
    if displacement is None:
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


def load_pickle(pickle_file):
    """
    Python 2 & 3 compatible way of loading a Python Pickle file
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# Set of modules for converting OpenQuake surface classes to dictionaries
def planar_fault_surface_to_dict(surface):
    """
    Parses a PlanarSurface object to a dictionary formatted for json export
    """
    output = OrderedDict([("type", "PlanarSurface")])
    # Top left
    for i, key in zip([0, 1, 3, 2],
        ["top_left", "top_right", "bottom_right", "bottom_left"]):
        output[key] = [surface.corner_lons[i],
                       surface.corner_lats[i],
                       surface.corner_depths[i]]
    return output


def simple_fault_surface_to_dict(surface):
    """
    Parsers a SimpleFaultSurface object ot a dictionary formatted for
    json export

    More complicated here as we need to use the surface nodes
    """
    output = OrderedDict([("type", "SimpleFaultSurface")])
    for node in surface.surface_nodes[0]:
        if "LineString" in node.tag:
            trace = node[0].text
            output["trace"] = [[trace[i], trace[i + 1]]
                                for i in range(0, len(trace), 2)]
        else:
            output[node.tag] = node.text
    return output


def complex_fault_surface_to_dict(surface):
    """
    Parses a ComplexFaultSurface object ot a dictionary formatted for json
    export
    """
    output = OrderedDict([("type", "ComplexFaultSurface")])
    intermediate_edges = []
    for node in surface.surface_nodes[0]:
        edge = node[0].nodes[0].text
        edge = [[edge[i], edge[i + 1], edge[i + 2]]
                for i in range(0, len(edge), 3)]
        if "intermediateEdge" in node.tag:
            intermediate_edges.append(edge)
        else:
            output[node.tag] = edge
    output["intermediateEdges"] = intermediate_edges
    return output


surfaces_to_dict = {
    "PlanarSurface": planar_fault_surface_to_dict,
    "SimpleFaultSurface": simple_fault_surface_to_dict,
    "ComplexFaultSurface": complex_fault_surface_to_dict,
    }


def multi_surface_to_dict(surface):
    """
    Parses a multi
    """
    output = OrderedDict([("type", "MultiSurface"), ("surfaces", [])])
    for sfc in surface.surfaces:
        output["surfaces"].append(
            surfaces_to_dict[sfc.__class__.__name__](sfc))
    return output


surfaces_to_dict["MultiSurface"] = multi_surface_to_dict


# Surfaces from json dict
def planar_fault_surface_from_dict(data, mesh_spacing=1.):
    """
    Builds a planar fault surface from the json load
    """
    assert "PlanarSurface" in data
    top_left = Point(data["top_left"][0],
                     data["top_left"][1],
                     data["top_left"][2])
    top_right = Point(data["top_right"][0],
                      data["top_right"][1],
                      data["top_right"][2])
    bottom_left = Point(data["bottom_left"][0],
                        data["bottom_left"][1],
                        data["bottom_left"][2])
    bottom_right = Point(data["bottom_right"][0],
                         data["bottom_right"][1],
                         data["bottom_right"][2])
    return PlanarSurface.from_corner_points(mesh_spacing, top_left, top_right,
                                            bottom_right, bottom_left)

def simple_fault_surface_from_dict(data, mesh_spacing=1.):
    """
    Builds a simple fault surface from the json load
    """
    assert "SimpleFaultSurface" in data
    trace = []
    for lon, lat in data["trace"]:
        trace.append(Point(lon, lat, 0.0))
    trace = Line(trace)
    return SimpleFaultSurface.from_fault_data(trace, data["upperSeismoDepth"],
                                              data["lowerSeismoDepth"],
                                              data["dip"], mesh_spacing)

def _3d_line_from_list(vals):
    """
    """
    vertices = []
    for lon, lat, depth in vals:
        vertices.append(Point(lon, lat, depth))
    return Line(vertices)

def complex_fault_surface_from_dict(data, mesh_spacing=1.):
    """
    Builds a complex fault surface from the json load
    """
    assert "ComplexFaultSurface" in data
    edges = [_3d_line_from_list(data["faultTopEdge"])]
    if len(data["intermediateEdges"]):
        for iedge in data["intermediateEdges"]:
            edges.append(_3d_line_from_list(iedge))
    edges.append(_3d_line_from_list(data["faultBottomEdge"]))
    return ComplexFaultSurface.from_fault_data(edges, mesh_spacing)

surfaces_from_dict = {
    "PlanarSurface": planar_fault_surface_from_dict,
    "SimpleFaultSurface": simple_fault_surface_from_dict,
    "ComplexFaultSurface": complex_fault_surface_from_dict}

def multi_surface_from_dict(data, mesh_spacing=1.):
    """
    Builds a multi-fault surface from the json load
    """
    assert "MultiSurface" in data
    surfaces = []
    for surface in data["surfaces"]:
        surfaces.append(surfaces_from_dict[surface["type"]](surface,
                                                            mesh_spacing))
    return MultiSurface(surfaces)

surfaces_from_dict["MultiSurface"] = multi_surface_from_dict



    


    


#surfaces_to_dict = {
#    "PlanarSurface": planar_fault_surface_to_dict,
#    "SimpleFaultSurface": simple_fault_surface_to_dict,
#    "ComplexFaultSurface": complex_fault_surface_to_dict,
#    "MultiSurface": multi_surface_to_dict
#    }

