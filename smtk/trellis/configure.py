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
...configure implement SourceConfigure, the class to set-up the source and
site configuration for comparing the trellis plots
"""

import numpy as np
from math import sqrt, pi, sin, cos, fabs, exp, log
from copy import deepcopy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from openquake.baselib.slots import with_slots
from openquake.hazardlib.geo import (Point, Line, Polygon, Mesh,
                                     PlanarSurface, NodalPlane)
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.site import Site, SiteCollection
#from openquake.hazardlib.source.rupture import Rupture
from openquake.hazardlib.source.point import PointSource
from openquake.hazardlib.gsim.base import (SitesContext,
                                           RuptureContext,
                                           DistancesContext)
from smtk.sm_utils import _save_image


MESH_SPACING = 1.0
TO_RAD = pi / 180.
FROM_RAD = 180. / pi
# Default point - some random location on Earth
DEFAULT_POINT = Point(45.18333, 9.15, 0.)


# This object was original from the OQ-Hazardlib. It has now been removed to
# a direct copy and paste is found here
@with_slots
class Rupture(object):
    """
    Rupture object represents a single earthquake rupture.

    :param mag:
        Magnitude of the rupture.
    :param rake:
        Rake value of the rupture.
        See :class:`~openquake.hazardlib.geo.nodalplane.NodalPlane`.
    :param tectonic_region_type:
        Rupture's tectonic regime. One of constants
        in :class:`openquake.hazardlib.const.TRT`.
    :param hypocenter:
        A :class:`~openquake.hazardlib.geo.point.Point`, rupture's hypocenter.
    :param surface:
        An instance of subclass of
        :class:`~openquake.hazardlib.geo.surface.base.BaseSurface`.
        Object representing the rupture surface geometry.
    :param source_typology:
        Subclass of :class:`~openquake.hazardlib.source.base.BaseSeismicSource`
        (class object, not an instance) referencing the typology
        of the source that produced this rupture.
    :param rupture_slip_direction:
        Angle describing rupture propagation direction in decimal degrees.

    :raises ValueError:
        If magnitude value is not positive, hypocenter is above the earth
        surface or tectonic region type is unknown.

    NB: if you want to convert the rupture into XML, you should set the
    attribute surface_nodes to an appropriate value.
    """
    _slots_ = '''mag rake tectonic_region_type hypocenter surface
    surface_nodes source_typology rupture_slip_direction'''.split()

    def __init__(self, mag, rake, tectonic_region_type, hypocenter,
                 surface, source_typology, rupture_slip_direction=None,
                 surface_nodes=()):
        if not mag > 0:
            raise ValueError('magnitude must be positive')
        if not hypocenter.depth > 0:
            raise ValueError('rupture hypocenter must have positive depth')
        NodalPlane.check_rake(rake)
        self.tectonic_region_type = tectonic_region_type
        self.rake = rake
        self.mag = mag
        self.hypocenter = hypocenter
        self.surface = surface
        self.source_typology = source_typology
        self.surface_nodes = surface_nodes
        self.rupture_slip_direction = rupture_slip_direction


def create_planar_surface(top_centroid, strike, dip, area, aspect):
    """
    Given a central location, create a simple planar rupture
    :param top_centroid:
        Centroid of trace of the rupture, as instance of :class:
            openquake.hazardlib.geo.point.Point
    :param float strike:
        Strike of rupture(Degrees)
    :param float dip:
        Dip of rupture (degrees)
    :param float area:
        Area of rupture (km^2)
    :param float aspect:
        Aspect ratio of rupture

    :returns: Rupture as an instance of the :class:
        openquake.hazardlib.geo.surface.planar.PlanarSurface
    """
    rad_dip = dip * pi / 180.
    width = sqrt(area / aspect)
    length = aspect * width
    # Get end points by moving the top_centroid along strike
    top_right = top_centroid.point_at(length / 2., 0., strike)
    top_left = top_centroid.point_at(length / 2.,
                                     0.,
                                     (strike + 180.) % 360.)
    # Along surface width
    surface_width = width * cos(rad_dip)
    vertical_depth = width * sin(rad_dip)
    dip_direction = (strike + 90.) % 360.

    bottom_right = top_right.point_at(surface_width,
                                      vertical_depth,
                                      dip_direction)
    bottom_left = top_left.point_at(surface_width,
                                    vertical_depth,
                                    dip_direction)

    # Create the rupture
    return PlanarSurface(MESH_SPACING, strike, dip, top_left, top_right,
                         bottom_right, bottom_left)


def get_hypocentre_on_planar_surface(plane, hypo_loc=None):
    """
    Determines the location of the hypocentre within the plane
    :param plane:
        Rupture plane as instance of :class:
        openquake.hazardlib.geo.surface.planar.PlanarSurface
    :param tuple hypo_loc:
        Hypocentre location as fraction of rupture plane, as a tuple of 
        (Along Strike, Down Dip), e.g. a hypocentre located in the centroid of
        the rupture plane would be input as (0.5, 0.5), whereas a hypocentre
        located in a position 3/4 along the length, and 1/4 of the way down
        dip of the rupture plane would be entered as (0.75, 0.25)
    :returns:
        Hypocentre location as instance of :class:
        openquake.hazardlib.geo.point.Point
    """

    centroid = plane.get_middle_point()
    if hypo_loc is None:
        return centroid

    along_strike_dist = (hypo_loc[0] * plane.length) - (0.5 * plane.length)
    down_dip_dist = (hypo_loc[1] * plane.width) - (0.5 * plane.width)
    if along_strike_dist >= 0.:
        along_strike_azimuth = plane.strike
    else:
        along_strike_azimuth = (plane.strike + 180.) % 360.
        along_strike_dist = (0.5 - hypo_loc[0]) * plane.length
    # Translate along strike
    hypocentre = centroid.point_at(along_strike_dist,
                                   0.,
                                   along_strike_azimuth)
    # Translate down dip
    horizontal_dist = down_dip_dist * cos(TO_RAD * plane.dip)
    vertical_dist = down_dip_dist * sin(TO_RAD * plane.dip)
    if down_dip_dist >= 0.:
        down_dip_azimuth = (plane.strike + 90.) % 360.
    else:
        down_dip_azimuth = (plane.strike - 90.) % 360.
        down_dip_dist = (0.5 - hypo_loc[1]) * plane.width
        horizontal_dist = down_dip_dist * cos(TO_RAD * plane.dip)

    return hypocentre.point_at(horizontal_dist,
                               vertical_dist,
                               down_dip_azimuth)


def vs30_to_z1pt0_as08(vs30):
    """
    Extracts a depth to 1.0 km/s velocity layer using the relationship
    proposed in Abrahamson & Silva 2008
    :param float vs30:
        Input Vs30 (m/s)
    """
    if vs30 < 180.:
        return np.exp(6.745)
    elif vs30 > 500.:
        return np.exp(5.394 - 4.48 * np.log(vs30 / 500.))
    else:
        return np.exp(6.745 - 1.35 * np.log(vs30 / 180.))


def vs30_to_z1pt0_cy08(vs30):
    """
    Extracts a depth to 1.0 km/s velocity layer using the relationship
    proposed in Chiou & Youngs 2008
    :param float vs30:
        Input Vs30 (m/s)
    """
    return np.exp(28.5 - (3.82 / 8.) * np.log((vs30 ** 8.) + (378.7 ** 8.)))


def z1pt0_to_z2pt5(z1pt0):
    """
    Calculates the depth to 2.5 km/s layer (km /s) using the model presented
    in Campbell & Bozorgnia (2007)
    :param float z1pt0:
        Depth (m) to the 1.0 km/s layer
    :returns:
        Depth (km) to 2.5 km/s layer
    """
    return 0.519 + 3.595 * (z1pt0 / 1000.)

def vs30_to_z1pt0_cy14(vs30, japan=False):
    """
    Returns the estimate depth to the 1.0 km/s velocity layer based on Vs30
    from Chiou & Youngs (2014) California model

    :param numpy.ndarray vs30:
        Input Vs30 values in m/s
    :param bool japan:
        If true returns the Japan model, otherwise the California model
    :returns:
        Z1.0 in m
    """
    if japan:
        c1 = 412. ** 2.
        c2 = 1360.0 ** 2.
        return np.exp((-5.23 / 2.0) * np.log((np.power(vs30,2.) + c1) / (c2 + c1)))
    else:
        c1 = 571 ** 4.
        c2 = 1360.0 ** 4.
        return np.exp((-7.15 / 4.0) * np.log((vs30 ** 4. + c1) / (c2 + c1)))

def vs30_to_z2pt5_cb14(vs30, japan=False):
    """
    Converts vs30 to depth to 2.5 km/s interface using model proposed by
    Campbell & Bozorgnia (2014)

    :param vs30:
        Vs30 values (numpy array or float)

    :param bool japan:
        Use Japan formula (True) or California formula (False)

    :returns:
        Z2.5 in km
    """
    if japan:
        return np.exp(5.359 - 1.102 * np.log(vs30))
    else:
        return np.exp(7.089 - 1.144 * np.log(vs30))


def _setup_site_peripherals(azimuth, origin_point, vs30, z1pt0, z2pt5, strike, 
                            surface):
    """
    For a given configuration determine the site periferal values
    """
    if not z1pt0:
        z1pt0 = vs30_to_z1pt0_cy14(vs30)
    #    z1pt0 = vs30_to_z1pt0_as08(vs30)
    if not z2pt5:
        #z2pt5 = z1pt0_to_z2pt5(z1pt0)
        z2pt5 = vs30_to_z2pt5_cb14(vs30)
    azimuth = (strike + azimuth) % 360.
    origin_location = get_hypocentre_on_planar_surface(surface,
                                                       origin_point)
    origin_location.depth = 0.0
    return azimuth, origin_location, z1pt0, z2pt5


def _rup_to_point(distance, surface, origin, azimuth, distance_type='rjb',
        iter_stop=1E-3, maxiter=1000):
    """
    Place a point at a given distance from a rupture along a specified azimuth
    """
    pt0 = origin
    pt1 = origin.point_at(distance, 0., azimuth)
    #print pt0, pt1
    r_diff = np.inf
    dip = surface.dip
    sin_dip = np.sin(np.radians(dip))
    dist_sin_dip = distance / sin_dip
    #max_surf_dist = surface.width / np.cos(np.radians(dip))
    iterval = 0
    while (np.fabs(r_diff) >= iter_stop) and (iterval <= maxiter):
        pt1mesh = Mesh(np.array([pt1.longitude]),
                       np.array([pt1.latitude]),
                       None)
        if distance_type == 'rjb' or np.fabs(dip - 90.0) < 1.0E-3:
            r_diff =  (distance -
                       surface.get_joyner_boore_distance(pt1mesh)).flatten()
            pt0 = Point(pt1.longitude, pt1.latitude)
            if r_diff > 0.:
                pt1 = pt0.point_at(r_diff, 0., azimuth)
            else:
                pt1 = pt0.point_at(np.fabs(r_diff), 0.,
                                   (azimuth + 180.) % 360.)
        elif distance_type == 'rrup':
            rrup = surface.get_min_distance(pt1mesh).flatten()
            if azimuth >= 0.0 and azimuth <= 180.0:
                # On hanging wall
                r_diff = dist_sin_dip - (rrup / sin_dip)   

            else:
                # On foot wall
                r_diff = distance - rrup 
            pt0 = Point(pt1.longitude, pt1.latitude)
            
            #print azimuth, (azimuth + 180.0) % 360,  rrup, r_diff, np.fabs(r_diff)
            if r_diff > 0.:
                pt1 = pt0.point_at(r_diff, 0., azimuth)
            else:
                pt1 = pt0.point_at(np.fabs(r_diff), 0.,
                                   (azimuth + 180.) % 360.)
            
        else:
            raise ValueError('Distance type must be rrup or rjb!')
        iterval += 1
    return pt1


class PointAtDistance(object):
    """
    Abstract Base class to implement set of methods for rendering a point at
    a given distance from the rupture
    """
    def point_at_distance(self, model, distance, vs30, line_azimuth=90.,
            origin_point=(0.5, 0.), vs30measured=True, z1pt0=None, z2pt5=None,
            backarc=False):
        """
        """
        raise NotImplementedError
 

class PointAtRuptureDistance(PointAtDistance):
    """
    Locate a point at a given Joyner-Boore distance
    """

    def point_at_distance(self, model, distance, vs30, line_azimuth=90., 
            origin_point=(0.5, 0.), vs30measured=True, z1pt0=None, z2pt5=None,
            backarc=False):
        """
        Generates a site given a specified rupture distance from the 
        rupture surface
        """
        azimuth, origin_location, z1pt0, z2pt5 = _setup_site_peripherals(
            line_azimuth, origin_point, vs30, z1pt0, z2pt5, model.strike,
            model.surface)
        selected_point = _rup_to_point(distance,
                                       model.surface,
                                       origin_location,
                                       azimuth,
                                       'rrup')
        target_sites = SiteCollection([Site(selected_point,
                                            vs30,
                                            vs30measured,
                                            z1pt0,
                                            z2pt5,
                                            backarc=backarc)])
        return target_sites


class PointAtJoynerBooreDistance(PointAtDistance):
    """
    Locate a point at a given Joyner-Boore distance
    """
    
    def point_at_distance(self, model, distance, vs30, line_azimuth=90.,
             origin_point=(0.5, 0.),  vs30measured=True, z1pt0=None,
             z2pt5=None, backarc=False):
        """
        Generates a site given a specified rupture distance from the 
        rupture surface
        """
        azimuth, origin_location, z1pt0, z2pt5 = _setup_site_peripherals(
            line_azimuth, origin_point, vs30, z1pt0, z2pt5, model.strike,
            model.surface)
        selected_point = _rup_to_point(distance,
                                       model.surface,
                                       origin_location,
                                       azimuth,
                                       'rjb')
        target_sites = SiteCollection([Site(selected_point,
                                            vs30,
                                            vs30measured,
                                            z1pt0,
                                            z2pt5,
                                            backarc=backarc)])
        return target_sites


class PointAtEpicentralDistance(PointAtDistance):
    """
    Locate at point at a given epicentral distance from a source
    """

    def point_at_distance(self, model, distance, vs30, line_azimuth=90., 
            origin_point=(0.5, 0.), vs30measured=True, z1pt0=None, z2pt5=None,
            backarc=False):
        """
        Generates a point at a given epicentral distance
        """
        azimuth, origin_point, z1pt0, z2pt5 = _setup_site_peripherals(
            line_azimuth, origin_point, vs30, z1pt0, z2pt5, model.strike,
            model.surface)
        return SiteCollection([Site(
            model.hypocentre.point_at(distance, 0., line_azimuth),
            vs30,
            vs30measured,
            z1pt0,
            z2pt5,
            backarc=backarc)])


class PointAtHypocentralDistance(PointAtDistance):
    """
    Locate a point at a given hypocentral distance from a source
    """
    def point_at_distance(self, model, distance, vs30, line_azimuth=90., 
            origin_point=(0.5, 0.), vs30measured=True, z1pt0=None, z2pt5=None,
            backarc=False):
        """
        Generates a point at a given hypocentral distance
        """
        azimuth, origin_point, z1pt0, z2pt5 = _setup_site_peripherals(
            line_azimuth, origin_point, vs30, z1pt0, z2pt5, model.strike,
            model.surface)

        xdist = sqrt(distance ** 2. - model.hypocentre.depth ** 2.)
        return SiteCollection([Site(
            model.hypocentre.point_at(xdist, -model.hypocentre.depth, azimuth),
            vs30,
            vs30measured,
            z1pt0,
            z2pt5,
            backarc=backarc)])


POINT_AT_MAPPING = {'rrup': PointAtRuptureDistance(),
                    'rjb': PointAtJoynerBooreDistance(),
                    'repi': PointAtEpicentralDistance(),
                    'rhypo': PointAtHypocentralDistance()}


class GSIMRupture(object):
    """
    Defines a rupture plane consistent with the properties specified for
    the trellis plotting. Also contains methods for configuring the site
    locations
    """
    def __init__(self, magnitude, dip, aspect, 
                 tectonic_region='Active Shallow Crust' , rake=0., ztor=0., 
                 strike=0., msr=WC1994(), initial_point=DEFAULT_POINT,
                 hypocentre_location=None):
        """
        Instantiate the rupture - requires a minimum of a magnitude, dip
        and aspect ratio
        """
        self.magnitude = magnitude
        self.dip = dip
        self.aspect = aspect
        self.rake = rake
        self.strike = strike
        self.location = initial_point
        self.ztor = ztor
        self.trt = tectonic_region
        self.hypo_loc = hypocentre_location
        # If the top of rupture depth in the initial
        if fabs(self.location.depth - self.ztor) > 1E-9:
            self.location.depth = ztor
        self.msr = msr
        self.area = self.msr.get_median_area(self.magnitude, self.rake)
        self.surface = create_planar_surface(self.location,
                                             self.strike,
                                             self.dip,
                                             self.area,
                                             self.aspect)
        self.hypocentre = get_hypocentre_on_planar_surface(self.surface,
                                                           self.hypo_loc)
        self.rupture = self.get_rupture()
        self.target_sites_config = None
        self.target_sites = None

    def get_rupture(self):
        """
        Returns the rupture as an instance of the
        openquake.hazardlib.source.rupture.Rupture class
        """
        return Rupture(self.magnitude, 
                       self.rake,
                       self.trt, 
                       self.hypocentre,
                       self.surface,
                       PointSource)

    def get_gsim_contexts(self):
        """
        Returns a comprehensive set of GMPE contecxt objects
        """
        assert isinstance(self.rupture, Rupture)
        assert isinstance(self.target_sites, SiteCollection)
        # Distances
        dctx = DistancesContext()
        # Rupture distance
        setattr(dctx, 
                'rrup',
                self.rupture.surface.get_min_distance(self.target_sites.mesh))
        # Rx
        setattr(dctx, 
                'rx',
                self.rupture.surface.get_rx_distance(self.target_sites.mesh))
        # Rjb
        setattr(dctx, 
                'rjb',
                self.rupture.surface.get_joyner_boore_distance(
                    self.target_sites.mesh))
        # Rhypo
        setattr(dctx,
                'rhypo',
                self.rupture.hypocenter.distance_to_mesh(
                    self.target_sites.mesh))
        # Repi
        setattr(dctx, 'repi',
                self.rupture.hypocenter.distance_to_mesh(
                    self.target_sites.mesh, 
                    with_depths=False))
        # Ry0
        setattr(dctx, 'ry0',
                self.rupture.surface.get_ry0_distance(self.target_sites.mesh))
        # Rcdpp - ignored at present
        setattr(dctx, 'rcdpp', None)
        # Azimuth - ignored at present
        setattr(dctx, 'azimuth', None)
        setattr(dctx, 'hanging_wall', None)
        # Rvolc
        setattr(dctx, "rvolc", np.zeros_like(self.target_sites.mesh.lons))
        # Sites
        sctx = SitesContext()
        key_list = ["lons", "lats", "vs30", "vs30measured", "z1pt0", "z2pt5",
                    "backarc"]
        for key in key_list:
            setattr(sctx, key, self.target_sites.array[key])

        # Rupture
        rctx = RuptureContext()
        setattr(rctx, 'mag', self.magnitude)
        setattr(rctx, 'strike', self.strike)
        setattr(rctx, 'dip', self.dip)
        setattr(rctx, 'rake', self.rake)
        setattr(rctx, 'ztor', self.ztor)
        setattr(rctx, 'hypo_depth', self.rupture.hypocenter.depth)
        setattr(rctx, 'hypo_lat', self.rupture.hypocenter.latitude)
        setattr(rctx, 'hypo_lon', self.rupture.hypocenter.longitude)
        setattr(rctx, 'hypo_loc', self.hypo_loc)
        setattr(rctx, 'width', self.rupture.surface.get_width())
        return sctx, rctx, dctx

    def get_target_sites_mesh(self, maximum_distance, spacing, vs30,
                              vs30measured=True, z1pt0=None, z2pt5=None,
                              backarc=False):
        """
        Renders a two dimensional mesh of points over the rupture surface
        """
        # Get bounding box of dilated rupture
        lowx, highx, lowy, highy = self._get_limits_maximum_rjb(
            maximum_distance)
        # Create bounding box lines and then resample at spacing
        ewline = Line([Point(lowx, highy, 0.), Point(highx, highy, 0.)])
        nsline = Line([Point(lowx, highy, 0.), Point(lowx, lowy, 0.)])
        ewline = ewline.resample(spacing)
        nsline = nsline.resample(spacing)
        xvals = np.array([pnt.longitude for pnt in ewline.points])
        yvals = np.array([pnt.latitude for pnt in nsline.points])

        gridx, gridy = np.meshgrid(xvals, yvals)

        numx, numy = np.shape(gridx)
        npts = numx * numy
        gridx = (np.reshape(gridx, npts, 1)).flatten()
        gridy = (np.reshape(gridy, npts, 1)).flatten()
        site_list = []

        if not z1pt0:
            #z1pt0 = vs30_to_z1pt0_as08(vs30)
            z1pt0 = vs30_to_z1pt0_cy14(vs30)

        if not z2pt5:
            #z2pt5 = z1pt0_to_z2pt5(z1pt0)
            z2pt5 = vs30_to_z2pt5_cb14(vs30)

        for iloc in range(0, npts):
            site_list.append(Site(Point(gridx[iloc], gridy[iloc], 0.),
                                  vs30,
                                  vs30measured,
                                  z1pt0,
                                  z2pt5,
                                  backarc=backarc))
        self.target_sites = SiteCollection(site_list)
        self.target_sites_config = {
            "TYPE": "Mesh",
            "RMAX": maximum_distance,
            "SPACING": spacing,
            "VS30": vs30,
            "VS30MEASURED": vs30measured,
            "Z1.0": z1pt0,
            "Z2.5": z2pt5,
            "BACKARC": backarc}
        return self.target_sites


    def get_target_sites_line(self, maximum_distance, spacing, vs30,
            line_azimuth=90., origin_point=(0.5, 0.5), as_log=False, 
            vs30measured=True, z1pt0=None, z2pt5=None, backarc=False):
        """
        Defines the target sites along a line with respect to the rupture
        """
        #input_origin_point = deepcopy(origin_point)
        azimuth, origin_location, z1pt0, z2pt5 = _setup_site_peripherals(
            line_azimuth,
            origin_point,
            vs30,
            z1pt0,
            z2pt5,
            self.strike,
            self.surface)

        self.target_sites = [Site(origin_location,
                                  vs30,
                                  vs30measured,
                                  z1pt0,
                                  z2pt5,
                                  backarc=backarc)]
        spacings = self._define_line_spacing(maximum_distance,
                                             spacing,
                                             as_log)
        for offset in spacings:
            target_loc= origin_location.point_at(offset, 0., azimuth)
            # Get Rupture distance
            temp_mesh = Mesh(np.array(target_loc.longitude),
                             np.array(target_loc.latitude),
                             np.array(target_loc.depth))
            distance = self.surface.get_min_distance(temp_mesh)
            self.target_sites.append(Site(target_loc, 
                                          vs30, 
                                          vs30measured, 
                                          z1pt0,
                                          z2pt5,
                                          backarc=backarc))
        self.target_sites_config = {
            "TYPE": "Line",
            "RMAX": maximum_distance,
            "SPACING": spacing,
            "AZIMUTH": line_azimuth,
            "ORIGIN": origin_point,
            "AS_LOG": as_log,
            "VS30": vs30,
            "VS30MEASURED": vs30measured,
            "Z1.0": z1pt0,
            "Z2.5": z2pt5,
            "BACKARC": backarc}
        self.target_sites = SiteCollection(self.target_sites)
        return self.target_sites

    def _define_line_spacing(self, maximum_distance, spacing, as_log=False):
        """
        The user may wish to define the line spacing in either log or
        linear space
        """
        nvals = int(maximum_distance / spacing) + 1
        if as_log:
            spacings = np.logspace(-3., np.log10(maximum_distance), nvals)
            spacings[0] = 0.0
        else:
            spacings = np.linspace(0.0, maximum_distance, nvals)

        if spacings[-1] < (maximum_distance - 1.0E-7):
            spacings = np.hstack([spacings, maximum_distance])

        return spacings


    def get_target_sites_point(self, distance, distance_type, vs30, 
            line_azimuth=90, origin_point=(0.5, 0.5), vs30measured=True, 
            z1pt0=None, z2pt5=None, backarc=False):
        """
        Returns a single target site at a fixed distance from the source,
        with distance defined according to a specific typology
        :param float distance:
            Distance (km) from the point to the source
        :param str distance_type:
            Type of distance {'rrup', 'rjb', 'repi', 'rhyp'}
        :param float vs30:
            Vs30 (m / s)
        :param float line_azimuth:
            Aziumth of the source-to-site line
        :param tuple origin point:
            Location (along strike, down dip) of the origin of the source-site
            line within the rupture
        :param bool vs30measured:
            Is vs30 measured (True) or inferred (False)
        :param float z1pt0:
            Depth to 1 km/s interface
        :param floar z2pt5:
            Depth to 2.5 km/s interface
        """
        if not distance_type in list(POINT_AT_MAPPING.keys()):
            raise ValueError("Distance type must be one of: Rupture ('rrup'), "
                             "Joyner-Boore ('rjb'), Epicentral ('repi') or "
                             "Hypocentral ('rhyp')")
        #input_origin_point = deepcopy(origin_point)
        azimuth, origin_location, z1pt0, z2pt5 = _setup_site_peripherals(
            line_azimuth,
            origin_point,
            vs30,
            z1pt0,
            z2pt5,
            self.strike,
            self.surface)
        self.target_sites_config = {
            "TYPE": "Point",
            "R": distance,
            "RTYPE": distance_type,
            "AZIMUTH": line_azimuth,
            "ORIGIN": origin_point,
            "VS30": vs30,
            "VS30MEASURED": vs30measured,
            "Z1.0": z1pt0,
            "Z2.5": z2pt5,
            "BACKARC": backarc}

        self.target_sites = POINT_AT_MAPPING[distance_type].point_at_distance(
            self, 
            distance,
            vs30,
            line_azimuth, 
            origin_point, 
            vs30measured, 
            z1pt0, 
            z2pt5,
            backarc=backarc)

    def _get_limits_maximum_rjb(self, maximum_distance):
        """
        Returns the bounding box of a polyon representing the locations of
        maximum distance from the rupture
        """
        top_left = deepcopy(self.surface.top_left)
        top_left.depth = 0.

        top_right = deepcopy(self.surface.top_right)
        top_right.depth = 0.

        bottom_left = deepcopy(self.surface.bottom_left)
        bottom_left.depth = 0.

        bottom_right = deepcopy(self.surface.bottom_right)
        bottom_right.depth = 0.

        surface_projection = Polygon([top_left,
                                      top_right,
                                      bottom_right,
                                      bottom_left])
        dilated_projection = surface_projection.dilate(maximum_distance)
        return (np.min(dilated_projection.lons),
                np.max(dilated_projection.lons),
                np.min(dilated_projection.lats),
                np.max(dilated_projection.lats))


    def filter_hanging_wall(self, filter_type=None):
        """
        Opt to consider only hanging wall or footwall sites 

        """
        if not filter_type:
            # Considers both footwall and hanging wall 
            return self.target_sites
        elif not filter_type in ('HW', 'FW'):
            raise ValueError('Hanging wall filter must be either "HW" or "FW"')
        else:
            pass

        # Gets the Rx distance
        r_x = self.surface.get_rx_distance(self.target_sites.mesh)
        selected_sites = []
        if filter_type == "HW":
            # Only hanging wall
            idx = np.where(r_x >= 0.)[0]
        else:
            # Only footwall
            idx = np.where(r_x < 0.)[0]
        for val in idx:
            selected_sites.append(self.target_sites.sites[val])
        self.target_sites = SiteCollection(selected_sites)
        return self.target_sites

    def _site_collection_to_mesh(self):
        """
        Returns a collection of sites as an instance of the :class:
        `openquake.hazardlib.geo.Mesh`
        """
        if isinstance(self.target_sites, SiteCollection):
            locations = np.array([len(self.target_sites.sites), 3], 
                                   dtype=float)
            for iloc, site in enumerate(self.target_sites.sites):
                locations[iloc, 0] = site.location.longitude
                locations[iloc, 1] = site.location.latitude
                locations[iloc, 2] = site.location.depth
            return Mesh(locations[:, 0], locations[:, 1], locations[:, 2])
        else:
            raise ValueError('Target sites must be an instance of '
                             'openquake.hazardlib.site.SiteCollection')

    def plot_distance_comparisons(self, distance1, distance2, logaxis=False,
        figure_size=(7, 5), filename=None, filetype="png", dpi=300):
        """
        Creates a plot comparing different distance metrics for the 
        specific rupture and target site combination
        """
        xdist = self._calculate_distance(distance1)       
        ydist = self._calculate_distance(distance2)       
        plt.figure(figsize=figure_size)

        if logaxis:
            plt.loglog(xdist, ydist, color='b', marker='o', linestyle='None')
        else:
            plt.plot(xdist, ydist, color='b', marker='o', linestyle='None')
        
        plt.xlabel("%s (km)" % distance1, size='medium')
        plt.ylabel("%s (km)" % distance2, size='medium')
        plt.title('Rupture: M=%6.1f, Dip=%3.0f, Ztor=%4.1f, Aspect=%5.2f'
                   % (self.magnitude, self.dip, self.ztor, self.aspect))
        _save_image(filename, filetype, dpi)
        plt.show()

    def _calculate_distance(self, distance_type):
        """
        Calculates the rupture to target site distances for the present 
        rupture and target site configuration
        """
        if distance_type == 'rrup':
            return self.surface.get_min_distance(self.target_sites.mesh)
        elif distance_type == 'rjb':
            return self.surface.get_joyner_boore_distance(
                self.target_sites.mesh)
        elif distance_type == 'rx':
            return self.surface.get_rx_distance(self.target_sites.mesh)
        elif distance_type == 'rhypo':
            return self.hypocentre.distance_to_mesh(self.target_sites.mesh)
        elif distance_type == 'repi':
            return self.hypocentre.distance_to_mesh(self.target_sites.mesh,
                                                    with_depths=False)
        else:
            raise ValueError('Unsupported Distance Measure: %s' 
                             % distance_type)

    def plot_model_configuration(self, marker_style="o", figure_size=(7, 5), 
            filename=None, filetype="png", dpi=300):
        """
        Produces a 3D plot of the current model configuration
        """
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection='3d')
        # Wireframe rupture surface mesh
        rupture_mesh = self.surface.get_mesh()
        ax.plot_wireframe(rupture_mesh.lons,
                          rupture_mesh.lats, 
                          -rupture_mesh.depths,
                          rstride=1,
                          cstride=1)

        # Scatter the target sites
        ax.scatter(self.target_sites.mesh.lons,
                   self.target_sites.mesh.lats,
                   np.ones_like(self.target_sites.mesh.lons),
                   c='r',
                   marker=marker_style)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth (km)')
        ax.set_zlim(-np.ceil(np.max(rupture_mesh.depths)), 0.0)
        _save_image(filename, filetype, dpi)
        plt.show()
        
