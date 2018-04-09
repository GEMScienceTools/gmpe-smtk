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
Example of conditional simulation of ground motion fields with the OQ-hazardlib
"""

import numpy as np
from collections import OrderedDict
from shapely import wkt
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.geo.geodetic import geodetic_distance
from openquake.hazardlib.correlation import jbcorrelation
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.gsim.base import ContextMaker
from openquake.hazardlib.sourceconverter import RuptureConverter
from openquake.hazardlib import nrml
from smtk.residuals.gmpe_residuals import Residuals


DEFAULT_CORRELATION = jbcorrelation
GSIM_LIST = get_available_gsims()


def build_planar_surface(geometry):
    """
    Builds the planar rupture surface from the openquake.nrmllib.models
    instance
    """
    # Read geometry from wkt
    geom = wkt.loads(geometry.wkt)
    top_left = Point(geom.xy[0][0],
                     geom.xy[1][0],
                     geometry.upper_seismo_depth)
    top_right = Point(geom.xy[0][1],
                      geom.xy[1][1],
                      geometry.upper_seismo_depth)
    strike = top_left.azimuth(top_right)
    dip_dir = (strike + 90.) % 360.
    depth_diff = geometry.lower_seismo_depth - geometry.upper_seismo_depth
    bottom_right = top_right.point_at(
        depth_diff / np.tan(geometry.dip * (np.pi / 180.)),
        depth_diff,
        dip_dir)
    bottom_left = top_left.point_at(
        depth_diff / np.tan(geometry.dip * (np.pi / 180.)),
        depth_diff,
        dip_dir)
    return PlanarSurface(1.0,
                         strike,
                         geometry.dip,
                         top_left,
                         top_right,
                         bottom_right,
                         bottom_left)


def build_rupture_from_file(rupture_file, simple_mesh_spacing=1.0,
                            complex_mesh_spacing=1.0):
    """
    Parses a rupture from the OpenQuake nrml 4.0 format and parses it to
    an instance of :class: openquake.hazardlib.source.rupture.Rupture
    """
    [rup_node] = nrml.read(rupture_file)
    conv = RuptureConverter(simple_mesh_spacing,
                            complex_mesh_spacing)
    return conv.convert_node(rup_node)


def get_regular_site_collection(limits, vs30, z1pt0=100.0, z2pt5=1.0):
    """
    Gets a collection of sites from a regularly spaced grid of points
    :param list limits:
        Limits of mesh as [lower_long, upper_long, long_spc, lower_lat,
                           upper_lat, lat_spc]
    :param float vs30:
        Vs30 values
    :param float z1pt0:
        Depth (m) to the 1 km/s interface
    :param float z2pt5:
        Depth (km) to the 2.5 km/s interface
    """
    xgrd = np.arange(limits[0], limits[1] + limits[2], limits[2])
    ygrd = np.arange(limits[3], limits[4] + limits[5], limits[5])
    gx, gy = np.meshgrid(xgrd, ygrd)
    nx, ny = np.shape(gx)
    ngp = nx * ny
    gx = np.reshape(gx, [ngp, 1]).flatten()
    gy = np.reshape(gy, [ngp, 1]).flatten()
    depths = np.zeros(ngp, dtype=float)
    return SiteCollection([
        Site(Point(gx[i], gy[i]), vs30, True, z1pt0, z2pt5, i)
        for i in range(0, ngp)])



def conditional_simulation(known_sites, residuals, unknown_sites, imt, nsim,
    correlation_model=DEFAULT_CORRELATION):
    """
    Generates the residuals for a set of sites, conditioned upon the
    known residuals at a set of observation locations
    :param known_sites:
        Locations of known sites as instance of :class: 
        openquake.hazardlib.sites.SiteCollection
    :param dict residuals:
        Dictionary of residuals for specifc GMPE and IMT
    :param unknown_sites:
        Locations of unknown sites as instance of :class:
        `openquake.hazardlib.sites.SiteCollection`
    :param imt:
        Intensity measure type
    :psram int nsim:
        Number of simulations
    :param correlation_model:
        Chosen correlation model, i.e. jbcorrelation

    """
    # Get site to site distances for known
    imt = from_string(imt)
    # Make sure that sites are at the surface (to check!)
    known_sites.depths = np.zeros_like(known_sites.depths)
    unknown_sites.depths = np.zeros_like(unknown_sites.depths)
    cov_kk = correlation_model(known_sites, imt).I
    cov_uu = correlation_model(unknown_sites, imt)
    d_k_uk = np.zeros([len(known_sites), len(unknown_sites)],
                      dtype=float)
    for iloc in range(len(known_sites)):
        d_k_uk[iloc, :] = geodetic_distance(known_sites.array["lons"][iloc],
                                            known_sites.array["lats"][iloc],
                                            unknown_sites.array["lons"],
                                            unknown_sites.array["lats"])
    cov_ku = correlation_model(d_k_uk, imt)
    mu = cov_ku.T * cov_kk * np.matrix(residuals).T
    stddev = cov_uu - (cov_ku.T * cov_kk * cov_ku)
    unknown_residuals = np.matrix(np.random.normal(
        0., 1., [len(unknown_sites), nsim]))
    lower_matrix = np.linalg.cholesky(stddev)
    output_residuals = np.zeros_like(unknown_residuals)
    for iloc in range(0, nsim):
        output_residuals[:, iloc] = mu + \
            lower_matrix * unknown_residuals[:, iloc]
    return output_residuals



def get_conditional_gmfs(database, rupture, sites, gsims, imts,
        number_simulations, truncation_level,
        correlation_model=DEFAULT_CORRELATION):
    """
    Get a set of random fields conditioned on a set of observations
    :param database:
        Ground motion records for the event as instance of :class:
        smtk.sm_database.GroundMotionDatabase
    :param rupture:
        Event rupture as instance of :class:
        openquake.hazardlib.source.rupture.Rupture
    :param sites:
        Target sites as instance of :class:
        openquake.hazardlib.site.SiteCollection
    :param list gsims:
        List of GMPEs required
    :param list imts:
        List of intensity measures required
    :param int number_simulations:
        Number of simulated fields required
    :param float truncation_level:
        Ground motion truncation level
    """

    # Get known sites mesh
    known_sites = database.get_site_collection()

    # Get Observed Residuals
    residuals = Residuals(gsims, imts)
    residuals.get_residuals(database)
    imt_dict = OrderedDict([
        (imtx,  np.zeros([len(sites.lons), number_simulations]))
        for imtx in imts])
    gmfs = OrderedDict([(gmpe, imt_dict) for gmpe in gsims])
    gmpe_list = [GSIM_LIST[gmpe]() for gmpe in gsims]
    cmaker = ContextMaker(gmpe_list)
    sctx, rctx, dctx = cmaker.make_contexts(sites, rupture)
    for gsim in gmpe_list:
        gmpe = gsim.__class__.__name__
        #gsim = GSIM_LIST[gmpe]()
        #sctx, rctx, dctx = gsim.make_contexts(sites, rupture)
        for imtx in imts:
            if truncation_level == 0:
                gmfs[gmpe][imtx], _ = gsim.get_mean_and_stddevs(sctx, rctx,
                    dctx, from_string(imtx), stddev_types=[])
                continue
            if "Intra event" in gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                epsilon = conditional_simulation(
                    known_sites,
                    residuals.residuals[gmpe][imtx]["Intra event"],
                    sites,
                    imtx,
                    number_simulations,
                    correlation_model)
                tau = np.unique(residuals.residuals[gmpe][imtx]["Inter event"])
                mean, [stddev_inter, stddev_intra] = gsim.get_mean_and_stddevs(
                    sctx,
                    rctx,
                    dctx, 
                    from_string(imtx), 
                    ["Inter event", "Intra event"])
                for iloc in range(0, number_simulations):
                    gmfs[gmpe][imtx][:, iloc] = np.exp(
                        mean +
                        (tau * stddev_inter) +
                        (epsilon[:, iloc].A1 * stddev_intra))
                        
            else:
                epsilon = conditional_simulation(
                    known_sites,
                    residuals.residuals[gmpe][imtx]["Total"],
                    sites,
                    imtx,
                    number_simulations,
                    correlation_model)
                tau = None
                mean, [stddev_total] = gsim.get_mean_and_stddevs(
                    sctx,
                    rctx,
                    dctx,
                    from_string(imtx),
                    ["Total"])
                for iloc in range(0, number_simulations):
                    gmfs[gmpe][imtx][:, iloc] = np.exp(
                        mean +
                        (epsilon[:, iloc].A1 * stddev_total.flatten()))
    return gmfs
