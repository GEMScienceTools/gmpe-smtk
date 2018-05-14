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
Basic Pseudo-database built on top of hdf5 for a set of processed strong
motion records
"""
import numpy as np
import json
from datetime import datetime
from collections import OrderedDict, namedtuple
from openquake.hazardlib.gsim.base import (SitesContext, DistancesContext,
                                          RuptureContext)
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.scalerel import PeerMSR
from smtk.trellis.configure import vs30_to_z1pt0_as08, z1pt0_to_z2pt5
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14
import smtk.sm_utils as utils

DEFAULT_MSR = PeerMSR()


class Magnitude(object):
    """
    Class to hold magnitude attributes
    :param float value:
        The magnitude value
    :param str mtype:
        Magnitude Type
    :param float sigma:
        The magnitude uncertainty (standard deviation)
    """
    def __init__(self, value, mtype, sigma=None, source=""):
        """
        Instantiate the class
        """
        self.value = value
        self.mtype = mtype
        self.sigma = sigma
        self.source = source

    @classmethod
    def from_dict(cls, d):
        """
        Instantiates the object from a dictionary
        """
        return cls(d["value"], d["mtype"], d["sigma"], d["source"])

    def to_dict(self):
        return self.__dict__

    def __eq__(self, m):
        """
        """
        same_value = np.fabs(self.value - m.value) < 1.0E-3
        if self.sigma and m.sigma:
            same_sigma = np.fabs(self.sigma - m.sigma) < 1.0E-7
        else:
            same_sigma = self.sigma == m.sigma
        return same_value and same_sigma and (self.mtype == m.mtype) and\
            (self.source == m.source)


class Rupture(object):
    """
    Class to hold rupture attributes
    :param str id:
        Rupture (earthquake) ID
    :param str name:
        Event Name
    :param magnitude:
        Earthquake magnitude as instance of Magnitude class
    :param float length:
        Total rupture length (km)
    :param float width:
        Total rupture width (km)
    :param float depth:
        Depth to the top of rupture (km)
    :param float area:
        Rupture area in km^2
    :param surface:
        Rupture surface as instance of :class:
        openquake.hazardlib.geo.surface.base.BaseRuptureSurface
    :param tuple hypo_loc:
        Hypocentral location within rupture surface as a fraction of
        (along-strike length, down-dip width)
    """
    def __init__(self, eq_id, eq_name, magnitude, length, width, depth,
                 hypocentre=None, area=None, surface=None, hypo_loc=None):
        """
        Instantiate
        """
        self.id = eq_id
        self.name = eq_name
        self.magnitude = magnitude
        self.length = length
        self.width = width
        self.area = area
        self.area = self.get_area()
        self.depth = depth
        self.surface = surface
        self.hypocentre = hypocentre
        self.hypo_loc = hypo_loc
        self.aspect = None
        self.aspect = self.get_aspect()
        # QuakeML parameters
        self.max_displacement = None  # in m
        self.mean_displacement = None  # in m
        self.moment_release_top_5km = None
        self.rise_time = None
        self.velocity = None
        self.shallow_asperity = False
        self.stress_drop = None
        self.vr_to_vs = None

    def get_area(self):
        """
        Returns the area of the rupture
        """
        if self.area:
            return self.area
        if self.length and self.width:
            return self.length * self.width
        else:
            return None

    def get_aspect(self):
        """
        Returns the aspect ratio
        """
        if self.aspect:
            # Trivial case
            return self.aspect
        if self.length and self.width:
            # If length and width both specified
            return self.length / self.width
        if self.length and self.area:
            # If length and area specified
            self.width = self.area / self.length
            return self.length / self.width
        if self.width and self.area:
            # If width and area specified
            self.length = self.area / self.width
            return self.length / self.width
        # out of options - returning None
        return None

    def to_dict(self):
        """
        """
        output = OrderedDict([])
        for key in self.__dict__:
            if key == "magnitude" and self.magnitude is not None:
                # Parse magnitude object to dictionary
                output[key] = self.magnitude.__dict__
            elif key == "hypocentre" and self.hypocentre is not None:
                output[key] = [self.hypocentre.longitude,
                               self.hypocentre.latitude,
                               self.hypocentre.depth]
            elif key == "surface" and self.surface is not None:
                output[key] = utils.surface_to_dict[
                    self.surface.__class__.__name__](self.surface)
            else:
                output[key] = getattr(self, key)
        return output

    @classmethod
    def from_dict(cls, data, mesh_spacing=1.):
        """
        Creates an instance of the class from a json load
        """
        rup = cls(data["id"], data["name"],
                  Magnitude.from_dict(data["magnitude"]),
                  data["length"], data["width"], data["depth"])
        for key in data:
            if key in ["id", "name", "magnitude", "length", "width", "depth"]:
                continue
            elif key == "surface" and data["surface"] is not None:
                rup.surface = utils.surfaces_from_dict(data[key]["type"],
                                                       mesh_spacing)
            else:
                setattr(rup, key, data[key])
        return rup


class GCMTNodalPlanes(object):
    """
    Class to represent the nodal plane distribution of the tensor
    Each nodal plane is represented as a dictionary of the form:
    {'strike':, 'dip':, 'rake':}
    :param dict nodal_plane_1:
        First nodal plane
    :param dict nodal_plane_2:
        Second nodal plane
    """
    def __init__(self):
        """
        Instantiate with empty nodal planes
        """
        self.nodal_plane_1 = None
        self.nodal_plane_2 = None

    def to_dict(self):
        """
        Returns a dictionary
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """
        """
        nps = cls()
        for key in data:
            setattr(nps, key, data[key])
        return nps

class GCMTPrincipalAxes(object):
    """
    Class to represent the eigensystem of the tensor in terms of  T-, B- and P-
    plunge and azimuth
    #_axis = {'eigenvalue':, 'azimuth':, 'plunge':}
    :param dict t_axis:
        The eigensystem of the T-axis
    :param dict b_axis:
        The eigensystem of the B-axis
    :param dict p_axis:
        The eigensystem of the P-axis
    """
    def __init__(self):
        """
        Instantiate
        """
        self.t_axis = None
        self.b_axis = None
        self.p_axis = None

    def to_dict(self):
        """
        Returns a dictionary
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """
        """
        pas = cls()
        for key in data:
            setattr(pas, key, data[key])
        return pas


MECHANISM_TYPE = {"Normal": -90.0,
                  "Strike-Slip": 0.0,
                  "Reverse": 90.0,
                  "Oblique": 0.0,
                  "Unknown": 0.0,
                  "N": -90.0, # Flatfile conventions
                  "S": 0.0,
                  "R": 90.0,
                  "U": 0.0,
                  "NF": -90., # ESM flatfile conventions
                  "SS": 0.,
                  "TF": 90.,
                  "NS": -45., # Normal with strike-slip component
                  "TS": 45., # Reverse with strike-slip component
                  "O": 0.0
                  }


DIP_TYPE = {"Normal": 60.0,
            "Strike-Slip": 90.0,
            "Reverse": 35.0,
            "Oblique": 60.0,
            "Unknown": 90.0,
            "N": 60.0, # Flatfile conventions
            "S": 90.0,
            "R": 35.0,
            "U": 90.0,
            "NF": 60., # ESM flatfile conventions
            "SS": 90.,
            "TF": 35.,
            "NS": 70., # Normal with strike-slip component
            "TS": 45., # Reverse with strike-slip component
            "O": 90.0
            }


class FocalMechanism(object):
    """
    Class to hold the full focal mechanism attribute set
    :param str eq_id:
        Identifier of the earthquake
    :param str name:
        Focal mechanism name
    :param nodal_planes:
        Nodal planes as instance of :class: GCMTNodalPlane
    :param eigenvalues:
        Eigenvalue decomposition as instance of :class: GCMTPrincipalAxes
    :param numpy.ndarray tensor:
        (3, 3) Moment Tensor
    :param str mechanism_type:
        Qualitative description of mechanism
    """
    def __init__(self, eq_id, name, nodal_planes, eigenvalues,
                 moment_tensor=None, mechanism_type=None):
        """
        Instantiate
        """
        self.id = eq_id
        self.name = name
        self.nodal_planes = nodal_planes
        self.eigenvalues = eigenvalues
        self.scalar_moment = None
        self.tensor = moment_tensor
        self.mechanism_type = mechanism_type

    def get_rake_from_mechanism_type(self):
        """
        Returns an idealised "rake" based on a qualitative description of the
        style of faulting
        """
        if self.mechanism_type in MECHANISM_TYPE:
            return MECHANISM_TYPE[self.mechanism_type]
        else:
            return 0.0

    def to_dict(self):
        """
        """
        output = OrderedDict([])
        for key in self.__dict__:
            if key in ("nodal_planes", "eigenvalues") and getattr(self, key)\
                is not None:
                output[key] = getattr(self, key).__dict__
            elif key == "tensor":
                output[key] = self._moment_tensor_to_list()
            else:
                output[key] = getattr(self, key)
        return output

    @classmethod
    def from_dict(cls, data):
        """
        """
        keys = list(data)
        if "nodal_planes" in keys and data["nodal_planes"]:
            nps = GCMTNodalPlanes.from_dict(data["nodal_planes"])
        else:
            nps = None
        if "eigenvalues" in keys and data["eigenvalues"]:
            eigs =  GCMTPrincipalAxes.from_dict(data[key])
        else:
            eigs = None
        if "tensor" in keys and data["tensor"]:
            tensor = np.array(data["tensor"])
        else:
            tensor = None

        foc_mech = cls(data["id"], data["name"], nps, eigs, tensor)
        if "mechanism_type" in data:
            setattr(foc_mech, "mechanism_type", data["mechanism_type"])
        return foc_mech

    def _moment_tensor_to_list(self):
        """
        """
        if self.tensor is None:
            return None
        else:
            return self.tensor.to_list() 


class Earthquake(object):
    """
    Class to hold earthquake event related information
    :param str id:
        Earthquake ID
    :param str name:
        Earthquake name
    :param datetime:
        Earthquake date and time as instance of :class: datetime.datetime
    :param float longitude:
        Earthquake hypocentre longitude
    :param float latitude:
        Earthquake hypocentre latitude
    :param float depth:
        Earthquake hypocentre depth (km)
    :param magnitude:
        Primary magnitude as instance of :class: Magnitude
    :param magnitude_list:
        Magnitude solutions for the earthquake as list of instances of the
        :class: Magntiude
    :param mechanism:
        Focal mechanism as instance of the :class: FocalMechanism
    :param rupture:
        Earthquake rupture as instance of the :class: Rupture
    """
    def __init__(self, eq_id, name, date_time, longitude, latitude, depth,
                 magnitude, focal_mechanism=None, eq_country=None,
                 tectonic_region=None):
        """
        Instantiate
        """
        self.id = eq_id
        assert isinstance(date_time, datetime)
        self.datetime = date_time
        self.name = name
        self.country = eq_country
        self.longitude = longitude
        self.latitude = latitude
        self.depth = depth
        self.magnitude = magnitude
        self.magnitude_list = []
        self.mechanism = focal_mechanism
        self.rupture = None
        self.tectonic_region = tectonic_region

    def to_dict(self):
        """
        Parses the information to a json compatible dictionary
        """
        output = OrderedDict([])
        for key in self.__dict__:
            if key == "datetime":
                output[key] = str(self.datetime)
            elif key == "magnitude":
                output[key] = self.magnitude.__dict__
            elif key == "magnitude_list":
                output[key] = [mag.__dict__ for mag in self.magnitude_list]
            elif key in ("mechanism", "rupture") and\
                getattr(self, key) is not None:
                output[key] = getattr(self, key).to_dict()
            else:
                output[key] = getattr(self, key)
        return output

    @classmethod
    def from_dict(cls, data):
        """
        Loads the class from a json compatible dictionary
        """
        reqs = ["id", "name", "datetime", "longitude", "latitude", "depth",
                "magnitude"]
        eqk = cls(data["id"], data["name"],
                  datetime.strptime(data["datetime"], "%Y-%m-%d %H:%M:%S"),
                  data["longitude"],
                  data["latitude"],
                  data["depth"],
                  Magnitude.from_dict(data["magnitude"]))
        
        for key in data:
            if key in reqs:
                continue
            elif key == "magnitude_list" and len(data[key]):
                setattr(eqk, key, [Magnitude.from_dict(mag)
                                   for mag in data[key]])
            elif key == "mechanism" and data[key]:
                setattr(eqk, key, FocalMechanism.from_dict(data[key]))
            elif key == "rupture" and data[key]:
                setattr(eqk, key, Rupture.from_dict(data[key]))
            else:
                setattr(eqk, key, data[key])
        return eqk


class RecordDistance(object):
    """
    Class to hold source to site distance information
    :param float repi:
        Epicentral distance (km)
    :param float rhypo:
        Hypocentral distance (km)
    :param float rjb:
        Joyner-Boore distance (km)
    :param float rrup:
        Rupture distance (km)
    :param float r_x:
        Cross-track distance from site to up-dip projection of fault plane
        to surface
    :param float ry0:
        Along-track distance from site to surface projection of fault plane
    :param float azimuth:
        Source to site azimuth (degrees)
    :param bool hanging_wall:
        True if site on hanging wall, False otherwise
    :param bool flag:
        Distance flagged according to SigmaDatabase definition
    :param float rcdpp:
        Direct point parameter for directivity effect centered on the site-
        and earthquake-specific
        average DPP used
    """
    def __init__(self, repi, rhypo, rjb=None, rrup=None, r_x=None, ry0=None,
                 flag=None, azimuth=None, rcdpp=None, rvolc=None):
        """
        Instantiates class
        """
        self.repi = repi
        self.rhypo = rhypo
        self.rjb = rjb
        self.rrup = rrup
        self.r_x = r_x
        self.ry0 = ry0
        self.azimuth = None
        self.hanging_wall = None
        self.flag = flag
        self.rcdpp = rcdpp
        self.rvolc = rvolc
        # QuakeML parameters
        self.pre_event_length = None
        self.post_event_length = None

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """
        """
        d = cls(data["repi"], data["rhypo"])
        for key in data:
            setattr(d, key, data[key])
        return d

# Eurocode 8 Site Class Vs30 boundaries
EC8_VS30_BOUNDARIES = {
    "A": (800.0, np.inf),
    "B": (360.0, 800.0),
    "C": (180.0, 360.0),
    "D": (100.0, 180.0),
    "S1": (-np.inf, 100)}

# Eurocode 8 Site Class NSPT boundaries
EC8_NSPT_BOUNDARIES = {
    "B": (50.0, np.inf),
    "C": (15.0, 50.0),
    "D": (-np.inf, 15.0)}

# NEHRP Site Class Vs30 boundaries
NEHRP_VS30_BOUNDARIES = {
    "A": (1500.0, np.inf),
    "B": (760.0, 1500.0),
    "C": (360.0, 760.0),
    "D": (180.0, 360.0),
    "E": (-np.inf, 180.0)}

# NEHRP Site Class NSPT boundaries
NEHRP_NSPT_BOUNDARIES = {
    "C": (50.0, np.inf),
    "D": (15.0, 50.0),
    "E": (-np.inf, 15.0)}

class RecordSite(object):
    """
    Class to hold attributes belonging to the site
    :param str site_id:
        Site identifier
    :param str site_code:
        Network site code
    :param site_name:
        Network site name
    :param float longitude:
        Site longitude
    :param float latitude:
        Site latitude
    :param float altitude:
        Site elevation (m)
    :param site_class:
        Qualitative description of site class ("Rock", "Stiff Soil" etc.)
    :param float vs30:
        30-m average shear wave velocity (m/s)
    :param str vs30_measured:
        Vs30 is "measured" or "Inferred"
    :param str vs30_measured_type:
        Method for measuring Vs30
    :param float vs30_uncertainty:
        Standard error of Vs30
    :param float nspt:
        Number of blows of standard penetration test
    :param str nehrp:
        NEHRP Site Class
    :param str ec8:
        Eurocode 8 Site Class
    :param str building_structure:
        Description of structure hosting the instrument
    :param int number_floors:
        Number of floors of structure hosting the instrument
    :param int floor:
        Floor number for location of instrument
    :param str instrument_type:
        Description of instrument type
    :param str digitiser:
        Description of digitiser
    :param str network_code:
        Code of strong motion recording network
    :param str country:
        Country of site
    :param float z1pt0:
        Depth (m) to 1.0 km/s shear-wave velocity interface
    :param float z1pt5:
        Depth (m) to 1.5 km/s shear-wave velocity interface
    :param float z2pt5:
        Depth (km) to 2.5 km/s shear-wave velocity interface
    :param book backarc:
        True if site is in subduction backarc, False otherwise

    """
    def __init__(self, site_id, site_code, site_name, longitude, latitude,
                 altitude, vs30=None, vs30_measured=None, network_code=None,
                 country=None, site_class=None, backarc=False):
        """

        """
        self.id = site_id
        self.name = site_name
        self.code = site_code
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.site_class = site_class
        self.vs30 = vs30
        self.vs30_measured = vs30_measured
        self.vs30_measured_type = None
        self.vs30_uncertainty = None
        self.nspt = None
        self.nehrp = None
        self.ec8 = None
        self.building_structure = None
        self.number_floors = None
        self.floor = None
        self.instrument_type = None
        self.digitiser = None
        self.network_code = network_code
        self.sensor_depth = None
        self.country = country
        self.z1pt0 = None
        self.z1pt5 = None
        self.z2pt5 = None
        self.backarc = backarc
        self.morphology = None
        self.slope = None

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """
        """
        reqs = ["id", "code", "name", "longitude", "latitude", "altitude"]
        site = cls(*[data[req] for req in reqs])
        for key in data:
            if key not in reqs:
                setattr(site, key, data[key])
        return site

    def to_openquake_site(self, missing_vs30=None):
        """
        Returns the site as an instance of the :class:
        openquake.hazardlib.site.Site
        """
        if self.vs30:
            vs30 = self.vs30
            vs30_measured = self.vs30_measured
        else:
            vs30 = missing_vs30
            vs30_measured = False
        
        if self.z1pt0:
            z1pt0 = self.z1pt0
        else:
            z1pt0 = vs30_to_z1pt0_as08(vs30)

        if self.z2pt5:
            z2pt5 = self.z2pt5
        else:
            z2pt5 = z1pt0_to_z2pt5(z1pt0)
        
        location = Point(self.longitude,
                         self.latitude,
                         -self.altitude / 1000.) # Elevation from m to km
        oq_site = Site(location,
                       vs30,
                       vs30_measured,
                       z1pt0,
                       z2pt5,
                       backarc=self.backarc)
        setattr(oq_site, "id", self.id)
        return oq_site

    def get_ec8_class(self):
        """
        Returns the EC8 class associated with a site given a Vs30
        """
        if self.ec8:
            return self.ec8
        if self.vs30:
            for key in EC8_VS30_BOUNDARIES:
                in_group = (self.vs30 >= EC8_VS30_BOUNDARIES[key][0]) and\
                    (self.vs30 < EC8_VS30_BOUNDARIES[key][1])
                if in_group:
                    self.ec8 = key
                    return self.ec8
        elif self.nspt:
            # Check to see if a site class can be determined from NSPT
            for key in EC8_NSPT_BOUNDARIES:
                in_group = (self.nspt >= EC8_NSPT_BOUNDARIES[key][0]) and\
                    (self.nspt < EC8_NSPT_BOUNDARIES[key][1])
                if in_group:
                    self.ec8 = key
                    return self.ec8
        else:
            print("Cannot determine EC8 site class - no Vs30 or NSPT measures!")
        return None
    
    def get_nehrp_class(self):
        """
        Returns the NEHRP class associated with a site given a Vs30 or NSPT
        """
        if self.nehrp:
            return self.nehrp
        if self.vs30:
            for key in NEHRP_VS30_BOUNDARIES:
                in_group = (self.vs30 >= NEHRP_VS30_BOUNDARIES[key][0]) and\
                    (self.vs30 < NEHRP_VS30_BOUNDARIES[key][1])
                if in_group:
                    self.nehrp = key
                    return self.nehrp
        elif self.nspt:
            # Check to see if a site class can be determined from NSPT
            for key in NEHRP_NSPT_BOUNDARIES:
                in_group = (self.nspt >= NEHRP_NSPT_BOUNDARIES[key][0]) and\
                    (self.nspt < NEHRP_NSPT_BOUNDARIES[key][1])
                if in_group:
                    self.nehrp = key
                    return self.nehrp
        else:
            print("Cannot determine NEHRP site class - no Vs30 or NSPT measures!")
        return None


    def vs30_from_ec8(self):
        """
        Returns an approximation of Vs30 given an EC8 site class (e.g. for the case
        when Vs30 is not measured but the site class is given).
        """
        if self.ec8 == 'A':
            return 900
        if self.ec8 == 'B':
            return 580
        if self.ec8 == 'C':
            return 220
        if self.ec8 == 'D':
            return 100
        if self.ec8 == 'E':
            return 100
        else:
            print("Cannot determine Vs30 from EC8 site class")

Filter = {'Type': None,
          'Order': None,
          'Passes': None,
          'Low-Cut': None,
          'High-Cut': None}

Baseline = {'Type': None,
            'Start': None,
            'End': None}

ims_dict = {'PGA': None,
            'PGV': None,
            'PGD': None,
            'CAV': None,
            'Ia': None,
            'CAV5': None,
            'arms': None,
            'd5_95': None,
            'd5_75': None}


class Component(object):
    """
    Contains the metadata relating to waveform of the record
    :param str id:
        Waveform unique identifier
    :param orientation:
        Orientation of record as either azimuth (degrees, float) or string
    :param dict ims:
        Intensity Measures of component
    :param float longest_period:
        Longest usable period (s)
    :param dict waveform_filter:
        Waveform filter properties as dictionary
    :param dict baseline:
        Baseline correction metadata
    :param str units:
        Units of record
        
    """
    def __init__(self, waveform_id, orientation, ims=None, longest_period=None,
                 waveform_filter=None, baseline=None, units=None):
        """
        Instantiate
        """
        self.id = waveform_id
        self.orientation = orientation
        self.lup = longest_period
        self.sup = None
        self.filter = waveform_filter
        self.baseline = baseline
        self.ims = ims
        self.units = units  # Equivalent to gain unit
        self.late_trigger = None
        # QuakeML compatible parameters (mostly unused)
        self.start_time = None
        self.duration = None
        self.resample_rate_denominator = None
        self.resample_rate_numerator = None
        self.owner = None
        self.creation_info = None

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """
        """
        comp = cls(data["id"], data["orientation"])
        for key in data:
            if not key in ["id", "orientation"]:
                setattr(comp, key, data[key])
        return comp


class GroundMotionRecord(object):
    """
    Class containing the full representation of the strong motion record
    :param str id:
        Ground motion record unique identifier
    :param str time_series_file:
        Path to time series file
    :param str spectra_file:
        Path to spectra file
    :param event:
        Earthquake event representation as :class: Earthquake
    :param distance:
        Distances representation as :class: RecordDistance
    :param site:
        Site representation as :class: RecordSite
    :param xrecord:
        x-component of record as instance of :class: Component
    :param yrecord:
        y-component of record as instance of :class: Component
    :param vertical:
         vertical component of record as instance of :class: Component
    :param float average_lup:
        Longest usable period of record-pair
    :param float average_sup:
        Shortest usable period of record-pair
    :param dict ims:
        Intensity measure of record
    :param directivity:
        ?
    :param str datafile:
        Data file for strong motion record
    """
    def __init__(self, gm_id, time_series_file, event, distance, record_site,
                 x_comp, y_comp, vertical=None, ims=None, longest_period=None,
                 shortest_period=None, spectra_file=None):
        """
        """
        self.id = gm_id
        self.time_series_file = time_series_file
        self.spectra_file = spectra_file
        assert isinstance(event, Earthquake)
        self.event = event
        assert isinstance(distance, RecordDistance)
        self.distance = distance
        assert isinstance(record_site, RecordSite)
        self.site = record_site
        assert isinstance(x_comp, Component) and isinstance(y_comp, Component)
        self.xrecord = x_comp
        self.yrecord = y_comp
        if vertical:
            assert isinstance(vertical, Component)
        self.vertical = vertical
        self.average_lup = longest_period
        self.average_sup = shortest_period
        self.ims = ims
        self.directivity = None
        self.datafile = None
        self.misc = None

    def to_dict(self):
        """
        """
        output = OrderedDict([])
        for key in self.__dict__:
            if key in ("event", "distance", "site", "xrecord", "yrecord"):
                output[key] = getattr(self, key).to_dict()
            elif key == "vertical" and self.vertical:
                output[key] = self.vertical.to_dict()
            else:
                output[key] = getattr(self, key)
        return output

    @classmethod
    def from_dict(cls, data):
        """
        """
        reqs = ["id", "time_series_file", "event", "distance", "site",
               "xrecord", "yrecord", "vertical"]
        evnt = cls(data["id"],
                   data["time_series_file"],
                   Earthquake.from_dict(data["event"]),
                   RecordDistance.from_dict(data["distance"]),
                   RecordSite.from_dict(data["site"]),
                   Component.from_dict(data["xrecord"]),
                   Component.from_dict(data["yrecord"]))

        if "vertical" in data and data["vertical"]:
            evnt.vertical = Component.from_dict(data["vertical"])

        for key in data:
            if key in reqs:
                continue
            elif not data[key]:
                setattr(evnt, key, None)
            else:
                setattr(evnt, key, data[key])
        return evnt
            

    def get_azimuth(self):
        """
        If the azimuth is missing, returns the epicentre to station azimuth
        """
        if self.distance.azimuth:
            return self.distance.azimuth
        else:
            self.distance.azimuth = geodetic.azimuth(
                self.event.longitude,
                self.event.latitude,
                self.site.longitude,
                self.site.latitude)
        return self.distance.azimuth


class GroundMotionDatabase(object):
    """
    Class to represent a database of strong motions
    :param str id:
        Database identifier
    :param str name:
        Database name
    :param str directory:
        Path to database directory
    :param list records:
        Strong motion data as list of :class: GroundMotionRecord
    :param list site_ids:
        List of site ids
    """
    def __init__(self, db_id, db_name, db_directory=None, records=[],
                 site_ids=[]):
        """
        """
        self.id = db_id
        self.name = db_name
        self.directory = db_directory
        #self.records = []
        #print(records)
        self.records = [rec for rec in records]
        self.site_ids = site_ids

    def to_json(self):
        """
        Exports the database to json
        """
        json_dict = {"id": self.id,
                     "name": self.name,
                     "directory": self.directory,
                     "records": []}
        for rec in self.records:
            json_dict["records"].append(rec.to_dict())
        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, filename):
        """
        """
        with open(filename, "r") as f:
            raw = json.load(f)
        gmdb = cls(raw["id"], raw["name"], raw["directory"])
        for record in raw["records"]:
            gmdb.records.append(GroundMotionRecord.from_dict(record))
        gmdb.site_ids = [rec.site.id for rec in gmdb.records]
        return gmdb
        

    def number_records(self):
        """
        Returns number of records
        """
        return len(self.records)

    def __len__(self):
        """
        Returns the number of records
        """
        return len(self.records)

    def __iter__(self):
        """
        Iterate of the records
        """
        for record in self.records:
            yield record

    def __repr__(self):
        """
        String with database ID and name
        """
        return "{:s} - ID({:s}) - Name ({:s})".format(self.__class__.__name__,
                                                      self.id,
                                                      self.name)

    def get_contexts(self, nodal_plane_index=1):
        """
        Returns a list of dictionaries, each containing the site, distance
        and rupture contexts for individual records
        """
        wfid_list = np.array([rec.event.id for rec in self.records])
        eqid_list = self._get_event_id_list()
        context_dicts = []
        for eqid in eqid_list:
            idx = np.where(wfid_list == eqid)[0]
            context_dicts.append({
                'EventID': eqid,
                'EventIndex': idx.tolist(),
                'Sites': self._get_sites_context_event(idx),
                'Distances': self._get_distances_context_event(idx),
                'Rupture': self._get_event_context(idx, nodal_plane_index)})
        return context_dicts
    
    def _get_event_id_list(self):
        """
        Returns the list of unique event keys from the database
        """
        event_list = []
        for record in self.records:
            if not record.event.id in event_list:
                event_list.append(record.event.id)
        return np.array(event_list)

    def _get_site_id(self, str_id):
        """
        TODO 
        """
        if not str_id in self.site_ids:
            self.site_ids.append(str_id)
        _id = np.argwhere(str_id == np.array(self.site_ids))[0]
        return _id[0]

    def _get_sites_context_event(self, idx):
        """
        Returns the site context for a particular event
        """
        sctx = SitesContext()
        longs = []
        lats = []
        depths = []
        vs30 = []
        vs30_measured = []
        z1pt0 = []
        z2pt5 = []
        backarc = []
        azimuth = []
        hanging_wall = []
        for idx_j in idx:
            # Site parameters
            rup = self.records[idx_j]
            longs.append(rup.site.longitude)
            lats.append(rup.site.latitude)
            if rup.site.altitude:
                depths.append(rup.site.altitude * -1.0E-3)
            else:
                depths.append(0.0)
            vs30.append(rup.site.vs30)
            if rup.site.vs30_measured is not None:
                vs30_measured.append(rup.site.vs30_measured)
            if rup.site.z1pt0 is not None:
                z1pt0.append(rup.site.z1pt0)
            else:
                z1pt0.append(vs30_to_z1pt0_cy14(rup.site.vs30))
            if rup.site.z2pt5 is not None:
                z2pt5.append(rup.site.z2pt5)
            else:
                z2pt5.append(vs30_to_z2pt5_cb14(rup.site.vs30))
            if ("backarc" in dir(rup.site)) and rup.site.backarc is not None:
                backarc.append(rup.site.backarc)
        setattr(sctx, 'vs30', np.array(vs30))
        if len(longs) > 0:
            setattr(sctx, 'lons', np.array(longs))
        if len(lats) > 0:
            setattr(sctx, 'lats', np.array(lats))
        if len(depths) > 0:
            setattr(sctx, 'depths', np.array(depths))
        if len(vs30_measured) > 0:
            setattr(sctx, 'vs30measured', np.array(vs30_measured))
        if len(z1pt0) > 0:
            setattr(sctx, 'z1pt0', np.array(z1pt0))
        if len(z2pt5) > 0:
            setattr(sctx, 'z2pt5', np.array(z2pt5))
        if len(backarc) > 0:
            setattr(sctx, 'backarc', np.array(backarc))
        return sctx

    def _get_distances_context_event(self, idx):
        """
        Returns the distance contexts for a specific event
        """
        dctx = DistancesContext()
        rrup = []
        rjb = []
        repi = []
        rhypo = []
        r_x = []
        ry0 = []
        rcdpp = []
        azimuth = []
        hanging_wall = []
        rvolc = []
        for idx_j in idx:
            # Distance parameters
            rup = self.records[idx_j]
            repi.append(rup.distance.repi)
            rhypo.append(rup.distance.rhypo)
            # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
            # is a hack! Need feedback on how to fix
            if rup.distance.rjb is not None:
                rjb.append(rup.distance.rjb)
            else:
                rjb.append(rup.distance.repi)
            if rup.distance.rrup is not None:
                rrup.append(rup.distance.rrup)
            else:
                rrup.append(rup.distance.rhypo)
            if rup.distance.r_x is not None:
                r_x.append(rup.distance.r_x)
            else:
                r_x.append(rup.distance.repi)
            if ("ry0" in dir(rup.distance)) and rup.distance.ry0 is not None:
                ry0.append(rup.distance.ry0)
            if ("rcdpp" in dir(rup.distance)) and\
                rup.distance.rcdpp is not None:
                rcdpp.append(rup.distance.rcdpp)
            if rup.distance.azimuth is not None:
                azimuth.append(rup.distance.azimuth)
            if rup.distance.hanging_wall is not None:
                hanging_wall.append(rup.distance.hanging_wall)
            if "rvolc" in dir(rup.distance) and\
                rup.distance.rvolc is not None:
                rvolc.append(rup.distance.rvolc)

        setattr(dctx, 'repi', np.array(repi))
        setattr(dctx, 'rhypo', np.array(rhypo))
        if len(rjb) > 0:
            setattr(dctx, 'rjb', np.array(rjb))
        if len(rrup) > 0:
            setattr(dctx, 'rrup', np.array(rrup))
        if len(r_x) > 0:
            setattr(dctx, 'rx', np.array(r_x))
        if len(ry0) > 0:
            setattr(dctx, 'ry0', np.array(ry0))
        if len(rcdpp) > 0:
            setattr(dctx, 'rcdpp', np.array(rcdpp))
        if len(azimuth) > 0:
            setattr(dctx, 'azimuth', np.array(azimuth))
        if len(hanging_wall) > 0:
            setattr(dctx, 'hanging_wall', np.array(hanging_wall))
        if len(rvolc) > 0:
            setattr(dctx, 'rvolc', np.array(rvolc))
        return dctx

    def _get_event_context(self, idx, nodal_plane_index=1):
        """
        Returns the event contexts for a specific event
        """
        idx = idx[0]
        rctx = RuptureContext()
        rup = self.records[idx]
        setattr(rctx, 'mag', rup.event.magnitude.value)
        if nodal_plane_index == 2:
            setattr(rctx, 'strike',
                rup.event.mechanism.nodal_planes.nodal_plane_2['strike'])
            setattr(rctx, 'dip',
                rup.event.mechanism.nodal_planes.nodal_plane_2['dip'])
            setattr(rctx, 'rake',
                rup.event.mechanism.nodal_planes.nodal_plane_2['rake'])
        else:
            setattr(rctx, 'strike', 0.0)
            setattr(rctx, 'dip', 90.0)
            rctx.rake = rup.event.mechanism.get_rake_from_mechanism_type()
        if rup.event.rupture.surface:
            setattr(rctx, 'ztor', rup.event.rupture.surface.get_top_edge_depth())
            setattr(rctx, 'width', rup.event.rupture.surface.width)
            setattr(rctx, 'hypo_loc', rup.event.rupture.surface.get_hypo_location(1000))
        else:
            setattr(rctx, 'ztor', rup.event.depth)
            # Use the PeerMSR to define the area and assuming an aspect ratio
            # of 1 get the width
            setattr(rctx, 'width',
                    np.sqrt(DEFAULT_MSR.get_median_area(rctx.mag, 0)))
            # Default hypocentre location to the middle of the rupture
            setattr(rctx, 'hypo_loc', (0.5, 0.5))
        setattr(rctx, 'hypo_depth', rup.event.depth)
        setattr(rctx, 'hypo_lat', rup.event.latitude)
        setattr(rctx, 'hypo_lon', rup.event.longitude)
        return rctx

    def get_site_collection(self, missing_vs30=None):
        """
        Returns the sites in the database as an instance of the :class:
        openquake.hazardlib.site.SiteCollection
        """
        return SiteCollection([rec.site.to_openquake_site(missing_vs30)
                               for rec in self.records])
