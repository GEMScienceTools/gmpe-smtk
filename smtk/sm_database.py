"""
Basic Pseudo-database built on top of hdf5 for a set of processed strong
motion records

.. moduleauthor::  R. Zaccarelli
"""
import os
import pickle
import json
from datetime import datetime
from collections import OrderedDict
import numpy as np
import h5py
from openquake.hazardlib import imt
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo.point import Point
from smtk.trellis.configure import vs30_to_z1pt0_as08, z1pt0_to_z2pt5
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14
import smtk.sm_utils as utils
from smtk.rcrs import ResidualsCompliantRecordSet


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
        if self.mechanism_type in utils.MECHANISM_TYPE:
            return utils.MECHANISM_TYPE[self.mechanism_type]
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
            eigs =  GCMTPrincipalAxes.from_dict(data["eigenvalues"])
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
                         -self.altitude / 1000.)  # Elevation from m to km
        oq_site = Site(location,
                       vs30,
                       z1pt0,
                       z2pt5,
                       vs30measured=vs30_measured,
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


class GroundMotionDatabase(ResidualsCompliantRecordSet):
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
        self.records = [rec for rec in records]
        self.site_ids = site_ids

    ##############################################################
    # Implementing ResidualsCompliantRecordSet ABSTRACT METHODS: #
    ##############################################################

    @property
    def records(self):
        '''Returns an iterable of records from this database (e.g. list, tuple,
        generator). Note that as any iterable, the returned object does not
        need to define a length whereby `len(self.records)` will work: this
        will depend on subclasses implementation
        '''
        return self._records

    # this is no abstract method, it makes `self.records = new_records` work:
    @records.setter
    def records(self, list_of_records):
        self._records = list_of_records

    def get_record_eventid(self, record):
        '''Returns the record event id (usually int) of the given record'''
        return record.event.id

    def update_sites_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.SitesContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g.: `context.vs30.append(record.vs30)`
        '''
        context.vs30.append(record.site.vs30)
        context.lons.append(record.site.longitude)
        context.lats.append(record.site.latitude)
        if record.site.altitude:
            depth = record.site.altitude * -1.0E-3
        else:
            depth = 0.0
        context.depths.append(depth)
        if record.site.vs30_measured is not None:
            vs30_measured = record.site.vs30_measured
        else:
            vs30_measured = 0
        context.vs30measured.append(vs30_measured)
        if record.site.z1pt0 is not None:
            z1pt0 = record.site.z1pt0
        else:
            z1pt0 = vs30_to_z1pt0_cy14(record.site.vs30)
        context.z1pt0.append(z1pt0)
        if record.site.z2pt5 is not None:
            z2pt5 = record.site.z2pt5
        else:
            z2pt5 = vs30_to_z2pt5_cb14(record.site.vs30)
        context.z2pt5.append(z2pt5)
        if getattr(record.site, "backarc", None) is not None:
            context.backarc.append(record.site.backarc)

    def update_distances_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.DistancesContext`
        object. In the typical implementation it has the attributes defined in
        `self.distances_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g.: `context.rjb.append(record.rjb)`
        '''
        context.repi.append(record.distance.repi)
        context.rhypo.append(record.distance.rhypo)
        # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
        # is a hack! Need feedback on how to fix
        if record.distance.rjb is not None:
            context.rjb.append(record.distance.rjb)
        else:
            context.rjb.append(record.distance.repi)
        if record.distance.rrup is not None:
            context.rrup.append(record.distance.rrup)
        else:
            context.rrup.append(record.distance.rhypo)
        if record.distance.r_x is not None:
            context.rx.append(record.distance.r_x)
        else:
            context.rx.append(record.distance.repi)
        if getattr(record.distance, "ry0", None) is not None:
            context.ry0.append(record.distance.ry0)
        if getattr(record.distance, "rcdpp", None) is not None:
            context.rcdpp.append(record.distance.rcdpp)
        if record.distance.azimuth is not None:
            context.azimuth.append(record.distance.azimuth)
        if record.distance.hanging_wall is not None:
            context.hanging_wall.append(record.distance.hanging_wall)
        if getattr(record.distance, "rvolc", None) is not None:
            context.rvolc.append(record.distance.rvolc)

    def update_rupture_context(self, record, context, nodal_plane_index=1):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.RuptureContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to NaN (numpy.nan).
        Here you should set those attributes with the relative record value,
        e.g.: `context.mag = record.event.mag`
        '''
        context.mag = record.event.magnitude.value
        if nodal_plane_index == 2:
            context.strike = \
                record.event.mechanism.nodal_planes.nodal_plane_2['strike']
            context.dip = \
                record.event.mechanism.nodal_planes.nodal_plane_2['dip']
            context.rake = \
                record.event.mechanism.nodal_planes.nodal_plane_2['rake']
        elif nodal_plane_index == 1:
            context.strike = \
                record.event.mechanism.nodal_planes.nodal_plane_1['strike']
            context.dip = \
                record.event.mechanism.nodal_planes.nodal_plane_1['dip']
            context.rake = \
                record.event.mechanism.nodal_planes.nodal_plane_1['rake']
        else:
            context.strike = 0.0
            context.dip = 90.0
            context.rake = \
                record.event.mechanism.get_rake_from_mechanism_type()

        if record.event.rupture.surface:
            context.ztor = record.event.rupture.surface.get_top_edge_depth()
            context.width = record.event.rupture.surface.width
            context.hypo_loc = \
                record.event.rupture.surface.get_hypo_location(1000)
        else:
            if record.event.rupture.depth is not None:
                context.ztor = record.event.rupture.depth
            else:
                context.ztor = record.event.depth

            if record.event.rupture.width is not None:
                context.width = record.event.rupture.width
            else:
                # Use the PeerMSR to define the area and assuming an aspect ratio
                # of 1 get the width
                context.width = \
                    np.sqrt(utils.DEFAULT_MSR.get_median_area(context.mag, 0))

            # Default hypocentre location to the middle of the rupture
            context.hypo_loc = (0.5, 0.5)
        context.hypo_depth = record.event.depth
        context.hypo_lat = record.event.latitude
        context.hypo_lon = record.event.longitude

    def update_observations(self, record, observations, component="Geometric"):
        '''Updates the observed intensity measures types (imt) with the given
        `record` data. `observations` is a `dict` of imts (string) mapped to
        numeric lists. Here you should append to each list the imt value
        derived from `record`, ususally numeric or NaN (`numpy.nan`):
        ```
            for imt, values in observations.items():
                if imtx in self.SCALAR_IMTS:  # currently, 'PGA' or 'PGV'
                    val = ... get the imt scalar value from record ...
                elif "SA(" in imtx:
                    val = ... get the SA numeric array / list from record ...
                else:
                    raise ValueError("IMT %s is unsupported!" % imtx)
                values.append(val)
        ```
        '''
        selection_string = "IMS/H/Spectra/Response/Acceleration/"
        fle = h5py.File(record.datafile, "r")
        for imtx, values in observations.items():
            if imtx in self.SCALAR_IMTS:
                values.append(self.get_scalar(fle, imtx, component))
            elif "SA(" in imtx:
                target_period = imt.from_string(imtx).period
                spectrum = fle[selection_string + component +
                               "/damping_05"].value
                periods = fle["IMS/H/Spectra/Response/Periods"].value
                values.append(utils.get_interpolated_period(
                    target_period, periods, spectrum))
            else:
                raise ValueError("IMT %s is unsupported!" % imtx)
        fle.close()

    ###########################
    # END OF ABSTRACT METHODS #
    ###########################

    # moved from smtk/residuals/gmpe_residuals.py:
    def get_scalar(self, fle, i_m, component="Geometric"):
        """
        Retrieves the scalar IM from the database
        :param fle:
            Instance of :class: h5py.File
        :param str i_m:
            Intensity measure
        :param str component:
            Horizontal component of IM
        """
        if not ("H" in fle["IMS"].keys()):
            x_im = fle["IMS/X/Scalar/" + i_m].value[0]
            y_im = fle["IMS/Y/Scalar/" + i_m].value[0]
            return utils.SCALAR_XY[component](x_im, y_im)
        else:
            if i_m in fle["IMS/H/Scalar"].keys():
                return fle["IMS/H/Scalar/" + i_m].value[0]
            else:
                raise ValueError("Scalar IM %s not in record database" % i_m)

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

    def __repr__(self):
        """
        String with database ID and name
        """
        return "{:s} - ID({:s}) - Name ({:s})".format(self.__class__.__name__,
                                                      self.id,
                                                      self.name)

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
        if str_id not in self.site_ids:
            self.site_ids.append(str_id)
        _id = np.argwhere(str_id == np.array(self.site_ids))[0]
        return _id[0]

    def get_site_collection(self, missing_vs30=None):
        """
        Returns the sites in the database as an instance of the :class:
        openquake.hazardlib.site.SiteCollection
        """
        return SiteCollection([rec.site.to_openquake_site(missing_vs30)
                               for rec in self.records])


def load_database(directory):
    """
    Wrapper function to load the metadata of a :class:`GroundMotionDatabase`
    according to the filetype
    """
    metadata_file = None
    filetype = None
    fileset = os.listdir(directory)
    for ftype in ["pkl", "json"]:
        if ("metadatafile.%s" % ftype) in fileset:
            metadata_file = "metadatafile.%s" % ftype
            filetype = ftype
            break
    if not metadata_file:
        raise IOError(
            "Expected metadata file of supported type not found in %s"
            % directory)
    metadata_path = os.path.join(directory, metadata_file)
    if filetype == "json":
        # json metadata filetype
        return GroundMotionDatabase.from_json(metadata_path)
    elif filetype == "pkl":
        # pkl file type
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Metadata filetype %s not supported" % ftype)
