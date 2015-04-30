#!/usr/bin/env/python

"""
Basic Pseudo-database built on top of hdf5 for a set of processed strong
motion records
"""
import os
import h5py
import numpy as np
from datetime import datetime
from openquake.hazardlib.gsim.base import (SitesContext, DistancesContext,
                                          RuptureContext)
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo.point import Point
from smtk.trellis.configure import vs30_to_z1pt0_as08, z1pt0_to_z2pt5

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
    def __init__(self, value, mtype, sigma=None):
        """
        Instantiate the class
        """
        self.value = value
        self.mtype = mtype
        self.sigma = None

class Rupture(object):
    """
    Class to hold rupture attributes
    :param str id:
        Rupture ID
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
    def __init__(self, eq_id, length, width, depth, area=None, surface=None,
            hypo_loc=None):
        """
        Instantiate
        """
        self.id = eq_id
        self.length = length
        self.width = width
        self.area = area
        self.depth = depth
        self.surface = surface
        self.hypo_loc = hypo_loc

    def get_area(self):
        """
        Returns the area of the rupture
        """
        if self.area:
            return self.area
        if self.length and self.width:
            self.area = self.length * self.width
        else:
            self.area = None
  

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

MECHANISM_TYPE = {"Normal": -90.0,
                  "Strike-Slip": 0.0,
                  "Reverse": 90.0,
                  "Oblique": 0.0,
                  "Unknown": 0.0}

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
        if self.mechanism_type in MECHANISM_TYPE.keys():
            return MECHANISM_TYPE[self.mechanism_type]
        else:
            return 0.0


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
        magnitude, focal_mechanism=None, eq_country=None):
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
        self.magnitude_list = None
        self.mechanism = focal_mechanism
        self.rupture = None

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
    :param flag:
        ?
    :param bool hanging_wall:
        True if site on hanging wall, False otherwise
    """
    def __init__(self, repi, rhypo, rjb=None, rrup=None, r_x=None, ry0=None,
            flag=None):
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
        self.flag = flag
        self.hanging_wall = None

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
        self.country = country
        self.z1pt0 = None
        self.z1pt5 = None
        self.z2pt5 = None
        self.backarc = backarc

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
        return Site(location,
                    vs30,
                    vs30_measured,
                    z1pt0,
                    z2pt5,
                    backarc=self.backarc,
                    id=self.id)

    def get_ec8_class(self):
        """
        Returns the EC8 class associated with a site given a Vs30
        """
        if self.ec8:
            return self.ec8
        if self.vs30:
            for key in EC8_VS30_BOUNDARIES.keys():
                in_group = (self.vs30 >= EC8_VS30_BOUNDARIES[key][0]) and\
                    (self.vs30 < EC8_VS30_BOUNDARIES[key][1])
                if in_group:
                    self.ec8 = key
                    return self.ec8
        elif self.nspt:
            # Check to see if a site class can be determined from NSPT
            for key in EC8_NSPT_BOUNDARIES.keys():
                in_group = (self.nspt >= EC8_NSPT_BOUNDARIES[key][0]) and\
                    (self.nspt < EC8_NSPT_BOUNDARIES[key][1])
                if in_group:
                    self.ec8 = key
                    return self.ec8
        else:
            print "Cannot determine EC8 site class - no Vs30 or NSPT measures!"
        return None
    
    def get_nehrp_class(self):
        """
        Returns the NEHRP class associated with a site given a Vs30 or NSPT
        """
        if self.nehrp:
            return self.nehrp
        if self.vs30:
            for key in NEHRP_VS30_BOUNDARIES.keys():
                in_group = (self.vs30 >= NEHRP_VS30_BOUNDARIES[key][0]) and\
                    (self.vs30 < NEHRP_VS30_BOUNDARIES[key][1])
                if in_group:
                    self.nehrp = key
                    return self.nehrp
        elif self.nspt:
            # Check to see if a site class can be determined from NSPT
            for key in NEHRP_NSPT_BOUNDARIES.keys():
                in_group = (self.nspt >= NEHRP_NSPT_BOUNDARIES[key][0]) and\
                    (self.nspt < NEHRP_NSPT_BOUNDARIES[key][1])
                if in_group:
                    self.nehrp = key
                    return self.nehrp
        else:
            print "Cannot determine NEHRP site class - no Vs30 or NSPT measures!"
        return None


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
        self.filter = waveform_filter
        self.baseline = baseline
        self.ims = ims
        self.units = None


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
        x_comp, y_comp, vertical=None, ims={}, longest_period=None,
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
        self.records = records
        self.site_ids = site_ids

    def number_records(self):
        """
        Returns number of records
        """
        return len(self.records)

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
        for idx_j in idx:
            # Site parameters
            rup = self.records[idx_j]
            longs.append(rup.site.longitude)
            lats.append(rup.site.latitude)
            depths.append(rup.site.altitude * -1.0E-3)
            vs30.append(rup.site.vs30)
            if rup.site.vs30_measured:
                vs30_measured.append(rup.site.vs30_measured)
            if rup.site.z1pt0:
                z1pt0.append(rup.site.z1pt0)
            if rup.site.z2pt5:
                z1pt0.append(rup.site.z2pt5)
            if ("backarc" in dir(rup.site)) and rup.site.backarc:
                backarc.append(rup.site.backarc)
        setattr(sctx, 'vs30', np.array(vs30))
        if len(longs) > 0:
            setattr(sctx, 'lons', np.array(longs))
        if len(lats) > 0:
            setattr(sctx, 'lats', np.array(lats))
        if len(depths) > 0:
            setattr(sctx, 'depths', np.array(lats))
        if len(vs30_measured) > 0:
            setattr(sctx, 'vs30measured', np.array(vs30))
        if len(z1pt0) > 0:
            setattr(sctx, 'z1pt0', np.array(z1pt0))
        if len(z2pt5) > 0:
            setattr(sctx, 'z2pt5', np.array(z1pt0))
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
        for idx_j in idx:
            # Distance parameters
            rup = self.records[idx_j]
            repi.append(rup.distance.repi)
            rhypo.append(rup.distance.rhypo)
            # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
            # is a hack! Need feedback on how to fix
            if rup.distance.rjb:
                rjb.append(rup.distance.rjb)
            else:
                rjb.append(rup.distance.repi)
            if rup.distance.rrup:
                rrup.append(rup.distance.rrup)
            else:
                rrup.append(rup.distance.rhypo)
            if rup.distance.r_x:
                r_x.append(rup.distance.r_x)
            if ("ry0" in dir(rup.distance)) and rup.distance.ry0:
                ry0.append(rup.distance.ry0)
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
            setattr(rctx, 'strike',
                rup.event.mechanism.nodal_planes.nodal_plane_1['strike'])
            setattr(rctx, 'dip',
                rup.event.mechanism.nodal_planes.nodal_plane_1['dip'])
            setattr(rctx, 'rake',
                rup.event.mechanism.nodal_planes.nodal_plane_1['rake'])
        if not rctx.rake:
            rctx.rake = rup.event.mechanism.get_rake_from_mechanism_type()
        if rup.event.rupture:
            setattr(rctx, 'ztor', rup.event.rupture.depth)
            setattr(rctx, 'width', rup.event.rupture.width)
            setattr(rctx, 'hypo_loc', rup.event.rupture.hypo_loc)
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
