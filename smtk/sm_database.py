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
    """
    def __init__(self, value, mtype, sigma=None):
        """

        """
        self.value = value
        self.mtype = mtype
        self.sigma = None

class Rupture(object):
    """
    Class to hold rupture attributes
    """
    def __init__(self, eq_id, length, width, depth, 
        area=None, surface=None, hypo_loc=None):
        """

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
    """
    def __init__(self):
        """
        """
        self.nodal_plane_1 = None
        self.nodal_plane_2 = None

class GCMTPrincipalAxes(object):
    """
    Class to represent the eigensystem of the tensor in terms of  T-, B- and P-
    plunge and azimuth
    #_axis = {'eigenvalue':, 'azimuth':, 'plunge':}
    """
    def __init__(self):
        """
        """
        self.t_axis = None
        self.b_axis = None
        self.p_axis = None

# MECHANISM_TYPE = {"Normal": -90.0,
#                   "Strike-Slip": 0.0,
#                   "Reverse": 90.0,
#                   "Oblique": 0.0}

MECHANISM_TYPE = {"N": -90.0,
                  "S": 0.0,
                  "R": 90.0,
                  "U": 0.0}

class FocalMechanism(object):
    """
    Class to hold the full focal mechanism attribute set
    """
    def __init__(self, eq_id, name, nodal_planes, eigenvalues,
        moment_tensor=None, mechanism_type=None):
        """

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
    Class to hold Event Related Information
    """
    def __init__(self, eq_id, name, date_time, longitude, latitude, depth,
        magnitude, focal_mechanism=None, eq_country=None):
        """

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
    Class to hold distance information
    """
    def __init__(self, repi, rhypo, rjb=None, rrup=None, r_x=None, ry0=None, flag=None):
        """
        """
        self.repi = repi
        self.rhypo = rhypo
        self.rjb = rjb
        self.rrup = rrup
        self.r_x = r_x
        self.ry0 = None
        self.azimuth = None
        self.flag = flag
        self.hanging_wall = None


class RecordSite(object):
    """
    Class to hold attributes belonging to the site
    """
    def __init__(self, site_id, site_code, site_name, longitude, latitude,
        altitude, vs30=None, vs30_measured=None, network_code=None,
        country=None, site_class=None):
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
        self.arc_location = None

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

        return Site(Point(self.longitude,
                          self.latitude,
                          # elev meters -> depth kilometers
                          self.altitude*(-1.e-3)), 
                    vs30,
                    vs30_measured,
                    z1pt0,
                    z2pt5,
                    self.id)


    def get_ec8_class(self):
        """
        Returns the EC8 class associated with a site given a Vs30
        """

        if not self.vs30:
            print "Cannot return EC8 site class - no Vs30!"
            return None

    def get_nehrp_class(self):
        """

        """
        raise NotImplementedError()


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
    """
    def __init__(self, waveform_id, orientation, ims=None, longest_period=None,
        waveform_filter=None, baseline=None, units=None):
        """

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
    Class to represent a databse of strong motions
    """
    def __init__(self, db_id, db_name, db_directory=None,
        records=[], site_ids=[]):
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
        and rupture contexts for individual
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
        Returns the list of unique event keys from the database
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
        for idx_j in idx:
            # Site parameters
            rup = self.records[idx_j]
            longs.append(rup.site.longitude)
            lats.append(rup.site.latitude)
            # site elevation (m) -> depth[km]
            depths.append(rup.site.altitude * (-1.e-3))
            vs30.append(rup.site.vs30)
            if rup.site.vs30_measured:
                vs30_measured.append(rup.site.vs30_measured)
            if rup.site.z1pt0:
                z1pt0.append(rup.site.z1pt0)
            if rup.site.z2pt5:
                z1pt0.append(rup.site.z2pt5)
        setattr(sctx, 'vs30', np.array(vs30))
        if len(longs) > 0:
            setattr(sctx, 'lons', np.array(longs))
        if len(lats) > 0:
            setattr(sctx, 'lats', np.array(lats))
        if len(depths) > 0:
            setattr(sctx, 'depths', np.array(depths))
        if len(vs30_measured) > 0:
            setattr(sctx, 'vs30measured', np.array(vs30))
        if len(z1pt0) > 0:
            setattr(sctx, 'z1pt0', np.array(z1pt0))
        if len(z2pt5) > 0:
            setattr(sctx, 'z2pt5', np.array(z1pt0))
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

        # TODO: Behaviour need to be checked
        #if rup.event.mechanism.fault_plane is None:
        #    setattr(rctx, 'strike', None)
        #    setattr(rctx, 'dip', None)
        #    setattr(rctx, 'rake', None)
        #elif rup.event.mechanism.fault_plane == 2:
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
