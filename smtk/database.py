#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation and G. Weatherill
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
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14

"""
Abstract class for a ground motion database. Any container of observations
used for the residuals calculation should inherit from :class:`GMDatabase`
"""
import sys
from collections import OrderedDict  # FIXME In Python3.7+, dict is sufficient

import numpy as np
from openquake.hazardlib.contexts import DistancesContext, RuptureContext, \
    SitesContext

from smtk.sm_utils import convert_accel_units

class ResidualsDatabase:

    sites_context_attrs = ('vs30', 'lons', 'lats', 'depths',
                           'vs30measured', 'z1pt0', 'z2pt5', 'backarc')
    distances_context_attrs = tuple(DistancesContext._slots_)
    rupture_context_attrs = tuple(RuptureContext._slots_)

    def get_contexts(self, records, nodal_plane_index=1,
                     imts=None, component="Geometric"):
        """
        Returns an iterable of dictionaries, each containing the site, distance
        and rupture contexts for individual records. Each context dict is of the
        form:
        ```
        {
         'EventID': earthquake id,
         'Sites': :class:`openquake.hazardlib.contexts.SitesContext`,
         'Distances': :class:`openquake.hazardlib.contexts.DistancesContext`,
         'Rupture': :class:`openquake.hazardlib.contexts.RuptureContext`
        }
        If `imts` is not None but a list of Intensity measure types,
        other two arguments are required 'Observation' (dict of imts mapped
        to each record imt value),a and 'Num. Sites' (the length of records)
        ```
        """
        # contexts are dicts which will temporarily be stored in a wrapping
        # dict `context_dicts` to easily retrieve a context from a record
        # event_id:
        context_dicts = {}
        compute_observations = imts is not None and len(imts)
        for rec in records:
            evt_id = self.get_record_eventid(rec)  # rec['event_id']
            dic = context_dicts.get(evt_id, None)
            if dic is None:
                # we might use defaultdict, but like this is more readable
                dic = {
                    'EventID': evt_id,
                    'EventIndex': [],
                    'Sites': self.create_sites_context(),
                    'Distances': self.create_distances_context(),
                    'Rupture': self.create_rupture_context(),
                    "Num. Sites": 0
                }
                if compute_observations:
                    dic["Observations"] = self.create_observations_dict(imts)
                # set Rupture only once:
                self.update_rupture_context(rec, dic['Rupture'],
                                            nodal_plane_index)
                # assign to wrapping `context_dicts`:
                context_dicts[evt_id] = dic

            dic['EventIndex'].append(self.get_record_id(rec))
            self.update_sites_context(rec, dic['Sites'])
            self.update_distances_context(rec, dic['Distances'])
            if compute_observations:
                self.update_observations(rec, dic['Observations'], component)
            dic["Num. Sites"] += 1

        # converts to numeric arrays (once at the end is faster, see
        # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
        # get default attributes not to be changed:
        for dic in context_dicts.values():
            self.finalize_sites_context(dic['Sites'])
            self.finalize_distances_context(dic['Distances'])
            self.finalize_rupture_context(dic['Rupture'])
            if compute_observations:
                self.finalize_observations(dic['Observations'])

        return context_dicts.values()

    def create_context(self, evt_id):
        '''
        Creates and intitializes a Context dict
        (by default, setting attributes o SitesContext, DistanceContext,
        RuptureContext). This method's return value is ignored if present

        :param evt_id: the event id (usually int) denoting the source event id
        '''
        context = {
            'EventID': evt_id,
            'EventIndex': [],
            'Sites': SitesContext(),
            'Distances': DistancesContext(),
            'Rupture': RuptureContext(),
            "Num. Sites": 0
        }
        sites_context = context['Sites']
        for _ in self.sites_context_attrs:
            setattr(sites_context, _, [])
        dist_context = context['Distances']
        for _ in self.disances_context_attrs:
            setattr(dist_context, _, [])
        rupt_context = context['Distances']
        for _ in self.rupture_context_attrs:
            setattr(rupt_context, _, np.nan)

    def create_sites_context(self):
        '''
        Creates, initializes and returns a sites context by setting the
        default values of the attributes defined in `self.sites_context_attrs`.
        The returned context is intended to be used in `self.get_contexts`.

        :return:  a :class:`openquake.hazardlib.contexts.SitesContext`
        '''
        ctx = SitesContext()
        for _ in self.sites_context_attrs:
            setattr(ctx, _, [])
        return ctx

    def create_distances_context(self):
        '''
        Creates, initializes and returns a distances context by setting the
        default values of the attributes defined in `self.distances_context_attrs`.
        The returned context is intended to be used in `self.get_contexts`.

        :return:  a :class:`openquake.hazardlib.contexts.DistancesContext`
        '''
        ctx = DistancesContext()
        for _ in self.distances_context_attrs:
            setattr(ctx, _, [])
        return ctx

    def create_rupture_context(self, evt_id):
        '''
        Creates, initializes and returns a rupture context by setting the
        default values of the attributes defined in `self.rupture_context_attrs`.
        The returned context is intended to be used in `self.get_contexts`.

        :return:  a :class:`openquake.hazardlib.contexts.RuptureContext`
        '''
        ctx = RuptureContext()
        for _ in self.rupture_context_attrs:
            setattr(ctx, _, np.nan)
        return ctx

    def finalize_sites_context(self, context):
        '''
        Finalizes the `context` object created with
        `self.create_sites_context` and populated in `self.get_contexts`.
        All operations should be performed inplace on `context`, this method
        is not expected to return any value

        :param context: a :class:`openquake.hazardlib.contexts.SitesContext`
        '''
        for _ in self.sites_context_attrs:
            setattr(context, _, np.array(getattr(context, _)))

    def finalize_distances_context(self, context):
        '''
        Finalizes the `context` object created with
        `self.create_distances_context` and populated in `self.get_contexts`.
        All operations should be performed inplace on `context`, this method
        is not expected to return any value

        :param context: a :class:`openquake.hazardlib.contexts.DistancesContext`
        '''
        for _ in self.distances_context_attrs:
            setattr(context, _, np.array(getattr(context, _)))

    def finalize_rupture_context(self, context):
        '''
        Finalizes the `context` object created with
        `self.create_rupture_context` and populated in `self.get_contexts`.
        All operations should be performed inplace on `context`, this method
        is not expected to return any value

        :param context: a :class:`openquake.hazardlib.contexts.RuptureContext`
        '''
        pass

    def get_observations(self, imts, records, component="Geometric"):
        """
        This method is not used but it's here for backward compatibility.
        Get the obsered ground motions from the database. *NOTE*: IMTs in
        acceleration units (e.g. PGA, SA) are supposed to return their
        values in cm/s/s (which is by default the unit in which they are
        stored)
        """
        observations = self.create_observations_dict(imts)
        for record in records:
            self.update_observations(observations, record, component)
        self.finalize_observations_dict(observations)
        return observations
#         select_records = \
#             sm_record_selector.select_from_event_id(context["EventID"])
#         observations = OrderedDict([(imtx, []) for imtx in self.imts])
#         selection_string = "IMS/H/Spectra/Response/Acceleration/"
#         for record in select_records:
#             fle = h5py.File(record.datafile, "r")
#             for imtx in self.imts:
#                 if imtx in SCALAR_IMTS:
#                     observations[imtx].append(
#                         get_scalar(fle, imtx, component))
#                 elif "SA(" in imtx:
#                     target_period = imt.from_string(imtx).period
#                     spectrum = fle[selection_string + component +
#                                    "/damping_05"].value
#                     periods = fle["IMS/H/Spectra/Response/Periods"].value
#                     observations[imtx].append(get_interpolated_period(
#                         target_period, periods, spectrum))
#                 else:
#                     raise "IMT %s is unsupported!" % imtx
#             fle.close()
#         for imtx in self.imts:
#             observations[imtx] = np.array(observations[imtx])
#         context["Observations"] = observations
#         context["Num. Sites"] = len(select_records)
#         return context

    def create_observations(self, imts):
        '''
        creates and returns a `dict` from the given imts
        '''
        return OrderedDict([(imtx, []) for imtx in imts])

    def finalize_observations(self, observations):
        '''
        finalizes a the observations `dict` after it has been populated
        with the database observed values. By default, it converts all dict
        values to numpy array
        '''
        for imtx, values in observations.items():
            observations[imtx] = np.asarray(values, dtype=float)

    def get_record_eventid(self, record):
        '''Returns the record event id (usually integer) from the given
        `record`'''
        raise NotImplementedError('')

    def get_record_id(self, record):
        '''Returns the record id (usually integer)'''
        raise NotImplementedError('')

    def update_observations(self, record, observations, component="Geometric"):
        raise NotImplementedError('')

    def update_sites_context(self, record, sites_context):
        '''Updates the attributes of `sites_context` with the given `record`
        data. In the typical implementation `sites_context` has the attributes
        defined in `self.sites_context_attrs`, all initialized as empty lists
        (intended to store numeric values). Subclasses should append here
        to those attributes the relative record value:
        ```
            vs30 = ... get the vs30 from `record` ...
            sites_context.vs30.append(vs30)
            ... and so on ...
        ```
        '''
        raise NotImplementedError('')

    def update_distances_context(self, record, distances_context):
        raise NotImplementedError('')

    def update_rupture_context(self, record, rupture_context,
                               nodal_plane_index=1):
        raise NotImplementedError('')


class GroundMotionDatabase(object):

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
        if str_id not in self.site_ids:
            self.site_ids.append(str_id)
        _id = np.argwhere(str_id == np.array(self.site_ids))[0]
        return _id[0]

    def update_sites_context(self, record, sites_context):
        """
        Returns the site context for a particular event
        """
        sites_context.vs30.append(record.site.vs30)
        sites_context.lons.append(record.site.longitude)
        sites_context.lats.append(record.site.latitude)
        if record.site.altitude:
            depth = record.site.altitude * -1.0E-3
        else:
            depth = 0.0
        sites_context.depths.append(depth)
        if record.site.vs30_measured is not None:
            vs30_measured = record.site.vs30_measured
        else:
            vs30_measured = 0
        sites_context.vs30measured.append(vs30_measured)
        if record.site.z1pt0 is not None:
            z1pt0 = record.site.z1pt0
        else:
            z1pt0 = vs30_to_z1pt0_cy14(record.site.vs30)
        sites_context.z1pt0.append(z1pt0)
        if record.site.z2pt5 is not None:
            z2pt5 = record.site.z2pt5
        else:
            z2pt5 = vs30_to_z2pt5_cb14(record.site.vs30)
        sites_context.z2pt5.append(z2pt5)
        if ("backarc" in dir(record.site)) and record.site.backarc is not None:
            sites_context.backarc.append(record.site.backarc)

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
        elif nodal_plane_index == 1:
            setattr(rctx, 'strike',
                    rup.event.mechanism.nodal_planes.nodal_plane_1['strike'])
            setattr(rctx, 'dip',
                    rup.event.mechanism.nodal_planes.nodal_plane_1['dip'])
            setattr(rctx, 'rake',
                    rup.event.mechanism.nodal_planes.nodal_plane_1['rake'])
        else:
            setattr(rctx, 'strike', 0.0)
            setattr(rctx, 'dip', 90.0)
            rctx.rake = rup.event.mechanism.get_rake_from_mechanism_type()

        if rup.event.rupture.surface:
            setattr(rctx, 'ztor', rup.event.rupture.surface.get_top_edge_depth())
            setattr(rctx, 'width', rup.event.rupture.surface.width)
            setattr(rctx, 'hypo_loc', rup.event.rupture.surface.get_hypo_location(1000))
        else:
            if rup.event.rupture.depth is not None:
                setattr(rctx, 'ztor', rup.event.rupture.depth)
            else:
                setattr(rctx, 'ztor', rup.event.depth)

            if rup.event.rupture.width is not None:
                setattr(rctx, 'width', rup.event.rupture.width)
            else:
                # Use the PeerMSR to define the area and assuming an aspect ratio
                # of 1 get the width
                setattr(rctx, 'width',
                        np.sqrt(utils.DEFAULT_MSR.get_median_area(rctx.mag, 0)))

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