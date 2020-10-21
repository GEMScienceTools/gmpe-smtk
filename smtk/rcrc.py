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
"""
module housing the ResidualsCompliantRecordsCollection abstract-like class which should
be inherited by any database/set/collection supporting residuals computation on
its records
"""
import sys
from collections import OrderedDict  # FIXME In Python3.7+, dict is sufficient

import numpy as np
import h5py
from openquake.hazardlib import imt
from openquake.hazardlib.contexts import DistancesContext, RuptureContext, \
    SitesContext

from smtk import sm_utils
from smtk.trellis.configure import vs30_to_z1pt0_cy14, vs30_to_z2pt5_cb14
from smtk.sm_utils import SCALAR_XY, get_interpolated_period,\
    convert_accel_units, MECHANISM_TYPE, DIP_TYPE, DEFAULT_MSR


class ResidualsCompliantRecordsCollection:
    '''This abstract-like class implements an iterables of records which can be
    used in :meth:`gmpe_residuals.Residuals.get_residuals` to compute the
    records residuals. Subclasses need to implement few abstract-like methods
    defining how to get the input data required for the residuals computation
    '''

    # SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
    SCALAR_IMTS = ["PGA", "PGV"]

    sites_context_attrs = ('vs30', 'lons', 'lats', 'depths',
                           'vs30measured', 'z1pt0', 'z2pt5', 'backarc')
    distances_context_attrs = tuple(DistancesContext._slots_)
    rupture_context_attrs = tuple(RuptureContext._slots_)

    ####################################################
    # ABSTRACT METHODS TO BE IMPLEMENTED IN SUBCLASSES #
    ####################################################

    @property
    def records(self):
        '''Returns an iterable of records from this database (e.g. list, tuple,
        generator). Note that as any iterable, the returned object does not
        need to define a length whereby `len(self.records)` will work: this
        will depend on subclasses implementation
        '''
        raise NotImplementedError('')

    def get_record_eventid(self, record):
        '''Returns the record event id (usually int) of the given record'''
        raise NotImplementedError('')

    def update_sites_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.SitesContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g. `context.vs30.append(record.vs30)`
        '''
        raise NotImplementedError('')

    def update_distances_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.DistancesContext`
        object. In the typical implementation it has the attributes defined in
        `self.distances_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g. `context.rjb.append(record.rjb)`
        '''
        raise NotImplementedError('')

    def update_rupture_context(self, record, context, nodal_plane_index=1):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.RuptureContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to NaN (numpy.nan).
        Here you should set those attributes with the relative record value,
        e.g. `context.mag = record.event.mag`
        '''
        raise NotImplementedError('')

    def update_observations(self, record, observations, component="Geometric"):
        '''Updates the observed intensity measures types (imt) with the given
        `record` data. `observations` is a `dict` of imts (string) mapped to
        numeric lists. Here you should append to each list the imt value
        derived from `record`, ususally numeric or NaN (`numpy.nan`):
        ```
            for imtx, values in observations.items():
                if imtx in self.SCALAR_IMTS:  # currently, 'PGA' or 'PGV'
                    val = ... get the imt scalar value from record ...
                elif "SA(" in imtx:
                    val = ... get the SA numeric array / list from record ...
                else:
                    raise ValueError("IMT %s is unsupported!" % imtx)
                values.append(val)
        ```
        *IMPORTANT*: IMTs in acceleration units (e.g. PGA, SA) are supposed to
        return their values in cm/s/s (which is generally the unit in which
        they are stored)
        '''
        raise NotImplementedError('')

    #################################
    # END OF ABSTRRACT-LIKE METHODS #
    #################################

    def __iter__(self):
        """
        Make this object iterable, i.e.
        `for rec in self` is equalto `for rec in self.records`
        """
        for record in self.records:
            yield record

    def get_contexts(self, nodal_plane_index=1,
                     imts=None, component="Geometric"):
        """Returns an iterable of dicts, each containing the site, distance
        and rupture contexts for individual records. Each context dict
        represents an earthquake event and is of the form:
        ```
        {
         'EventID': earthquake id,
         'Sites': :class:`openquake.hazardlib.contexts.SitesContext`,
         'Distances': :class:`openquake.hazardlib.contexts.DistancesContext`,
         'Rupture': :class:`openquake.hazardlib.contexts.RuptureContext`
        }
        Additionally, if `imts` is not None but a list of Intensity measure
        types (strings), each dict will contain two additional keys:
        'Observation' (dict of imts mapped to a numpy array of imt values,
        one per record) and 'Num. Sites' (the records count)
        ```
        """
        # contexts are dicts which will temporarily be stored in a wrapping
        # dict `context_dicts` to easily retrieve a context from a record
        # event_id:
        context_dicts = {}
        compute_observations = imts is not None and len(imts)
        for rec in self.records:
            evt_id = self.get_record_eventid(rec)  # rec['event_id']
            dic = context_dicts.get(evt_id, None)
            if dic is None:
                # we might use defaultdict, but like this is more readable
                dic = {
                    'EventID': evt_id,
                     # 'EventIndex': [],  # FIXME: we do not use it right?
                    'Sites': self.create_sites_context(),
                    'Distances': self.create_distances_context(),
                    'Rupture': self.create_rupture_context()
                }
                if compute_observations:
                    dic["Observations"] = self.create_observations_dict(imts)
                    dic["Num. Sites"] = 0
                # set Rupture only once:
                self.update_rupture_context(rec, dic['Rupture'],
                                            nodal_plane_index)
                # assign to wrapping `context_dicts`:
                context_dicts[evt_id] = dic

            # dic['EventIndex'].append(self.get_record_id(rec))
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

    def create_sites_context(self):
        '''Creates, initializes and returns a sites context by setting the
        default values of the attributes defined in `self.sites_context_attrs`.
        The returned context is intended to be used in `self.get_contexts`.

        :return:  a :class:`openquake.hazardlib.contexts.SitesContext`
        '''
        ctx = SitesContext()
        for _ in self.sites_context_attrs:
            setattr(ctx, _, [])
        return ctx

    def create_distances_context(self):
        '''Creates, initializes and returns a distances context by setting the
        default values of the attributes defined in `self.distances_context_attrs`.
        The returned context is intended to be used in `self.get_contexts`.

        :return:  a :class:`openquake.hazardlib.contexts.DistancesContext`
        '''
        ctx = DistancesContext()
        for _ in self.distances_context_attrs:
            setattr(ctx, _, [])
        return ctx

    def create_rupture_context(self, evt_id):
        '''Creates, initializes and returns a rupture context by setting the
        default values of the attributes defined in `self.rupture_context_attrs`.
        The returned context is intended to be used in `self.get_contexts`.

        :return:  a :class:`openquake.hazardlib.contexts.RuptureContext`
        '''
        ctx = RuptureContext()
        for _ in self.rupture_context_attrs:
            setattr(ctx, _, np.nan)
        return ctx

    def finalize_sites_context(self, context):
        '''Finalizes the `context` object created with
        `self.create_sites_context` and populated in `self.get_contexts`.
        All operations should be performed inplace on `context`, this method
        is not expected to return any value

        :param context: a :class:`openquake.hazardlib.contexts.SitesContext`
        '''
        for attname in self.sites_context_attrs:
            attval = getattr(context, attname)
            # remove attribute if its value is empty-like
            if attval is None or not len(attval):
                delattr(context, attname)
            else:
                # FIXME: dtype=float forces Nones to be safely converted to nan
                # but it assumes obviously all attval elements to be numeric
                setattr(context, attname, np.asarray(attval, dtype=float))

    def finalize_distances_context(self, context):
        '''Finalizes the `context` object created with
        `self.create_distances_context` and populated in `self.get_contexts`.
        All operations should be performed inplace on `context`, this method
        is not expected to return any value

        :param context: a :class:`openquake.hazardlib.contexts.DistancesContext`
        '''
        for attname in self.distances_context_attrs:
            attval = getattr(context, attname)
            # remove attribute if its value is empty-like
            if attval is None or not len(attval):
                delattr(context, attname)
            else:
                # FIXME: dtype=float forces Nones to be safely converted to nan
                # but it assumes obviously all attval elements to be numeric
                setattr(context, attname, np.asarray(attval, dtype=float))

    def finalize_rupture_context(self, context):
        '''Finalizes the `context` object created with
        `self.create_rupture_context` and populated in `self.get_contexts`.
        All operations should be performed inplace on `context`, this method
        is not expected to return any value

        :param context: a :class:`openquake.hazardlib.contexts.RuptureContext`
        '''
        pass

    def get_observations(self, imts, component="Geometric"):
        """Get the observed intensity measure values from the database records,
        returning a dict mapping each imt in `imts` (iterable of strings) to
        the numpy array of the records intensity measure values.
        This method is implemented for legacy code compatibility: it is not
        called by `self.get_context`, although its returned value is the same
        of the 'Observation' key returned by `self.get_context`
        """
        observations = self.create_observations_dict(imts)
        for record in self.records:
            self.update_observations(observations, record, component)
        self.finalize_observations_dict(observations)
        return observations

    def create_observations(self, imts):
        '''creates and returns an observations `dict` from the given imts'''
        return OrderedDict([(imtx, []) for imtx in imts])

    def finalize_observations(self, observations):
        '''
        finalizes a the observations `dict` after it has been populated
        with the database observed values. By default, it converts all dict
        values to numpy array
        '''
        for imtx, values in observations.items():
            observations[imtx] = np.asarray(values, dtype=float)


class GroundMotionDatabase(ResidualsCompliantRecordsCollection):

    def __init__(self, db_id, db_name, db_directory=None, records=[],
                 site_ids=[]):
        """
        """
        self.id = db_id
        self.name = db_name
        self.directory = db_directory
        self._records = [rec for rec in records]
        self.site_ids = site_ids

    ####################################################
    # ABSTRACT METHODS TO BE IMPLEMENTED IN SUBCLASSES #
    ####################################################

    @property
    def records(self):
        '''Returns an iterable of records from this database (e.g. list, tuple,
        generator). Note that as any iterable, the returned object does not
        need to define a length whereby `len(self.records)` will work: this
        will depend on subclasses implementation
        '''
        return self._records

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
                    np.sqrt(sm_utils.DEFAULT_MSR.get_median_area(context.mag,
                                                                 0))

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
        for imtx in self.imts:
            if imtx in self.SCALAR_IMTS:
                observations[imtx].append(
                    self.get_scalar(fle, imtx, component))
            elif "SA(" in imtx:
                target_period = imt.from_string(imtx).period
                spectrum = fle[selection_string + component +
                               "/damping_05"].value
                periods = fle["IMS/H/Spectra/Response/Periods"].value
                observations[imtx].append(get_interpolated_period(
                    target_period, periods, spectrum))
            else:
                raise ValueError("IMT %s is unsupported!" % imtx)
        fle.close()

    ###########################
    # END OF ABSTRACT METHODS #
    ###########################

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
            return SCALAR_XY[component](x_im, y_im)
        else:
            if i_m in fle["IMS/H/Scalar"].keys():
                return fle["IMS/H/Scalar/" + i_m].value[0]
            else:
                raise ValueError("Scalar IM %s not in record database" % i_m)


class GroundMotionTable(ResidualsCompliantRecordsCollection):

    def get_contexts(self, nodal_plane_index=1, imts=None, component="Geometric"):
        '''
        overrides super implementation to execute it inside a `with self`
        opening the underlying hdf and assuring it is not already open
        '''
        with self:
            super().get_contexts(nodal_plane_index, imts, component)

    ####################################################
    # ABSTRACT METHODS TO BE IMPLEMENTED IN SUBCLASSES #
    ####################################################
    
    @property
    def records(self):
        '''Yields an iterator of the records according to the specified filter
        `condition`. The underlying HDF file (including each yielded record)
        must not be modified while accessing this property, and thus must
        be opened in read mode.
        ```
        with GroundMotionTable(filepath, name, 'r').filter(condition) as dbase:
            # ... do your operation here
            for record in dbase.records:
                ...
        ```
        '''
        return records_where(self.table, self._condition)

    def get_record_eventid(self, record):
        '''Returns the record event id (usually int) of the given record'''
        return record['event_id']

    def update_sites_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.SitesContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g. `context.vs30.append(record.vs30)`
        '''
        isnan = np.isnan
        context.lons.append(record['station_longitude'])
        context.lats.append(record['station_latitude'])
        context.depths.append(0.0 if isnan(record['station_elevation'])
                              else record['station_elevation'] * -1.0E-3)
        vs30 = record['vs30']
        context.vs30.append(vs30)
        context.vs30measured.append(record['vs30_measured'])
        context.z1pt0.append(vs30_to_z1pt0_cy14(vs30)
                             if isnan(record['z1']) else record['z1'])
        context.z2pt5.append(vs30_to_z2pt5_cb14(vs30)
                             if isnan(record['z2pt5']) else record['z2pt5'])
        context.backarc.append(record['backarc'])

    def update_distances_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.DistancesContext`
        object. In the typical implementation it has the attributes defined in
        `self.distances_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value,
        e.g. `context.rjb.append(record.rjb)`
        '''
        isnan = np.isnan
        # TODO Setting Rjb == Repi and Rrup == Rhypo when missing value
        # is a hack! Need feedback on how to fix
        context.repi.append(record['repi'])
        context.rhypo.append(record['rhypo'])
        context.rjb.append(record['repi'] if isnan(record['rjb']) else
                           record['rjb'])
        context.rrup.append(record['rhypo'] if isnan(record['rrup']) else
                            record['rrup'])
        context.rx.append(-record['repi'] if isnan(record['rx']) else
                          record['rx'])
        context.ry0.append(record['repi'] if isnan(record['ry0']) else
                           record['ry0'])
        context.rcdpp.append(0.0)
        context.rvolc.append(0.0)
        context.azimuth.append(record['azimuth'])

    def update_rupture_context(self, record, context, nodal_plane_index=1):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.RuptureContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to NaN (numpy.nan).
        Here you should set those attributes with the relative record value,
        e.g. `context.mag = record.event.mag`
        '''
        # FIXME: nodal_plane_index not used??
        isnan = np.isnan

        strike, dip, rake = \
            record['strike_1'], record['dip_1'], record['rake_1']

        if np.isnan([strike, dip, rake]).any():
            strike, dip, rake = \
                record['strike_2'], record['dip_2'], record['rake_2']

        if np.isnan([strike, dip, rake]).any():
            strike = 0.0
            dip = 90.0
            try:
                sof = record['style_of_faulting']
                # might be bytes:
                if hasattr(sof, 'decode'):
                    sof = sof.decode('utf8')
                rake = MECHANISM_TYPE[sof]
                dip = DIP_TYPE[sof]
            except KeyError:
                rake = 0.0

        context.mag = record['magnitude']
        context.strike = strike
        context.dip = dip
        context.rake = rake
        context.hypo_depth = record['hypocenter_depth']
        _ = record['depth_top_of_rupture']
        context.ztor = context.hypo_depth if isnan(_) else _
        rec_width = record['rupture_width']
        if np.isnan(rec_width):
            rec_width = np.sqrt(DEFAULT_MSR.get_median_area(context.mag, 0))
        context.width = rec_width
        context.hypo_lat = record['event_latitude']
        context.hypo_lon = record['event_longitude']
        context.hypo_loc = (0.5, 0.5)

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
        has_imt_components = self.has_imt_components
        scalar_func = SCALAR_XY[component]
        sa_periods = self.table.attrs.sa_periods
        for imtx in observations.keys():
            value = np.nan
            components = [np.nan, np.nan]
            if "SA(" in imtx:
                target_period = imt.from_string(imtx).period
                if has_imt_components:
                    spectrum = record['sa_components'][:2]
                    components[0] = get_interpolated_period(target_period,
                                                            sa_periods,
                                                            spectrum[0])
                    components[1] = get_interpolated_period(target_period,
                                                            sa_periods,
                                                            spectrum[1])
                else:
                    spectrum = record['sa']
                    value = get_interpolated_period(target_period, sa_periods,
                                                    spectrum)
            else:
                imtx_ = imtx.lower()
                if has_imt_components:
                    components = record['%s_components' % imtx_][:2]
                value = record[imtx_]

            if has_imt_components:
                value = scalar_func(*components)
            observations[imtx].append(value)

    ###########################
    # END OF ABSTRACT METHODS #
    ###########################
