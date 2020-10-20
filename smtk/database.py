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
Abstract class for a ground motion database. Any container of observations
used for the residuals calculation should inherit from :class:`GMDatabase`
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
    convert_accel_units


class ResidualsDatabase:
    
    SCALAR_IMTS = ["PGA", "PGV"]

    sites_context_attrs = ('vs30', 'lons', 'lats', 'depths',
                           'vs30measured', 'z1pt0', 'z2pt5', 'backarc')
    distances_context_attrs = tuple(DistancesContext._slots_)
    rupture_context_attrs = tuple(RuptureContext._slots_)

    ####################################################
    # ABSTRACT METHODS TO BE IMPLEMENTED IN SUBCLASSES #
    ####################################################

    def get_record_eventid(self, record):
        '''Returns the record event id (usually int) of the given record'''
        raise NotImplementedError('')

#     def get_record_id(self, record):
#         '''Returns the unique id (usually int) of the given record'''
#         raise NotImplementedError('')

    def update_sites_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.SitesContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value:
        ```
            vs30 = ... get the vs30 from `record` ...
            context.vs30.append(vs30)
            ... and so on ...
        ```
        '''
        raise NotImplementedError('')

    def update_distances_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.DistancesContext`
        object. In the typical implementation it has the attributes defined in
        `self.distances_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value:
        ```
            rjb = ... get the rjb from `record` ...
            context.rjb.append(rjb)
            ... and so on ...
        ```
        '''
        raise NotImplementedError('')

    def update_rupture_context(self, record, context, nodal_plane_index=1):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.RuptureContext`
        object. In the typical implementation it  has the attributes defined in
        `self.sites_context_attrs` all initialized to NaN (numpy.nan).
        Here you should set those attributes with the relative record value:
        ```
            mag = ... get the magnitude from `record` ...
            context.mag = mag
            ... and so on ...
        ```
        '''
        raise NotImplementedError('')

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
        raise NotImplementedError('')

    #################################
    # END OF ABSTRRACT-LIKE METHODS #
    #################################

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
                     # 'EventIndex': [],  # FIXME: we do not use it right?
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


class GroundMotionDatabase(object):

    ####################################################
    # ABSTRACT METHODS TO BE IMPLEMENTED IN SUBCLASSES #
    ####################################################

    def get_record_eventid(self, record):
        '''Returns the record event id (usually int) of the given record'''
        return record.event.id

    def update_sites_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.SitesContext`
        object. In the typical implementation it has the attributes defined in
        `self.sites_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value:
        ```
            vs30 = ... get the vs30 from `record` ...
            context.vs30.append(vs30)
            ... and so on ...
        ```
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
        if ("backarc" in dir(record.site)) and record.site.backarc is not None:
            context.backarc.append(record.site.backarc)

    def update_distances_context(self, record, context):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.DistancesContext`
        object. In the typical implementation it has the attributes defined in
        `self.distances_context_attrs` all initialized to empty lists.
        Here you should append to those attributes the relative record value:
        ```
            rjb = ... get the rjb from `record` ...
            context.rjb.append(rjb)
            ... and so on ...
        ```
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

    def update_rupture_context(self, rup, rctx, nodal_plane_index=1):
        '''Updates the attributes of `context` with the given `record` data.
        `context` is a :class:`openquake.hazardlib.contexts.RuptureContext`
        object. In the typical implementation it  has the attributes defined in
        `self.sites_context_attrs` all initialized to NaN (numpy.nan).
        Here you should set those attributes with the relative record value:
        ```
            mag = ... get the magnitude from `record` ...
            context.mag = mag
            ... and so on ...
        ```
        '''
        rctx.mag = rup.event.magnitude.value
        if nodal_plane_index == 2:
            rctx.strike = \
                rup.event.mechanism.nodal_planes.nodal_plane_2['strike']
            rctx.dip = \
                rup.event.mechanism.nodal_planes.nodal_plane_2['dip']
            rctx.rake = \
                rup.event.mechanism.nodal_planes.nodal_plane_2['rake']
        elif nodal_plane_index == 1:
            rctx.strike = \
                rup.event.mechanism.nodal_planes.nodal_plane_1['strike']
            rctx.dip = \
                rup.event.mechanism.nodal_planes.nodal_plane_1['dip']
            rctx.rake = \
                rup.event.mechanism.nodal_planes.nodal_plane_1['rake']
        else:
            rctx.strike = 0.0
            rctx.dip = 90.0
            rctx.rake = rup.event.mechanism.get_rake_from_mechanism_type()

        if rup.event.rupture.surface:
            rctx.ztor = rup.event.rupture.surface.get_top_edge_depth()
            rctx.width = rup.event.rupture.surface.width
            rctx.hypo_loc = rup.event.rupture.surface.get_hypo_location(1000)
        else:
            if rup.event.rupture.depth is not None:
                rctx.ztor = rup.event.rupture.depth
            else:
                rctx.ztor = rup.event.depth

            if rup.event.rupture.width is not None:
                rctx.width = rup.event.rupture.width
            else:
                # Use the PeerMSR to define the area and assuming an aspect ratio
                # of 1 get the width
                rctx.width = \
                    np.sqrt(sm_utils.DEFAULT_MSR.get_median_area(rctx.mag, 0))

            # Default hypocentre location to the middle of the rupture
            rctx.hypo_loc = (0.5, 0.5)
        rctx.hypo_depth = rup.event.depth
        rctx.hypo_lat = rup.event.latitude
        rctx.hypo_lon = rup.event.longitude

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
