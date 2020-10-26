"""
Implementing the abstract-like interface to be inherited by any
database/set/collection aiming to support residuals computation on its records

.. moduleauthor::  R. Zaccarelli
"""
import sys
from collections import OrderedDict  # FIXME In Python3.7+, dict is sufficient

import numpy as np
from openquake.hazardlib.contexts import DistancesContext, RuptureContext, \
    SitesContext


class ResidualsCompliantRecordSet:
    '''This abstract-like class represents an iterables of records which can be
    used in :meth:`gmpe_residuals.Residuals.get_residuals` to compute the
    records residuals. Subclasses need to implement few abstract-like methods
    defining how to get the input data required for the residuals computation
    For subclassing examples, see:
    :class:`smtk.sm_database.GroundMotionDatabase`
    :class:`smtk.sm_table.GroundMotionTable`
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
        return their values in cm/s/s (which should be by convention the unit
        in which they are stored)
        '''
        raise NotImplementedError('')

    #################################
    # END OF ABSTRRACT-LIKE METHODS #
    #################################

    def __iter__(self):
        """
        Make this object iterable, i.e.
        `for rec in self` is equal to `for rec in self.records`
        """
        for record in self.records:
            yield record

    def get_contexts(self, nodal_plane_index=1,
                     imts=None, component="Geometric"):
        """Returns an iterable of `dict`s, each containing the site, distance
        and rupture contexts for individual records. Each context dict
        represents an earthquake event and is of the form:
        ```
        {
         'EventID': earthquake id,
         'Sites': :class:`openquake.hazardlib.contexts.SitesContext`,
         'Distances': :class:`openquake.hazardlib.contexts.DistancesContext`,
         'Rupture': :class:`openquake.hazardlib.contexts.RuptureContext`
        }
        ```
        Additionally, if `imts` is not None but a list of Intensity measure
        types (strings), each dict will contain two additional keys:
        'Observations' (dict of imts mapped to a numpy array of imt values,
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
                    'Rupture': self.create_rupture_context(evt_id)
                }
                if compute_observations:
                    dic["Observations"] = self.create_observations(imts)
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

    def create_observations(self, imts):
        '''creates and returns an observations `dict` from the given imts'''
        return OrderedDict([(imtx, []) for imtx in imts])

    def finalize_observations(self, observations):
        '''Finalizes a the observations `dict` after it has been populated
        with the database observed values. By default, it converts all dict
        values to numpy array
        '''
        for imtx, values in observations.items():
            observations[imtx] = np.asarray(values, dtype=float)

    def get_observations(self, imts, component="Geometric"):
        """Get the observed intensity measure values from the database records,
        returning a `dict` of imts mapped to a numpy array of the imt values,
        one per record.
        This method is implemented for legacy code compatibility and it is not
        called by `self.get_context`, although it returns the same value as
        `self.get_context(..., imts, component)["Observations"]`
        """
        observations = self.create_observations(imts)
        for record in self.records:
            self.update_observations(record, observations, component)
        self.finalize_observations(observations)
        return observations
