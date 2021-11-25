"""
Implementing the abstract-like interface to be inherited by any
database/set/collection aiming to support residuals computation on its records
"""
from collections import OrderedDict  # FIXME In Python3.7+, dict is sufficient

import numpy as np
from openquake.hazardlib.contexts import DistancesContext, RuptureContext


class ContextsDB:
    """This abstract-like class represents a Database (DB) of Contexts suitable
    for Residual analysis: subclasses of `ContextDB` can be passed as argument
    to :meth:`gmpe_residuals.Residuals.get_residuals`

    Typical implementation sketch:
    ```
        def get_context(nodal_plane_index=1, imts=None, component="Geometric"):

            # loop through

    ```
    To implement a custom `ContextDB` you have to subclass `get_contexts`.
    Therein, you iteratively create `Context` dictionaries via `create_context`,
    and then populate the dict with rupture, site, distance information, as
    well as observed imt, if present.

    For subclassing examples, see:
    :class:`smtk.sm_database.GroundMotionDatabase`
    """

    # SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
    # SCALAR_IMTS = ["PGA", "PGV"]

    sites_context_attrs = ('vs30', 'lons', 'lats', 'depths',
                           'vs30measured', 'z1pt0', 'z2pt5', 'backarc')
    distances_context_attrs = tuple(DistancesContext._slots_)  # noqa
    rupture_context_attrs = tuple(RuptureContext._slots_)  # noqa

    def get_contexts(self, nodal_plane_index=1,
                     imts=None, component="Geometric"):
        """Returns an iterable of `dict`s, each `dict` describing a given
        earthquake-based context "Ctx" (with sites, distances and rupture information)
        and optionally its observed IMT values "Observations". This method is used by
        :meth:`gmpe_residuals.Residuals.get_residuals`
        and should not be overwritten unless for very specific reasons.
        """
        # contexts are dicts which will temporarily be stored in a wrapping
        # dict `context_dicts` to easily retrieve a context from a record
        # event_id:
        # context_dicts = {}
        compute_observations = imts is not None and len(imts)
        for evt_id, records in self.event_records_iter():
            dic = self.create_context(evt_id, imts)
            ctx = dic['Ctx']
            self.update_context(ctx, records, nodal_plane_index)
            if compute_observations:
                observations = dic['Observations']
                for imtx, values in observations.items():
                    values = self.get_observations(imtx, records, component)
                    observations[imtx] = np.asarray(values, dtype=float)
                dic["Num. Sites"] = len(records)
            # Legacy code??  FIXME: kind of redundant with "Num. Sites" above
            dic['Ctx'].sids = np.arange(len(records), dtype=np.uint32)
            yield dic

        # converts to numeric arrays (once at the end is faster, see
        # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
        # get default attributes not to be changed:
        # for dic in context_dicts.values():
        #     self.finalize_context(dic)

#        return context_dicts.values()

    def create_context(self, evt_id, imts=None):
        """Create a new context dict.

        :param evt_id: the event id (e.g. int, or str)
        :param imts: a list of strings denoting the imts to be included in the
            context. If missing or None, the returned dict **will NOT** have
            the keys "Observations" and "Num. Sites"

        :return: the dict with keys:
            ```
            {
            'EventID': evt_id,
            'Ctx: a new :class:`openquake.hazardlib.contexts.RuptureContext`
            'Observations": dict[str, list] # (each imt in imts mapped to `[]`)
            'Num. Sites': 0
            }
            ```
            NOTE: Remember 'Observations' and 'Num. Sites' are missing if `imts`
            is missing, None or an emtpy sequence.
        """
        dic = {
            'EventID': evt_id,
            'Ctx': RuptureContext()
        }
        if imts is not None and len(imts):
            dic["Observations"] = OrderedDict([(imt, []) for imt in imts])
            dic["Num. Sites"] = 0
        return dic


# class ResidualsCompliantRecordSet:
#     '''This abstract-like class represents an iterables of records which can be
#     used in :meth:`gmpe_residuals.Residuals.get_residuals` to compute the
#     records residuals. Subclasses need to implement few abstract-like methods
#     defining how to get the input data required for the residuals computation
#     For subclassing examples, see:
#     :class:`smtk.sm_database.GroundMotionDatabase`
#     :class:`smtk.sm_table.GroundMotionTable`
#     '''

    # SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
    # SCALAR_IMTS = ["PGA", "PGV"]
    #
    # sites_context_attrs = ('vs30', 'lons', 'lats', 'depths',
    #                        'vs30measured', 'z1pt0', 'z2pt5', 'backarc')
    # distances_context_attrs = tuple(DistancesContext._slots_)
    # rupture_context_attrs = tuple(RuptureContext._slots_)

    ####################################################
    # ABSTRACT METHODS TO BE IMPLEMENTED IN SUBCLASSES #
    ####################################################

    # @property
    # def records(self):
    #     '''Returns an iterable of records from this database (e.g. list, tuple,
    #     generator). Note that as any iterable, the returned object does not
    #     need to define a length whereby `len(self.records)` will work: this
    #     will depend on subclasses implementation
    #     '''
    #     raise NotImplementedError('')
    #
    # def get_record_eventid(self, record):
    #     '''Returns the record event id (usually int) of the given record'''
    #     raise NotImplementedError('')

    def event_records_iter(self):
        """Yield the tuple (event_id:Any, records:Sequence)

        where:
         - `event_id` is a scalar denoting the unique earthquake id, and
         - `records` are the database records related to the given event. `records`
            must be a sequence with given length (`len(records)` must work) and
            its elements are user-defined according to the type of dataset
            implemented. For instance, `records` can be a pandas DataFrame, or
            a list of: `dict`s, `h5py` Datasets, `pytables` rows, `django` model
            instances, and so on
        """
        raise NotImplementedError('')

    def update_context(self, ctx, records, nodal_plane_index=1):
        """Update the attributes of the earthquake-related context `ctx` with
        the earthquake data `records`.
        See `rupture_context_attrs`, `sites_context_attrs`,
        `distances_context_attrs`, for a list of possible attributes. Any
        attribute of `ctx` that is non-scalar should be given as numpy array.

        :param: a :class:`openquake.hazardlib.contexts.RuptureContext`
            created for a specific event in :meth:`event_records_iter`
        :param records: sequence (e.g., list, tuple, pandas DataFrame) of records
            related to the given context `ctx` (see :meth:`event_records_iter`)
        """
        raise NotImplementedError('')

    def get_observations(self, imtx, records, component="Geometric"):
        """Return the observed values of the given IMT `imtx` from `records`,
        as numpy array. This method is not called `get_contexts` if the latter
        is called with the `imts` argument missing or `None`.

        *IMPORTANT*: IMTs in acceleration units (e.g. PGA, SA) are supposed to
        return their values in cm/s/s (which should be by convention the unit
        in which they are stored)

        :param imtx: a string denoting a given Intensity measure type.
        :param records: sequence (e.g., list, tuple, pandas DataFrame) of records
            related to the given context `ctx` (see :meth:`event_records_iter`)

        :return: a numpy array the same length of records
        """
        # NOTE: imtx otherwise it shadows paclkage imt!!
        raise NotImplementedError('')

    # def update_distances_context(self, ctx, records):
    #     """Update the attributes of the earthquake-related context `ctx` with
    #     the earthquake data `records`.
    #     See `distances_context_attrs` for a list of possible attributes. Any
    #     attribute of `ctx` that is non-scalar should be given as numpy array.
    #
    #     :param: a :class:`openquake.hazardlib.contexts.RuptureContext`
    #         created for a specific event in :meth:`event_records_iter`
    #     :param records: sequence (e.g., list, tuple, pandas DataFrame) of records
    #         related to the given context `ctx` (see :meth:`event_records_iter`)
    #     """
    #     raise NotImplementedError('')
    #
    # def update_rupture_context(self, ctx, records, nodal_plane_index=1):
    #     """Update the attributes of the earthquake-related context `ctx` with
    #     the earthquake data `records`.
    #     See `rupture_context_attrs` for a list of possible attributes. Any
    #     attribute of `ctx` that is non-scalar should be given as numpy array.
    #
    #     :param: a :class:`openquake.hazardlib.contexts.RuptureContext`
    #         created for a specific event in :meth:`event_records_iter`
    #     :param records: sequence (e.g., list, tuple, pandas DataFrame) of records
    #         related to the given context `ctx` (see :meth:`event_records_iter`)
    #     """
    #     raise NotImplementedError('')

    #################################
    # END OF ABSTRRACT-LIKE METHODS #
    #################################

    # def __iter__(self):
    #     """
    #     Make this object iterable, i.e.
    #     `for rec in self` is equal to `for rec in self.records`
    #     """
    #     for record in self.records:
    #         yield record

    # def finalize_context(self, context):
    #     '''Finalizes the `context` object created with
    #     `self.create_sites_context` and populated in `self.get_contexts`.
    #     All operations should be performed inplace on `context`, this method
    #     is not expected to return any value
    #
    #     :param context: a :class:`openquake.hazardlib.contexts.SitesContext`
    #     '''
    #     ctx = context['Ctx']
    #     for attname in self.sites_context_attrs:
    #         attval = getattr(ctx, attname)
    #         # remove attribute if its value is empty-like
    #         if attval is None or not len(attval):
    #             delattr(ctx, attname)
    #         elif attname in ('vs30measured', 'backarc'):
    #             setattr(ctx, attname, np.asarray(attval, dtype=bool))
    #         else:
    #             # dtype=float forces Nones to be safely converted to nan
    #             setattr(ctx, attname, np.asarray(attval, dtype=float))
    #
    #     for attname in self.distances_context_attrs:
    #         attval = getattr(ctx, attname)
    #         # remove attribute if its value is empty-like
    #         if attval is None or not len(attval):
    #             delattr(ctx, attname)
    #         else:
    #             # FIXME: dtype=float forces Nones to be safely converted to nan
    #             # but it assumes obviously all attval elements to be numeric
    #             setattr(ctx, attname, np.asarray(attval, dtype=float))
    #     ctx.sids = np.arange(len(ctx.vs30), dtype=np.uint32)
    #
    #     observations = context.get('Observations', None)
    #     if observations:
    #         for imtx, values in observations.items():
    #             observations[imtx] = np.asarray(values, dtype=float)

    # def create_observations(self, imts):
    #     '''creates and returns an observations `dict` from the given imts'''
    #     return OrderedDict([(imtx, []) for imtx in imts])

    # def finalize_observations(self, observations):
    #     '''Finalizes a the observations `dict` after it has been populated
    #     with the database observed values. By default, it converts all dict
    #     values to numpy array
    #     '''
    #     for imtx, values in observations.items():
    #         observations[imtx] = np.asarray(values, dtype=float)

    # def get_observations(self, imts, component="Geometric"):
    #     """Get the observed intensity measure values from the database records,
    #     returning a `dict` of imts mapped to a numpy array of the imt values,
    #     one per record.
    #     This method is implemented for legacy code compatibility and it is not
    #     called by `self.get_context`, although it returns the same value as
    #     `self.get_context(..., imts, component)["Observations"]`
    #     """
    #     observations = self.create_observations(imts)
    #     for record in self.records:
    #         self.update_observations(record, observations, component)
    #     self.finalize_observations(observations)
    #     return observations



    # def get_contexts(self, nodal_plane_index=1,
    #                  imts=None, component="Geometric"):
    #     """Returns an iterable of `dict`s, each containing the site, distance
    #     and rupture contexts for individual records. Each context dict
    #     represents an earthquake event and is of the form:
    #     ```
    #     {
    #      'EventID': earthquake id,
    #      'Ctx': :class:`openquake.hazardlib.contexts.RuptureContext`
    #     }
    #     ```
    #     Additionally, if `imts` is not None but a list of Intensity measure
    #     types (strings), each dict will contain two additional keys:
    #     'Observations' (dict of imts mapped to a numpy array of imt values,
    #     one per record) and 'Num. Sites' (the records count)
    #     ```
    #     """
    #     # contexts are dicts which will temporarily be stored in a wrapping
    #     # dict `context_dicts` to easily retrieve a context from a record
    #     # event_id:
    #     context_dicts = {}
    #     compute_observations = imts is not None and len(imts)
    #     for rec in self.records:
    #         evt_id = self.get_record_eventid(rec)  # rec['event_id']
    #         dic = context_dicts.get(evt_id, None)
    #         if dic is None:
    #             dic = self.create_context(evt_id, imts)
    #             # we might use defaultdict, but like this is more readable
    #             # dic = {
    #             #     'EventID': evt_id,
    #             #     'Ctx': self.create_context(evt_id)
    #             # }
    #             # if compute_observations:
    #             #     dic["Observations"] = self.create_observations(imts)
    #             #     dic["Num. Sites"] = 0
    #
    #             # set Rupture only once:
    #             self.update_rupture_context(rec, dic['Ctx'],
    #                                         nodal_plane_index)
    #             # assign to wrapping `context_dicts`:
    #             context_dicts[evt_id] = dic
    #
    #         # dic['EventIndex'].append(self.get_record_id(rec))
    #         self.update_sites_context(rec, dic['Ctx'])
    #         self.update_distances_context(rec, dic['Ctx'])
    #         if compute_observations:
    #             self.update_observations(rec, dic['Observations'], component)
    #             dic["Num. Sites"] += 1
    #
    #     # converts to numeric arrays (once at the end is faster, see
    #     # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    #     # get default attributes not to be changed:
    #     for dic in context_dicts.values():
    #         self.finalize_context(dic)
    #
    #     return context_dicts.values()