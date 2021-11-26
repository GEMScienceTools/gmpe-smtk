"""
Module defining the interface of a Context Database (ContextDB), a database of
data capable of yielding Contexts and Observations suitable for Residual analysis
"""
from collections import OrderedDict  # compatibility with dicts in Python <3.7

import numpy as np
from openquake.hazardlib.contexts import DistancesContext, RuptureContext


class ContextDB:
    """This abstract-like class represents a Database (DB) of data capable of
    yielding Contexts and Observations suitable for Residual analysis (see
    argument `ctx_database` of :meth:`gmpe_residuals.Residuals.get_residuals`)

    Concrete subclasses of `ContextDB` must implement three abstract methods
    (e.g. :class:`smtk.sm_database.GroundMotionDatabase`):
     - get_event_and_records(self)
     - update_context(self, ctx, records, nodal_plane_index=1)
     - get_observations(self, imtx, records, component="Geometric")
       (which is called only if `imts` is given in :meth:`self.get_contexts`)

    Please refer to the functions docstring for further details
    """

    rupture_context_attrs = tuple(RuptureContext._slots_)  # noqa
    distances_context_attrs = tuple(DistancesContext._slots_)  # noqa
    sites_context_attrs = ('vs30', 'lons', 'lats', 'depths',
                           'vs30measured', 'z1pt0', 'z2pt5', 'backarc')

    def get_contexts(self, nodal_plane_index=1,
                     imts=None, component="Geometric"):
        """Return an iterable of Contexts. Each Context is a `dict` with
        earthquake, sites and distances information (`dict["Ctx"]`)
        and optionally arrays of observed IMT values (`dict["Observations"]`).
        See `create_context` for details.

        This is the only method required by
        :meth:`gmpe_residuals.Residuals.get_residuals`
        and should not be overwritten only in very specific circumstances.
        """
        compute_observations = imts is not None and len(imts)
        for evt_id, records in self.get_event_and_records():
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

    def create_context(self, evt_id, imts=None):
        """Create a new Context `dict`. Objects of this type will be yielded
        by `get_context`.

        :param evt_id: the earthquake id (e.g. int, or str)
        :param imts: a list of strings denoting the IMTs to be included in the
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

    def get_event_and_records(self):
        """Yield the tuple (event_id:Any, records:Sequence)

        where:
         - `event_id` is a scalar denoting the unique earthquake id, and
         - `records` are the database records related to the given event: it
            must be a sequence with given length (i.e., `len(records)` must
            work) and its elements can be any user-defined data type according
            to the current user implementations. For instance, `records` can be
            a pandas DataFrame, or a list of several data types such as `dict`s
            `h5py` Datasets, `pytables` rows, `django` model instances.
        """
        raise NotImplementedError('')

    def update_context(self, ctx, records, nodal_plane_index=1):
        """Update the attributes of the earthquake-based context `ctx` with
        the data in `records`.
        See `rupture_context_attrs`, `sites_context_attrs`,
        `distances_context_attrs`, for a list of possible attributes. Any
        attribute of `ctx` that is non-scalar should be given as numpy array.

        :param ctx: a :class:`openquake.hazardlib.contexts.RuptureContext`
            created for a specific event. It is the key 'Ctx' of the Context dict
            returned by `self.create_context`
        :param records: sequence (e.g., list, tuple, pandas DataFrame) of records
            related to the given event (see :meth:`get_event_and_records`)
        """
        raise NotImplementedError('')

    def get_observations(self, imtx, records, component="Geometric"):
        """Return the observed values of the given IMT `imtx` from `records`,
        as numpy array. This method is not called if `get_contexts`is called
        with the `imts` argument missing or `None`.

        *IMPORTANT*: IMTs in acceleration units (e.g. PGA, SA) are supposed to
        return their values in cm/s/s (which should be by convention the unit
        in which they are stored)

        :param imtx: a string denoting a given Intensity measure type.
        :param records: sequence (e.g., list, tuple, pandas DataFrame) of records
            related to a given event (see :meth:`get_event_and_records`)

        :return: a numpy array of observations, the same length of `records`
        """
        # NOTE: imtx, not imt, otherwise it shadows the package imt!!
        raise NotImplementedError('')
