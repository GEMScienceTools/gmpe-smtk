'''
NgaWest2 try (FIXME write doc)
'''
from smtk.gm_database import GMDatabaseParser


class NgaWest2(GMDatabaseParser):
    _mappings: {}

    @classmethod
    def process_flatfile_row(cls, rowdict):
        '''do any further processing of the given `rowdict`, a dict
        represenitng a parsed csv row. At this point, `rowdict` keys are
        already mapped to the :class:`GMDatabaseTable` columns (see `_mappings`
        class attribute), spectra values are already set in `rowdict['sa']`
        (interpolating csv spectra columns, if needed).
        This method should process `rowdict` in place, the returned value
        is ignored. Any exception is wrapped in the caller method.

        :param rowdict: a row of the csv flatfile, as Python dict. Values
            are strings and will be casted to the matching Table column type
            after this method call
        '''
        # convert event time from cells into a datetime string:
        evtime = cls.datetime(rowdict.pop('Year'),
                              rowdict.pop('Month'),
                              rowdict.pop('Day'),
                              rowdict.pop('Hour', None) or 0,
                              rowdict.pop('Minute', None) or 0,
                              rowdict.pop('Second', None) or 0)
        rowdict['event_time'] = evtime


class NgaEast2(GMDatabaseParser):
    _mappings: {}


class Esm(GMDatabaseParser):
    _mappings: {}


# class KikNet(GMDatabase):
#     _mappings: {}


class WeakMotion(GMDatabaseParser):
    _mappings: {}

