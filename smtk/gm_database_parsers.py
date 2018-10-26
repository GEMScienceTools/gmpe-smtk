'''
NgaWest2 try (FIXME write doc)
'''
from smtk.gm_database import GMDatabase


class NgaWest2(GMDatabase):
    _mappings: {}

    @classmethod
    def process_flatfile_row(cls, rowdict):
        '''do any further processing of the given rowdict.
        spectra are already set.
        This method should process rowdict in place, the returned value
        is ignored.
        Any exception is wrapped in the caller method. Try catch it here
        if you want different behaviour

        :param rowdict: a row of the csv flatfile, as Python dict. Values
            are strings and will be casted to the right Table column type
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


class NgaEast2(GMDatabase):
    _mappings: {}


class Esm(GMDatabase):
    _mappings: {}


# class KikNet(GMDatabase):
#     _mappings: {}


class WeakMotion(GMDatabase):
    _mappings: {}

