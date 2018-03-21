import os
import unittest
from smtk.parsers.asa_database_parser import ASADatabaseMetadataReader
from openquake.hazardlib import valid


#  Test that metadata is parsed correctly (case when it metadata is given)
#  Consider two ASA files, one from UNAM and the other from CICESE
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class ASA_MetadataParsertest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filepath = os.path.join(BASE_DATA_PATH,
                                os.path.join("correct_ASA_files"))
        instance = ASADatabaseMetadataReader(db_id='1',
                                             db_name='db',
                                             filename=filepath,
                                             record_folder=filepath)
        cls.database = instance.parse()
        del instance

    def test_pref_mags(self):
        parsed_mags = []
        for i in self.database.records:
            parsed_mags.append(i.event.magnitude.value)
        self.assertEqual(parsed_mags, [4.8, 7.2])

    def test_event_coords(self):
        parsed_lons = []
        for i in self.database.records:
            parsed_lons.append(str(i.event.longitude))
        parsed_lats = []
        for i in self.database.records:
            parsed_lats.append(str(i.event.latitude))
        self.assertEqual(valid.longitudes(','.join(parsed_lons)),
                         [101.089, 115.33])
        self.assertEqual(valid.latitudes(','.join(parsed_lats)),
                         [17.446, 32.32])

    def test_site_coords(self):
        parsed_lons = []
        for i in self.database.records:
            parsed_lons.append(str(i.site.longitude))
        parsed_lats = []
        for i in self.database.records:
            parsed_lats.append(str(i.site.latitude))
        self.assertEqual(valid.longitudes(','.join(parsed_lons)),
                         [99.85157, 116.301])
        self.assertEqual(valid.latitudes(','.join(parsed_lats)),
                         [16.84851, 32.02])

    def test_event_id(self):
        parsed_evt_ids = []
        for i in self.database.records:
            parsed_evt_ids.append(i.event.id)
        self.assertEqual(parsed_evt_ids,
                         ['1989-03-10_05:19:51', '2010-04-04_22:40:04'])

    def test_instrument(self):
        parsed_instrument_types = []
        for i in self.database.records:
            parsed_instrument_types.append(i.site.instrument_type)
        self.assertEqual(parsed_instrument_types, ['DCA-333', 'GMS-18'])

    def test_morphology(self):
        parsed_morphology = []
        for i in self.database.records:
            parsed_morphology.append(i.site.morphology)
        self.assertEqual(
            parsed_morphology, ['ARENA - LIMO - ARCILLA  ',
                                'Rocas graniticas no diferenciadas'])

    @classmethod
    def tearDownClass(cls):
        cls.database = None
