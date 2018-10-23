"""
Test suite for the `residual_plotter` module responsible for plotting the
plot data defined in `residual_plots`
"""
import os
import sys
import shutil
import unittest
from unittest.mock import patch, MagicMock

from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser
import smtk.residuals.gmpe_residuals as res
from smtk.residuals.residual_plotter import ResidualPlot, LikelihoodPlot,\
    ResidualWithMagnitude, ResidualWithDepth, ResidualWithVs30, \
    ResidualWithDistance
from smtk.database_visualiser import DISTANCES


if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


EXPECTED_IDS = [
"EMSC_20040918_0000026_RA_PYAS_0", "EMSC_20040918_0000026_RA_PYAT_0",
"EMSC_20040918_0000026_RA_PYLI_0", "EMSC_20040918_0000026_RA_PYLL_0",
"EMSC_20041205_0000033_CH_BNALP_0", "EMSC_20041205_0000033_CH_BOURR_0",
"EMSC_20041205_0000033_CH_DIX_0", "EMSC_20041205_0000033_CH_EMV_0",
"EMSC_20041205_0000033_CH_LIENZ_0", "EMSC_20041205_0000033_CH_LLS_0",
"EMSC_20041205_0000033_CH_MMK_0", "EMSC_20041205_0000033_CH_SENIN_0",
"EMSC_20041205_0000033_CH_SULZ_0", "EMSC_20041205_0000033_CH_VDL_0",
"EMSC_20041205_0000033_CH_ZUR_0", "EMSC_20041205_0000033_RA_STBO_0",
"EMSC_20130103_0000020_HL_SIVA_0", "EMSC_20130103_0000020_HL_ZKR_0",
"EMSC_20130108_0000044_HL_ALNA_0", "EMSC_20130108_0000044_HL_AMGA_0",
"EMSC_20130108_0000044_HL_DLFA_0", "EMSC_20130108_0000044_HL_EFSA_0",
"EMSC_20130108_0000044_HL_KVLA_0", "EMSC_20130108_0000044_HL_LIA_0",
"EMSC_20130108_0000044_HL_NOAC_0", "EMSC_20130108_0000044_HL_PLG_0",
"EMSC_20130108_0000044_HL_PRK_0", "EMSC_20130108_0000044_HL_PSRA_0", 
"EMSC_20130108_0000044_HL_SMTH_0", "EMSC_20130108_0000044_HL_TNSA_0",
"EMSC_20130108_0000044_HL_YDRA_0", "EMSC_20130108_0000044_KO_ENZZ_0",
"EMSC_20130108_0000044_KO_FOCM_0", "EMSC_20130108_0000044_KO_GMLD_0",
"EMSC_20130108_0000044_KO_GOKC_0", "EMSC_20130108_0000044_KO_GOMA_0",
"EMSC_20130108_0000044_KO_GPNR_0", "EMSC_20130108_0000044_KO_KIYI_0",
"EMSC_20130108_0000044_KO_KRBN_0", "EMSC_20130108_0000044_KO_ORLT_0", 
"EMSC_20130108_0000044_KO_SHAP_0"]


class ResidualsTestCase(unittest.TestCase):
    """
    Core test case for the residuals objects
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup constructs the database from the ESM test data
        """
        ifile = os.path.join(BASE_DATA_PATH, "residual_tests_esm_data.csv")
        cls.out_location = os.path.join(BASE_DATA_PATH, "residual_tests")
        if os.path.exists(cls.out_location):
            shutil.rmtree(cls.out_location)
        parser = ESMFlatfileParser.autobuild("000", "ESM ALL",
                                             cls.out_location, ifile)
        del parser
        cls.database_file = os.path.join(cls.out_location,
                                         "metadatafile.pkl")
        cls.database = None
        with open(cls.database_file, "rb") as f:
            cls.database = pickle.load(f)
        cls.gsims = ["AkkarEtAlRjb2014",  "ChiouYoungs2014"]
        cls.imts = ["PGA", "SA(1.0)"]

    @patch('smtk.residuals.residual_plotter.plt.subplot')
    @patch('smtk.residuals.residual_plotter.plt')
    def tests_residual_plotter(self, mock_pyplot, mock_pyplot_subplot):
        """
        Tests basic execution of residual plot.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                ResidualPlot(residuals, gsim, imt, bin_width=0.1)
                # assert we called pyplot show:
                self.assertTrue(mock_pyplot.show.call_count == 1)
                ResidualPlot(residuals, gsim, imt, bin_width=0.1,
                             show=False)
                # assert we did NOT call pyplot show (call count still 1):
                self.assertTrue(mock_pyplot.show.call_count == 1)
                # reset mock:
                mock_pyplot.show.reset_mock()

                # assert we called the right matplotlib plotting functions:
                self.assertTrue(mocked_axes_obj.bar.called)
                self.assertTrue(mocked_axes_obj.plot.called)
                self.assertFalse(mocked_axes_obj.semilogx.called)
                # reset mock:
                mocked_axes_obj.reset_mock()

    @patch('smtk.residuals.residual_plotter.plt.subplot')
    @patch('smtk.residuals.residual_plotter.plt')
    def tests_likelihood_plotter(self, mock_pyplot, mock_pyplot_subplot):
        """
        Tests basic execution of Likelihood plotD.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                LikelihoodPlot(residuals, gsim, imt, bin_width=0.1)
                # assert we called pyplot show:
                self.assertTrue(mock_pyplot.show.call_count == 1)
                LikelihoodPlot(residuals, gsim, imt, bin_width=0.1,
                               show=False)
                # assert we did NOT call pyplot show (call count still 1):
                self.assertTrue(mock_pyplot.show.call_count == 1)
                # reset mock:
                mock_pyplot.show.reset_mock()

                # assert we called the right matplotlib plotting functions:
                self.assertTrue(mocked_axes_obj.bar.called)
                self.assertFalse(mocked_axes_obj.plot.called)
                self.assertFalse(mocked_axes_obj.semilogx.called)
                # reset mock:
                mocked_axes_obj.reset_mock()

    @patch('smtk.residuals.residual_plotter.plt.subplot')
    @patch('smtk.residuals.residual_plotter.plt')
    def tests_with_mag_vs30_depth_plotter(self, mock_pyplot,
                                          mock_pyplot_subplot):
        """
        Tests basic execution of residual with (magnitude, vs30, depth) plots.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                for plotClass in [ResidualWithMagnitude,
                                  ResidualWithDepth,
                                  ResidualWithVs30]:
                    plotClass(residuals, gsim, imt, bin_width=0.1)
                    # assert we called pyplot show:
                    self.assertTrue(mock_pyplot.show.call_count == 1)
                    plotClass(residuals, gsim, imt, bin_width=0.1, show=False)
                    # assert we did NOT call pyplot show (call count still 1):
                    self.assertTrue(mock_pyplot.show.call_count == 1)
                    # reset mock:
                    mock_pyplot.show.reset_mock()

                    # assert we called the right matplotlib plotting functions:
                    self.assertFalse(mocked_axes_obj.bar.called)
                    self.assertTrue(mocked_axes_obj.plot.called)
                    self.assertFalse(mocked_axes_obj.semilogx.called)

                    # check plot type:
                    plotClass(residuals, gsim, imt, plot_type='log',
                              bin_width=0.1, show=False)
                    self.assertTrue(mocked_axes_obj.semilogx.called)

                    # reset mock:
                    mocked_axes_obj.reset_mock()

    @patch('smtk.residuals.residual_plotter.plt.subplot')
    @patch('smtk.residuals.residual_plotter.plt')
    def tests_with_distance(self, mock_pyplot, mock_pyplot_subplot):
        """
        Tests basic execution of residual with distance plots.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                for dist in DISTANCES.keys():

                    if dist == 'r_x':
                        # as for residual_plots_test, we should confirm
                        # with scientific expertise that this is the case:
                        with self.assertRaises(AttributeError):
                            ResidualWithDistance(residuals, gsim, imt,
                                                 distance_type=dist,
                                                 show=True)
                        continue

                    ResidualWithDistance(residuals, gsim, imt, bin_width=0.1)
                    # assert we called pyplot show:
                    self.assertTrue(mock_pyplot.show.call_count == 1)
                    ResidualWithDistance(residuals, gsim, imt, bin_width=0.1,
                                         show=False)
                    # assert we did NOT call pyplot show (call count still 1):
                    self.assertTrue(mock_pyplot.show.call_count == 1)
                    # reset mock:
                    mock_pyplot.show.reset_mock()

                    # assert we called the right matplotlib plotting functions:
                    self.assertFalse(mocked_axes_obj.bar.called)
                    self.assertFalse(mocked_axes_obj.plot.called)
                    self.assertTrue(mocked_axes_obj.semilogx.called)

                    # check plot type:
                    ResidualWithDistance(residuals, gsim, imt,
                                         plot_type='', bin_width=0.1,
                                         show=False)
                    self.assertTrue(mocked_axes_obj.plot.called)

                    # reset mock:
                    mocked_axes_obj.reset_mock()

    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database
        """
        shutil.rmtree(cls.out_location)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
