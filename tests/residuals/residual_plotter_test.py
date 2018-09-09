"""
Core test suite for the database and residuals construction
"""
import os
import sys
import shutil
import unittest
from unittest.mock import patch
import numpy as np

from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser
import smtk.residuals.gmpe_residuals as res
from smtk.residuals.residual_plots import residuals_density_distribution
from smtk.residuals.residual_plotter import ResidualPlot, LikelihoodPlot,\
    ResidualWithMagnitude, ResidualWithDepth, ResidualWithVs30, ResidualWithDistance
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

    def _plot_data_check(self, plot_data, plot_data_is_json,
                         additional_keys=None):
        allkeys = ['x', 'y', 'xlabel', 'ylabel'] + \
            ([] if not additional_keys else list(additional_keys))

        for res_type in plot_data:
            res_data = plot_data[res_type]
            # assert we have the specified keys:
            assert sorted(res_data.keys()) == sorted(allkeys)
            assert len(res_data['x']) == len(res_data['y'])
            assert isinstance(res_data['xlabel'], str)
            assert isinstance(res_data['ylabel'], str)
            array_type = list if plot_data_is_json else np.ndarray
            assert isinstance(res_data['x'], array_type)
            assert isinstance(res_data['y'], array_type)

    def _hist_data_check(self, residuals, gsim, imt, plot_data):
        for res_type, res_data in plot_data.items():
            assert len(residuals.residuals[gsim][imt][res_type]) > \
                len(res_data['x'])

    def _scatter_data_check(self, residuals, gsim, imt, plot_data):
        for res_type, res_data in plot_data.items():
            assert len(residuals.residuals[gsim][imt][res_type]) == \
                len(res_data['x'])

    @patch('smtk.residuals.residual_plotter.plt')
    def tests_residual_plotter(self, mock_plt):
        """
        Tests basic execution of residual plot.
        Simply tests pyplot show is called by mocking its `show` method
        """
        residuals = res.Residuals(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        plt_show_call_count = 0
        for gsim in self.gsims:
            for imt in self.imts:
                ResidualPlot(residuals, gsim, imt, bin_width=0.1)
                # assert we called pyplot show:
                assert mock_plt.show.call_count == plt_show_call_count+1
                ResidualPlot(residuals, gsim, imt, bin_width=0.1, show=False)
                # assert we did NOT call pyplot show:
                assert mock_plt.show.call_count == plt_show_call_count+1
                # increment counter:
                plt_show_call_count += 1

    @patch('smtk.residuals.residual_plotter.plt')
    def tests_likelihood_plotter(self, mock_plt):
        """
        Tests basic execution of Likelihood plotD.
        Simply tests pyplot show is called by mocking its `show` method
        """
        residuals = res.Likelihood(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        plt_show_call_count = 0
        for gsim in self.gsims:
            for imt in self.imts:
                LikelihoodPlot(residuals, gsim, imt, bin_width=0.1)
                # assert we called pyplot show:
                assert mock_plt.show.call_count == plt_show_call_count+1
                LikelihoodPlot(residuals, gsim, imt, bin_width=0.1, show=False)
                # assert we did NOT call pyplot show:
                assert mock_plt.show.call_count == plt_show_call_count+1
                # increment counter:
                plt_show_call_count += 1

    @patch('smtk.residuals.residual_plotter.plt')
    def tests_with_mag_vs30_depth_plotter(self, mock_plt):
        """
        Tests basic execution of residual with (magnitude, vs30, depth) plots.
        Simply tests pyplot show is called by mocking its `show` method
        """
        residuals = res.Likelihood(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        plt_show_call_count = 0
        for gsim in self.gsims:
            for imt in self.imts:
                for plotClass in [ResidualWithMagnitude,
                                  ResidualWithDepth,
                                  ResidualWithVs30]:
                    # FIXME: we should mock Axes plot and semilogx
                    # to test the input parameter 'plot_type'
                    plotClass(residuals, gsim, imt, show=True)
                    # assert we called pyplot show:
                    assert mock_plt.show.call_count == plt_show_call_count+1
                    plotClass(residuals, gsim, imt, show=False)
                    # assert we did NOT call pyplot show:
                    assert mock_plt.show.call_count == plt_show_call_count+1
                    # increment counter:
                    plt_show_call_count += 1

    @patch('smtk.residuals.residual_plotter.plt')
    def tests_with_distance(self, mock_plt):
        """
        Tests basic execution of residual with distance plots.
        Simply tests pyplot show is called by mocking its `show` method
        """
        residuals = res.Likelihood(self.gsims, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        plt_show_call_count = 0
        for gsim in self.gsims:
            for imt in self.imts:
                for dist in DISTANCES.keys():

                    if dist == 'r_x':
                        # as for residual_plots_test, we should confirm
                        # with scientific expertise that this is the case:
                        with self.assertRaises(AttributeError):
                            ResidualWithDistance(residuals, gsim, imt,
                                         distance_type=dist, show=True)
                        continue

                    ResidualWithDistance(residuals, gsim, imt,
                                         distance_type=dist, show=True)
                    # assert we called pyplot show:
                    assert mock_plt.show.call_count == plt_show_call_count+1
                    ResidualWithDistance(residuals, gsim, imt,
                                         distance_type=dist, show=False)
                    # assert we did NOT call pyplot show:
                    assert mock_plt.show.call_count == plt_show_call_count+1
                    # increment counter:
                    plt_show_call_count += 1


#         self._check_residual_dictionary_correctness(residuals.residuals)
#         residuals.get_residual_statistics()

#     def tests_likelihood_execution(self):
#         """
#         Tests basic execution of residuals - not correctness of values
#         """
#         lkh = res.Likelihood(self.gsims, self.imts)
#         lkh.get_residuals(self.database, component="Geometric")
#         self._check_residual_dictionary_correctness(lkh.residuals)
#         lkh.get_likelihood_values()
# 
#     def tests_llh_execution(self):
#         """
#         Tests execution of LLH - not correctness of values
#         """
#         llh = res.LLH(self.gsims, self.imts)
#         llh.get_residuals(self.database, component="Geometric")
#         self._check_residual_dictionary_correctness(llh.residuals)
#         llh.get_loglikelihood_values(self.imts)
# 
#     def tests_multivariate_llh_execution(self):
#         """
#         Tests execution of multivariate llh - not correctness of values
#         """
#         multi_llh = res.MultivariateLLH(self.gsims, self.imts)
#         multi_llh.get_residuals(self.database, component="Geometric")
#         self._check_residual_dictionary_correctness(multi_llh.residuals)
#         multi_llh.get_likelihood_values()
# 
#     def tests_edr_execution(self):
#         """
#         Tests execution of EDR - not correctness of values
#         """
#         edr = res.EDR(self.gsims, self.imts)
#         edr.get_residuals(self.database, component="Geometric")
#         self._check_residual_dictionary_correctness(edr.residuals)
#         edr.get_edr_values()

    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database
        """
        shutil.rmtree(cls.out_location)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()