"""
Core test suite for the database and residuals construction
when created from sm_database.GroundMotionDatabase and
sm_table.GroundMotionTable (contexts should be equal)
"""
import os
import sys
import shutil
import unittest

import numpy as np
from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser
import smtk.residuals.gmpe_residuals as res

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

    def test_correct_build_load(self):
        """
        Verifies that the database has been built and loaded correctly
        """
        self.assertEqual(len(self.database), 41)
        self.assertListEqual([rec.id for rec in self.database],
                             EXPECTED_IDS)

    def _check_residual_dictionary_correctness(self, res_dict):
        """
        Basic check for correctness of the residual dictionary
        """
        for i, gsim in enumerate(res_dict):
            self.assertEqual(gsim, self.gsims[i])
            for j, imt in enumerate(res_dict[gsim]):
                self.assertEqual(imt, self.imts[j])
                if gsim == "AkkarEtAlRjb2014":
                    # For Akkar et al - inter-event residuals should have
                    # 4 elements and the intra-event residuals 41
                    self.assertEqual(
                        len(res_dict[gsim][imt]["Inter event"]), 4)
                elif gsim == "ChiouYoungs2014":
                    # For Chiou & Youngs - inter-event residuals should have
                    # 41 elements and the intra-event residuals 41 too
                    self.assertEqual(
                        len(res_dict[gsim][imt]["Inter event"]), 41)
                else:
                    pass
                self.assertEqual(
                        len(res_dict[gsim][imt]["Intra event"]), 41)
                self.assertEqual(
                        len(res_dict[gsim][imt]["Total"]), 41)

    def test_residuals_execution(self):
        """
        Tests basic execution of residuals - not correctness of values -
        to be the same when created from sm_database.GroundMotionDatabase and
        sm_table.GroundMotionTable
        """
        residuals1 = res.Residuals(self.gsims, self.imts)
        residuals1.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(residuals1.residuals)
        stats1 = residuals1.get_residual_statistics()

        # yes, we said we do not check for correctness of values.
        # However, this simple check warn us in case
        # code modification will unexpectedly change residuals calculations:
        expected_res_obs = {'PGA': np.array([2.02129254e-04,
                                             1.05953800e-03,
                                             1.89631889e-04,
                                             7.53992832e-05]),
                            'SA(1.0)': np.array([0.00012647,
                                                 0.00073836,
                                                 0.00019964,
                                                 0.00014166])}
        for imtx in expected_res_obs:
            assert np.allclose(residuals1.contexts[0]['Observations'][imtx],
                               expected_res_obs[imtx], rtol=.5e-3)

    def test_likelihood_execution(self):
        """
        Tests basic execution of residuals - not correctness of values -
        to be the same when created from sm_database.GroundMotionDatabase and
        sm_table.GroundMotionTable
        """
        lkh1 = res.Residuals(self.gsims, self.imts)
        lkh1.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(lkh1.residuals)
        lkh1.get_likelihood_values()

    def test_llh_execution(self):
        """
        Tests execution of LLH - not correctness of values -
        to be the same when created from sm_database.GroundMotionDatabase and
        sm_table.GroundMotionTable
        """
        llh1 = res.Residuals(self.gsims, self.imts)
        llh1.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(llh1.residuals)
        llh1.get_loglikelihood_values(self.imts)

    def test_multivariate_llh_execution(self):
        """
        Tests execution of multivariate llh - not correctness of values -
        to be the same when created from sm_database.GroundMotionDatabase and
        sm_table.GroundMotionTable
        """
        multi_llh1 = res.Residuals(self.gsims, self.imts)
        multi_llh1.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(multi_llh1.residuals)
        multi_llh1.get_multivariate_loglikelihood_values()

    def test_edr_execution(self):
        """
        Tests execution of EDR - not correctness of values -
        to be the same when created from sm_database.GroundMotionDatabase and
        sm_table.GroundMotionTable
        """
        edr1 = res.Residuals(self.gsims, self.imts)
        edr1.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(edr1.residuals)
        edr1.get_edr_values()

    def test_multiple_metrics(self):
        """
        Tests the execution running multiple metrics in one call
        with sm_table.GroundMotionTable instead of
        sm_database.GroundMotionDatabase
        """
        # OLD CODE:
        # residuals = res.Residuals(self.gsims, self.imts)
        # residuals.get_residuals(self.database, component="Geometric")
        # config = {}
        # for key in ["Residuals", "Likelihood", "LLH",
        #             "MultivariateLLH", "EDR"]:
        #     _ = res.GSIM_MODEL_DATA_TESTS[key](residuals, config)


    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database
        """
        shutil.rmtree(cls.out_location)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
