"""
Test for functions added to gmpe_residuals (get_edr_values_wrt_spectral_period,
_get_edr_gmpe_information_wrt_spectral_period and _get_edr_wrt_spectral_period)
and residual_plotter (PlotLoglikelihoodWithSpectralPeriod,
PlotModelWeightsWithSpectralPeriod, PlotEDRWithSpectralPeriod, LoglikelihoodTable,
WeightsTable and EDRTable) to output loglikelihood values and sample loglikelihood
based GMPE weightings (Scherbaum et al., 2009) and EDR metrics (Kale and Akkar, 2013)
w.r.t. spectral period (rather than aggregated over all intensity measures as before).
"""

import numpy as np
import pandas as pd
import os
from smtk.parsers.esm_flatfile_parser import ESMFlatfileParser
import smtk.residuals.gmpe_residuals as res
import smtk.residuals.residual_plotter as rspl
import pickle
import shutil


#Parse test flatfile and load metadata
DATA = os.path.dirname(__file__)
input_fileset = os.path.join(DATA,'data','Ranking_Metrics_Test_Flatfile.csv')
output_database = os.path.join(DATA,'data','metadata')
if os.path.exists(output_database):
   shutil.rmtree(output_database)
   
parser = ESMFlatfileParser.autobuild("000", "ranking metrics wrt period test", output_database, input_fileset)  

metadata_directory = os.path.join(DATA,'data','metadata') #Specify metadata directory
metadata_file = '\metadatafile.pkl' #Specify metadata file
metadata = metadata_directory + metadata_file
sm_database = pickle.load(open(metadata,"rb"))

#Specify gmpes to consider
gmpe_list = ["ChiouYoungs2014",
             "CampbellBozorgnia2014",
             "BooreEtAl2014",
             "AbrahamsonEtAl2014"]

imt_list = ["PGA","SA(0.1)","SA(0.2)","SA(0.5)","SA(1.0)","SA(2.0)"]

residuals = res.Residuals(gmpe_list,imt_list)
residuals.get_residuals(sm_database)

os.mkdir('generated_csv_files_for_test')
output_directory=os.path.join(DATA,'generated_csv_files_for_test')
filename = output_directory + '\\test.csv'

"""
Get loglikelihood and sample of loglikelihood based model weights first
wrt period using modified function and then compute average over all considered
spectral periods to show match to outputs of original function
"""

#Get original loglikelihood and sample of loglikelihood based model weights
original_metrics=residuals.get_loglikelihood_values(residuals.imts)
original_llh=original_metrics[0]
avg_model_weights_original=original_metrics[1]

original_avg_llh={}
for gmpe in residuals.gmpe_list:
    original_avg_llh[gmpe]=original_llh[gmpe]['All']
avg_llh_original=np.array(pd.Series(original_avg_llh))
    
rspl.LoglikelihoodTable(residuals,filename) #Get loglikelihood wrt spectral period (from table function)
llh_wrt_period=residuals.final_llh_df 
avg_llh_from_wrt_period_function=llh_wrt_period.loc['Avg over all periods'] #Get values within table
avg_llh_new=np.array(pd.Series(avg_llh_from_wrt_period_function))

rspl.WeightsTable(residuals,filename) #Get model weights wrt spectral period (from table function)
model_weights_wrt_period=residuals.final_model_weights_df 
avg_model_weights_from_wrt_period_function=model_weights_wrt_period.loc['Avg over all periods'] #Get values within table
avg_model_weights_new=np.array(pd.Series(avg_model_weights_from_wrt_period_function))

#Evaluate equivalencies of values computed from w.r.t. imt function outputs and original function outputs
test_equivalency_llh=avg_llh_new/avg_llh_original
test_equivalency_model_weights=np.array(avg_model_weights_new)/np.array(pd.Series(avg_model_weights_original))
test_array = np.array((test_equivalency_llh,test_equivalency_model_weights))

"""
Compute equal values of kappa using (1) all observations, expected and stddev
values as within original function and (2) using all subsets (indexed by imt)
reassembled from 'new' function.

Note: cumulative dist. functions are used to compute MDE norm and EDR
within Kale and Akkar (2013) methodology. Therefore, the average
MDE Norm, sqrt(kappa) and EDR values based on aggregation of CDFs for each imt
will not equal the value computed within the original functions
(due to computing a CDF using observations, expected and stddev for all
imts within a single array in the original function, rather than one array per imt).

Therefore, here, the subsets of observations, expected and stddev for each
imt per gmpe are recombined, and then kappa values matching those
provided by the original functions are attained to test these functions. 
"""

#Compute equal kappa from original and wrt spectral period functions

#Function for EDR metrics wrt spectral period
for gmpe in residuals.gmpe_list:        
    #_get_edr_gmpe_information_wrt_spectral_period(self,gmpe)
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE (per imt)
        """  
        #Remove non-acceleration imts from residuals.imts for generation of residuals
        imt_append=pd.DataFrame(residuals.imts,index=residuals.imts)
        imt_append.columns=['imt']
        for imt_idx in range(0,np.size(residuals.imts)):
             if residuals.imts[imt_idx]=='PGV':
                 imt_append=imt_append.drop(imt_append.loc['PGV'])
             if residuals.imts[imt_idx]=='PGD':
                 imt_append=imt_append.drop(imt_append.loc['PGD'])
             if residuals.imts[imt_idx]=='CAV':
                 imt_append=imt_append.drop(imt_append.loc['CAV'])
             if residuals.imts[imt_idx]=='Ia':
                 imt_append=imt_append.drop(imt_append.loc['Ia'])
            
        imt_append_list=pd.DataFrame()
        for idx in range(0,len(imt_append)):
             imt_append_list[idx]=imt_append.iloc[idx]
        imt_append=imt_append.reset_index()
        imt_append_list.columns=imt_append.imt
        residuals.imts_appended=list(imt_append_list)
        
        obs_wrt_imt={}
        expected_wrt_imt={}
        stddev_wrt_imt={}
        for imtx in residuals.imts_appended:
            obs = np.array([], dtype=float)
            expected = np.array([], dtype=float)
            stddev = np.array([], dtype=float)
            for context in residuals.contexts:
                obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                expected = np.hstack([expected,context["Expected"][gmpe][imtx]["Mean"]])
                stddev = np.hstack([stddev,context["Expected"][gmpe][imtx]["Total"]])
            obs_wrt_imt[imtx]=obs
            expected_wrt_imt[imtx]=expected
            stddev_wrt_imt[imtx]=stddev
            
#Function for EDR metrics aggregating over all imts (original function)
for gmpe in residuals.gmpe_list:
    #def _get_edr_gmpe_information(self, gmpe):
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE (aggregating over all IMS)
        """
        obs = np.array([], dtype=float)
        expected = np.array([], dtype=float)
        stddev = np.array([], dtype=float)
        for imtx in residuals.imts:
            for context in residuals.contexts:
                obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                expected = np.hstack([expected,
                                      context["Expected"][gmpe][imtx]["Mean"]])
                stddev = np.hstack([stddev,
                                    context["Expected"][gmpe][imtx]["Total"]])

#Compute original kappa ratio
mu_a = np.mean(obs)
mu_y = np.mean(expected)
b_1 = np.sum((obs - mu_a) * (expected - mu_y)) /\
np.sum((obs - mu_a) ** 2.)
b_0 = mu_y - b_1 * mu_a
y_c = expected - ((b_0 + b_1 * obs) - obs)
de_orig = np.sum((obs - expected) ** 2.)
de_corr = np.sum((obs - y_c) ** 2.)
original_kappa = de_orig / de_corr

#Compute 'new' kappa from reassembled subsets
all_new_function_obs=pd.DataFrame()
all_new_function_expected=pd.DataFrame()
for imt in residuals.imts_appended:
    all_new_function_obs[imt]=obs_wrt_imt[imt]
    all_new_function_expected[imt]=expected_wrt_imt[imt]
all_new_function_expected

all_new_function_obs_1 = np.array(all_new_function_obs)
all_new_function_expected_1 = np.array(all_new_function_expected)

mu_a = np.mean(all_new_function_obs_1)
mu_y = np.mean(all_new_function_expected_1)
b_1 = np.sum((all_new_function_obs_1 - mu_a) * (all_new_function_expected_1 - mu_y)) /\
          np.sum((all_new_function_obs_1 - mu_a) ** 2.)
b_0 = mu_y - b_1 * mu_a
y_c = all_new_function_expected_1 - ((b_0 + b_1 * all_new_function_obs_1) - all_new_function_obs_1)
de_orig = np.sum((all_new_function_obs_1 - all_new_function_expected_1) ** 2.)
de_corr = np.sum((all_new_function_obs_1 - y_c) ** 2.)

new_kappa=de_orig/de_corr

#Evaluate 'new' and original kappa
kappa_ratio=new_kappa/original_kappa

if np.all(test_array) < 1.005 and np.all(test_array) > 0.995 and kappa_ratio==1:
    print ('Pass') #acceptable match observed for all metrics
    
shutil.rmtree(output_directory)