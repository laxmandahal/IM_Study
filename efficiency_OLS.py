import os 
import sys
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import lognorm
from scipy.stats import norm 
from scipy import integrate
from scipy.stats import gmean

from scipy.stats import pearsonr

import statsmodels.api as sm 
import statsmodels.formula.api as smf

class OLS:
    def __init__(self, EDP, IM):
        self.EDP = EDP
        self.IM = IM
        self.Y = np.log(EDP)
        self.X = sm.add_constant(np.log(IM))
        
        self.residual_mean = None
        self.residual_std = None
        self.xmin = None
        self.xmax = None
        
        self.fitModel()
        self.get_summary()
        
#         self.plot_residuals_hist()
#         self.plot_model_fit()
        
    def fitModel(self):
        model = sm.OLS(self.Y, self.X)
        self.result = model.fit()
        
    def get_summary(self):
        return self.result.summary()
    
    def plot_residuals_hist(self):
        fig, ax = plt.subplots()
        sns.histplot(x = self.result.resid, ax=ax, stat = 'density', linewidth = 0, kde=True)
        ax.set(title = 'Distribution of residuals', xlabel = 'residuals')
        
        self.residual_mean, self.residual_std = norm.fit(self.result.resid)
        ##plotting normal pdf
        self.xmin, self.xmax = plt.xlim() # using maxi/min values from histogram
        x = np.linspace(self.xmin, self.xmax, 200)
        pdf = norm.pdf(x, self.residual_mean, self.residual_std)
        sns.lineplot(x = x, y = pdf, color = 'red', ax = ax)
        plt.show()
    
    def get_efficiency(self):
        return np.std(self.result.resid)
            
    def qqplot(self):
        sm.qqplot(self.result.resid, line = 's')


    def fitplot(self):
        sm.graphics.plot_fit(self.result, 1, vlines = False);
        
    def plot_model_fit(self):
        Ymin = self.Y.min()
        Ymax = self.Y.max()
#         ax = sns.subplots()
        ax = sns.scatterplot(x = self.result.fittedvalues, y = self.Y)
#         ax.set_ylim(Ymin, Ymax)
#         ax.set_xlim(self.xmin, self.xmax)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Observed Values')
        
        X_ref = Y_ref = np.linspace(Ymin, Ymax, 200)
        plt.plot(X_ref, Y_ref, color = 'red', linewidth = 1.3)
        plt.show()

def SummaryResutls_efficiency(buildingIndex, df_IMs, IMs=['SaT1', 'PGA', 'PGV', 'Sa_avg'],
                              pairingID = 1, average_EDP = False, rotate_EDP = False):
    '''
    pairingID: if 1, GM_h1 applied in X-direction, GM_h2 applied in Y-direction
               if 2, GM_h2 applied in X-direction, GM_h1 applied in Y-direction
               
    Note:
    Care must be taken while using gminfo and pairing ID
    -------------
    df_IMs = gminfo_RotD50 can be used with pairing ID 1 OR 2 
    df_IMs = gminfo_h1 can only be used with pairing ID 1
    df_IMs = gminfo_h2 can only be used with pairing ID 2
    '''
    dataDir = os.path.join(baseDir, *['Results', 'IM_study_826GMs', BuildingList[buildingIndex]])
    os.chdir(dataDir)
    sdr = pd.read_csv('SDR.csv', header = None)
    pfa = pd.read_csv('PFA.csv', header = None)
    
    numStory = int(BuildingList[buildingIndex].split('_')[0][1])
    tempdf = []
    d = []
    keys = ['1st Story', '2nd Story', '3rd Story', '4th Story']
    temp_sdr = {}
    
    if pairingID == 1:
        start_index_multiplier = 0
        end_index_multiplier = 1
    else:
        start_index_multiplier = 2
        end_index_multiplier = 3
    
    for i in range(len(IMs)):
        if IMs[i] == 'SaT1':
            IM = df_IMs['T_%s'%T[buildingIndex]].values
            ######### result for SaT1, SvT1 and SdT1 is the same b/c they are constantly relate
#         elif IMs[i] == 'SvT1':
#             IM = sv_rotD50['T_%s'%T[buildingIndex]].values
#         elif IMs[i] == 'SdT1':
#             IM = sd_rotD50['T_%s'%T[buildingIndex]].values
        else:
            IM = df_IMs[IMs[i]]

        for j in range(numStory):
            
            sdrX = sdr[3+j].values[numGM * start_index_multiplier : numGM * end_index_multiplier]
            sdrZ = sdr[3+j].values[numGM * end_index_multiplier : numGM * (end_index_multiplier + 1)]

            pfaX = pfa[4+j].values[numGM * start_index_multiplier : numGM * end_index_multiplier]
            pfaZ = pfa[4+j].values[numGM * end_index_multiplier : numGM * (end_index_multiplier + 1)]
            
            if average_EDP:
                sdrX_avg = gmean([sdr[3+j].values[0:numGM], sdr[3+j].values[numGM*2:numGM*3]])
                pfaX_avg = gmean([pfa[4+j].values[0:numGM], pfa[4+j].values[numGM*2:numGM*3]])
                
                sdrZ_avg = gmean([sdr[3+j].values[numGM:numGM*2], sdr[3+j].values[numGM*3:numGM*4]])
                pfaZ_avg = gmean([pfa[4+j].values[numGM:numGM*2], pfa[4+j].values[numGM*3:numGM*4]])
#                 compute_RotDxx_EDP(sdrX, sdrZ, percentile=50)
#                 pfa_rotD50 = compute_RotDxx_EDP(pfaX, pfaZ, percentile=50)
                if rotate_EDP:
                    sdr_rotD50 = compute_RotDxx_EDP(sdrX_avg, sdrZ_avg, percentile=50)
                    pfa_rotD50 = compute_RotDxx_EDP(pfaX_avg, pfaZ_avg, percentile=50)
                    ols_sdrX = OLS(sdr_rotD50, IM)
                    ols_pfaZ = OLS(pfa_rotD50, IM)
                    temp_sdr[keys[j]] = {'SDR_RotD50':ols_sdrX.get_efficiency(),
                                         'PFA_RotD50': ols_pfaZ.get_efficiency()}
                else:
                    ols_sdrX = OLS(sdrX_avg, IM)
                    ols_sdrZ = OLS(sdrZ_avg, IM)
                    ols_pfaX = OLS(pfaX_avg, IM)
                    ols_pfaZ = OLS(pfaZ_avg, IM)
                    temp_sdr[keys[j]] = {'SDR_X_Avg':ols_sdrX.get_efficiency(),
                                         'SDR_Z_Avg':ols_sdrZ.get_efficiency(),
                                         'PFA_X_Avg': ols_pfaX.get_efficiency(),
                                         'PFA_Z_Avg': ols_pfaZ.get_efficiency()}
            else:
                ols_sdrX = OLS(sdrX, IM)
                ols_sdrZ = OLS(sdrZ, IM)
                ols_pfaX = OLS(pfaX, IM)
                ols_pfaZ = OLS(pfaZ, IM)
                temp_sdr[keys[j]] = {'SDR_X':ols_sdrX.get_efficiency(),
                                     'SDR_Z':ols_sdrZ.get_efficiency(),
                                     'PFA_X': ols_pfaX.get_efficiency(),
                                     'PFA_Z': ols_pfaZ.get_efficiency()}
        reform = {(outerKey, innerKey): values for outerKey, innerDict in temp_sdr.items() for innerKey, values in innerDict.items()}
        df = pd.DataFrame.from_dict(reform, orient='index').transpose()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df['IM'] = IMs[i]
        df = df.set_index('IM')
        tempdf.append(df)

    return pd.concat(tempdf)
