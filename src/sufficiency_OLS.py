import numpy as np 
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os 
from scipy.stats import gmean
from efficiency_OLS import OLS


class Sufficiency():
    def __init__(self, ln_EDP, ln_IM, Rjb, Mag):
        self.EDP = ln_EDP
        self.IM = ln_IM
        self.X = sm.add_constant(self.IM)
        
        temp = {'I': np.ones(shape = len(Mag)),
                'lnEDP': self.EDP,
                'lnIM': self.IM,
                'lnRjb': np.log(Rjb),
                'M': Mag}
        self.dummydf = pd.DataFrame(temp, index = None)
        
        self.sufficiency_against_R()
        self.sufficiency_against_M()
        self.sufficiency_against_M_and_R()
        
    def sufficiency_against_R(self):
        self.model_IM_vs_R = smf.ols('lnEDP ~ lnIM + lnRjb', data = self.dummydf)
        model_res = self.model_IM_vs_R.fit()
        self.summary_against_R = model_res.summary()
        return model_res.pvalues['lnRjb']
        
    def sufficiency_against_M(self):
        self.model_IM_vs_M = smf.ols('lnEDP ~ lnIM + M', data = self.dummydf)
        model_res = self.model_IM_vs_M.fit()
        self.summary_against_M = model_res.summary()
        return model_res.pvalues['M']

    def sufficiency_against_M_and_R(self):
        self.model_IM_vs_R_and_M = smf.ols('lnEDP ~ lnIM + M + lnRjb', data = self.dummydf)
        model_res = self.model_IM_vs_R_and_M.fit()
        self.summary_against_M_and_R = model_res.summary()
        return model_res.pvalues
    
    def qqplot(self, model):
        sm.qqplot(model.fit(), line = 's')


def SummaryResutls_sufficiency(baseDir, BuildingList, buildingIndex, df_IMs, IMs=['SaT1', 'PGA', 'PGV', 'Sa_avg'],
                               pairingID = 1, fit_residual = True, average_EDP = False):
    numGM = len(df_IMs)
    T = np.array([0.13, 0.12, 0.16, 0.15, 0.22, 0.22, 0.26, 0.25, 0.49, 0.49])
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
        else:
            IM = df_IMs[IMs[i]]

        for j in range(numStory):
                ## geometric average of EDP (in X and Z direction) between pairing ID 1 and 2
            if average_EDP: 
                sdrX = gmean([sdr[3+j].values[0:numGM], sdr[3+j].values[numGM*2:numGM*3]])
                pfaX = gmean([pfa[4+j].values[0:numGM], pfa[4+j].values[numGM*2:numGM*3]])
                
                sdrZ = gmean([sdr[3+j].values[numGM:numGM*2], sdr[3+j].values[numGM*3:numGM*4]])
                pfaZ = gmean([pfa[4+j].values[numGM:numGM*2], pfa[4+j].values[numGM*3:numGM*4]])
            else:
                sdrX = sdr[3+j].values[numGM * start_index_multiplier : numGM * end_index_multiplier]
                sdrZ = sdr[3+j].values[numGM * end_index_multiplier : numGM * (end_index_multiplier + 1)]

                pfaX = pfa[4+j].values[numGM * start_index_multiplier : numGM * end_index_multiplier]
                pfaZ = pfa[4+j].values[numGM * end_index_multiplier : numGM * (end_index_multiplier + 1)]
            
            residuals_sdrX = OLS(sdrX, IM).result.resid
            residuals_sdrZ = OLS(sdrZ, IM).result.resid
            residuals_pfaX = OLS(pfaX, IM).result.resid
            residuals_pfaZ = OLS(pfaZ, IM).result.resid
            
            if fit_residual:
                suff_sdrX = Sufficiency(residuals_sdrX, np.log(IM),
                                    df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
                suff_sdrZ = Sufficiency(residuals_sdrZ, np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
                suff_pfaX = Sufficiency(residuals_pfaX, np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
                suff_pfaZ = Sufficiency(residuals_pfaZ, np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
            else:
                suff_sdrX = Sufficiency(np.log(sdrX), np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
                suff_sdrZ = Sufficiency(np.log(sdrZ), np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
                suff_pfaX = Sufficiency(np.log(pfaX), np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
                suff_pfaZ = Sufficiency(np.log(pfaZ), np.log(IM),
                                        df_IMs['Distance_Rjb'].values, df_IMs['Magnitude'].values)
            #### sufficiency against R
            p_sdrX_R = np.round(suff_sdrX.sufficiency_against_R(), 4)
            p_sdrZ_R = np.round(suff_sdrZ.sufficiency_against_R(), 4)
            p_pfaX_R = np.round(suff_pfaX.sufficiency_against_R(), 4)
            p_pfaZ_R = np.round(suff_pfaZ.sufficiency_against_R(), 4)
            #### sufficiency against M 
            p_sdrX_M = np.round(suff_sdrX.sufficiency_against_M(), 4)
            p_sdrZ_M = np.round(suff_sdrZ.sufficiency_against_M(), 4)
            p_pfaX_M = np.round(suff_pfaX.sufficiency_against_M(), 4)
            p_pfaZ_M = np.round(suff_pfaZ.sufficiency_against_M(), 4)
            temp_sdr[keys[j]] = {'SDR_X vs M':'YES(%s)'%p_sdrX_M if p_sdrX_M >= 0.05 else 'NO(%s)'%p_sdrX_M,
                                 'SDR_Z vs M':'YES(%s)'%p_sdrZ_M if p_sdrZ_M >= 0.05 else 'NO(%s)'%p_sdrZ_M,
                                 'PFA_X vs M':'YES(%s)'%p_pfaX_M if p_pfaX_M >= 0.05 else 'NO(%s)'%p_pfaX_M,
                                 'PFA_Z vs M':'YES(%s)'%p_pfaZ_M if p_pfaZ_M >= 0.05 else 'NO(%s)'%p_pfaZ_M,
                                 'SDR_X vs R':'YES(%s)'%p_sdrX_R if p_sdrX_R >= 0.05 else 'NO(%s)'%p_sdrX_R,
                                 'SDR_Z vs R':'YES(%s)'%p_sdrZ_R if p_sdrZ_R >= 0.05 else 'NO(%s)'%p_sdrZ_R,
                                 'PFA_X vs R':'YES(%s)'%p_pfaX_R if p_pfaX_R >= 0.05 else 'NO(%s)'%p_pfaX_R,
                                 'PFA_Z vs R':'YES(%s)'%p_pfaZ_R if p_pfaZ_R >= 0.05 else 'NO(%s)'%p_pfaZ_R
                                 
                                }
        reform = {(outerKey, innerKey): values for outerKey, innerDict in temp_sdr.items() for innerKey, values in innerDict.items()}
        df = pd.DataFrame.from_dict(reform, orient='index').transpose()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df['IM'] = IMs[i]
        df = df.set_index('IM')
        tempdf.append(df)

    return pd.concat(tempdf)
