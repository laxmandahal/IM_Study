import os 
import numpy as np 
import pandas as pd 
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from scipy.stats import gmean


def compute_RotDxx_EDP(edpX, edpZ, percentile = 50):
    angles = np.arange(0, 180, step=1)
    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]
    
    edp_stacked = np.vstack([edpX, edpZ])
    rotated_edp = np.dot(coeffs, edp_stacked)
    percentile_edp = np.percentile(rotated_edp, q = percentile, axis = 0, interpolation='linear')
    return percentile_edp

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def calc_mutual_information(x, y, bins):
    '''
    Calculates mutual information
    '''
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood", correction=False)
    mi = 0.5 * g / c_xy.sum()
    return mi



def calc_mutual_information_sklearn(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def save_data_for_R(baseDir, BuildingList, buildingIndex, df_IMs, IMs=['SaT1', 'PGA', 'PGV', 'Sa_avg', 'CAV'], pairingID = 1, 
                   average_EDP = False, rotate_EDP = False, save_to_csv=False):
    dataDir = os.path.join(baseDir, *['Results', 'IM_study_826GMs', BuildingList[buildingIndex]])
    os.chdir(dataDir)
    sdr = pd.read_csv('SDR.csv', header = None)
    pfa = pd.read_csv('PFA.csv', header = None)
    
    if pairingID == 1:
        start_index_multiplier = 0
        end_index_multiplier = 1
    else:
        start_index_multiplier = 2
        end_index_multiplier = 3
    
    numGM = len(df_IMs)
    T = np.array([0.13, 0.12, 0.16, 0.15, 0.22, 0.22, 0.26, 0.25, 0.49, 0.49])

    numStory = int(BuildingList[buildingIndex].split('_')[0][1])
    temp = {}
    for i in range(numStory):
        ## geometric average of EDP (in X and Z direction) between pairing ID 1 and 2
        if average_EDP:
            name_suffix = 'avg_EDP'
            sdrX = gmean([sdr[3+i].values[0:numGM], sdr[3+i].values[numGM*2:numGM*3]])
            pfaX = gmean([pfa[4+i].values[0:numGM], pfa[4+i].values[numGM*2:numGM*3]])

            sdrZ = gmean([sdr[3+i].values[numGM:numGM*2], sdr[3+i].values[numGM*3:numGM*4]])
            pfaZ = gmean([pfa[4+i].values[numGM:numGM*2], pfa[4+i].values[numGM*3:numGM*4]])
            
            if rotate_EDP:
                name_suffix = 'rot_EDP'
                sdr_rotD50 = compute_RotDxx_EDP(sdrX, sdrZ, percentile=50)
                pfa_rotD50 = compute_RotDxx_EDP(pfaX, pfaZ, percentile=50)
            
        else:
            name_suffix = 'pID%s'%pairingID
            sdrX = sdr[3+i].values[numGM * start_index_multiplier : numGM * end_index_multiplier]
            sdrZ = sdr[3+i].values[numGM * end_index_multiplier : numGM * (end_index_multiplier + 1)]

            pfaX = pfa[4+i].values[numGM * start_index_multiplier : numGM * end_index_multiplier]
            pfaZ = pfa[4+i].values[numGM * end_index_multiplier : numGM * (end_index_multiplier + 1)]
        
        if rotate_EDP:
            temp['story_%s_SDR_rotD50'%(i+1)] = sdr_rotD50
            temp['story_%s_PFA_rotD50'%(i+1)] = pfa_rotD50
        else:
            temp['story_%s_sdrX'%(i+1)] = sdrX
            temp['story_%s_sdrZ'%(i+1)] = sdrZ
            temp['story_%s_pfaX'%(i+1)] = pfaX
            temp['story_%s_pfaZ'%(i+1)] = pfaZ
    
    for i in range(len(IMs)):
        if IMs[i] == 'SaT1':
            temp[IMs[i]] = df_IMs['T_%s'%T[buildingIndex]].values
        else:
            temp[IMs[i]] = df_IMs[IMs[i]]
    temp['Magnitude'] = df_IMs['Magnitude']
    temp['Distance_Rjb'] = df_IMs['Distance_Rjb']
    temp['Distance_rup'] = df_IMs['Distance_rup']
    final_df = pd.DataFrame(temp)
    if save_to_csv:
        final_df.to_csv('Inputs_for_%s_%s.csv'%(BuildingList[buildingIndex], name_suffix))
    return final_df  