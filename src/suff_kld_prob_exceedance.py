import os 
import numpy as np 
import pandas as pd
from scipy import stats

from extract_edp_data import extract_EDP_data
from efficiency_OLS import OLS
from Linear_model_fits import fit_against_IM_and_M, fit_against_IM_and_M_Rjb, fit_against_IM_and_Rjb
from Linear_model_fits import fit_IM_against_M, fit_IM_against_Rjb, fit_IM_against_M_and_Rjb

from KLD_multivariate_normal import kl_divergence_mvn


def compute_KL_divergence(p, q):
    mu1 = np.mean(p)
    mu2 = np.mean(q)
    std1 = np.std(p, ddof=0)
    std2 = np.std(q, ddof=0)
    
    kl = np.log(std2 / std1) + (std1 ** 2 + (mu1 - mu2)**2) / (2 * std2 ** 2) - 0.5
    return kl



def compile_kld_prob_exceedance(BuildingList, building_Index, baseDir, IM_List, df_IMs, pairingID = 1,
                          numGM = 826, remove_collapse = False, KL_on_residual=False,
                         use_predicted_IM=False):
    ## time period of the buildings, used to get Sa(T1)
    # T = np.array([0.13, 0.12, 0.22, 0.22, 0.16, 0.15, 0.26, 0.25, 0.49, 0.49])
    T = np.array([0.13, 0.12, 0.16, 0.15, 0.22, 0.22, 0.26, 0.25, 0.49, 0.49])
    
    sdr_all = extract_EDP_data(baseDir, BuildingList, building_Index, edp_type='SDR', pairingID=pairingID,
                           numGM=numGM, remove_collapse=False)
    sdr = extract_EDP_data(baseDir, BuildingList, building_Index, edp_type='SDR', pairingID=pairingID,
                           numGM=numGM, remove_collapse=remove_collapse)
    
    sdrX = sdr[sdr['Direction'] == 'X']['Max_EDP'].values
    sdrZ = sdr[sdr['Direction'] == 'Z']['Max_EDP'].values
    pfa = extract_EDP_data(baseDir, BuildingList, building_Index, edp_type='PFA', pairingID=pairingID, numGM=numGM)
    pfa = pfa[pfa['GM_ID'].isin(sdr['GM_ID'].values)] # if collapse is remove, this gives consistent pfa data 
    pfaX = pfa[pfa['Direction'] == 'X']['Max_EDP'].values
    pfaZ = pfa[pfa['Direction'] == 'Z']['Max_EDP'].values
    
    gm_ID_index_X = [sdr_all[sdr_all['Direction']=='X']['GM_ID'].tolist().index(x) for x in sdr[sdr['Direction'] == 'X']['GM_ID'].values]
    gm_ID_index_Z = [sdr_all[sdr_all['Direction']=='Z']['GM_ID'].tolist().index(x) for x in sdr[sdr['Direction'] == 'Z']['GM_ID'].values]
    
    M = df_IMs['Magnitude'].values
    Rjb = df_IMs['Distance_Rjb'].values
    
    sdrX_range = np.linspace(min(sdrX), 0.1, 500)
    sdrZ_range = np.linspace(min(sdrZ), 0.1, 500)
    pfaX_range = np.linspace(min(pfaX), max(pfaX), 500)
    pfaZ_range = np.linspace(min(pfaZ), max(pfaZ), 500)

    kld_sdrX = []
    kld_sdrZ = []
    kld_pfaX = []
    kld_pfaZ = []
    
    kld_sdrX_R = []
    kld_sdrZ_R = []
    kld_pfaX_R = []
    kld_pfaZ_R = []
    
    kld_sdrX_M_R = []
    kld_sdrZ_M_R = []
    kld_pfaX_M_R = []
    kld_pfaZ_M_R = []
    
    for i in range(len(IM_List)):

        if IM_List[i] == 'SaT1':
            IM = df_IMs['T_%s'%T[building_Index]].values
        else:
            IM = df_IMs[IM_List[i]]
        IM_M = IM 
        IM_R = IM
        IM_M_and_R = IM
        if use_predicted_IM:
            IM_M = fit_IM_against_M(IM, M).get_prediction().predicted_mean
            IM_M = fit_IM_against_Rjb(IM, Rjb).get_prediction().predicted_mean
            IM_M_and_R = fit_IM_against_M_and_Rjb(IM, M, Rjb).get_prediction().predicted_mean

        ols_sdrX_im = OLS(sdrX, IM[gm_ID_index_X])
        ols_sdrZ_im = OLS(sdrZ, IM[gm_ID_index_Z])
        ols_pfaX_im = OLS(pfaX, IM[gm_ID_index_X])
        ols_pfaZ_im = OLS(pfaZ, IM[gm_ID_index_Z])
        
        sdrX_pred = ols_sdrX_im.get_pred_values()
        sdrZ_pred = ols_sdrZ_im.get_pred_values()
        pfaX_pred = ols_pfaX_im.get_pred_values()
        pfaZ_pred = ols_pfaZ_im.get_pred_values()

        prob_exced_sdrX = np.ones(len(sdrX_range)) - stats.norm.cdf((np.log(sdrX_range) - np.mean(sdrX_pred)) / ols_sdrX_im.get_efficiency())
        prob_exced_sdrZ = np.ones(len(sdrZ_range)) - stats.norm.cdf((np.log(sdrZ_range) - np.mean(sdrZ_pred)) / ols_sdrZ_im.get_efficiency())
        prob_exced_pfaX = np.ones(len(pfaX_range)) - stats.norm.cdf((np.log(pfaX_range) - np.mean(pfaX_pred)) / ols_pfaX_im.get_efficiency())
        prob_exced_pfaZ = np.ones(len(pfaZ_range)) - stats.norm.cdf((np.log(pfaZ_range) - np.mean(pfaZ_pred)) / ols_pfaZ_im.get_efficiency())

        ols_sdrX_im_M = fit_against_IM_and_M(sdrX, IM_M[gm_ID_index_X], M[gm_ID_index_X])
        ols_sdrZ_im_M = fit_against_IM_and_M(sdrZ, IM_M[gm_ID_index_Z], M[gm_ID_index_Z])
        ols_pfaX_im_M = fit_against_IM_and_M(pfaX, IM_M[gm_ID_index_X], M[gm_ID_index_X])
        ols_pfaZ_im_M = fit_against_IM_and_M(pfaZ, IM_M[gm_ID_index_Z], M[gm_ID_index_Z])
        
        sdrX_pred_M = ols_sdrX_im_M.get_prediction().predicted_mean
        sdrZ_pred_M = ols_sdrZ_im_M.get_prediction().predicted_mean
        pfaX_pred_M = ols_pfaX_im_M.get_prediction().predicted_mean
        pfaZ_pred_M = ols_pfaZ_im_M.get_prediction().predicted_mean

        prob_exced_sdrX_M = np.ones(len(sdrX_range)) - stats.norm.cdf((np.log(sdrX_range) - np.mean(sdrX_pred_M)) / np.std(ols_sdrX_im_M.resid))
        prob_exced_sdrZ_M = np.ones(len(sdrZ_range)) - stats.norm.cdf((np.log(sdrZ_range) - np.mean(sdrZ_pred_M)) / np.std(ols_sdrZ_im_M.resid))
        prob_exced_pfaX_M = np.ones(len(pfaX_range)) - stats.norm.cdf((np.log(pfaX_range) - np.mean(pfaX_pred_M)) / np.std(ols_pfaX_im_M.resid))
        prob_exced_pfaZ_M = np.ones(len(pfaZ_range)) - stats.norm.cdf((np.log(pfaZ_range) - np.mean(pfaZ_pred_M)) / np.std(ols_pfaZ_im_M.resid))
        
        ## fit against causal parameter R only 
        ols_sdrX_im_R = fit_against_IM_and_Rjb(sdrX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X])
        ols_sdrZ_im_R = fit_against_IM_and_Rjb(sdrZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z])
        ols_pfaX_im_R = fit_against_IM_and_Rjb(pfaX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X])
        ols_pfaZ_im_R = fit_against_IM_and_Rjb(pfaZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z])
        
        sdrX_pred_R = ols_sdrX_im_R.get_prediction().predicted_mean
        sdrZ_pred_R = ols_sdrZ_im_R.get_prediction().predicted_mean
        pfaX_pred_R = ols_pfaX_im_R.get_prediction().predicted_mean
        pfaZ_pred_R = ols_pfaZ_im_R.get_prediction().predicted_mean

        prob_exced_sdrX_R = np.ones(len(sdrX_range)) - stats.norm.cdf((np.log(sdrX_range) - np.mean(sdrX_pred_R)) / np.std(ols_sdrX_im_R.resid))
        prob_exced_sdrZ_R = np.ones(len(sdrZ_range)) - stats.norm.cdf((np.log(sdrZ_range) - np.mean(sdrZ_pred_R)) / np.std(ols_sdrZ_im_R.resid))
        prob_exced_pfaX_R = np.ones(len(pfaX_range)) - stats.norm.cdf((np.log(pfaX_range) - np.mean(pfaX_pred_R)) / np.std(ols_pfaX_im_R.resid))
        prob_exced_pfaZ_R = np.ones(len(pfaZ_range)) - stats.norm.cdf((np.log(pfaZ_range) - np.mean(pfaZ_pred_R)) / np.std(ols_pfaZ_im_R.resid))

        ## fit against multivariate causal parameters M and R
        ols_sdrX_im_M_R = fit_against_IM_and_M_Rjb(sdrX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X], Rjb[gm_ID_index_X])
        ols_sdrZ_im_M_R = fit_against_IM_and_M_Rjb(sdrZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z], Rjb[gm_ID_index_Z])
        ols_pfaX_im_M_R = fit_against_IM_and_M_Rjb(pfaX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X], Rjb[gm_ID_index_X])
        ols_pfaZ_im_M_R = fit_against_IM_and_M_Rjb(pfaZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z], Rjb[gm_ID_index_Z])
        
        sdrX_pred_M_R = ols_sdrX_im_M_R.get_prediction().predicted_mean
        sdrZ_pred_M_R = ols_sdrZ_im_M_R.get_prediction().predicted_mean
        pfaX_pred_M_R = ols_pfaX_im_M_R.get_prediction().predicted_mean
        pfaZ_pred_M_R = ols_pfaZ_im_M_R.get_prediction().predicted_mean

        prob_exced_sdrX_M_R = np.ones(len(sdrX_range)) - stats.norm.cdf((np.log(sdrX_range) - np.mean(sdrX_pred_M_R)) / np.std(ols_sdrX_im_M_R.resid))
        prob_exced_sdrZ_M_R = np.ones(len(sdrZ_range)) - stats.norm.cdf((np.log(sdrZ_range) - np.mean(sdrZ_pred_M_R)) / np.std(ols_sdrZ_im_M_R.resid))
        prob_exced_pfaX_M_R = np.ones(len(pfaX_range)) - stats.norm.cdf((np.log(pfaX_range) - np.mean(pfaX_pred_M_R)) / np.std(ols_pfaX_im_M_R.resid))
        prob_exced_pfaZ_M_R = np.ones(len(pfaZ_range)) - stats.norm.cdf((np.log(pfaZ_range) - np.mean(pfaZ_pred_M_R)) / np.std(ols_pfaZ_im_M_R.resid))


        # sdrX_pred_R = fit_against_IM_and_Rjb(sdrX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        # sdrZ_pred_R = fit_against_IM_and_Rjb(sdrZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z]).get_prediction().predicted_mean
        # pfaX_pred_R = fit_against_IM_and_Rjb(pfaX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        # pfaZ_pred_R = fit_against_IM_and_Rjb(pfaZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z]).get_prediction().predicted_mean
        
        # sdrX_pred_M_R = fit_against_IM_and_M_Rjb(sdrX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X], Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        # sdrZ_pred_M_R = fit_against_IM_and_M_Rjb(sdrZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z], Rjb[gm_ID_index_Z]).get_prediction().predicted_mean
        # pfaX_pred_M_R = fit_against_IM_and_M_Rjb(pfaX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X], Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        # pfaZ_pred_M_R = fit_against_IM_and_M_Rjb(pfaZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z], Rjb[gm_ID_index_Z]).get_prediction().predicted_mean

        
        # if KL_on_residual:
        #     sdrX_pred = OLS(sdrX, IM[gm_ID_index_X]).get_residual()
        #     sdrZ_pred = OLS(sdrZ, IM[gm_ID_index_Z]).get_residual()
        #     pfaX_pred = OLS(pfaX, IM[gm_ID_index_X]).get_residual()
        #     pfaZ_pred = OLS(pfaZ, IM[gm_ID_index_Z]).get_residual()

        #     sdrX_pred_M = fit_against_IM_and_M(sdrX, IM_M[gm_ID_index_X], M[gm_ID_index_X]).resid
        #     sdrZ_pred_M = fit_against_IM_and_M(sdrZ, IM_M[gm_ID_index_Z], M[gm_ID_index_Z]).resid
        #     pfaX_pred_M = fit_against_IM_and_M(pfaX, IM_M[gm_ID_index_X], M[gm_ID_index_X]).resid
        #     pfaZ_pred_M = fit_against_IM_and_M(pfaZ, IM_M[gm_ID_index_Z], M[gm_ID_index_Z]).resid

        #     sdrX_pred_R = fit_against_IM_and_Rjb(sdrX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X]).resid
        #     sdrZ_pred_R = fit_against_IM_and_Rjb(sdrZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z]).resid
        #     pfaX_pred_R = fit_against_IM_and_Rjb(pfaX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X]).resid
        #     pfaZ_pred_R = fit_against_IM_and_Rjb(pfaZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z]).resid
            
        #     sdrX_pred_M_R = fit_against_IM_and_M_Rjb(sdrX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X], Rjb[gm_ID_index_X]).resid
        #     sdrZ_pred_M_R = fit_against_IM_and_M_Rjb(sdrZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z], Rjb[gm_ID_index_Z]).resid
        #     pfaX_pred_M_R = fit_against_IM_and_M_Rjb(pfaX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X], Rjb[gm_ID_index_X]).resid
        #     pfaZ_pred_M_R = fit_against_IM_and_M_Rjb(pfaZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z], Rjb[gm_ID_index_Z]).resid
            
            ## scipy.stats.entropy normalizes pk and qk if they don't sum to 1
#         kld_sdrX.append(stats.entropy(pk=sdrX_pred, qk=sdrX_pred_M))
#         kld_sdrZ.append(stats.entropy(pk=sdrZ_pred, qk=sdrZ_pred_M))
#         kld_pfaX.append(stats.entropy(pk=pfaX_pred, qk=pfaX_pred_M))
#         kld_pfaZ.append(stats.entropy(pk=pfaZ_pred, qk=pfaZ_pred_M))
        
#         kld_sdrX_R.append(stats.entropy(pk=sdrX_pred, qk=sdrX_pred_R))
#         kld_sdrZ_R.append(stats.entropy(pk=sdrZ_pred, qk=sdrZ_pred_R))
#         kld_pfaX_R.append(stats.entropy(pk=pfaX_pred, qk=pfaX_pred_R))
#         kld_pfaZ_R.append(stats.entropy(pk=pfaZ_pred, qk=pfaZ_pred_R))
        
        kld_sdrX.append(compute_KL_divergence(p=prob_exced_sdrX, q=prob_exced_sdrX_M))
        kld_sdrZ.append(compute_KL_divergence(p=prob_exced_sdrZ, q=prob_exced_sdrZ_M))
        kld_pfaX.append(compute_KL_divergence(p=prob_exced_pfaX, q=prob_exced_pfaX_M))
        kld_pfaZ.append(compute_KL_divergence(p=prob_exced_pfaZ, q=prob_exced_pfaZ_M))
        
        kld_sdrX_R.append(compute_KL_divergence(p=prob_exced_sdrX, q=prob_exced_sdrX_R))
        kld_sdrZ_R.append(compute_KL_divergence(p=prob_exced_sdrZ, q=prob_exced_sdrZ_R))
        kld_pfaX_R.append(compute_KL_divergence(p=prob_exced_pfaX, q=prob_exced_pfaX_R))
        kld_pfaZ_R.append(compute_KL_divergence(p=prob_exced_pfaZ, q=prob_exced_pfaZ_R))
        
        kld_sdrX_M_R.append(compute_KL_divergence(p=prob_exced_sdrX, q=prob_exced_sdrX_M_R))
        kld_sdrZ_M_R.append(compute_KL_divergence(p=prob_exced_sdrZ, q=prob_exced_sdrZ_M_R))
        kld_pfaX_M_R.append(compute_KL_divergence(p=prob_exced_pfaX, q=prob_exced_pfaX_M_R))
        kld_pfaZ_M_R.append(compute_KL_divergence(p=prob_exced_pfaZ, q=prob_exced_pfaZ_M_R))

    d = {
        'IM': IM_List,
        'KL(sdrX|im, sdrX|im,M)': kld_sdrX,
        'KL(sdrZ|im, sdrZ|im,M)': kld_sdrZ,
        'KL(pfaX|im, pfaX|im,M)': kld_pfaX,
        'KL(pfaZ|im, pfaZ|im,M)': kld_pfaZ,
        'KL(sdrX|im, sdrX|im,R)': kld_sdrX_R,
        'KL(sdrZ|im, sdrZ|im,R)': kld_sdrZ_R,
        'KL(pfaX|im, pfaX|im,R)': kld_pfaX_R,
        'KL(pfaZ|im, pfaZ|im,R)': kld_pfaZ_R,
        'KL(sdrX|im, sdrX|im,M,R)': kld_sdrX_M_R,
        'KL(sdrZ|im, sdrZ|im,M,R)': kld_sdrZ_M_R,
        'KL(pfaX|im, pfaX|im,M,R)': kld_pfaX_M_R,
        'KL(pfaZ|im, pfaZ|im,M,R)': kld_pfaZ_M_R
    }
    df = pd.DataFrame(d)
    df = df.set_index('IM')
    return df.T



def summarize_kld_results(BuildingList, baseDir, IM_List, df_IMs, pairingID = 1,
                          numGM = 826, remove_collapse = False, KL_on_residual=False,
                         use_predicted_IM=False, mv_causal_params=False):
    d = {}

    for idx in range(len(BuildingList)):
        tempdf = compile_kld_prob_exceedance(BuildingList, idx, baseDir, IM_List, df_IMs, pairingID = pairingID,
                          numGM = numGM, remove_collapse = remove_collapse, KL_on_residual=KL_on_residual,
                         use_predicted_IM=use_predicted_IM)
        d[BuildingList[idx]] = tempdf.iloc[:8].sum().values
        if mv_causal_params:
            d[BuildingList[idx]] = tempdf.iloc[8:].sum().values

    d['IM'] = tempdf.iloc[:8].sum().keys()
    
    df = pd.DataFrame(d)
    df = df.set_index('IM')
    return df.T



def compile_kld_mvn_EDPs(BuildingList, building_Index, baseDir, IM_List, df_IMs, pairingID = 1,
                          numGM = 826, remove_collapse = False, KL_on_residual=False,
                         use_predicted_IM=False):
    ## time period of the buildings, used to get Sa(T1)
    # T = np.array([0.13, 0.12, 0.22, 0.22, 0.16, 0.15, 0.26, 0.25, 0.49, 0.49])
    T = np.array([0.13, 0.12, 0.16, 0.15, 0.22, 0.22, 0.26, 0.25, 0.49, 0.49])
    
    sdr_all = extract_EDP_data(baseDir, BuildingList, building_Index, edp_type='SDR', pairingID=pairingID,
                           numGM=numGM, remove_collapse=False)
    sdr = extract_EDP_data(baseDir, BuildingList, building_Index, edp_type='SDR', pairingID=pairingID,
                           numGM=numGM, remove_collapse=remove_collapse)
    
    sdrX = sdr[sdr['Direction'] == 'X']['Max_EDP'].values
    sdrZ = sdr[sdr['Direction'] == 'Z']['Max_EDP'].values
    pfa = extract_EDP_data(baseDir, BuildingList, building_Index, edp_type='PFA', pairingID=pairingID, numGM=numGM)
    pfa = pfa[pfa['GM_ID'].isin(sdr['GM_ID'].values)] # if collapse is remove, this gives consistent pfa data 
    pfaX = pfa[pfa['Direction'] == 'X']['Max_EDP'].values
    pfaZ = pfa[pfa['Direction'] == 'Z']['Max_EDP'].values
    
    gm_ID_index_X = [sdr_all[sdr_all['Direction']=='X']['GM_ID'].tolist().index(x) for x in sdr[sdr['Direction'] == 'X']['GM_ID'].values]
    gm_ID_index_Z = [sdr_all[sdr_all['Direction']=='Z']['GM_ID'].tolist().index(x) for x in sdr[sdr['Direction'] == 'Z']['GM_ID'].values]
    
    M = df_IMs['Magnitude'].values
    Rjb = df_IMs['Distance_Rjb'].values
    

    kld_edpX = []
    kld_edpZ = []
    # kld_pfaX = []
    # kld_pfaZ = []
    
    kld_edpX_R = []
    kld_edpZ_R = []
    # kld_pfaX_R = []
    # kld_pfaZ_R = []
    
    kld_edpX_M_R = []
    kld_edpZ_M_R = []
    # kld_pfaX_M_R = []
    # kld_pfaZ_M_R = []
    
    for i in range(len(IM_List)):

        if IM_List[i] == 'SaT1':
            IM = df_IMs['T_%s'%T[building_Index]].values
        else:
            IM = df_IMs[IM_List[i]]
        IM_M = IM 
        IM_R = IM
        IM_M_and_R = IM
        if use_predicted_IM:
            IM_M = fit_IM_against_M(IM, M).get_prediction().predicted_mean
            IM_M = fit_IM_against_Rjb(IM, Rjb).get_prediction().predicted_mean
            IM_M_and_R = fit_IM_against_M_and_Rjb(IM, M, Rjb).get_prediction().predicted_mean
#         print(len(IM[gm_ID_index_Z]), len(sdrZ))
#         print(len(IM[gm_ID_index_X]), len(sdrX))
        sdrX_pred = OLS(sdrX, IM[gm_ID_index_X]).get_pred_values()
        sdrZ_pred = OLS(sdrZ, IM[gm_ID_index_Z]).get_pred_values()
        pfaX_pred = OLS(pfaX, IM[gm_ID_index_X]).get_pred_values()
        pfaZ_pred = OLS(pfaZ, IM[gm_ID_index_Z]).get_pred_values()

        sdrX_pred_M = fit_against_IM_and_M(sdrX, IM_M[gm_ID_index_X],
                                           M[gm_ID_index_X]).get_prediction().predicted_mean
        sdrZ_pred_M = fit_against_IM_and_M(sdrZ, IM_M[gm_ID_index_Z],
                                           M[gm_ID_index_Z]).get_prediction().predicted_mean
        pfaX_pred_M = fit_against_IM_and_M(pfaX, IM_M[gm_ID_index_X],
                                           M[gm_ID_index_X]).get_prediction().predicted_mean
        pfaZ_pred_M = fit_against_IM_and_M(pfaZ, IM_M[gm_ID_index_Z],
                                           M[gm_ID_index_Z]).get_prediction().predicted_mean
        
        sdrX_pred_R = fit_against_IM_and_Rjb(sdrX, IM_R[gm_ID_index_X],
                                             Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        sdrZ_pred_R = fit_against_IM_and_Rjb(sdrZ, IM_R[gm_ID_index_Z],
                                             Rjb[gm_ID_index_Z]).get_prediction().predicted_mean
        pfaX_pred_R = fit_against_IM_and_Rjb(pfaX, IM_R[gm_ID_index_X],
                                             Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        pfaZ_pred_R = fit_against_IM_and_Rjb(pfaZ, IM_R[gm_ID_index_Z],
                                             Rjb[gm_ID_index_Z]).get_prediction().predicted_mean
        
        sdrX_pred_M_R = fit_against_IM_and_M_Rjb(sdrX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X],
                                             Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        sdrZ_pred_M_R = fit_against_IM_and_M_Rjb(sdrZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z],
                                             Rjb[gm_ID_index_Z]).get_prediction().predicted_mean
        pfaX_pred_M_R = fit_against_IM_and_M_Rjb(pfaX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X],
                                             Rjb[gm_ID_index_X]).get_prediction().predicted_mean
        pfaZ_pred_M_R = fit_against_IM_and_M_Rjb(pfaZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z],
                                             Rjb[gm_ID_index_Z]).get_prediction().predicted_mean

        
        if KL_on_residual:
            sdrX_pred = OLS(sdrX, IM[gm_ID_index_X]).get_residual()
            sdrZ_pred = OLS(sdrZ, IM[gm_ID_index_Z]).get_residual()
            pfaX_pred = OLS(pfaX, IM[gm_ID_index_X]).get_residual()
            pfaZ_pred = OLS(pfaZ, IM[gm_ID_index_Z]).get_residual()

            sdrX_pred_M = fit_against_IM_and_M(sdrX, IM_M[gm_ID_index_X], M[gm_ID_index_X]).resid
            sdrZ_pred_M = fit_against_IM_and_M(sdrZ, IM_M[gm_ID_index_Z], M[gm_ID_index_Z]).resid
            pfaX_pred_M = fit_against_IM_and_M(pfaX, IM_M[gm_ID_index_X], M[gm_ID_index_X]).resid
            pfaZ_pred_M = fit_against_IM_and_M(pfaZ, IM_M[gm_ID_index_Z], M[gm_ID_index_Z]).resid

            sdrX_pred_R = fit_against_IM_and_Rjb(sdrX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X]).resid
            sdrZ_pred_R = fit_against_IM_and_Rjb(sdrZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z]).resid
            pfaX_pred_R = fit_against_IM_and_Rjb(pfaX, IM_R[gm_ID_index_X], Rjb[gm_ID_index_X]).resid
            pfaZ_pred_R = fit_against_IM_and_Rjb(pfaZ, IM_R[gm_ID_index_Z], Rjb[gm_ID_index_Z]).resid
            
            sdrX_pred_M_R = fit_against_IM_and_M_Rjb(sdrX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X],
                                             Rjb[gm_ID_index_X]).resid
            sdrZ_pred_M_R = fit_against_IM_and_M_Rjb(sdrZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z],
                                                 Rjb[gm_ID_index_Z]).resid
            pfaX_pred_M_R = fit_against_IM_and_M_Rjb(pfaX, IM_M_and_R[gm_ID_index_X], M[gm_ID_index_X],
                                                 Rjb[gm_ID_index_X]).resid
            pfaZ_pred_M_R = fit_against_IM_and_M_Rjb(pfaZ, IM_M_and_R[gm_ID_index_Z], M[gm_ID_index_Z],
                                                 Rjb[gm_ID_index_Z]).resid
            
            ## scipy.stats.entropy normalizes pk and qk if they don't sum to 1
#         kld_sdrX.append(stats.entropy(pk=sdrX_pred, qk=sdrX_pred_M))
#         kld_sdrZ.append(stats.entropy(pk=sdrZ_pred, qk=sdrZ_pred_M))
#         kld_pfaX.append(stats.entropy(pk=pfaX_pred, qk=pfaX_pred_M))
#         kld_pfaZ.append(stats.entropy(pk=pfaZ_pred, qk=pfaZ_pred_M))
        
#         kld_sdrX_R.append(stats.entropy(pk=sdrX_pred, qk=sdrX_pred_R))
#         kld_sdrZ_R.append(stats.entropy(pk=sdrZ_pred, qk=sdrZ_pred_R))
#         kld_pfaX_R.append(stats.entropy(pk=pfaX_pred, qk=pfaX_pred_R))
#         kld_pfaZ_R.append(stats.entropy(pk=pfaZ_pred, qk=pfaZ_pred_R))
        
        kld_edpX.append(kl_divergence_mvn(edp1=sdrX_pred, edp2=pfaX_pred, edp3=sdrX_pred_M, edp4=pfaX_pred_M))
        kld_edpZ.append(kl_divergence_mvn(edp1=sdrZ_pred, edp2=pfaZ_pred, edp3=sdrZ_pred_M, edp4=pfaZ_pred_M))
        # kld_pfaX.append(compute_KL_divergence(p=pfaX_pred, q=pfaX_pred_M))
        # kld_pfaZ.append(compute_KL_divergence(p=pfaZ_pred, q=pfaZ_pred_M))
        
        kld_edpX_R.append(kl_divergence_mvn(edp1=sdrX_pred, edp2=pfaX_pred, edp3=sdrX_pred_R, edp4=pfaX_pred_R))
        kld_edpZ_R.append(kl_divergence_mvn(edp1=sdrZ_pred, edp2=pfaZ_pred, edp3=sdrZ_pred_R, edp4=pfaZ_pred_R))
        # kld_pfaX_R.append(compute_KL_divergence(p=pfaX_pred, q=pfaX_pred_R))
        # kld_pfaZ_R.append(compute_KL_divergence(p=pfaZ_pred, q=pfaZ_pred_R))
        
        kld_edpX_M_R.append(kl_divergence_mvn(edp1=sdrX_pred, edp2=pfaX_pred, edp3=sdrX_pred_M_R, edp4=pfaX_pred_M_R))
        kld_edpZ_M_R.append(kl_divergence_mvn(edp1=sdrZ_pred, edp2=pfaZ_pred, edp3=sdrZ_pred_M_R, edp4=pfaZ_pred_M_R))
        # kld_pfaX_M_R.append(compute_KL_divergence(p=pfaX_pred, q=pfaX_pred_M_R))
        # kld_pfaZ_M_R.append(compute_KL_divergence(p=pfaZ_pred, q=pfaZ_pred_M_R))

    d = {
        'IM': IM_List,
        'KL(sdr+pfaX|im, sdr+pfaX|im,M)': kld_edpX,
        'KL(sdr+pfaZ|im, sdr+pfaZ|im,M)': kld_edpZ,
        # 'KL(pfaX|im, pfaX|im,M)': kld_pfaX,
        # 'KL(pfaZ|im, pfaZ|im,M)': kld_pfaZ,
        'KL(sdr+pfaX|im, sdr+pfaX|im,R)': kld_edpX_R,
        'KL(sdr+pfaZ|im, sdr+pfaZ|im,R)': kld_edpZ_R,
        # 'KL(pfaX|im, pfaX|im,R)': kld_pfaX_R,
        # 'KL(pfaZ|im, pfaZ|im,R)': kld_pfaZ_R,
        'KL(sdr+pfaX|im, sdr+pfaX|im,M,R)': kld_edpX_M_R,
        'KL(sdr+pfaZ|im, sdr+pfaZ|im,M,R)': kld_edpZ_M_R#,
        # 'KL(pfaX|im, pfaX|im,M,R)': kld_pfaX_M_R,
        # 'KL(pfaZ|im, pfaZ|im,M,R)': kld_pfaZ_M_R
    }
    df = pd.DataFrame(d)
    df = df.set_index('IM')
    return df.T


def summarize_agg_kld_mvn(BuildingList, baseDir, IM_List, df_IMs, pairingID = 1,
                          numGM = 826, remove_collapse = False, KL_on_residual=False,
                         use_predicted_IM=False, mv_causal_params=False):
    d = {}

    for idx in range(len(BuildingList)):
        tempdf = compile_kld_mvn_EDPs(BuildingList, idx, baseDir, IM_List, df_IMs, pairingID = pairingID,
                          numGM = numGM, remove_collapse = remove_collapse, KL_on_residual=KL_on_residual,
                         use_predicted_IM=use_predicted_IM)
        d[BuildingList[idx]] = tempdf.iloc[:4].sum().values
        if mv_causal_params:
            d[BuildingList[idx]] = tempdf.iloc[4:].sum().values

    d['IM'] = tempdf.iloc[:8].sum().keys()
    
    df = pd.DataFrame(d)
    df = df.set_index('IM')
    return df.T