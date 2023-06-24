import numpy as np 
import pandas as pd
import statsmodels.formula.api as smf

def fit_against_IM_and_M(edp, IM, M):
    '''
    return Model fit result
    '''
    temp = {'I': np.ones(shape = len(edp)),
        'lnEDP': np.log(edp),
        'lnSaT1': np.log(IM),
        'M': M}
    dummydf = pd.DataFrame(temp, index = None)
    model = smf.ols('lnEDP ~ lnSaT1 + M', data = dummydf)
    model_res = model.fit()
    # model_res = model.fit(cov_type = 'hc0', optim_hessian = 'eim')
    return model_res
    
def fit_against_IM_and_Rjb(edp, IM, Rjb):
    '''
    return Model fit result
    '''
    temp = {'I': np.ones(shape = len(edp)),
        'lnEDP': np.log(edp),
        'lnSaT1': np.log(IM),
        'Rjb': np.log(Rjb)}
    dummydf = pd.DataFrame(temp, index = None)
    model = smf.ols('lnEDP ~ lnSaT1 + Rjb', data = dummydf)
    model_res = model.fit()
    return model_res

def fit_against_IM_and_M_Rjb(edp, IM, M, Rjb):
    '''
    return Model fit result
    '''
    temp = {'I': np.ones(shape = len(edp)),
        'lnEDP': np.log(edp),
        'lnSaT1': np.log(IM),
        'M': M,
        'Rjb': np.log(Rjb)}
    dummydf = pd.DataFrame(temp, index = None)
    model = smf.ols('lnEDP ~ lnSaT1 + M + Rjb', data = dummydf)
    model_res = model.fit()
    return model_res

def fit_IM_against_M(IM, M):
    '''
    return Model fit result
    '''
    temp = {'I': np.ones(shape = len(IM)),
        'lnSaT1': np.log(IM),
        'M': np.log(M)
        }
    dummydf = pd.DataFrame(temp, index = None)
    model = smf.ols('lnSaT1 ~ M ', data = dummydf)
    model_res = model.fit()
    return model_res

def fit_IM_against_Rjb(IM, Rjb):
    '''
    return Model fit result
    '''
    temp = {'I': np.ones(shape = len(IM)),
        'lnSaT1': np.log(IM),
        'Rjb': np.log(Rjb)
        }
    dummydf = pd.DataFrame(temp, index = None)
    model = smf.ols('lnSaT1 ~ Rjb', data = dummydf)
    model_res = model.fit()
    return model_res

def fit_IM_against_M_and_Rjb(IM, M, Rjb):
    '''
    return Model fit result
    '''
    temp = {'I': np.ones(shape = len(IM)),
        'lnSaT1': np.log(IM),
        'M': np.log(M),
        'Rjb': np.log(Rjb)}
    dummydf = pd.DataFrame(temp, index = None)
    model = smf.ols('lnSaT1 ~ M + Rjb', data = dummydf)
    model_res = model.fit()
    return model_res
