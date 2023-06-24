"""
This file implements Generalized Linear Models (GLM) to fit the collapse fragility function. 

"""

import numpy as np 
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm 


        
def ProbitFit(hazardLevel, collapseCount, numGM):
    '''
    This function implements GLM with Probit Link function. Statsmodel.api python package is used. 

    :param hazardLevel: Intensity Measures (IM) used to conduct nonlinear dynamic analysis. type: array
    :param collapseCount: the count of collapses at each hazard level. type: array
    :param numGM: the total number of ground motion records at each hazard level. type: array
    '''
    logIM = np.log(hazardLevel)

    nonCollapseCount = numGM - collapseCount
    data = pd.DataFrame({ 'logIM':logIM, 'numCollapse':collapseCount, 'nonCollapse':nonCollapseCount })
    formula = 'numCollapse + nonCollapse ~  logIM'
    
    model = sm.GLM.from_formula( formula = formula, data = data, 
                        family=sm.families.Binomial(link = sm.families.links.probit()))
    

    fit = model.fit()
    summaryReport = fit.summary()
    theta = np.exp(-fit.params[0]/fit.params[1])
    beta = 1/fit.params[1]

    fittedProbCollapse = norm.cdf(fit.params[0] + fit.params[1] * logIM)
    return theta, beta


        
   

if __name__ == '__main__':
    #the inputs have to be an numpy array not list
    hazard_levels = np.array([0.178, 0.274, 0.444, 0.56, 0.652, 0.79, 0.982, 1.246, 1.564, 2.014, 2.417, 3.021, 3.625, 4.028, 4.431, 5.035])
    collapse_counts = np.array([0, 0, 0, 0, 0, 1, 4, 6, 14, 14, 22, 31, 34, 36, 37, 43])
    numGM = np.array([45] * len(hazard_levels))
    theta, beta = ProbitFit(hazard_levels, collapse_counts, numGM)
    print(theta, beta)
