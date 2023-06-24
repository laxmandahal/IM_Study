import numpy as np 
import os 
import sys
import pandas as pd 

sys.path.append('/u/home/l/laxmanda/project-hvburton/IM_study/ATC116_archetypes/Codes/Hoffman')
import ExtractMaxEDP as extractedps


cwd = r'/u/home/l/laxmanda/project-hvburton/IM_study/ATC116_archetypes'


NumGM = np.array([50, 47, 47, 48, 47])
HazardLevel = np.array([0.403, 0.975, 1.307, 1.676, 2.237])


CollapseCriteria = 0.1
DemolitionCriteria = 0.01


dynamicDirectory = os.path.join(cwd, *['BuildingModels', 's4_96x48_veryhigh','DynamicAnalysis'])

NumStory = 4

SDR = extractedps.ExtractSDR(dynamicDirectory, HazardLevel, NumGM, NumStory)
numCount = extractedps.Count(SDR, CollapseCriteria, NumGM)
CollapseCount = pd.DataFrame(numCount)

os.chdir(os.path.join(cwd, *['Results', 'IM_study']))
SDR.to_csv('SDR.csv', sep=',', header = False, index = False)
CollapseCount.to_csv('CollapseCount.csv', header = False, index = False)