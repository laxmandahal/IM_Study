import os 
import numpy as np 
import pandas as pd 


# with open('BuildingNames.txt', 'r') as f:
#     BuildingList = f.read() 
# BuildingList = BuildingList.split('\n')
# BuildingList


def extract_EDP_data(baseDir, BuildingList, buildingIndex, edp_type = 'SDR', pairingID = 1, numGM = 826, 
                    remove_collapse = False):
    
    dataDir = os.path.join(baseDir, *['Results', 'IM_study_826GMs', BuildingList[buildingIndex]])
#     dataDir = dataDir = os.path.join(baseDir, *['Results', 'IM_study', BuildingList[buildingIndex]])
    edp = pd.read_csv(os.path.join(dataDir, '%s.csv'%edp_type), header = None)
    
    numStory = int(BuildingList[buildingIndex].split('_')[0][1])
    base_col_names = ['Hazard_Level', 'Pairing_ID', 'GM_ID']
    if edp_type == 'PFA':
        story_col_name = ['Story_%i'%x for x in range(numStory + 1)]
    else:
        story_col_name = ['Story_%i'%x for x in range(1, numStory + 1)]
    col_names = base_col_names + story_col_name
    edp = edp.rename(columns = dict(zip(edp.columns, col_names)))
    
    if pairingID == 1:
        edp = edp[edp['Pairing_ID'] == pairingID]
        start_index_multiplier = 0
        end_index_multiplier = 1
    elif pairingID == 2:
        edp = edp[edp['Pairing_ID'] == pairingID]
        start_index_multiplier = 2
        end_index_multiplier = 3
    else:
        pass
        
    direction = []
    for i in range(len(edp)):
        if ((edp['GM_ID'].values[i]> numGM * start_index_multiplier) &
            (edp['GM_ID'].values[i]<= numGM * end_index_multiplier)):
            direction.append('X')
        elif ((edp['GM_ID'].values[i] > numGM * end_index_multiplier) &
              (edp['GM_ID'].values[i] <= numGM * (end_index_multiplier + 1))):
            direction.append('Z')
        else:
            pass

    edp['Direction'] = direction
    if edp_type == 'SDR':
        if remove_collapse:
            edp = edp[edp['Story_1'] < 0.1]
    edp['Max_EDP'] = edp.loc[:,story_col_name].max(axis=1)
#     print(len(direction))
    return edp