import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
import seaborn as sns

from utility_functions import calc_mutual_information, calc_mutual_information_sklearn
from efficiency_OLS import SummaryResutls_efficiency

label_mapping = {
    'SaT1': r'Sa$_{T1}$',
    'PGA': 'PGA', 
    'PGV': 'PGV',
    'Sa_avg': r'Sa$_{avg}$',
    'CAV': 'CAV',
    'SI':'SI',
    'ASI':'ASI',
    'DSI':'DSI',
    'DS_5to75': r'$DS_{5-75}$',
    'DS_5to95': r'$DS_{5-95}$'
}

colorList = ['b', 'green', 'darkorange', 'k', 'dodgerblue', 'mediumseagreen', 'r', 'darkviolet', 'hotpink', 'y']

def plot_efficiency_OLS(baseDir, BuildingList, buildingIndex, df_IMs, IM=['SaT1', 'PGA', 'PGV', 'Sa_avg', 'CAV'], savefig = False,
                   pairingID = 1, average_EDP = False, Uni_Direction = False, fileName = 'efficiency_OLS'):
    numStory = int(BuildingList[buildingIndex].split('_')[0][1])
    floor = np.arange(1, numStory + 1)
    
    summaryResult = SummaryResutls_efficiency(baseDir, BuildingList, buildingIndex, df_IMs, IM, pairingID = pairingID,
                                              average_EDP=average_EDP, rotate_EDP=Uni_Direction)
    minEDP = summaryResult.min().min()
    maxEDP = summaryResult.max().max()
    xtick_value = np.arange(np.round(minEDP, 1), 1, 0.1)
    xtick_value = np.arange(0.1, 1, 0.1)
    markerList = ['.', '*', '+', 'o', 's', 'p', 'd', 'X', 'v', 'D', 'P']
    # colorList = ['b', 'g', 'darkorange', 'k', 'dodgerblue', 'lime', 'r', 'c', 'm', 'y']
    # colorList = ['b', 'green', 'darkorange', 'k', 'dodgerblue', 'mediumseagreen', 'r', 'powderblue', 'm', 'y']
    colorList = ['b', 'green', 'darkorange', 'k', 'dodgerblue', 'mediumseagreen', 'r', 'darkviolet', 'hotpink', 'y']
    # labelList = [r'Sa$_{T1}$', 'PGA', 'PGV', r'$Sa_{avg}$', 'CAV', 'SI', 'ASI', 'DSI', r'$DS_{5-75}$',  r'$DS_{5-95}$' ]
    labelList = [label_mapping[s] for s in IM]
    
    if Uni_Direction:
        # fileName = 'RotEDP_RotD50'
        fig, axs = plt.subplots(1, 2, figsize=(10,6), sharey = True)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
#         fig.suptitle('%s Avg Rot (X+Z) EDP (pID 1&2) and GM RotD50'%BuildingList[buildingIndex], fontsize = 16)
        for i in range(len(IM)):
            axs[0].plot(summaryResult.loc['%s'%IM[i]][::2].values, floor, label = labelList[i], marker = markerList[i],
                       linewidth =1.65, markersize=5, color = colorList[i])
            axs[0].set_yticks(floor)
            axs[0].set_xticks(xtick_value)
            axs[0].set_xlabel(r'Dispersion ($\sigma_{SDR_{RotD50}|IM}$)', fontsize = 16)
            axs[0].set_ylabel('Floor Level', fontsize = 16)
            axs[1].plot(summaryResult.loc['%s'%IM[i]][1::2], floor, label = labelList[i], marker = markerList[i],
                       linewidth =1.65, markersize=5, color = colorList[i])
            axs[1].set_yticks(floor)
            axs[1].set_xticks(xtick_value)
            axs[1].set_xlabel(r'Dispersion ($\sigma_{PFA_{RotD50}|IM}$)', fontsize = 16)
#             axs[1].set_ylabel('Floor Level', fontsize = 16)
        plt.legend(bbox_to_anchor=(1.25, 0.5), loc='center', ncol=1, fontsize = 13)

    else:
        # fileName = 'AvgEDP_RotD50'
        fig, axs = plt.subplots(2, 2, figsize=(10,12), sharey = True)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        
#         fig.suptitle('%s EDP Pairing ID 1 and GM H1'%BuildingList[buildingIndex], fontsize = 16)
        for i in range(len(IM)):
            axs[0, 0].plot(summaryResult.loc['%s'%IM[i]][::4], floor, label = labelList[i], marker = markerList[i],
                           linewidth =1.55, markersize=5, color = colorList[i])
            axs[0, 0].set_yticks(floor)
            axs[0, 0].set_xticks(xtick_value)
            axs[0, 0].set_xlabel(r'Dispersion ($\sigma_{SDR_X|IM}$)', fontsize = 20)
            axs[0, 0].set_ylabel('Floor Level', fontsize = 20)
#             axs[0, 0].legend()
            axs[1, 0].plot(summaryResult.loc['%s'%IM[i]][1::4], floor, label = labelList[i], marker = markerList[i],
                           linewidth =1.65, markersize=5, color = colorList[i])
            axs[1, 0].set_yticks(floor)
            axs[1, 0].set_xticks(xtick_value)
            axs[1, 0].set_xlabel(r'Dispersion ($\sigma_{SDR_Z|IM}$)', fontsize = 20)
            axs[1, 0].set_ylabel('Floor Level', fontsize = 20)
            axs[0, 1].plot(summaryResult.loc['%s'%IM[i]][2::4], floor, label = labelList[i], marker = markerList[i],
                           linewidth =1.65, markersize=5, color = colorList[i])
            axs[0, 1].set_yticks(floor)
            axs[0, 1].set_xticks(xtick_value)
            axs[0, 1].set_xlabel(r'Dispersion ($\sigma_{PFA_X|IM}$)', fontsize = 20)
            axs[1, 1].plot(summaryResult.loc['%s'%IM[i]][3::4], floor, label = labelList[i], marker = markerList[i],
                           linewidth =1.55, markersize=5, color = colorList[i])
            axs[1, 1].set_yticks(floor)
            axs[1, 1].set_xticks(xtick_value)
            axs[1, 1].set_xlabel(r'Dispersion ($\sigma_{PFA_Z|IM}$)', fontsize = 20)
        plt.legend(bbox_to_anchor=(1.3, 1.1), loc='center', ncol=1, fontsize = 15)
    # dataDir = os.path.join(baseDir, *['Results', 'IM_study_826GMs', BuildingList[buildingIndex]])
    # os.chdir(dataDir)
    
    if savefig:
        # os.chdir(baseDir)
        plt.savefig('%s/Codes/plots/%s_%s.png'%(baseDir, BuildingList[buildingIndex], fileName), bbox_inches="tight")
    else:
        plt.show()


def plot_efficiency_OLS_portfolio(baseDir, efficiency_df, IM_list, EDP_type='SDR',
                                    savefig = False, fileName='efficiency_OLS_portfolio'):

    meanSFD = efficiency_df.loc[IM_list, 'meanSFD'].values
    meanMFD = efficiency_df.loc[IM_list, 'meanMFD'].values

    labels = [label_mapping[s] for s in IM_list]
    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots(figsize = (10,6.5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    rects1 = ax.bar(x - width/2, meanSFD, width, label='SFD', color = 'orangered')
    rects2 = ax.bar(x + width/2, meanMFD, width, label='MFD', color = 'royalblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if EDP_type == 'SDR':
        ax.set_ylabel(r'Average Dispersion ($\sigma_{SDR|IM}$)', fontsize = 22)
    elif EDP_type == 'PFA':
        ax.set_ylabel(r'Average Dispersion ($\sigma_{PFA|IM}$)', fontsize = 22)
    else: pass
    ax.set_xlabel('Intensity Measures', fontsize = 22)
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, rotation=90, ha = 'right')
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.legend(fontsize = 15)
    ax.grid(which='both', axis='y',  linewidth=1)
    ax.set_axisbelow(True)

    if savefig:
        plt.savefig('%s/Codes/plots/%s_%s.png'%(baseDir, fileName, EDP_type), bbox_inches="tight")
    else:
        plt.show()


def plot_efficiency_entropy(baseDir, BuildingList, buildingIndex, df_entropy, IM_list,
                            savefig = False, fileName = 'entropy_efficiency'):
    fig, ax = plt.subplots(figsize=(9,6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    # colorList = ['b', 'green', 'darkorange', 'k', 'dodgerblue', 'mediumseagreen', 'r', 'powderblue', 'm', 'y']
    
    labels = df_entropy.loc[BuildingList[buildingIndex], IM_list].keys()
    labels = [label_mapping[s] for s in labels]
    heights = np.round(df_entropy.loc[BuildingList[buildingIndex], IM_list].values, 2)
    bars = ax.barh(labels, heights, color=colorList)
    ax.set_xlim(np.min(heights) - 1.5, 0)
    ax.set_xlabel('Joint Entropy', fontsize = 20)
    ax.bar_label(bars, fontsize = 14, label_type='edge',fontweight='bold', padding=3)
    
    if savefig:
        plt.savefig('%s/Codes/plots/%s_%s.png'%(baseDir, BuildingList[buildingIndex], fileName), bbox_inches="tight")
    else:
        plt.show()


def plot_efficiency_entropy_portfolio(baseDir, df_entropy, IM_list,
                            savefig = False, fileName = 'entropy_efficiency_portfolio'):
    
    if 'mean' not in df_entropy.index:
        df_entropy.loc['mean'] = df_entropy.mean(axis=0)
    if 'meanSFD' not in df_entropy.index:
        df_entropy.loc['meanSFD'] = df_entropy.loc[['s1_48x32_high',
                                                    's1_48x32_veryhigh',
                                                    's2_48x32_high',
                                                    's2_48x32_veryhigh']].mean(axis=0)
    if 'meanMFD' not in df_entropy.index:
       df_entropy.loc['meanMFD'] =df_entropy.loc[['s1_96x48_high',
                                                    's1_96x48_veryhigh',
                                                    's2_96x48_high',
                                                    's2_96x48_veryhigh',
                                                    's4_96x48_high',
                                                    's4_96x48_veryhigh']].mean(axis=0)
    fig, ax = plt.subplots(figsize=(10,8))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    colorList = ['b', 'green', 'darkorange', 'k', 'dodgerblue', 'mediumseagreen', 'r', 'darkviolet', 'hotpink', 'y']

    labels = df_entropy.loc['meanSFD', IM_list].keys()
    labels = [label_mapping[s] for s in labels]
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    meanSFD = np.round(df_entropy.loc['meanSFD'].values, 2)
    meanMFD = np.round(df_entropy.loc['meanMFD'].values, 2)
    rects1 = ax.barh(x - width/2, meanSFD, width, label='SFD', color = 'darkorange')
    rects2 = ax.barh(x + width/2, meanMFD, width, label='MFD', color = 'forestgreen')
    # ax.set_xlim(np.min([min(meanSFD), min(meanMFD)]) - 1, 0)
    ax.set_xlabel('Average Joint Entropy', fontsize = 20)
    # ax.set_ylabel('Intensity Measures', fontsize = 20)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.legend(fontsize = 15)
    ax.grid(which='both', axis='x',  linewidth=1)
    ax.set_axisbelow(True)
    # ax.bar_label(bars, label_type='edge',fontweight='bold', padding=3)
    
    if savefig:
        os.chdir(baseDir)
        plt.savefig('Codes/plots/%s.png'%fileName, bbox_inches="tight")
    else:
        plt.show()



def plot_mutual_information(baseDir, building_df, hide_upper= False, savefig = False, figname='MFD6B'):

    a11 = calc_mutual_information(np.log(building_df.story_1_sdrX), np.log(building_df.story_1_sdrX), bins=15)
    a12 = calc_mutual_information(np.log(building_df.story_1_sdrX), np.log(building_df.story_1_pfaX), bins=15)
    a13 = calc_mutual_information(np.log(building_df.story_1_sdrX), np.log(building_df.SaT1), bins=15)
    a14 = calc_mutual_information(np.log(building_df.story_1_sdrX), building_df.Magnitude, bins=15)
    a15 = calc_mutual_information(np.log(building_df.story_1_sdrX), building_df.Distance_Rjb, bins=15)
    a22 = calc_mutual_information(np.log(building_df.story_1_pfaX), np.log(building_df.story_1_pfaX), bins=15)
    a23 = calc_mutual_information(np.log(building_df.story_1_pfaX), np.log(building_df.SaT1), bins=15)
    a24 = calc_mutual_information(np.log(building_df.story_1_pfaX), building_df.Magnitude, bins=15)
    a25 = calc_mutual_information(np.log(building_df.story_1_pfaX), building_df.Distance_Rjb, bins=15)
    a33 = calc_mutual_information(np.log(building_df.SaT1), np.log(building_df.SaT1), bins=15)
    a34 = calc_mutual_information(np.log(building_df.SaT1), building_df.Magnitude, bins=15)
    a35 = calc_mutual_information(np.log(building_df.SaT1), building_df.Distance_Rjb, bins=15)
    a44 = calc_mutual_information(building_df.Magnitude, building_df.Magnitude, bins=15)
    a45 = calc_mutual_information(building_df.Magnitude, building_df.Distance_Rjb, bins=15)
    a55 = calc_mutual_information(building_df.Distance_Rjb, building_df.Distance_Rjb, bins=15)

    tickList = [r'$\ln SDR_X$', r'$\lnPFA_X$', r'$\lnSa_{T1}$', 'M', 'R']
    idx = ['lnSDRX', 'lnPFAX', 'lnSaT1', 'M', 'R']
    d = {
        # 'Index': idx, 
        'lnSDRX': [a11, a12, a13, a14, a15],
        'lnPFAX': [a12, a22, a23, a24, a25],
        'lnSaT1': [a13, a23, a33, a34, a35],
        'M': [a14, a24, a34, a44, a45],
        'R': [a15, a25, a35, a45, a55]
    }
    df = pd.DataFrame(d)
    # df = df.set_index('Index')

    # plt.figure(figsize=(6,6))
    fig, ax = plt.subplots(figsize=(6,5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    if hide_upper:
        mask = np.triu(np.ones_like(df, dtype=bool))
#         mask = np.fill_diagonal(mask, False)
        # Create a custom divergin palette
#         cmap = sns.diverging_palette(100, 7, s=75, l=40, n=5, center="light", as_cmap=True)
        sns.heatmap(df, cmap="Greens",annot=True, mask=mask, annot_kws={"fontsize":'large'},
                xticklabels=tickList, yticklabels=tickList)
    else:
        sns.heatmap(df, cmap="Greens",annot=True, annot_kws={"fontsize":'large'},
                xticklabels=tickList, yticklabels=tickList)
        # ax.set_xticklabels()
        ax.set_xticklabels(tickList, rotation=90, ha = 'right')
        ax.set_yticklabels(tickList, rotation=0)
    if savefig:
        os.chdir(baseDir)
        plt.savefig('Codes/plots/MI_heatmap_%s.png'%figname, bbox_inches="tight")
    else:
        plt.show()

    return df