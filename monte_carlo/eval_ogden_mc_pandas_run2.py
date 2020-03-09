import os
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import scipy.stats as stats
from scipy.stats import kruskal
from my_functions import *


results_1 = pd.read_csv('results-orthoK_1.csv')
results_2 = pd.read_csv('results-orthoK_2.csv')
results_3 = pd.read_csv('contact_time_and_8h_cor.csv')
results_temp = pd.concat((results_1[:1000], results_2[1000:]), axis=0)
results = pd.concat((results_temp, results_3), axis=1)
results = pd.DataFrame.drop(results, columns=['R_n 1d', 'R_n 4d', 'contact time -2.75 D', 'thickness_central 4d',
                                              'thickness_midperi 4d', 'power_eye 4d'])
results['E_epi'] = results['E_epi']*1e3
results['thickness_central 1d'] = results['thickness_central 1d']*1e3
results['thickness_midperi 1d'] = results['thickness_midperi 1d']*1e3
results['contact time -2 D'] = results['contact time -2 D']/3600
# print(np.min(results['thickness_central 1d']))

column_name = list(results.columns)
rejected = pd.DataFrame(np.zeros([1, len(column_name)]), columns=column_name)
rejected = pd.DataFrame.drop(rejected, 0)
i = 0
j = 0
## paper values
# for ii in range(len(results)):
#     results.loc[ii, 'error'] = (results.loc[ii, 'thickness_central 1d']/9+1)**2 + (results.loc[ii, 'power_eye 1d']/2.25+0.6)**2
# for ii in range(len(results)):
#     if results['thickness_central 1d'][i] > -4.5 or results['power_eye 1d'][i] > -1.125 or results['power_eye 1d'][i] < -1.8 \
#             or results['contact time -2 D'][i] > 1*60 or results['contact time -2 D'][i] < 0.15*60 or \
#             results['keratometry 8h'][i] > -0.5 or results['keratometry 8h'][i] < -2.35:
#         rejected = pd.DataFrame.append(rejected, results.iloc[j])
#         results = pd.DataFrame.drop(results, i)
#     else:
#         j +=1
# 
#     i += 1

## patient 1 and 2 values
for ii in range(len(results)):
    results.loc[ii, 'error'] = (results.loc[ii, 'thickness_central 1d']/1.5+1)**2 + (results.loc[ii, 'power_eye 1d']/2.25+0.22)**2
for ii in range(len(results)):
    if results['contact time -2 D'][i] < 0:
        results['contact time -2 D'][i] = np.nan

    if np.isnan(results['power_eye 1d'][i]):
        results = pd.DataFrame.drop(results, i)
    elif results['thickness_central 1d'][i] < -3.5 or results['power_eye 1d'][i] > -0.13*2.25 or results['power_eye 1d'][i] < -0.43*2.25\
               or results['keratometry 8h'][i] > -0.2*2.25 or results['keratometry 8h'][i] < -2.25*0.75:
        rejected = pd.DataFrame.append(rejected, results.iloc[j])
        results = pd.DataFrame.drop(results, i)
    else:
        j +=1

    i += 1


plot_res = pd.DataFrame.drop(results, columns=['Unnamed: 0']) #'keratometry 8h',
total_res = pd.concat([results, rejected])
plot_rej = pd.DataFrame.drop(total_res, columns=['Unnamed: 0'])


plot_res = pd.DataFrame.rename(plot_res, columns={"thickness_central 1d": "$ECT_{16h}$",
                                                  "thickness_central 4d": "$ECT_{4d}$", "thickness_midperi 1d": "$EMT_{16h}$",
                                                  "thickness_midperi 4d": "$EMT_{4d}$", 'power_eye 1d': "$k_{mean 16h}$",
                                                  'power_eye 4d': '$k_{mean 4d}$', 'keratometry 8h': '$k_{mean 8h}$',
                                                  'contact time -2 D': '$t_{con}$', 'E_epi': '$E_{epi}$',
                                                  'k_epi': '$k_{epi}$', 'k_stroma': '$k_{stroma}$'})
plot_rej = pd.DataFrame.rename(plot_rej, columns={"thickness_central 1d": "$ECT_{16h}$",
                                                  "thickness_central 4d": "$ECT_{4d}$", "thickness_midperi 1d": "$EMT_{16h}$",
                                                  "thickness_midperi 4d": "$EMT_{4d}$", 'power_eye 1d': "$k_{mean 16h}$",
                                                  'power_eye 4d': '$k_{mean 4d}$', 'keratometry 8h': '$k_{mean 8h}$',
                                                  'contact time -2 D': '$t_{con}$', 'E_epi': '$E_{epi}$',
                                                  'k_epi': '$k_{epi}$', 'k_stroma': '$k_{stroma}$'})
##

myBasicCorr = plot_res.corr()
mask = np.zeros(myBasicCorr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
#fig, (ax1, ax2) = plt.subplots(1, 2)
plt.figure()
plt.suptitle('Monte Carlo based on Patient 1 and 2', size=20)
sns.heatmap(plot_res.corr(method='spearman'), annot=True, mask=mask)  # , ax=ax1)
axes = pd.plotting.scatter_matrix(plot_res, diagonal='kde')  # , ax=ax2)
plt.suptitle('Monte Carlo based on Patient 1 and 2', size=20)
#plt.suptitle('$p_{lid} = 0.18+/-0.05 kPa$', size=20)
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        if i < j:
            axes[i, j].set_visible(False)

axes = pd.plotting.scatter_matrix(plot_rej, diagonal='kde')
plt.suptitle('Unfiltered Monte Carlo', size=20)
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        if i < j:
            axes[i, j].set_visible(False)

print('E epi t-test:', kruskal(rejected['E_epi'].values, results['E_epi'].values))
print('k epi t-test:', kruskal(np.log10(rejected['k_epi'].values), np.log10(results['k_epi'].values)))
print('k stroma t-test:', kruskal(np.log10(rejected['k_stroma'].values), np.log10(results['k_stroma'].values)))
print('min error accepted bin: \n', results.loc[results['error'].argmin(), :], '\nmin error rejected bin: \n', rejected.loc[rejected['error'].argmin(), :] )

E_range = np.linspace(0.75, 1.5, 200)
k_epi_range = np.logspace(-8, -5.5, 200)
k_stroma_range = np.logspace(-3.5, -2.2, 200)

nparam_density_E = stats.kde.gaussian_kde(plot_res['$E_{epi}$'].values.ravel())
nparam_density_E = nparam_density_E(E_range).argmax()
nparam_density_k_epi = stats.kde.gaussian_kde(plot_res['$k_{epi}$'].values.ravel())
nparam_density_k_epi = nparam_density_k_epi(k_epi_range).argmax()
nparam_density_k_stroma = stats.kde.gaussian_kde(plot_res['$k_{stroma}$'].values.ravel())
nparam_density_k_stroma = nparam_density_k_stroma(k_stroma_range).argmax()

print('E_mean:', E_range[nparam_density_E], '\nk_epi_mean:', k_epi_range[nparam_density_k_epi], '\nk_stroma_mean:',
      k_stroma_range[nparam_density_k_stroma])
