import os
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from scipy.stats import kruskal
from my_functions import *


results_1 = pd.read_csv('results-orthoK_1.csv')
results_2 = pd.read_csv('results-orthoK_2.csv')
results_3 = pd.read_csv('contact_time_and_8h_cor.csv')
results_temp = pd.concat((results_1[:1000], results_2[1000:]), axis=0)
results = pd.concat((results_temp, results_3), axis=1)
results = pd.DataFrame.drop(results, columns=['R_n 1d', 'R_n 4d', 'contact time -2.75 D'])
results['Eepi'] = results['Eepi']*1e3
results['eyelid-pressure'] = results['eyelid-pressure']*1e3
results['thickness_central 1d'] = results['thickness_central 1d']*1e3
results['thickness_central 4d'] = results['thickness_central 4d']*1e3
results['thickness_midperi 1d'] = results['thickness_midperi 1d']*1e3
results['thickness_midperi 4d'] = results['thickness_midperi 4d']*1e3
results['contact time -2 D'] = results['contact time -2 D']/60
print(np.min(results['thickness_central 1d']))

column_name = list(results.columns)
rejected = pd.DataFrame(np.zeros([1, len(column_name)]), columns=column_name)
rejected = pd.DataFrame.drop(rejected, 0)
i = 0
j = 0
## paper values
for ii in range(len(results)):
    results.loc[ii, 'error'] = (results.loc[ii, 'thickness_central 1d']/9+1)**2 + (results.loc[ii, 'power_eye 1d']/2.25+0.6)**2
for ii in range(len(results)):
    if results['thickness_central 1d'][i] > -4.5 or results['power_eye 1d'][i] > -1.125 or results['power_eye 1d'][i] < -1.8 \
            or results['contact time -2 D'][i] > 1*60 or results['contact time -2 D'][i] < 0.15*60 or \
            results['keratometry 8h'][i] > -0.5 or results['keratometry 8h'][i] < -2.35:
        rejected = pd.DataFrame.append(rejected, results.iloc[j])
        results = pd.DataFrame.drop(results, i)
    else:
        j +=1

    i += 1

## patient 1 and 2 values
# for ii in range(len(results)):
#     results.loc[ii, 'error'] = (results.loc[ii, 'thickness_central 1d']/9+1)**2 + (results.loc[ii, 'power_eye 1d']/2.25+0.17)**2
# for ii in range(len(results)):
#     if results['thickness_central 1d'][i] > -4.5 or results['power_eye 1d'][i] > -0.17*2.25 or results['power_eye 1d'][i] < -0.75*2.25\
#             or results['contact time -2 D'][i] > 1*60 or results['contact time -2 D'][i] < 0.15*60 or \
#             results['keratometry 8h'][i] > -0.5 or results['keratometry 8h'][i] < -2.35:
#         rejected = pd.DataFrame.append(rejected, results.iloc[j])
#         results = pd.DataFrame.drop(results, i)
#     else:
#         j +=1
#
#     i += 1


plot_res = pd.DataFrame.drop(results, columns=['Unnamed: 0']) #'keratometry 8h',
plot_rej = pd.DataFrame.drop(rejected, columns=['Unnamed: 0'])

plot_res = pd.DataFrame.rename(plot_res, columns={'eyelid-pressure': "$lid_p$", "thickness_central 1d": "$ECT_{16h}$",
                                                  "thickness_central 4d": "$ECT_{4d}$", "thickness_midperi 1d": "$EMT_{16h}$",
                                                  "thickness_midperi 4d": "$EMT_{4d}$", 'power_eye 1d': "$k_{mean 16h}$",
                                                  'power_eye 4d': '$k_{mean 4d}$', 'keratometry 8h': '$k_{mean 8h}$',
                                                  'contact time -2 D': '$t_{con}$', 'Eepi': '$E_{epi}$',
                                                  'Kepi': '$k_{epi}$', 'Kstroma': '$k_{stroma}$'})
plot_rej = pd.DataFrame.rename(plot_rej, columns={'eyelid-pressure': "$lid_p$", "thickness_central 1d": "$ECT_{16h}$",
                                                  "thickness_central 4d": "$ECT_{4d}$", "thickness_midperi 1d": "$EMT_{16h}$",
                                                  "thickness_midperi 4d": "$EMT_{4d}$", 'power_eye 1d': "$k_{mean 16h}$",
                                                  'power_eye 4d': '$k_{mean 4d}$', 'keratometry 8h': '$k_{mean 8h}$',
                                                  'contact time -2 D': '$t_{con}$', 'Eepi': '$E_{epi}$',
                                                  'Kepi': '$k_{epi}$', 'Kstroma': '$k_{stroma}$'})
##
fixed_value_0 = 0.18
fixed_value_1 = 0.20
fixed_value_2 = 0.22
tol = 0.05
check_bool_0 = np.logical_and(plot_res["$lid_p$"] >= fixed_value_0*(1-tol), plot_res["$lid_p$"] <= fixed_value_0*(1+tol))
check_bool_1 = np.logical_and(plot_res["$lid_p$"] >= fixed_value_1*(1-tol), plot_res["$lid_p$"] <= fixed_value_1*(1+tol))
check_bool_2 = np.logical_and(plot_res["$lid_p$"] >= fixed_value_2*(1-tol), plot_res["$lid_p$"] <= fixed_value_2*(1+tol))

plot_res_0 = plot_res.loc[check_bool_0, :]

myBasicCorr = plot_res_0.corr()
mask = np.zeros(myBasicCorr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
#fig, (ax1, ax2) = plt.subplots(1, 2)
plt.figure()
plt.suptitle('$p_{lid} = 0.18+/-0.05 kPa$', size=20)
sns.heatmap(plot_res_0.corr(method='spearman'), annot=True, mask=mask)  # , ax=ax1)
axes = pd.plotting.scatter_matrix(plot_res_0, diagonal='kde')  # , ax=ax2)
#plt.suptitle('$p_{lid} = 0.18+/-0.05 kPa$', size=20)
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        if i < j:
            axes[i, j].set_visible(False)

plot_res_1 = plot_res.loc[check_bool_1, :]
myBasicCorr = plot_res_1.corr()
mask = np.zeros(myBasicCorr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
plt.figure()
plt.suptitle('$p_{lid} = 0.2+/-0.05 kPa$', size=20)
sns.heatmap(plot_res_1.corr(method='spearman'), annot=True, mask=mask)
axes = pd.plotting.scatter_matrix(plot_res_1, diagonal='kde')
plt.suptitle('$p_{lid} = 0.2+/-0.05 kPa$', size=20)
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        if i < j:
            axes[i, j].set_visible(False)

plot_res_2 = plot_res.loc[check_bool_2, :]
myBasicCorr = plot_res_2.corr()
mask = np.zeros(myBasicCorr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
plt.figure()
plt.suptitle('$p_{lid} = 0.22+/-0.05 kPa$', size=20)
sns.heatmap(plot_res_2.corr(method='spearman'), annot=True, mask = mask)

axes = pd.plotting.scatter_matrix(plot_res_2, diagonal='kde')
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        if i < j:
            axes[i, j].set_visible(False)
plt.suptitle('$p_{lid} = 0.22+/-0.011 kPa$', size=20)

axes = pd.plotting.scatter_matrix(plot_rej, diagonal='kde')
for i in range(np.shape(axes)[0]):
    for j in range(np.shape(axes)[1]):
        if i < j:
            axes[i, j].set_visible(False)

print('eyelid pressure:', kruskal(rejected['eyelid-pressure'].values, results['eyelid-pressure'].values))
print('E epi t-test:', kruskal(rejected['Eepi'].values, results['Eepi'].values))
print('k epi t-test:', kruskal(np.log10(rejected['Kepi'].values), np.log10(results['Kepi'].values)))
print('k stroma t-test:', kruskal(np.log10(rejected['Kstroma'].values), np.log10(results['Kstroma'].values)))
print('accepted bin: \n', results.loc[results['error'].argmin(), :], '\n rejected bin: \n', rejected.loc[rejected['error'].argmin(), :] )