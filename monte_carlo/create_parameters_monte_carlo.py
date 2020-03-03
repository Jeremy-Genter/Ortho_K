import numpy as np
import os
import matplotlib as plt
import pickle
from copy import deepcopy
from scipy.stats import truncnorm
from my_functions import *

# create random values (normal distributed (bounded))

number_bins = 12
number_mc_simulations = number_bins**3
E_epi = np.zeros([number_bins, number_bins, number_bins])
k_epi = np.zeros([number_bins, number_bins, number_bins])
k_stroma = np.zeros([number_bins, number_bins, number_bins])
for i in range(number_bins):
    for ii in range(number_bins):
        for iii in range(number_bins):
            E_epi[i, ii, iii] = np.random.uniform(low=0.75e-3+i/number_bins*0.75e-3,
                                                  high=1.5e-3-(number_bins-1-i)/number_bins*0.75e-3,
                                                  size=(1, 1))
            k_stroma[i, ii, iii] = 10**np.random.uniform(low=-3.15 + ii / number_bins * 0.9,
                                                         high=-2.25 - (number_bins - 1 - ii)/number_bins*0.9,
                                                         size=(1, 1))
            if iii < number_bins/2:
                k_epi[i, ii, iii] = 10**np.random.uniform(low=-7.6 + iii/(number_bins/2)*1.2,
                                                          high=-6.4 - (number_bins/2 - 1 - iii)/(number_bins/2)*1.2,
                                                          size=(1, 1))
            else:
                k_epi[i, ii, iii] = 10**np.random.uniform(low=-5.5 + iii / (number_bins/2)*0.9,
                                                          high=-4.6 - (number_bins/2 - 1 - iii)/(number_bins/2)*0.9,
                                                          size=(1, 1))
E_epi = E_epi.reshape((-1, 1))
k_epi = k_epi.reshape((-1, 1))
k_stroma = k_stroma.reshape((-1, 1))
df = pd.DataFrame(np.concatenate((E_epi, k_epi, k_stroma), axis=1),
                  columns=['E_epi', 'k_epi', 'k_stroma'])

# histogram on linear scale
# plt.subplot(411)
# plt.hist(df['E_epi'], bins=100)
# plt.subplot(412)
# plt.hist(df['eye_lid_pressure'], bins=100)
#
# # histogram on log scale.
# # Use non-equal bin sizes, such that they look equal on log scale.
# ax1 = plt.subplot(413)
# hist, bins, _ = plt.hist(df['k_epi'], bins=100)
# ax1.cla()
# logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
# plt.hist(df['k_epi'], bins=logbins)
# plt.xscale('log')
# ax2 = plt.subplot(414)
# hist, bins, _ = plt.hist(df['k_stroma'], bins=100)
# ax2.cla()
# logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
# plt.hist(df['k_stroma'], bins=logbins)
# plt.xscale('log')
# plt.show()

axes = pd.plotting.scatter_matrix(df, diagonal='kde')
df.to_csv('combinations-orthoK_run2.csv')

