import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from my_functions import *
from copy import deepcopy

folder_patient_1_pre = 'Patient_1/PRE/'
folder_patient_1_post = 'Patient_1/POST/'
folder_patient_2_pre = 'Patient_2/PRE/'
folder_patient_2_post = 'Patient_2/POST_1day/'
folder = [folder_patient_1_pre, folder_patient_1_post, folder_patient_2_pre, folder_patient_2_post]

file_names = ['autoOD_', 'autoOI_']

R = np.zeros([8, 1])

epi_mean_l = np.zeros([8, 1])
epi_std_l = np.zeros([8, 1])
epi_mean_r = np.zeros([8, 1])
epi_std_r = np.zeros([8, 1])
fig, axs = plt.subplots()
fig.suptitle('Left Eye')
fig1, axs1 = plt.subplots()
fig1.suptitle('Right Eye')
for loop in range(4):
    epi_l = np.zeros([5001, 12])
    epi_r = np.zeros([5001, 12])
    for meas in range(1, 13):
        if meas < 10:
            epi_l_temp = np.asarray(pd.read_csv(folder[loop] + file_names[0] + '0' + str(meas) + '.dat'))
            epi_r_temp = np.asarray(pd.read_csv(folder[loop] + file_names[1] + '0' + str(meas) + '.dat'))
        else:
            epi_l_temp = np.asarray(pd.read_csv(folder[loop] + file_names[0] + str(meas) + '.dat'))
            epi_r_temp = np.asarray(pd.read_csv(folder[loop] + file_names[1] + str(meas) + '.dat'))
        epi_l[:, meas-1] = epi_l_temp[:, 1]
        epi_r[:, meas-1] = epi_r_temp[:, 1]
    axs.plot(epi_l_temp[:, 0], epi_l[:, 1:].mean(axis=1), label=folder[loop])
    leg = axs.legend(loc='lower right', fontsize=9)
    axs1.plot(epi_r_temp[:, 0], epi_r[:, 1:].mean(axis=1), label=folder[loop])
    leg = axs1.legend(loc='lower right', fontsize=9)

    epi_mean_l[loop] = epi_l[2500, :].mean()
    epi_std_l[loop] = epi_l[2500, :].std()
    epi_mean_r[loop] = epi_r[2500, :].mean()
    epi_std_r[loop] = epi_r[2500, :].std()

    epi_mean_l[loop+4] = epi_l[[0, -1], :].mean()
    epi_std_l[loop+4] = epi_l[[0, -1], :].std()
    epi_mean_r[loop+4] = epi_r[[0, -1], :].mean()
    epi_std_r[loop+4] = epi_r[[0, -1], :].std()

epi = np.zeros([4,1])
for i in range(2):
    epi[i] = epi_mean_l[2*i]-epi_mean_l[2*i+1]
    print('left eye central:', epi_mean_l[2*i]-epi_mean_l[2*i+1], '+/-', epi_std_l[i], 'mid-periphery:', epi_mean_l[i+4], '+/-', epi_std_l[i+4])
for i in range(2):
    epi[i+2] = epi_mean_r[2 * i] - epi_mean_r[2 * i + 1]
    print('right eye central:', epi_mean_r[2*i]-epi_mean_r[2*i+1], '+/-', epi_std_r[i], 'mid-periphery:', epi_mean_r[i+4], '+/-', epi_std_r[i+4])
print(epi.mean(), '+/-' + str(epi.std()))
fig, axs = plt.subplots()
fig.suptitle('Thickness Epithelium')
axs.set_ylabel('thickness epithleium [$\mu m$]')
axs.errorbar(['PreP1 central', 'PostP1 central', 'PreP 2 central', 'PostP 2 central',
              'PreP1 mid-periphery', 'PostP1 mid-periphery', 'PreP 2 mid-periphery', 'PostP2 mid-periphery'],
             epi_mean_l, yerr=epi_std_l, linestyle='None', marker='^', color='k', label='left eye')
axs.errorbar(['PreP1 central', 'PostP1 central', 'PreP 2 central', 'PostP 2 central',
              'PreP1 mid-periphery', 'PostP1 mid-periphery', 'PreP 2 mid-periphery', 'PostP2 mid-periphery'],
             epi_mean_r, yerr=epi_std_r, linestyle='None', marker='^', color='r', label='right eye')
leg = axs.legend(loc='lower right', fontsize=9)
