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

file_names = ['TopoOIpre.csv', 'TopoODpre.csv', 'TopoOIpost.csv', 'TopoODpost.csv']

R = np.zeros([4, 4])

an_surf_l = np.zeros([31, 256])
an_surf_r = np.zeros([31, 256])
an_power_l = np.zeros([31, 256])
an_power_r = np.zeros([31, 256])
thickness_l = np.zeros([31, 256])
thickness_r = np.zeros([31, 256])
pos_surf_l = np.zeros([31, 256])
pos_surf_r = np.zeros([31, 256])
fig1, axs1 = plt.subplots(1, 2)
fig2, axs2 = plt.subplots(1, 2)
fig1.suptitle('Left Eye Patient 1', fontsize=16)
fig2.suptitle('Right Eye Patient 1', fontsize=16)
fig3, axs3 = plt.subplots(1, 2)
fig4, axs4 = plt.subplots(1, 2)
fig3.suptitle('Left Eye Patient 2', fontsize=16)
fig4.suptitle('Right Eye Patient 2', fontsize=16)
for loop in range(1):
    if loop == 0:
        p_data_l = pd.read_csv(folder_patient_1_pre + file_names[0], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_1_pre + file_names[1], skiprows=2)
    elif loop == 1:
        p_data_l = pd.read_csv(folder_patient_1_post + file_names[2], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_1_post + file_names[3], skiprows=2)
    elif loop == 2:
        p_data_l = pd.read_csv(folder_patient_2_pre + file_names[0], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_2_pre + file_names[1], skiprows=2)
    else:
        p_data_l = pd.read_csv(folder_patient_2_post + file_names[2], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_2_post + file_names[3], skiprows=2)

    for i in range(31):
        an_surf_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i + 32].split(';'))[:-1].astype(float)
        an_surf_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i + 32].split(';'))[:-1].astype(float)
        thickness_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i].split(';'))[:-1].astype(float)
        thickness_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i].split(';'))[:-1].astype(float)
        an_power_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i + 128].split(';'))[:-1].astype(float)
        an_power_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i + 128].split(';'))[:-1].astype(float)
        #pos_surf_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i + 64].split(';'))[:-1].astype(float)[:-1]
        #pos_surf_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i + 64].split(';'))[:-1].astype(float)[:-1]

    pos_surf_l = an_surf_l + thickness_l*1e-3
    print(thickness_l[0, 0], thickness_r[0, 0])
    pos_surf_l[pos_surf_l < -5] = np.nan
    pos_surf_l[0, :] = pos_surf_l[0, 0]

    pos_surf_r = an_surf_r + thickness_r*1e-3
    pos_surf_r[pos_surf_r < -5] = np.nan
    pos_surf_r[0, :] = pos_surf_r[0, 0]

    an_surf_l[an_surf_l < -5] = np.nan
    an_surf_l[0, :] = an_surf_l[0, 0]

    an_power_l[an_power_l < -5] = np.nan
    an_power_l[0, :] = an_power_l[0, 0]

    an_surf_r[an_surf_r < -10] = np.nan
    an_surf_r[0, :] = an_surf_r[0, 0]

    an_power_r[an_power_r < -5] = np.nan
    an_power_r[0, :] = an_power_r[0, 0]

    y_2_an_l = np.nanmean(an_surf_l[15, :])
    y_3_an_l = np.nanmean(an_surf_l[22, :])
    y_4_an_l = np.nanmean(an_surf_l[-1, :])
    y_2_an_r = np.nanmean(an_surf_r[15, :])
    y_3_an_r = np.nanmean(an_surf_r[22, :])
    y_4_an_r = np.nanmean(an_surf_l[-1, :])

    y_1_pos_l = np.nanmean(pos_surf_l[7, :])-pos_surf_l[0,0]
    y_2_pos_l = np.nanmean(pos_surf_l[15, :])-pos_surf_l[0,0]
    y_3_pos_l = np.nanmean(pos_surf_l[25, 0])-pos_surf_l[0,0]
    y_1_pos_r = np.nanmean(pos_surf_r[7, :])-pos_surf_r[0,0]
    y_2_pos_r = np.nanmean(pos_surf_r[15, :])-pos_surf_r[0,0]
    y_3_pos_r = np.nanmean(pos_surf_r[25, 0])-pos_surf_r[0,0]

    print('y2 - y3 l/r anterior surface loop' + str(loop), y_2_an_l, y_3_an_l, y_4_an_l, y_2_an_r, y_3_an_r, y_4_an_l)
    print('y1 - y3 l/r posterior surface loop' + str(loop), y_1_pos_l, y_2_pos_l, y_3_pos_l, y_1_pos_r, y_2_pos_r, y_3_pos_r)

    theta = np.linspace(0, 2*np.pi, 256)
    r = np.linspace(0, 6, 31)
    theta, r = np.meshgrid(theta, r)
    x, y = pol2cart(r, theta)
    r_out_index = 7
    data_an_l = np.zeros([256*r_out_index, 3])
    data_an_l[:, 0] = np.reshape(x[:r_out_index, :], (-1, 1))[:, 0]
    data_an_l[:, 1] = np.reshape(y[:r_out_index, :], (-1, 1))[:, 0]
    data_an_l[:, 2] = np.reshape(an_surf_l[:r_out_index, :], (-1, 1))[:, 0]
    j = 0
    data_an_r = deepcopy(data_an_l)
    data_an_r[:, 2] = np.reshape(an_surf_r[:r_out_index, :], (-1, 1))[:, 0]
    data_pos_l = deepcopy(data_an_l)
    data_pos_l[:, 2] = np.reshape(pos_surf_l[:r_out_index, :], (-1, 1))[:, 0]
    data_pos_r = deepcopy(data_an_l)
    data_pos_r[:, 2] = np.reshape(pos_surf_r[:r_out_index, :], (-1, 1))[:, 0]
    while j < len(data_an_l[:, 0]):
        if np.sum(np.isnan(data_an_l[j, :])) > 0:
            data_an_l = np.delete(data_an_l, j, axis=0)
            continue
        j += 1
    while j < len(data_an_r[:, 0]):
        if np.sum(np.isnan(data_an_r[j, :])) > 0:
            data_an_r = np.delete(data_an_r, j, axis=0)
            continue
        j += 1
    while j < len(data_pos_l[:, 0]):
        if np.sum(np.isnan(data_pos_l[j, :])) > 0:
            data_an_r = np.delete(data_pos_l, j, axis=0)
            continue
        j += 1
    while j < len(data_pos_r[:, 0]):
        if np.sum(np.isnan(data_pos_r[j, :])) > 0:
            data_pos_r = np.delete(data_pos_r, j, axis=0)
            continue
        j += 1
    data_pos_l[:, 2] = data_pos_l[:, 2] - (data_pos_l[:256, 2]).min()
    data_pos_r[:, 2] = data_pos_r[:, 2] - (data_pos_r[:256, 2]).min()
    for n in range(4):
        if n == 0:
            data_temp = data_an_l[256:, :]
        elif n == 1:
            data_temp = data_an_r[256:, :]
        elif n == 2:
            data_temp = data_pos_l[256:, :]
        elif n == 3:
            data_temp = data_pos_r[256:, :]
        R_temp = keratometry(data_temp)  # , mode='sphere')
        #R_temp = nm_biconic_fit(data_temp)
        #R[loop, n] = sphere_fit(data_temp)[0]
        #R[loop, n] = (R_temp[0] + R_temp[1]) / 2
        R[loop, n] = (R_temp[3]['Rx'] + R_temp[3]['Ry'])/2
        print(R_temp)

    if loop == 0:
        # levels = np.linspace(0, 3, 150)
        # CS = axs1[0, 0].contourf(x, y, an_surf_l, levels=levels, cmap='rainbow')
        # cbar = fig1.colorbar(CS, ax=axs1[0, 0])
        # cbar.ax.set_ylabel('elevation [$\mu m$]')
        # axs1[0, 0].set_title('Anterior Surface Elevation Before Treatment')
        # axs1[0, 0].set_ylabel('Y-Coordinates [mm]')

        levels = np.linspace(40, 55, 150)
        CS = axs1[0].contourf(x, y, an_power_l, levels=levels, cmap='rainbow')
        cbar = fig1.colorbar(CS, ax=axs1[0])
        cbar.ax.set_ylabel('refractive power [D]')
        axs1[0].set_title('Anterior Surface Power Left Before Treatment')
        axs1[0].set_xlabel('X-Coordinates [mm]')
        axs1[0].set_ylabel('Y-Coordinates [mm]')

        # levels = np.linspace(0, 3, 150)
        # CS = axs2[0, 0].contourf(x, y, an_surf_r, levels=levels, cmap='rainbow')
        # cbar = fig2.colorbar(CS, ax=axs2[0, 0])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs2[0, 0].set_title('Anterior Surface Elevation Before Treatment')
        # axs2[0, 0].set_ylabel('Y-Coordinates [mm]')

        levels = np.linspace(40, 55, 150)
        CS = axs2[0].contourf(x, y, an_power_r, levels=levels, cmap='rainbow')
        cbar = fig2.colorbar(CS, ax=axs2[0])
        cbar.ax.set_ylabel('refractive power [D]')
        axs2[0].set_title('Anterior Surface Power Left Before Treatment')
        axs2[0].set_xlabel('X-Coordinates [mm]')
        axs2[0].set_ylabel('Y-Coordinates [mm]')
    elif loop == 1:
        # levels = np.linspace(0, 3, 150)
        # CS = axs1[0, 1].contourf(x, y, an_surf_l, levels=levels, cmap='rainbow')
        # cbar = fig1.colorbar(CS, ax=axs1[0, 1])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs1[0, 1].set_title('Anterior Surface Elevation Right After Treatment')

        levels = np.linspace(40, 55, 150)
        CS = axs1[1].contourf(x, y, an_power_l, levels=levels, cmap='rainbow')
        cbar = fig1.colorbar(CS, ax=axs1[1])
        cbar.ax.set_ylabel('refractive power [D]')
        axs1[1].set_title('Anterior Surface Power After Treatment')
        axs1[1].set_xlabel('X-Coordinates [mm]')

        # levels = np.linspace(0, 3, 150)
        # CS = axs2[0, 1].contourf(x, y, an_surf_r, levels=levels, cmap='rainbow')
        # cbar = fig2.colorbar(CS, ax=axs2[0, 1])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs2[0, 1].set_title('Anterior Surface Elevation Right After Treatment')

        levels = np.linspace(40, 55, 150)
        CS = axs2[1].contourf(x, y, an_power_r, levels=levels, cmap='rainbow')
        cbar = fig2.colorbar(CS, ax=axs2[1])
        cbar.ax.set_ylabel('refractive power [D]')
        axs2[1].set_title('Anterior Surface Power After Treatment')
        axs2[1].set_xlabel('X-Coordinates [mm]')

    elif loop == 2:
        # levels = np.linspace(0, 3, 150)
        # CS = axs3[0, 0].contourf(x, y, an_surf_l, levels=levels, cmap='rainbow')
        # cbar = fig3.colorbar(CS, ax=axs3[0, 0])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs3[0, 0].set_title('Anterior Surface Elevation Before Treatment')
        # axs3[0, 0].set_ylabel('Y-Coordinates [mm]')

        levels = np.linspace(40, 55, 150)
        CS = axs3[0].contourf(x, y, an_power_l, levels=levels, cmap='rainbow')
        cbar = fig3.colorbar(CS, ax=axs3[0])
        cbar.ax.set_ylabel('refractive power [D]')
        axs3[0].set_title('Anterior Surface Power Left Before Treatment')
        axs3[0].set_xlabel('X-Coordinates [mm]')
        axs3[0].set_ylabel('Y-Coordinates [mm]')

        # levels = np.linspace(0, 3, 150)
        # CS = axs4[0, 0].contourf(x, y, an_surf_r, levels=levels, cmap='rainbow')
        # cbar = fig4.colorbar(CS, ax=axs4[0, 0])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs4[0, 0].set_title('Anterior Surface Elevation Before Treatment')
        # axs4[0, 0].set_ylabel('Y-Coordinates [mm]')

        levels = np.linspace(40, 55, 150)
        CS = axs4[0].contourf(x, y, an_power_r, levels=levels, cmap='rainbow')
        cbar = fig4.colorbar(CS, ax=axs4[0])
        cbar.ax.set_ylabel('refractive power [D]')
        axs4[0].set_title('Anterior Surface Power Left Before Treatment')
        axs4[0].set_xlabel('X-Coordinates [mm]')
        axs4[0].set_ylabel('Y-Coordinates [mm]')

    else:
        # levels = np.linspace(0, 3, 150)
        # CS = axs3[0, 1].contourf(x, y, an_surf_l, levels=levels, cmap='rainbow')
        # cbar = fig1.colorbar(CS, ax=axs3[0, 1])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs3[0, 1].set_title('Anterior Surface Elevation Right After Treatment')

        levels = np.linspace(40, 55, 150)
        CS = axs3[1].contourf(x, y, an_power_l, levels=levels, cmap='rainbow')
        cbar = fig3.colorbar(CS, ax=axs3[1])
        cbar.ax.set_ylabel('refractive power [D]')
        axs3[1].set_title('Anterior Surface Power After Treatment')
        axs3[1].set_xlabel('X-Coordinates [mm]')

        # levels = np.linspace(0, 3, 150)
        # CS = axs4[0, 1].contourf(x, y, an_surf_r, levels=levels, cmap='rainbow')
        # cbar = fig4.colorbar(CS, ax=axs4[0, 1])
        # cbar.ax.set_ylabel('elevation [mm]')
        # axs4[0, 1].set_title('Anterior Surface Elevation Right After Treatment')

        levels = np.linspace(40, 55, 150)
        CS = axs4[1].contourf(x, y, an_power_r, levels=levels, cmap='rainbow')
        cbar = fig4.colorbar(CS, ax=axs4[1])
        cbar.ax.set_ylabel('refractive power [D]')
        axs4[1].set_title('Anterior Surface Power After Treatment')
        axs4[1].set_xlabel('X-Coordinates [mm]')
print(R)
print(0.3375/(R*1e-3))



