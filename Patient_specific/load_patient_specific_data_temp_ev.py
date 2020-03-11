import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from my_functions import *
from scipy.interpolate import griddata
from copy import deepcopy
from mpl_toolkits import mplot3d

folders = ['Patient_3/pre/', 'Patient_3/lens_out/', 'Patient_3/20min/', 'Patient_3/40min/', 'Patient_3/60min/',
           'Patient_3/4hours/']

file_names = ['ELE_OD.CSV', 'ELE_OS.CSV', 'PAC_OD.CSV', 'PAC_OS.CSV']

R = np.zeros([6, 2])
R_x = np.zeros([6, 2])
R_y = np.zeros([6, 2])
power_eye = np.zeros([6, 2])
power_eye_x = np.zeros([6, 2])
power_eye_y = np.zeros([6, 2])

fig1, axs1 = plt.subplots(2, 3)
fig1.suptitle('Left Eye Patient 3', fontsize=16)
fig2, axs2 = plt.subplots(2, 3)
fig2.suptitle('Right Eye Patient 3', fontsize=16)
t = np.round([-8, 0, 36/60, 53/60, 80/60, 4+23/60], 2)+8
n = 1.3375

for loop in range(6):
    for loop_2 in range(2):

        an_surf = pd.read_csv(folders[loop] + file_names[loop_2], nrows=141, delimiter=';').values[:, 1:-1]*1e-3
        thickness = pd.read_csv(folders[loop] + file_names[loop_2+2], nrows=141, delimiter=';').values[:, 1:-1]*1e-3
        pos_surf = an_surf + thickness

        x = np.linspace(-7, 7, 141)
        y = np.linspace(-7, 7, 141)
        x, y = np.meshgrid(x, y)
        rho, phi = cart2pol(x, y)
        index_sim_ker = np.where(rho < 1.5)
        points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1)) ), axis=1)
        values = an_surf.reshape((-1, 1))
        j = 0
        while j < len(values[:, 0]):
            if np.sum(np.isnan(values[j, :])) > 0:
                values = np.delete(values, j, axis=0)
                points = np.delete(points, j, axis=0)
                continue
            j += 1
        rho_new = np.linspace(-6, 6, 40)
        phi_new = np.linspace(0, 2*np.pi, 15)
        rho_new, phi_new = np.meshgrid(rho_new, phi_new)
        grid_x, grid_y = pol2cart(rho_new, phi_new)

        ref_power = np.full((grid_x.shape), np.nan)
        R_map = np.full((grid_x.shape), np.nan)
        for i in range(ref_power.shape[0]):
            for ii in range(3, ref_power.shape[1]-3):
                temp = np.zeros([7, 2])
                temp[:, 0] = rho_new[i, ii-3:ii+4].reshape((-1, 1))[:, 0]
                temp[:, -1] = griddata(points, values, (grid_x[i, ii-3:ii+4], grid_y[i, ii-3:ii+4]),
                                       method='cubic')[:, 0]
                if np.isnan(temp[:, -1]).any() == False:
                    R_map[i, ii] = circ2_fit(temp)[0]
                    ref_power[i, ii] = (n-1)/(R_map[i, ii]*1e-3)
        # index_mean_R = np.where(rho_new < 1.5)
        # R[loop, loop_2] = np.nanmedian(R_map[index_mean_R])

        data_an = np.zeros([len(index_sim_ker[0]), 3])
        data_an[:, 0] = np.reshape(x[index_sim_ker], (-1, 1))[:, 0]
        data_an[:, 1] = np.reshape(y[index_sim_ker], (-1, 1))[:, 0]
        data_an[:, 2] = np.reshape(an_surf[index_sim_ker], (-1, 1))[:, 0]

        data_pos = np.zeros([len(index_sim_ker[0]), 3])
        data_pos[:, 0] = np.reshape(x[index_sim_ker], (-1, 1))[:, 0]
        data_pos[:, 1] = np.reshape(y[index_sim_ker], (-1, 1))[:, 0]
        data_pos[:, 2] = np.reshape(pos_surf[index_sim_ker], (-1, 1))[:, 0]

        data_an_plt = np.zeros([x.shape[0]*x.shape[1], 3])
        data_an_plt[:, 0] = np.reshape(x, (-1, 1))[:, 0]
        data_an_plt[:, 1] = np.reshape(y, (-1, 1))[:, 0]
        data_an_plt[:, 2] = np.reshape(an_surf, (-1, 1))[:, 0]

        data_pos_plt = np.zeros([x.shape[0]*x.shape[1], 3])
        data_pos_plt[:, 0] = np.reshape(x, (-1, 1))[:, 0]
        data_pos_plt[:, 1] = np.reshape(y, (-1, 1))[:, 0]
        data_pos_plt[:, 2] = np.reshape(pos_surf, (-1, 1))[:, 0]

        j = 0
        while j < len(data_an[:, 0]):
            if np.sum(np.isnan(data_an[j, :])) > 0:
                data_an = np.delete(data_an, j, axis=0)
                continue
            j += 1
        data_an = np.delete(data_an, 351, axis=0)  # 36
        j = 0
        while j < len(data_pos[:, 0]):
            if np.sum(np.isnan(data_pos[j, :])) > 0:
                data_pos = np.delete(data_pos, j, axis=0)
                continue
            j += 1


        # R_temp = keratometry(data_an)  # , mode='sphere')
        # R_temp = keratometry(data_an, mode='sphere')
        # R_x[loop, loop_2] = R_temp[3]['Rx']
        # R_y[loop, loop_2] = R_temp[3]['Ry']
        # R[loop, loop_2] = (R_x[loop, loop_2] + R_y[loop, loop_2])/2

        R[loop, loop_2] = sphere_fit(data_an)[0]

        # R_temp = ellipsoid_fit(data_an)
        # R[loop, loop_2] = (R_temp[0] + R_temp[1])/2
        # R_x[loop, loop_2] = R_temp[0]
        # R_y[loop, loop_2] = R_temp[1]

        power_eye[loop, loop_2] = (n - 1) / (R[loop, loop_2] * 1e-3)
        #power_eye_x[loop, loop_2] = (n - 1) / (R_x[loop, loop_2] * 1e-3)
        #power_eye_y[loop, loop_2] = (n - 1) / (R_y[loop, loop_2] * 1e-3)
        levels = np.linspace(38, 50, 150)
        if loop_2 == 0:

            if loop < 3:
                CS = axs1[0, loop].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
                if loop == 2:
                    cbar = fig1.colorbar(CS, ax=axs1[0, 2])
                    cbar.ax.set_ylabel('refractive power [D]')
                axs1[0, loop].set_title(str(t[loop]) + 'h')
                axs1[0, loop].set_xlabel('X-Coordinates [mm]')
                axs1[0, loop].set_ylabel('Y-Coordinates [mm]')
            else:
                CS = axs1[1, loop-3].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
                if loop == 5:
                    cbar = fig1.colorbar(CS, ax=axs1[1, 2])
                    cbar.ax.set_ylabel('refractive power [D]')
                axs1[1, loop-3].set_title(str(t[loop]) + 'h')
                axs1[1, loop-3].set_xlabel('X-Coordinates [mm]')
                axs1[1, loop-3].set_ylabel('Y-Coordinates [mm]')

        elif loop_2 == 1:

            if loop < 3:
                CS = axs2[0, loop].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
                if loop == 2:
                    cbar = fig2.colorbar(CS, ax=axs2[0, 2])
                    cbar.ax.set_ylabel('refractive power [D]')
                axs2[0, loop].set_title(str(t[loop]) + 'h')
                axs2[0, loop].set_xlabel('X-Coordinates [mm]')
                axs2[0, loop].set_ylabel('Y-Coordinates [mm]')
            else:
                CS = axs2[1, loop-3].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
                if loop == 5:
                    cbar = fig2.colorbar(CS, ax=axs2[1, 2])
                    cbar.ax.set_ylabel('refractive power [D]')
                axs2[1, loop-3].set_title(str(t[loop]) + 'h')
                axs2[1, loop-3].set_xlabel('X-Coordinates [mm]')
                axs2[1, loop-3].set_ylabel('Y-Coordinates [mm]')


        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(data_an_plt[:, 0], data_an_plt[:, 1], data_an_plt[:, 2], c=data_an_plt[:, 2])
        # ax.scatter3D(data_pos_plt[:, 0], data_pos_plt[:, 1], data_pos_plt[:, 2], c=data_pos_plt[:, 2])

fig, axs = plt.subplots()

l_name = ['Left Eye ', 'Right Eye ']
fig.suptitle('Temproal evolution Patient 3', fontsize=16)
r_x = np.zeros([6, 2])
r_y = np.zeros([6, 2])
r_x[:, 0] = [(7.43), (7.65), (7.58), (7.55), (7.56), (7.56)]
r_x[:, 1] = [(7.47), (7.63), (7.6), (7.56), (7.56), (7.54)]
r_y[:, 0] = [(7.33), 7.45, (7.38), (7.34), 7.31, 7.38]
r_y[:, 1] = [(7.3), (7.4), (7.38), (7.34), (7.29), (7.31)]
p_x = (n-1)/(r_x*1e-3)
p_y = (n-1)/(r_y*1e-3)

for i in range(2):
    axs.plot(t, power_eye[:, i]-power_eye[0, i], label=l_name[i] + 'R', marker='+')
    # axs.plot(t, power_eye_x[:, i] - power_eye_x[0, i], label=l_name[i] + 'R_x')
    # axs.plot(t, power_eye_y[:, i] - power_eye_y[0, i], label=l_name[i] + 'R_y')
    axs.plot(t, p_x[:, i] - p_x[0, i], label=l_name[i] + 'R_x pentacam', marker='+')
    axs.plot(t, p_y[:, i] - p_y[0, i], label=l_name[i] + 'R_y pentacam', marker='+')

leg = axs.legend(loc='lower right', fontsize=9)
axs.set_xlabel('time [h]', Fontsize=12)
axs.set_ylabel('refractive power change [D]', Fontsize=12)




