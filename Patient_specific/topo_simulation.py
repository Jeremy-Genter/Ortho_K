import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from my_functions import *
from scipy.interpolate import griddata
from copy import deepcopy
from mpl_toolkits import mplot3d

folders = ['Patient_1/PRE/', 'Patient_1/POST/']

file_names = ['TopoOIpre.csv', 'TopoOIpost.csv']

fig1, axs1 = plt.subplots(2, 2)
fig1.suptitle('Left Eye Patient 1: Quater Model', fontsize=16)
t_meas = [64, 16*3600+64]
n = 1.3375

dir_list = ['dir00']  # os.listdir(os.getcwd())

rho_new = np.linspace(-6, 6, 40)
phi_new = np.linspace(0, 2*np.pi, 15)
rho_new, phi_new = np.meshgrid(rho_new, phi_new)
grid_x, grid_y = pol2cart(rho_new, phi_new)
levels = np.linspace(38, 50, 150)

for k in dir_list:

    # read dat files
    path2file = os.path.join(k, 'anterior_surface.dat')
    t, nodes_t = load_output_dat_file(path2file)

    # reshape results from dat file
    t = np.asarray(t)
    nodes_t = np.asarray(nodes_t)
    nodes_index = nodes_t[0:int(len(nodes_t) / len(t)), 0]
    nodes_t = np.reshape(np.asarray(nodes_t), (len(t), len(nodes_index) * 4), order='C')

    x_tot = np.zeros((len(nodes_index), 3 * len(t)))

    iii = 0
    # remove index
    for steps in t:
        ii = 0
        for i in nodes_index:
            temp = nodes_t[iii,
                   int(np.where(nodes_t[iii, :] == i)[0][0]): int(np.where(nodes_t[iii, :] == i)[0][0]) + 4]
            x_tot[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        iii += 1
    kk_ = 0
    for time in t_meas:
        t_index = np.argmin(np.abs(t - time))
        pos = x_tot[:, t_index * 3:(t_index + 1) * 3]
        pos_2 = deepcopy(pos)
        pos_2[:, 1] = -pos_2[:, 1]
        pos = np.concatenate((pos, pos_2), axis=0)
        pos_2 = deepcopy(pos)
        pos_2[:, 0] = -pos_2[:, 0]
        pos = np.concatenate((pos, pos_2), axis=0)

        x = pos[:,0]
        y = pos[:,1]
        z = pos[:, 2] - np.min(pos[:, 2])

        rho, phi = cart2pol(x, y)
        index_sim_ker = np.where(rho < 1.5)
        points = pos[:, :-1]
        values = z
        j = 0
        while j < len(values):
            if np.sum(np.isnan(values[j])) > 0:
                values = np.delete(values, j, axis=0)
                points = np.delete(points, j, axis=0)
                continue
            j += 1

        ref_power = np.full((grid_x.shape), np.nan)
        R_map = np.full((grid_x.shape), np.nan)
        for i in range(ref_power.shape[0]):
            for ii in range(3, ref_power.shape[1]-3):
                temp = np.zeros([7, 2])
                temp[:, 0] = rho_new[i, ii-3:ii+4].reshape((-1, 1))[:, 0]
                temp[:, -1] = griddata(points, values, (grid_x[i, ii-3:ii+4], grid_y[i, ii-3:ii+4]), method='cubic')
                if np.isnan(temp[:, -1]).any() == False:
                    R_map[i, ii] = circ2_fit(temp)[0]
                    ref_power[i, ii] = (n-1)/(R_map[i, ii]*1e-3)




        CS = axs1[kk_, 0].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
        axs1[kk_, 0].set_title(str(t_meas[kk_]) + 'h')
        axs1[kk_, 0].set_xlabel('X-Coordinates [mm]')
        axs1[kk_, 0].set_ylabel('Y-Coordinates [mm]')
        kk_ += 1

    kk_ = 0
    an_surf = np.zeros([31, 256])
    for time in t_meas:
        an_surf[1:, :] = pd.read_csv(folders[kk_] + file_names[kk_], skiprows=35, nrows=30, delimiter=';').values[:, :-1]
        an_surf[an_surf < -5] = np.nan
        an_surf[:, 0] = (an_surf[:, 0] + an_surf[:, -1])/2
        an_surf[:, -1] = an_surf[:, 0]
        theta = np.linspace(0, 2 * np.pi, 256)
        r = np.linspace(0, 6, 31)
        theta, r = np.meshgrid(theta, r)
        x, y = pol2cart(r, theta)
        points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1)) ), axis=1)
        values = an_surf.reshape((-1, 1))

        j = 0
        while j < len(values):
            if np.sum(np.isnan(values[j][0])) > 0:
                values = np.delete(values, j, axis=0)
                points = np.delete(points, j, axis=0)
                continue
            j += 1

        ref_power = np.full((grid_x.shape), np.nan)
        R_map = np.full((grid_x.shape), np.nan)
        for i in range(ref_power.shape[0]):
            for ii in range(3, ref_power.shape[1]-3):
                temp = np.zeros([7, 2])
                temp[:, 0] = rho_new[i, ii-3:ii+4].reshape((-1, 1))[:, 0]
                temp[:, -1] = griddata(points, values, (grid_x[i, ii-3:ii+4], grid_y[i, ii-3:ii+4]), method='cubic')[:,0]
                if np.isnan(temp[:, -1]).any() == False:
                    R_map[i, ii] = circ2_fit(temp)[0]
                    ref_power[i, ii] = (n-1)/(R_map[i, ii]*1e-3)

        CS = axs1[kk_, 1].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
        cbar = fig1.colorbar(CS, ax=axs1[kk_, 1])
        cbar.ax.set_ylabel('refractive power [D]')
        axs1[kk_, 1].set_title(str(t_meas[kk_]) + 'h')
        axs1[kk_, 1].set_xlabel('X-Coordinates [mm]')
        axs1[kk_, 1].set_ylabel('Y-Coordinates [mm]')

        kk_ += 1






