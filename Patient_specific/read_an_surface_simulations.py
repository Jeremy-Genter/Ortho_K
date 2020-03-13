import numpy as np
import pandas as pd
from my_functions import *
from scipy.interpolate import griddata
from copy import deepcopy
import os


folder = os.path.join('Patient_3_OD', 'simulations', 'initial_parameters')

t_meas = np.asarray([0, 8*3600, (8+36/60)*3600, (8+53/60)*3600, (8+80/60)*360, (8+4+23/60)*3600])+64
n = 1.3375
os.chdir(folder)
dir_list = os.listdir(os.getcwd())

rho_new = np.linspace(-6, 6, 40)
phi_new = np.linspace(0, 2*np.pi, 15)
rho_new, phi_new = np.meshgrid(rho_new, phi_new)
grid_x, grid_y = pol2cart(rho_new, phi_new)

for k in dir_list:
    if os.path.isfile(os.path.join(k, 'anterior_surface.dat')) == 0:
        continue
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
    os.system('mkdir ' + os.path.join(k, 'topography_data'))
    for time in t_meas:
        t_index = np.argmin(np.abs(t - time))
        pos = x_tot[:, t_index * 3:(t_index + 1) * 3]

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2] - np.min(pos[:, 2])

        rho, phi = cart2pol(x, y)
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

        np.savetxt(os.path.join(k, 'topography_data', 'power_topo' + str(np.round(time, 1))), ref_power)

        kk_ += 1
