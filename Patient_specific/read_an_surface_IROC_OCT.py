import numpy as np
import pandas as pd
from my_functions import *
from scipy.interpolate import griddata
from copy import deepcopy
import os


folder = 'JG/measurements'  # MN/measurements'

n = 1.3375
os.chdir(folder)
dir_list = os.listdir(os.getcwd())
os.system('mkdir topography_data')
rho_new = np.linspace(-6, 6, 40)
phi_new = np.linspace(0, 2*np.pi, 15)
rho_new, phi_new = np.meshgrid(rho_new, phi_new)
grid_x, grid_y = pol2cart(rho_new, phi_new)

kk_ = 0
for k in dir_list:
    if k.lower().endswith('.csv') == False:
        continue

    an_surf = np.zeros([31, 256])
    an_surf[1:, :] = pd.read_csv(k, nrows=30, delimiter=';', skiprows=60).values[:, :-1]

    theta = np.linspace(0, 2 * np.pi, 256)
    r = np.linspace(0, 6, 31)
    theta, r = np.meshgrid(theta, r)
    x, y = pol2cart(r, theta)
    points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
    values = an_surf.reshape((-1, 1))
    j = 0
    while j < len(values[:, 0]):
        if np.sum(np.isnan(values[j, :])) > 0:
            values = np.delete(values, j, axis=0)
            points = np.delete(points, j, axis=0)
            continue
        j += 1

    ref_power = np.full((grid_x.shape), np.nan)
    R_map = np.full((grid_x.shape), np.nan)
    for i in range(ref_power.shape[0]):
        for ii in range(3, ref_power.shape[1] - 3):
            temp = np.zeros([7, 2])
            temp[:, 0] = rho_new[i, ii - 3:ii + 4].reshape((-1, 1))[:, 0]
            temp[:, -1] = griddata(points, values, (grid_x[i, ii - 3:ii + 4], grid_y[i, ii - 3:ii + 4]),
                                   method='cubic')[:, 0]
            if np.isnan(temp[:, -1]).any() == False:
                R_map[i, ii] = circ2_fit(temp)[0]
                ref_power[i, ii] = (n - 1) / (R_map[i, ii] * 1e-3)

    np.savetxt(os.path.join('topography_data', 'power_topo' + k[:-4]), ref_power)

    kk_ += 1
