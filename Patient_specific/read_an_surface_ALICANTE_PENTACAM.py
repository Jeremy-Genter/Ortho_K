import numpy as np
import pandas as pd
from my_functions import *
from scipy.interpolate import griddata
import os

folders = ['Patient_3/pre/', 'Patient_3/lens_out/', 'Patient_3/20min/', 'Patient_3/40min/', 'Patient_3/60min/',
           'Patient_3/4hours/']
file_names = ['ELE_OD.CSV', 'ELE_OS.CSV', 'PAC_OD.CSV', 'PAC_OS.CSV']
os.system('mkdir ' + os.path.join('Patient_3_OD', 'measurements', 'topography_data'))

t_meas = np.asarray([0, 8*3600, (8+36/60)*3600, (8+53/60)*3600, (8+80/60)*360, (8+4+23/60)*3600])+64
n = 1.3375
rho_new = np.linspace(-6, 6, 40)
phi_new = np.linspace(0, 2*np.pi, 15)
rho_new, phi_new = np.meshgrid(rho_new, phi_new)
grid_x, grid_y = pol2cart(rho_new, phi_new)

for k in range(1):
    kk_ = 0
    for time in t_meas:
        an_surf = pd.read_csv(folders[kk_] + file_names[0], nrows=141, delimiter=';').values[:, 1:-1] * 1e-3

        x = np.linspace(-7, 7, 141)
        y = np.linspace(-7, 7, 141)
        x, y = np.meshgrid(x, y)
        rho, phi = cart2pol(x, y)
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

        np.savetxt(os.path.join('Patient_3_OD', 'measurements', 'topography_data', 'power_topo' + str(np.round((time-64)/60, 1)) + 'min'), ref_power)

        kk_ += 1






