import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_functions import *
from scipy.interpolate import griddata
import os


folders = [os.path.join('Patient_3_OD', 'simulations', 'dir02', 'topography_data'),
           os.path.join('Patient_3_OD', 'measurements', 'topography_data')] # [os.path.join('MN', 'measurements'), os.path.join('MN', 'simulations')]

fig1, axs1 = plt.subplots(6, 2)
fig1.subplots_adjust(top=0.925, wspace=0.1, hspace=0.5)
fig1.suptitle('Left Eye Patient 4', fontsize=16)

t_meas = np.asarray([0, 8*3600, (8+36/60)*3600, (8+53/60)*3600, (8+80/60)*360, (8+4+23/60)*3600])+64

dir_list_sim = os.listdir(folders[0])[::-1]
dir_list_meas = os.listdir(folders[1])[::-1]

n = 1.3375
rho_new = np.linspace(-6, 6, 40)
phi_new = np.linspace(0, 2*np.pi, 15)
rho_new, phi_new = np.meshgrid(rho_new, phi_new)
grid_x, grid_y = pol2cart(rho_new, phi_new)
levels = np.linspace(35, 52.5, 150)

kk_ = 0
ref_power_init = np.loadtxt(os.path.join(folders[0], dir_list_sim[0]))
for k in dir_list_sim:
    ref_power = np.loadtxt(os.path.join(folders[0], k))
    CS = axs1[kk_, 0].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
    axs1[kk_, 0].set_title(str(np.round((t_meas[kk_] - 64) / 3600, 2)) + 'h')
    if kk_ + 1 == len(t_meas):
        axs1[kk_, 0].set_xlabel('X [mm]')
    axs1[kk_, 0].set_ylabel('Y [mm]')
    kk_ += 1

kk_ = 0
ref_power_init = np.loadtxt(os.path.join(folders[1], dir_list_meas[0]))
for k in dir_list_meas:
    ref_power = np.loadtxt(os.path.join(folders[1], k))
    CS = axs1[kk_, 1].contourf(grid_x, grid_y, ref_power, levels=levels, cmap='rainbow')
    if kk_ + 1 == len(t_meas):
        axs1[kk_, 1].set_xlabel('X [mm]')
    axs1[kk_, 1].set_title(str(np.round((t_meas[kk_] - 64) / 3600, 2)) + 'h')

    kk_ += 1
fig1.subplots_adjust(right=0.775)
cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig1.colorbar(CS, cax=cbar_ax)
cbar.ax.set_ylabel('refractive power [D]')