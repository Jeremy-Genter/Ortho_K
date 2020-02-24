import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from my_functions import *

os.chdir('simulations-mc')
dir_list = os.listdir(os.getcwd())
os.chdir('..')
thickness_central = np.full([2000, 2], np.nan)
thickness_midperi = np.full([2000, 2], np.nan)
R_n = np.full([2000, 2], np.nan)
power_eye = np.full([2000, 2], np.nan)
kk_ = 0
for k in dir_list:
    if os.path.isfile(os.path.join('simulations-mc', k, 'anterior_surface.dat')) == 0:
        kk_ += 1
        continue
  
    # read dat files
    path2file = os.path.join('simulations-mc', k, 'anterior_surface.dat')
    _, nodes_t = load_output_dat_file(path2file)
    path2file = os.path.join('simulations-mc', k, 'anterior_surface_stroma.dat')
    _, nodes_t_an_str = load_output_dat_file(path2file)
    path2file = os.path.join('simulations-mc', k, 'posterior_surface.dat')
    t, nodes_t_pos = load_output_dat_file(path2file)
    # reshape results from dat file
    t = np.asarray(t)
    nodes_t = np.asarray(nodes_t)
    nodes_index = nodes_t[0:int(len(nodes_t) / len(t)), 0]
    nodes_t = np.reshape(np.asarray(nodes_t), (len(t), len(nodes_index) * 4), order='C')

    nodes_t_an_str = np.asarray(nodes_t_an_str)
    nodes_index_an_str = nodes_t_an_str[0:int(len(nodes_t_an_str) / len(t)), 0]
    nodes_t_an_str = np.reshape(np.asarray(nodes_t_an_str), (len(t), len(nodes_index_an_str) * 4), order='C')

    nodes_t_pos = np.asarray(nodes_t_pos)
    nodes_index_pos = nodes_t_pos[0:int(len(nodes_t_pos) / len(t)), 0]
    nodes_t_pos = np.reshape(np.asarray(nodes_t_pos), (len(t), len(nodes_index_pos) * 4), order='C')

    x = np.zeros((len(nodes_index), 3 * len(t)))
    x_an_str = np.zeros((len(nodes_index_an_str), 3 * len(t)))
    x_pos = np.zeros((len(nodes_index_pos), 3 * len(t)))
    iii = 0
    # remove index
    for steps in t:
        ii = 0
        for i in nodes_index:
            temp = nodes_t[iii,
                   int(np.where(nodes_t[iii, :] == i)[0][0]): int(np.where(nodes_t[iii, :] == i)[0][0]) + 4]
            x[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        ii = 0
        for i in nodes_index_an_str:
            temp = nodes_t_an_str[iii, int(np.where(nodes_t_an_str[iii, :] == i)[0][0]): int(
                np.where(nodes_t_an_str[iii, :] == i)[0][0]) + 4]
            x_an_str[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        ii = 0
        for i in nodes_index_pos:
            temp = nodes_t_pos[iii,
                   int(np.where(nodes_t_pos[iii, :] == i)[0][0]): int(np.where(nodes_t_pos[iii, :] == i)[0][0]) + 4]
            x_pos[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        iii += 1

    steps = 0
    # calculate epi thickness
    for meas in [3600*16, 3*24*3600+3600*16]:
        iii = np.abs(t-meas).argmin()
        if np.abs(t[iii]-meas) > 3600:
            continue
        iii_control = np.abs(t-64).argmin()
        thickness_central_temp = x[np.abs(x[:, iii*3]-0.02).argmin(), iii*3:(iii + 1) * 3] - x_an_str[np.abs(x_an_str[:, iii*3]-0.02).argmin(), iii*3:(iii + 1) * 3]
        thickness_central_temp_con = x[np.abs(x[:, iii_control * 3] - 0.02).argmin(), iii_control * 3:(iii_control + 1) * 3]\
                                     - x_an_str[np.abs(x_an_str[:, iii_control * 3] - 0.02).argmin(), iii_control * 3:(iii_control + 1) * 3]
        thickness_central[kk_, steps] = np.linalg.norm(thickness_central_temp)-np.linalg.norm(thickness_central_temp_con)

        thickness_midperi_temp = x[np.abs(x[:, iii*3]-3).argmin(), iii*3:(iii + 1) * 3] - x_an_str[np.abs(x_an_str[:, iii*3]-2.97).argmin(), iii*3:(iii + 1) * 3]
        thickness_midperi_temp_con = x[np.abs(x[:, iii_control*3]-3).argmin(), iii_control*3:(iii_control + 1) * 3]\
                                     - x_an_str[np.abs(x_an_str[:, iii_control*3]-2.97).argmin(), iii_control*3:(iii_control + 1) * 3]
        thickness_midperi[kk_, steps] = np.linalg.norm(thickness_midperi_temp)-np.linalg.norm(thickness_midperi_temp_con)
        steps += 1

    # calculate Radius and power of the eye
    x = np.array(sorted(x, key=lambda x_column: x_column[0]))
    n = 1.3375
    steps = 0
    for meas in [3600*16, 3*24*3600+3600*16]:
        iii = np.abs(t-meas).argmin()
        if np.abs(t[iii]-meas) > 3600:
            continue
        pos = x[5:np.abs(x[:, 3*iii]-1.5).argmin(), iii*3:(iii + 1)*3]
        r = circ_fit(pos)
        R_n[kk_, steps] = r[0]
        power_eye[kk_, steps] = (n - 1)/(R_n[kk_, steps]*1e-3) - (n - 1)/(7.6*1e-3)
        steps += 1
    kk_ += 1

error = np.asarray((thickness_central[:, 0]+0.009)**2 + (thickness_central[:, 1]+0.015)**2 + (thickness_midperi[:, 0]-0.006)**2+\
        (thickness_midperi[:, 1]+0.0105)**2 + (power_eye[:, 0]+1.5)**2 + (power_eye[:, 1]+2)**2)
error.shape = (2000, 1)
index_err_min = error.argmin()
f = open('index_optimal_parameters.txt', "w+")
f.write(str(index_err_min))
f.close()

df_1 = pd.read_csv('combinations-orthoK.csv')
df_2 = pd.DataFrame(np.concatenate((thickness_central, thickness_midperi, power_eye, R_n, error), axis=1),
                  columns=['thickness_central 1d', 'thickness_central 4d', 'thickness_midperi 1d', 'thickness_midperi 4d',
                           'power_eye 1d', 'power_eye 4d', 'R_n 1d', 'R_n 4d', 'error'])
df = pd.concat([df_1, df_2], axis=1, sort=False)
df.to_csv('results-orthoK.csv', index=False)
