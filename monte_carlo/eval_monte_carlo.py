import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from my_functions import *

fig00, ax00 = plt.subplots()
fig00.suptitle('Refractive Correction: lid_p range 0.21 - 0.23\,$kPa$', fontsize=16)
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
    t, nodes_t = load_output_dat_file(path2file)

    # reshape results from dat file
    t = np.asarray(t)
    nodes_t = np.asarray(nodes_t)
    nodes_index = nodes_t[0:int(len(nodes_t) / len(t)), 0]
    nodes_t = np.reshape(np.asarray(nodes_t), (len(t), len(nodes_index) * 4), order='C')

    x = np.zeros((len(nodes_index), 3 * len(t)))

    iii = 0
    # remove index
    for steps in t:
        ii = 0
        for i in nodes_index:
            temp = nodes_t[iii,
                   int(np.where(nodes_t[iii, :] == i)[0][0]): int(np.where(nodes_t[iii, :] == i)[0][0]) + 4]
            x[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        iii += 1

    # revovle for spherical/biconical fit
    rot_angle = 5 / 180 * np.pi
    slices = int(2 * np.pi / rot_angle)
    skip = 5  # at least one
    index_15 = (np.abs(x[:, 0] - 1.5)).argmin()
    x_revolved = np.zeros([np.int(np.ceil((index_15 - 5) / skip)) * slices, 3 * len(t)])
    contact_diameter = np.zeros([4, 1])
    kk = 0
    for jj in range(slices):
        R_y = np.matrix(
            [[np.round(np.cos(jj * rot_angle), decimals=6), 0, np.round(np.sin(jj * rot_angle), decimals=6)],
             [0, 1, 0],
             [-np.round(np.sin(jj * rot_angle), decimals=6), 0, np.round(np.cos(jj * rot_angle), decimals=6)]])
        for j in range(len(t)):
            # biconic fitting describes surface in function of z; turn coordinate system accordingly
            temp = np.transpose(np.dot(R_y, np.transpose(x[5:index_15:skip, j * 3:(j + 1) * 3])))
            temp[:, 1] = -(temp[:, 1] - np.max(temp[:, 1], axis=0))
            x_revolved[
            jj * np.int(np.ceil((index_15 - 5) / skip)):(jj + 1) * np.int(np.ceil((index_15 - 5) / skip)),
            j * 3:(j + 1) * 3] = np.concatenate([temp[:, 2], temp[:, 0], temp[:, 1]], axis=1)

    # calculate Radius and power of the eye
    n = 1.3375
    index_1d = np.argmin(np.abs(t-18*3600))
    t = t[:index_1d]
    R_n = np.zeros(np.array([len(t), 1]))
    R_ny = np.zeros(np.array([len(t), 1]))
    power_eye = np.zeros(np.array([len(t), 1]))
    R = np.zeros(np.array([len(t), 1]))
    for j in range(len(t)):
        pos = x_revolved[:, j * 3:(j + 1) * 3]
        r = sphere_fit(pos)
        R_n[j] = r[0]
        power_eye[j] = (n - 1) / (R_n[j] * 1e-3)
    # plot radius eye
    index_prestr = np.argmin(np.abs(t-64))
    ax00.plot(t[index_prestr:] / 3600, power_eye[index_prestr:] - (n - 1) / 0.0076)
    ax00.set_xlabel('time [h]', Fontsize=12)
    ax00.set_ylabel('refractive power change [D]', Fontsize=12)
    plt.xticks((np.arange(0, 20, 2)))

ax00.plot([0, 18], [-1.35, -1.35], color='black', lw=0.75)
ax00.plot([16, 16], [0, -3], color='black', lw=0.75)