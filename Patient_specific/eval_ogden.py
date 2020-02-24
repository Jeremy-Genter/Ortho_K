import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from my_functions import *


nodes_disp_file = ['anterior_surface.dat']

fig00, ax00 = plt.subplots()
fig00.suptitle('Refractive Correction: Patient 1', fontsize=16)

fig3, ax3 = plt.subplots()
fig3.suptitle('Change in Epithelial Thickness', fontsize=16)


section = ['<Nodes name="Cornea"', '<NodeSet name="anterior_surface">', '<NodeSet name="anterior_surface_stroma">',
           '<NodeSet name="posterior_surface">']

E_epi = np.asarray([0.8, 0.8])
k_epi = np.round(np.logspace(-6.5, -5.2, 5), 8)
k_stroma = np.round(np.logspace(-3.4, -2.3, 5), 5)
l_name = np.ndarray.tolist(E_epi) + np.ndarray.tolist(k_epi) + np.ndarray.tolist(k_stroma)
l_name = ['E: 0.86 kPa; $k_{epi}$: 6e-6 $\dfrac{mm^4}{Ns}$; k_stroma:$k_{stroma}$: 4e-3 $\dfrac{mm^4}{Ns}$ left', 'right',
          'E: 0.8 kPa; $k_{epi}$: 8e-7 $\dfrac{mm^4}{Ns}$; k_stroma:$k_{stroma}$: 1.7e-3 $\dfrac{mm^4}{Ns}$ left', 'right']
# l_name = ['fixed IOP', 'fluid cavity']
# l_name = np.ndarray.tolist(E_epi) + ['$p_{eyelid}=1\,kPa$ $E_{epi} = 4$', 'sealed Boundary']
dir = [0, 1, 2, 3]  # [3, 7]#
thickness_central = np.zeros([len(dir), 1])
thickness_midperi = np.zeros([len(dir), 1])
kk_ = 0
for k in dir:
    if k < 10:
        path2file = os.path.join('dir' + "0" + str(k), 'anterior_surface.dat')
        _, nodes_t = load_output_dat_file(path2file)
        path2file = os.path.join('dir' + "0" + str(k), 'anterior_surface_stroma.dat')
        _, nodes_t_an_str = load_output_dat_file(path2file)
        path2file = os.path.join('dir' + "0" + str(k), 'posterior_surface.dat')
        t, nodes_t_pos = load_output_dat_file(path2file)
    else:
        path2file = os.path.join('dir' + str(k), 'anterior_surface.dat')
        _, nodes_t = load_output_dat_file(path2file)
        path2file = os.path.join('dir' + str(k), 'anterior_surface_stroma.dat')
        _, nodes_t_an_str = load_output_dat_file(path2file)
        path2file = os.path.join('dir' + str(k), 'posterior_surface.dat')
        t, nodes_t_pos  = load_output_dat_file(path2file)

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
    # stitch disp and pos together
    for steps in t:
        ii = 0
        for i in nodes_index:
            temp = nodes_t[iii, int(np.where(nodes_t[iii, :] == i)[0][0]): int(np.where(nodes_t[iii, :] == i)[0][0]) + 4]
            x[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        ii = 0
        for i in nodes_index_an_str:
            temp = nodes_t_an_str[iii, int(np.where(nodes_t_an_str[iii, :] == i)[0][0]): int(np.where(nodes_t_an_str[iii, :] == i)[0][0]) + 4]
            x_an_str[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        ii = 0
        for i in nodes_index_pos:
            temp = nodes_t_pos[iii, int(np.where(nodes_t_pos[iii, :] == i)[0][0]): int(np.where(nodes_t_pos[iii, :] == i)[0][0]) + 4]
            x_pos[ii, iii * 3:(iii + 1) * 3] = temp[1:]
            ii += 1
        iii += 1

    iii = np.abs(t-16*3600).argmin()
    iii_control = np.abs(t-64).argmin()
    thickness_central_temp = x[np.abs(x[:, iii*3]-0.02).argmin(), iii*3:(iii + 1) * 3] - x_an_str[np.abs(x_an_str[:, iii*3]-0.02).argmin(), iii*3:(iii + 1) * 3]
    thickness_central_temp_con = x[np.abs(x[:, iii_control * 3] - 0.02).argmin(), iii_control * 3:(iii_control + 1) * 3]\
                                 - x_an_str[np.abs(x_an_str[:, iii_control * 3] - 0.02).argmin(), iii_control * 3:(iii_control + 1) * 3]
    thickness_central[kk_] = np.linalg.norm(thickness_central_temp)-np.linalg.norm(thickness_central_temp_con)
    thickness_midperi_temp = x[np.abs(x[:, iii*3]-3).argmin(), iii*3:(iii + 1) * 3] - x_an_str[np.abs(x_an_str[:, iii*3]-2.97).argmin(), iii*3:(iii + 1) * 3]
    thickness_midperi_temp_con = x[np.abs(x[:, iii_control*3]-3).argmin(), iii_control*3:(iii_control + 1) * 3]\
                                 - x_an_str[np.abs(x_an_str[:, iii_control*3]-2.97).argmin(), iii_control*3:(iii_control + 1) * 3]
    thickness_midperi[kk_] = np.linalg.norm(thickness_midperi_temp)-np.linalg.norm(thickness_midperi_temp_con)

    x = np.array(sorted(x, key=lambda x_column: x_column[0]))

    # revovle for spherical/biconical fit
    rot_angle = 5 / 180 * np.pi
    slices = int(2 * np.pi / rot_angle)
    skip = 5  # at least one
    index_15 = (np.abs(x[:, 0] - 1.5)).argmin()
    x_revolved = np.zeros([np.int(np.ceil((index_15-5) / skip)) * slices, 3 * len(t)])
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
            x_revolved[jj * np.int(np.ceil((index_15-5) / skip)):(jj + 1) * np.int(np.ceil((index_15-5) / skip)),
            j * 3:(j + 1) * 3] = np.concatenate([temp[:, 2], temp[:, 0], temp[:, 1]], axis=1)

    # calculate Radius and power of the eye
    n = 1.3375
    R_n = np.zeros(np.array([len(t), 1]))
    R_ny = np.zeros(np.array([len(t), 1]))
    power_eye = np.zeros(np.array([len(t), 1]))
    R = np.zeros(np.array([len(t), 1]))
    for j in range(int(np.float(len(t)))):
        pos = x_revolved[:, j * 3:(j + 1) * 3]
        r = sphere_fit(pos)
        # r = biconic_fitting(pos)
        R_n[j] = r[0]
        # R_ny[j] = r[1]
        # R[j] = (R_n[j] + R_ny[j])/2
        power_eye[j] = (n - 1) / (R_n[j] * 1e-3)
    # plot radius eye
    # ax1 = plt.subplot(211)
    if k < 10:
        label_name = str(l_name[kk_])
        ax00.plot(t[30:] / 3600, power_eye[30:] - (n - 1) / 0.0076, label=label_name)
        leg = ax00.legend(loc='lower right', fontsize=9)
        ax00.set_xlabel('time [h]', Fontsize=12)
        ax00.set_ylabel('refractive power change [D]', Fontsize=12)
        # plt.ylim([-3, 0])
        plt.xticks((np.arange(0, 20, 2)))
    kk_ += 1

ax00.plot([0, 20], [-1, -1], color='black', lw=1)
ax00.plot([0, 20], [-0.6, -0.6], color='black', lw=1)
ax00.plot([16, 16], [0, -5], color='black', lw=1)

ax3.boxplot(thickness_central*1e3)
ax3.boxplot(thickness_midperi*1e3)
ax3.set_xlabel(['central epithelium, mid-peripheral epithelium'], Fontsize=12)
ax3.set_ylabel('epithelial thickness [$\mu m$]', Fontsize=12)
leg = ax3.legend(loc='lower right', fontsize=9)
plt.xticks((np.arange(0, 4, 0.25)))
