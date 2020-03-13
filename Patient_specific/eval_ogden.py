import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from my_functions import *
from mpl_toolkits import mplot3d


nodes_disp_file = ['anterior_surface.dat']

fig00, ax00 = plt.subplots()
fig00.suptitle('Refractive Correction: Patient 3', fontsize=16)

#fig3, ax3 = plt.subplots()
#fig3.suptitle('Change in Epithelial Thickness', fontsize=16)


section = ['<Nodes name="Cornea"', '<NodeSet name="anterior_surface">', '<NodeSet name="anterior_surface_stroma">',
           '<NodeSet name="posterior_surface">']

#E_epi = np.asarray([0.8, 0.8])
#k_epi = np.round(np.logspace(-6.5, -5.2, 5), 8)
#k_stroma = np.round(np.logspace(-3.4, -2.3, 5), 5)
#l_name = np.ndarray.tolist(E_epi) + np.ndarray.tolist(k_epi) + np.ndarray.tolist(k_stroma)
#l_name = ['E: 1.5 kPa; $k_{epi}$: 7.5e-6 $\dfrac{mm^4}{Ns}$; k_stroma:$k_{stroma}$: 2.66e-3 $\dfrac{mm^4}{Ns}$ left']
l_name = ['Patient 3 OD']
folder = 'Patient_3_OD'
# l_name = np.ndarray.tolist(E_epi) + ['$p_{eyelid}=1\,kPa$ $E_{epi} = 4$', 'sealed Boundary']
os.chdir(folder)
dir = os.listdir(os.getcwd())
os.chdir('..')

#thickness_central = np.zeros([len(dir), 1])
#thickness_midperi = np.zeros([len(dir), 1])
kk_ = 0
for k in dir:
    if os.path.isfile(os.path.join(folder, k, 'anterior_surface.dat')) == 0:
        kk_ += 1
        continue
    path2file = os.path.join(folder, k, 'anterior_surface.dat')
    _, nodes_t = load_output_dat_file(path2file)
    path2file = os.path.join(folder, k, 'anterior_surface_stroma.dat')
    t, nodes_t_an_str = load_output_dat_file(path2file)
    # path2file = os.path.join(folder, k, 'posterior_surface.dat')
    # t, nodes_t_pos = load_output_dat_file(path2file)

    t = np.asarray(t)
    nodes_t = np.asarray(nodes_t)
    nodes_index = nodes_t[0:int(len(nodes_t) / len(t)), 0]
    nodes_t = np.reshape(np.asarray(nodes_t), (len(t), len(nodes_index) * 4), order='C')

    nodes_t_an_str = np.asarray(nodes_t_an_str)
    nodes_index_an_str = nodes_t_an_str[0:int(len(nodes_t_an_str) / len(t)), 0]
    nodes_t_an_str = np.reshape(np.asarray(nodes_t_an_str), (len(t), len(nodes_index_an_str) * 4), order='C')

    #nodes_t_pos = np.asarray(nodes_t_pos)
    #nodes_index_pos = nodes_t_pos[0:int(len(nodes_t_pos) / len(t)), 0]
    #nodes_t_pos = np.reshape(np.asarray(nodes_t_pos), (len(t), len(nodes_index_pos) * 4), order='C')

    x = np.zeros((len(nodes_index), 3 * len(t)))
    x_an_str = np.zeros((len(nodes_index_an_str), 3 * len(t)))
    #x_pos = np.zeros((len(nodes_index_pos), 3 * len(t)))
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
        #for i in nodes_index_pos:
        #    temp = nodes_t_pos[iii, int(np.where(nodes_t_pos[iii, :] == i)[0][0]): int(np.where(nodes_t_pos[iii, :] == i)[0][0]) + 4]
        #    x_pos[ii, iii * 3:(iii + 1) * 3] = temp[1:]
        #    ii += 1
        iii += 1

    #iii = np.abs(t-16*3600).argmin()
    #iii_control = np.abs(t-64).argmin()
    #thickness_central_temp = x[np.abs(x[:, iii*3]-0.02).argmin(), iii*3:(iii + 1) * 3] - x_an_str[np.abs(x_an_str[:, iii*3]-0.02).argmin(), iii*3:(iii + 1) * 3]
    #thickness_central_temp_con = x[np.abs(x[:, iii_control * 3] - 0.02).argmin(), iii_control * 3:(iii_control + 1) * 3]\
    #                             - x_an_str[np.abs(x_an_str[:, iii_control * 3] - 0.02).argmin(), iii_control * 3:(iii_control + 1) * 3]
    #thickness_central[kk_] = np.linalg.norm(thickness_central_temp)-np.linalg.norm(thickness_central_temp_con)
    #thickness_midperi_temp = x[np.abs(x[:, iii*3]-3).argmin(), iii*3:(iii + 1) * 3] - x_an_str[np.abs(x_an_str[:, iii*3]-2.97).argmin(), iii*3:(iii + 1) * 3]
    #thickness_midperi_temp_con = x[np.abs(x[:, iii_control*3]-3).argmin(), iii_control*3:(iii_control + 1) * 3]\
    #                             - x_an_str[np.abs(x_an_str[:, iii_control*3]-2.97).argmin(), iii_control*3:(iii_control + 1) * 3]
    #thickness_midperi[kk_] = np.linalg.norm(thickness_midperi_temp)-np.linalg.norm(thickness_midperi_temp_con)
    for i in range(len(t)):
        r_temp, phi_temp = cart2pol(x[:, i*3], x[:, i*3+1])
        x_zyl = np.concatenate((r_temp.reshape(-1, 1), phi_temp.reshape(-1, 1), x[:, i*3+2].reshape(-1, 1)), axis=1)
        x_zyl = np.array(sorted(x_zyl, key=lambda x_column: x_column[0]))
        #x_zyl_R_x = x_zyl[np.nonzero(x_zyl[:, 1] > -0.001)[0], :]
        #x_zyl_R_y = x_zyl[np.nonzero(x_zyl[:, 1] < -1.57)[0], :]
        if i == 0:
            index_1dot5 = np.abs(x_zyl[:, 0] - 1.5).argmin()
            #index_1dot5_x = np.abs(x_zyl_R_x[:, 0] - 1).argmin()
            #index_1dot5_y = np.abs(x_zyl_R_y[:, 0] - 1).argmin()
            x_1dot5 = np.zeros([index_1dot5, len(t)*3])
            #x_1dot5_x = np.zeros([index_1dot5_x, len(t) * 3])
            #x_1dot5_y = np.zeros([index_1dot5_y, len(t) * 3])
        z_offset = x_zyl[:, 2].min()
        x_zyl[:, 2] = x_zyl[:, 2] - z_offset
        x_temp, y_temp = pol2cart(x_zyl[:index_1dot5, 0], x_zyl[:index_1dot5, 1])
        x_1dot5[:, i * 3:i * 3 + 3] = np.concatenate(
            (x_temp.reshape(-1, 1), y_temp.reshape(-1, 1), x_zyl[:index_1dot5, 2].reshape(-1, 1)), axis=1)

        #z_offset = x_zyl_R_x[:, 2].min()
        #x_zyl_R_x[:, 2] = x_zyl_R_x[:, 2] - z_offset
        #x_temp, y_temp = pol2cart(x_zyl_R_x[:index_1dot5_x, 0], x_zyl_R_x[:index_1dot5_x, 1])
        #x_1dot5_x[:, i * 3:i * 3 + 3] = np.concatenate(
        #    (x_temp.reshape(-1, 1), y_temp.reshape(-1, 1), x_zyl_R_x[:index_1dot5_x, 2].reshape(-1, 1)), axis=1)

        #z_offset = x_zyl_R_y[:, 2].min()
        #x_zyl_R_y[:, 2] = x_zyl_R_y[:, 2] - z_offset
        #x_temp, y_temp = pol2cart(x_zyl_R_y[:index_1dot5_y, 0], x_zyl_R_y[:index_1dot5_y, 1])
        #x_1dot5_y[:, i * 3:i * 3 + 3] = np.concatenate(
        #    (x_temp.reshape(-1, 1), y_temp.reshape(-1, 1), x_zyl_R_y[:index_1dot5_y, 2].reshape(-1, 1)), axis=1)

    ## revovle for spherical/biconical fit
    # rot_angle = 5 / 180 * np.pi
    # slices = int(2 * np.pi / rot_angle)
    # skip = 5  # at least one
    # index_15_x = (np.abs(x_1dot5_x[:, 0] - 1.5)).argmin()
    # index_15_y = (np.abs(x_1dot5_y[:, 0] - 1.5)).argmin()
    # x_revolved_x = np.zeros([np.int(np.ceil((index_15_x-5) / skip)) * slices, 3 * len(t)])
    # x_revolved_y = np.zeros([np.int(np.ceil((index_15_y - 5) / skip)) * slices, 3 * len(t)])
    #
    # kk = 0
    # for jj in range(slices):
    #     R_y = np.matrix(
    #         [[np.round(np.cos(jj * rot_angle), decimals=6), 0, np.round(np.sin(jj * rot_angle), decimals=6)],
    #          [0, 1, 0],
    #          [-np.round(np.sin(jj * rot_angle), decimals=6), 0, np.round(np.cos(jj * rot_angle), decimals=6)]])
    #     for j in range(len(t)):
    #         temp = np.transpose(np.dot(R_y, np.transpose(x_1dot5_x[5:index_15_x:skip, j * 3:(j + 1) * 3])))
    #         x_revolved_x[
    #         jj * np.int(np.ceil((index_15_x - 5) / skip)):(jj + 1) * np.int(np.ceil((index_15_x - 5) / skip)),
    #         j * 3:(j + 1) * 3] = temp
    #         temp = np.transpose(np.dot(R_y, np.transpose(x_1dot5_y[5:index_15_y:skip, j * 3:(j + 1) * 3])))
    #         x_revolved_y[
    #         jj * np.int(np.ceil((index_15_y - 5) / skip)):(jj + 1) * np.int(np.ceil((index_15_y - 5) / skip)),
    #         j * 3:(j + 1) * 3] = temp

    # calculate Radius and power of the eye
    n = 1.3375
    R_n = np.full([len(t), 1], np.nan)
    R_n_x = np.full([len(t), 1], np.nan)
    R_n_y = np.full([len(t), 1], np.nan)
    power_eye = np.full([len(t), 1], np.nan)
    power_eye_x = np.full([len(t), 1], np.nan)
    power_eye_y = np.full([len(t), 1], np.nan)
    index_prestr = np.argmin(np.abs(t - 64))
    for j in range(index_prestr, (len(t)), 2):
        pos = x_1dot5[5:, j * 3:(j + 1) * 3]
        r = sphere_fit(pos)
        R_n[j] = r[0]
        # pos_2 = deepcopy(pos)
        # pos_2[:, 1] = -pos_2[:, 1]
        # pos = np.concatenate((pos, pos_2), axis=0)
        # pos_2 = deepcopy(pos)
        # pos_2[:, 0] = -pos_2[:, 0]
        # pos = np.concatenate((pos, pos_2), axis=0)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], c=pos[:, 2])
        # pos_x = x_revolved_x[5:, j * 3:(j + 1) * 3]
        # pos_y = x_revolved_y[5:, j * 3:(j + 1) * 3]
        # r_x = sphere_fit(pos_x)
        # r_y = sphere_fit(pos_y)
        # R_n_x[j] = r_x[0]
        # R_n_y[j] = r_y[0]
        power_eye[j] = (n - 1) / (R_n[j] * 1e-3)
        # power_eye_x[j] = (n - 1) / (R_n_x[j] * 1e-3)
        # power_eye_y[j] = (n - 1) / (R_n_y[j] * 1e-3)
    # plot radius eye
    # ax1 = plt.subplot(211)
    t_plot =  t[~np.isnan(power_eye).any(axis=1)]
    power_eye = power_eye[~np.isnan(power_eye).any(axis=1)]
    power_eye_x = power_eye_x[~np.isnan(power_eye_x).any(axis=1)]
    power_eye_y = power_eye_y[~np.isnan(power_eye_y).any(axis=1)]
    R_n = R_n[~np.isnan(R_n).any(axis=1)]
    R_n_x = R_n_x[~np.isnan(R_n_x).any(axis=1)]
    R_n_y = R_n_y[~np.isnan(R_n_y).any(axis=1)]


    label_name = str(l_name[0])

    ax00.plot(t_plot/3600, power_eye - power_eye[0], label=label_name+k)
    # ax00.plot(t_plot/3600, power_eye_x - power_eye_x[0], label=label_name+'R_x')
    # ax00.plot(t_plot/3600, power_eye_y - power_eye_y[0], label=label_name+'R_y')
    leg = ax00.legend(loc='lower right', fontsize=9)
    ax00.set_xlabel('time [h]', Fontsize=12)
    ax00.set_ylabel('refractive power change [D]', Fontsize=12)
    # plt.ylim([-3, 0])
    plt.xticks((np.arange(0, 20, 2)))
    #ax3.scatter([kk_], thickness_central[kk_] * 1e3, label='central epithelium'+label_name[kk_])
    #ax3.scatter([kk_], thickness_midperi[kk_] * 1e3, label='mid-peripheral epithelium'+label_name[kk_])
    #ax3.set_ylabel('epithelial thickness [$\mu m$]', Fontsize=12)
    #leg = ax3.legend(loc='lower right', fontsize=9)
    #plt.xticks((np.arange(0, 4, 0.25)))

    kk_ += 1

n = 1.3375
r_x = np.zeros([6, 1])
r_y = np.zeros([6, 1])
r_x[:, 0] = [(7.43), (7.65), (7.58), (7.55), (7.56), (7.56)]
r_y[:, 0] = [(7.33), 7.45, (7.38), (7.34), 7.31, 7.38]

p_x = (n-1)/(r_x*1e-3)
p_y = (n-1)/(r_y*1e-3)
p = np.mean(np.concatenate((p_x, p_y), axis=1), axis=1)
t_meas = np.asarray([0, 8*3600, (8+36/60)*3600, (8+53/60)*3600, (8+80/60)*3600, (8+4+23/60)*3600])/3600

ax00.scatter(t_meas, p - p[0], label='$k_{mean}$ measurment', marker='+')

# ax00.plot([0, 20], [-1.86, -1.86], lw=1, label='R_x measurment')
# ax00.plot([0, 20], [-0.15, -0.15], lw=1, label='R_y measurment')
# ax00.plot([0, 20], [-1, -1], lw=1, label='R_{mean} measurment')
# ax00.plot([16, 16], [0, -5], color='black', lw=1)
leg = ax00.legend(loc='lower right', fontsize=9)


