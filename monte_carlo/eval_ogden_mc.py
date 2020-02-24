import os
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from my_functions import *


results_1 = np.asanyarray(pd.read_csv('results-orthoK_1.csv'))
results_2 = np.asanyarray(pd.read_csv('results-orthoK_2.csv'))
results_3 = np.asanyarray(pd.read_csv('contact_time_and_8h_cor.csv'))
results_temp = np.concatenate((results_1[:1000, :], results_2[1000:, :]), axis=0)
results = np.concatenate((results_temp, results_3), axis=1)
results[:, 1] = results[:, 1]*1e3
results[:, 4] = results[:, 4]*1e3
results[:, 5] = results[:, 5]*1e3
results[:, 6] = results[:, 6]*1e3
results[:, 7] = results[:, 7]*1e3
results[:, 8] = results[:, 8]*1e3
results_temp = deepcopy(results)

## uncommment following for loop to filter data
i = 0
ii = 0
while i < len(results[:, 0]):
    if results[i, 5] > -4.5 or results[i, 9] > -1.3 or results[i, 15] > 1*3600 or results[i, 15] < 0.6*3600 or results[i, -1] > -2:
        results = np.delete(results, i, axis=0)
        results_temp[ii, :] = np.full([1, 17], np.nan)
    else:
        i += 1
    ii += 1
k = 5
idx_min = np.argpartition(results_temp[:, 9], k)
idx_max = np.argpartition(results_temp[:, 9], -k)
idx_min = np.reshape(idx_min, (-1, 1))
idx_max = np.reshape(idx_max, (-1, 1))
idx_opt = np.nanargmin(results_temp[:, 13])

f = open('index_optimal_parameters.txt', 'w+')
f.write(str(idx_opt))
f.close()
df = pd.DataFrame(np.concatenate((idx_min[:k], idx_max[-k:]), axis=1),
                  columns=['idx_min', 'idx_max'])
df.to_csv('index_limit_cases.csv', index=False)


## figure hist plot

fig1, ax1 = plt.subplots()
ax1.hist(results[:, 1],  bins='auto')
plt.xlabel('Young\'s modulus [kPa]')
plt.ylabel('number of simulations [-]')

fig2, ax2 = plt.subplots()
hist, bins, _ = ax2.hist(results[:, 2], bins='auto')
ax2.cla()
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
ax2.hist(results[:, 2], bins=logbins)
ax2.semilogx()
plt.xlabel('permeability epithelium [$\dfrac{mm^{4}}{Ns} $]')
plt.ylabel('number of simulations [-]')

fig3, ax3 = plt.subplots()
hist, bins, _ = ax3.hist(results[:, 3], bins='auto')
ax3.cla()
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
ax3.hist(results[:, 3], bins=logbins)
ax3.semilogx()
plt.xlabel('permeability stroma [$\dfrac{mm^{4}}{Ns} $]')
plt.ylabel('number of simulations [-]')

fig4, ax4 = plt.subplots()
ax4.hist(results[:, 4],  bins='auto')
plt.xlabel('eyelid pressure [kPa]')
plt.ylabel('number of simulations [-]')

## figure correlation plot
fig, axs = plt.subplots(4, 4)
fig.suptitle('Monte Carlo Analysis Day 1', fontsize=16)

axs[0, 0].scatter(results[:, 1], results[:, 5])
axs[0, 1].scatter(results[:, 2], results[:, 5])
axs[0, 2].scatter(results[:, 3], results[:, 5])
axs[0, 3].scatter(results[:, 4], results[:, 5])

axs[1, 0].scatter(results[:, 1], results[:, 7])
axs[1, 1].scatter(results[:, 2], results[:, 7])
axs[1, 2].scatter(results[:, 3], results[:, 7])
axs[1, 3].scatter(results[:, 4], results[:, 7])

axs[2, 0].scatter(results[:, 1], results[:, 9])
axs[2, 1].scatter(results[:, 2], results[:, 9])
axs[2, 2].scatter(results[:, 3], results[:, 9])
axs[2, 3].scatter(results[:, 4], results[:, 9])


axs[3, 0].scatter(results[:, 1], results[:, 15]/3600)
axs[3, 1].scatter(results[:, 2], results[:, 15]/3600)
axs[3, 2].scatter(results[:, 3], results[:, 15]/3600)
axs[3, 3].scatter(results[:, 4], results[:, 15]/3600)


y_label = ['central thickness [$\mu m$]', 'central thickness [$\mu m$]',
           'central thickness [$\mu m$]', 'central thickness [$\mu m$]',
           'peripheral thickness [$\mu m$]', 'peripheral thickness [$\mu m$]',
           'peripheral thickness [$\mu m$]', 'peripheral thickness [$\mu m$]',
           'refractive correction [$D$]', 'refractive correction [$D$]',
           'refractive correction [$D$]', 'refractive correction [$D$]',
           'contact time [min]', 'contact time [min]',
           'contact time [min]', 'contact time [min]']
x_label = ['Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]',
           'Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]',
           'Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]',
           'Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]']
i = 0
for ax in axs.flat:
    ax.set(xlabel=x_label[i], ylabel=y_label[i])
    if np.sum(i == np.asarray([1, 5, 9, 13])) == 1:
        ax.semilogx()

    elif np.sum(i == np.asarray([2, 6, 10, 14])) == 1:
        ax.semilogx()

    i += 1
axs[0, 0].set_ylim(bottom=-8, top=-0)
axs[0, 1].set_ylim(bottom=-8, top=-0)
axs[0, 2].set_ylim(bottom=-8, top=-0)
axs[0, 3].set_ylim(bottom=-8, top=-0)

axs[1, 0].set_ylim(bottom=0, top=2)
axs[1, 1].set_ylim(bottom=0, top=2)
axs[1, 2].set_ylim(bottom=0, top=2)
axs[1, 3].set_ylim(bottom=0, top=2)

axs[2, 0].set_ylim(bottom=-2.5, top=-0.75)
axs[2, 1].set_ylim(bottom=-2.5, top=-0.75)
axs[2, 2].set_ylim(bottom=-2.5, top=-0.75)
axs[2, 3].set_ylim(bottom=-2.5, top=-0.75)


axs[0, 0].set_xlim(left=5*10**(-1), right=1.75)
axs[1, 0].set_xlim(left=5*10**(-1), right=1.75)
axs[2, 0].set_xlim(left=5*10**(-1), right=1.75)
axs[3, 0].set_xlim(left=5*10**(-1), right=1.75)

axs[0, 1].set_xlim(left=5*10**(-7), right=10**(-5))
axs[1, 1].set_xlim(left=5*10**(-7), right=10**(-5))
axs[2, 1].set_xlim(left=5*10**(-7), right=10**(-5))
axs[3, 1].set_xlim(left=5*10**(-7), right=10**(-5))

axs[0, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))
axs[1, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))
axs[2, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))
axs[3, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))

axs[0, 3].set_xlim(left=10**(-1), right=0.275)
axs[1, 3].set_xlim(left=10**(-1), right=0.275)
axs[2, 3].set_xlim(left=10**(-1), right=0.275)
axs[3, 3].set_xlim(left=10**(-1), right=0.275)

for ax in axs.flat:
    ax.label_outer()

plt.show()



fig2, axs = plt.subplots(4, 4)
fig2.suptitle('Monte Carlo Analysis Day 4', fontsize=16)

axs[0, 0].scatter(results[:, 1], results[:, 6])
axs[0, 1].scatter(results[:, 2], results[:, 6])
axs[0, 2].scatter(results[:, 3], results[:, 6])
axs[0, 3].scatter(results[:, 4], results[:, 6])

axs[1, 0].scatter(results[:, 1], results[:, 8])
axs[1, 1].scatter(results[:, 2], results[:, 8])
axs[1, 2].scatter(results[:, 3], results[:, 8])
axs[1, 3].scatter(results[:, 4], results[:, 8])

axs[2, 0].scatter(results[:, 1], results[:, 10])
axs[2, 1].scatter(results[:, 2], results[:, 10])
axs[2, 2].scatter(results[:, 3], results[:, 10])
axs[2, 3].scatter(results[:, 4], results[:, 10])


axs[3, 0].scatter(results[:, 1], results[:, 13])
axs[3, 1].scatter(results[:, 2], results[:, 13])
axs[3, 2].scatter(results[:, 3], results[:, 13])
axs[3, 3].scatter(results[:, 4], results[:, 13])


y_label = ['central thickness [$\mu m$]', 'central thickness [$\mu m$]',
           'central thickness [$\mu m$]', 'central thickness [$\mu m$]',
           'peripheral thickness [$\mu m$]', 'peripheral thickness [$\mu m$]',
           'peripheral thickness [$\mu m$]', 'peripheral thickness [$\mu m$]',
           'refractive correction [$D$]', 'refractive correction [$D$]',
           'refractive correction [$D$]', 'refractive correction [$D$]',
           'error [-]', 'error [-]',
           'error [-]', 'error [-]']
x_label = ['Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]',
           'Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]',
           'Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]',
           'Young\'s modulus [kPa]', 'permeability epithelium [$\dfrac{mm^{4}}{Ns} $]',
           'permeability stroma [$\dfrac{mm^{4}}{Ns} $]', 'eye lid pressure [kPa]']
i = 0
for ax in axs.flat:
    ax.set(xlabel=x_label[i], ylabel=y_label[i])
    if np.sum(i == np.asarray([1, 5, 9, 13])) == 1:
        ax.semilogx()

    elif np.sum(i == np.asarray([2, 6, 10, 14])) == 1:
        ax.semilogx()

    #elif np.sum(i == np.asarray([0, 4, 8, 12])) == 1:

    #elif np.sum(i == np.asarray([3, 7, 11, 15])) == 1:

    i += 1
axs[0, 0].set_ylim(bottom=-8, top=-0)
axs[0, 1].set_ylim(bottom=-8, top=-0)
axs[0, 2].set_ylim(bottom=-8, top=-0)
axs[0, 3].set_ylim(bottom=-8, top=-0)

axs[1, 0].set_ylim(bottom=0, top=2)
axs[1, 1].set_ylim(bottom=0, top=2)
axs[1, 2].set_ylim(bottom=0, top=2)
axs[1, 3].set_ylim(bottom=0, top=2)

axs[2, 0].set_ylim(bottom=-2.5, top=-0.75)
axs[2, 1].set_ylim(bottom=-2.5, top=-0.75)
axs[2, 2].set_ylim(bottom=-2.5, top=-0.75)
axs[2, 3].set_ylim(bottom=-2.5, top=-0.75)


axs[0, 0].set_xlim(left=5*10**(-1), right=1.75)
axs[1, 0].set_xlim(left=5*10**(-1), right=1.75)
axs[2, 0].set_xlim(left=5*10**(-1), right=1.75)
axs[3, 0].set_xlim(left=5*10**(-1), right=1.75)

axs[0, 1].set_xlim(left=5*10**(-7), right=10**(-5))
axs[1, 1].set_xlim(left=5*10**(-7), right=10**(-5))
axs[2, 1].set_xlim(left=5*10**(-7), right=10**(-5))
axs[3, 1].set_xlim(left=5*10**(-7), right=10**(-5))

axs[0, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))
axs[1, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))
axs[2, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))
axs[3, 2].set_xlim(left=5*10 ** (-4), right=10 ** (-2))

axs[0, 3].set_xlim(left=10**(-1), right=0.275)
axs[1, 3].set_xlim(left=10**(-1), right=0.275)
axs[2, 3].set_xlim(left=10**(-1), right=0.275)
axs[3, 3].set_xlim(left=10**(-1), right=0.275)

for ax in axs.flat:
    ax.label_outer()

#
#
# fig00, ax00 = plt.subplots()
# fig00.suptitle('Refractive Correction: range of Monte-Carlo', fontsize=16)
#
# dir = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# dir = [0, 9, 10]
# # dir = [20, 29, 30]
# p_eye = {}
# t_eye = {}
# kk_ = 0
# for k in dir:
#     if k < 10:
#         path2file = os.path.join('dir' + "0" + str(k), 'anterior_surface.dat')
#         t, nodes_t = load_output_dat_file(path2file)
#     else:
#         path2file = os.path.join('dir' + str(k), 'anterior_surface.dat')
#         t, nodes_t = load_output_dat_file(path2file)
#
#
#     t = np.asarray(t)
#     nodes_t = np.asarray(nodes_t)
#     nodes_index = nodes_t[0:int(len(nodes_t) / len(t)), 0]
#     nodes_t = np.reshape(np.asarray(nodes_t), (len(t), len(nodes_index) * 4), order='C')
#
#     x = np.zeros((len(nodes_index), 3 * len(t)))
#     iii = 0
#     # stitch disp and pos together
#     for steps in t:
#         ii = 0
#         for i in nodes_index:
#             temp = nodes_t[iii, int(np.where(nodes_t[iii, :] == i)[0][0]): int(np.where(nodes_t[iii, :] == i)[0][0]) + 4]
#             x[ii, iii * 3:(iii + 1) * 3] = temp[1:]
#             ii += 1
#         iii += 1
#
#     x = np.array(sorted(x, key=lambda x_column: x_column[0]))
#
#     # # revovle for spherical/biconical fit
#     # rot_angle = 5 / 180 * np.pi
#     # slices = int(2 * np.pi / rot_angle)
#     # skip = 5  # at least one
#     # index_15 = (np.abs(x[:, 0] - 1.5)).argmin()
#     # x_revolved = np.zeros([np.int(np.ceil((index_15-5) / skip)) * slices, 3 * len(t)])
#     # contact_diameter = np.zeros([4, 1])
#     # kk = 0
#     # for jj in range(slices):
#     #     R_y = np.matrix(
#     #         [[np.round(np.cos(jj * rot_angle), decimals=6), 0, np.round(np.sin(jj * rot_angle), decimals=6)],
#     #          [0, 1, 0],
#     #          [-np.round(np.sin(jj * rot_angle), decimals=6), 0, np.round(np.cos(jj * rot_angle), decimals=6)]])
#     #     for j in range(len(t)):
#     #         # biconic fitting describes surface in function of z; turn coordinate system accordingly
#     #         temp = np.transpose(np.dot(R_y, np.transpose(x[5:index_15:skip, j * 3:(j + 1) * 3])))
#     #         temp[:, 1] = -(temp[:, 1] - np.max(temp[:, 1], axis=0))
#     #         x_revolved[jj * np.int(np.ceil((index_15-5) / skip)):(jj + 1) * np.int(np.ceil((index_15-5) / skip)),
#     #         j * 3:(j + 1) * 3] = np.concatenate([temp[:, 2], temp[:, 0], temp[:, 1]], axis=1)
#
#     # calculate Radius and power of the eye
#     n = 1.3375
#     R_n = np.zeros(np.array([len(t), 1]))
#     R_ny = np.zeros(np.array([len(t), 1]))
#     power_eye = np.zeros(np.array([len(t), 1]))
#     R = np.zeros(np.array([len(t), 1]))
#     for j in range((len(t))):
#         # pos = x_revolved[5:, j * 3:(j + 1) * 3]
#         # r = sphere_fit(pos)
#         pos = x[5:np.abs(x[:, 3*j]-1.5).argmin(), j*3:(j + 1)*3]
#         r = circ_fit(pos)
#         R_n[j] = r[0]
#         # R_ny[j] = r[1]
#         # R[j] = (R_n[j] + R_ny[j])/2
#         power_eye[j] = (n - 1) / (R_n[j] * 1e-3)
#     # plot radius eye
#     # ax1 = plt.subplot(211)
#     if k < 10:
#         p_eye[str(k)] = power_eye[30:] - (n - 1) / 0.0076
#         t_eye[str(k)] = t[30:]
#         #ax00.plot(t[30:] / 3600, power_eye[30:] - (n - 1) / 0.0076)
#     else:
#         ax00.plot(t[30:] / 3600, power_eye[30:] - (n - 1) / 0.0076)
#     ax00.set_xlabel('time [h]', Fontsize=12)
#     ax00.set_ylabel('refractive power change [D]', Fontsize=12)
#     # plt.ylim([-3, 0])
#     plt.xticks((np.arange(0, 98, 2)))
#
#
#     kk_ += 1
# ax00.plot([0, 96], [-1.5, -1.5], color='black', lw=1)
# ax00.plot([16, 16], [0, -3], color='black', lw=1)
# ax00.plot([24+16, 24+16], [0, -3], color='black', lw=1)
# ax00.plot([2*24+16, 2*24+16], [0, -3], color='black', lw=1)
# ax00.plot([3*24+16, 3*24+16], [0, -3], color='black', lw=1)
#
# t = t_eye[str(dir[1])]
# t_ = t_eye[str(dir[0])]
# p_temp = np.interp(t, t_, p_eye[str(dir[0])][:, 0])
#
# ax00.fill_between(t_eye[str(dir[1])] / 3600, p_eye[str(dir[1])][:, 0], p_temp, alpha=0.6, facecolors='gray')
plt.show()