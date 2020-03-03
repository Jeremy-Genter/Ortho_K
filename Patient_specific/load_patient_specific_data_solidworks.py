import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from my_functions import *
from copy import deepcopy

folder_patient_1_pre = 'Patient_1/PRE/'
folder_patient_1_post = 'Patient_1/POST/'
folder_patient_2_pre = 'Patient_2/PRE/'
folder_patient_2_post = 'Patient_2/POST_1day/'

file_names = ['TopoOIpre.csv', 'TopoODpre.csv', 'TopoOIpost.csv', 'TopoODpost.csv']

R = np.zeros([4, 4])

an_surf_l = np.zeros([31, 256])
an_surf_r = np.zeros([31, 256])
thickness_l = np.zeros([31, 256])
thickness_r = np.zeros([31, 256])
pos_surf_l = np.zeros([31, 256])
pos_surf_r = np.zeros([31, 256])
for loop in range(4):
    if loop == 0:
        p_data_l = pd.read_csv(folder_patient_1_pre + file_names[0], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_1_pre + file_names[1], skiprows=2)
    elif loop == 1:
        p_data_l = pd.read_csv(folder_patient_1_post + file_names[2], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_1_post + file_names[3], skiprows=2)
    elif loop == 2:
        p_data_l = pd.read_csv(folder_patient_2_pre + file_names[0], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_2_pre + file_names[1], skiprows=2)
    else:
        p_data_l = pd.read_csv(folder_patient_2_post + file_names[2], skiprows=2)
        p_data_r = pd.read_csv(folder_patient_2_post + file_names[3], skiprows=2)

    for i in range(31):
        an_surf_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i + 32].split(';'))[:-1].astype(float)
        an_surf_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i + 32].split(';'))[:-1].astype(float)
        thickness_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i].split(';'))[:-1].astype(float)
        thickness_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i].split(';'))[:-1].astype(float)
        #pos_surf_l[i, :] = np.asarray(p_data_l['CornealThickness [um]'][i + 64].split(';'))[:-1].astype(float)[:-1]
        #pos_surf_r[i, :] = np.asarray(p_data_r['CornealThickness [um]'][i + 64].split(';'))[:-1].astype(float)[:-1]

    pos_surf_l = an_surf_l + thickness_l*1e-3
    pos_surf_l[pos_surf_l < -5] = np.nan
    pos_surf_l[0, :] = pos_surf_l[0, 0]

    pos_surf_r = an_surf_r + thickness_r*1e-3
    pos_surf_r[pos_surf_r < -5] = np.nan
    pos_surf_r[0, :] = pos_surf_r[0, 0]

    an_surf_l[an_surf_l < -5] = np.nan
    an_surf_l[0, :] = np.nanmean(an_surf_l[0, :])
    an_surf_l[-1, :] = np.nanmean(an_surf_l[-1, :])


    theta = np.linspace(0, 2*np.pi, 256)
    r = np.linspace(0, 6, 31)
    theta, r = np.meshgrid(theta, r)
    x, y = pol2cart(r, theta)
    data_an_l = np.zeros([x.shape[0]*x.shape[1], 3])
    data_an_l[:, 0] = np.reshape(x, (-1, 1))[:, 0]
    data_an_l[:, 1] = np.reshape(y, (-1, 1))[:, 0]
    data_an_l[:, 2] = np.reshape(an_surf_l, (-1, 1))[:, 0]
    j = 0
    data_an_r = deepcopy(data_an_l)
    data_an_r[:, 2] = np.reshape(an_surf_r, (-1, 1))[:, 0]
    data_pos_l = deepcopy(data_an_l)
    data_pos_l[:, 2] = np.reshape(pos_surf_l, (-1, 1))[:, 0]
    data_pos_r = deepcopy(data_an_l)
    data_pos_r[:, 2] = np.reshape(pos_surf_r, (-1, 1))[:, 0]
    while j < len(data_an_l[:, 0]):
        if np.sum(np.isnan(data_an_l[j, :])) > 0:
            data_an_l = np.delete(data_an_l, j, axis=0)
            continue
        j += 1
    j = 0
    while j < len(data_an_r[:, 0]):
        if np.sum(np.isnan(data_an_r[j, :])) > 0:
            data_an_r = np.delete(data_an_r, j, axis=0)
            continue
        j += 1
    j = 0
    while j < len(data_pos_l[:, 0]):
        if np.sum(np.isnan(data_pos_l[j, :])) > 0:
            data_pos_l = np.delete(data_pos_l, j, axis=0)
            continue
        j += 1
    j = 0
    while j < len(data_pos_r[:, 0]):
        if np.sum(np.isnan(data_pos_r[j, :])) > 0:
            data_pos_r = np.delete(data_pos_r, j, axis=0)
            continue
        j += 1
    data_pos_l[:, 2] = data_pos_l[:, 2] - (data_pos_l[:256, 2]).min()
    data_pos_r[:, 2] = data_pos_r[:, 2] - (data_pos_r[:256, 2]).min()

    if loop == 0:
        np.savetxt('anterior__surf_l_P1_pre', data_an_l[255:, :])
        np.savetxt('anterior__surf_r_P1_pre', data_an_r[255:, :])
        np.savetxt('posterior_surf_l_P1_pre', data_pos_l[255:, :])
        np.savetxt('posterior_surf_r_P1_pre', data_pos_r[255:, :])
    elif loop == 1:
        np.savetxt('anterior__surf_l_P1_post', data_an_l[255:, :])
        np.savetxt('anterior__surf_r_P1_post', data_an_r[255:, :])
        np.savetxt('posterior_surf_l_P1_post', data_pos_l[255:, :])
        np.savetxt('posterior_surf_r_P1_post', data_pos_r[255:, :])
    elif loop == 2:
        np.savetxt('anterior__surf_l_P2_pre', data_an_l[255:, :])
        np.savetxt('anterior__surf_r_P2_pre', data_an_r[255:, :])
        np.savetxt('posterior_surf_l_P2_pre', data_pos_l[255:, :])
        np.savetxt('posterior_surf_r_P2_pre', data_pos_r[255:, :])
    elif loop == 3:
        np.savetxt('anterior__surf_l_P2_post', data_an_l[255:, :])
        np.savetxt('anterior__surf_r_P2_post', data_an_r[255:, :])
        np.savetxt('posterior_surf_l_P2_post', data_pos_l[255:, :])
        np.savetxt('posterior_surf_r_P2_post', data_pos_r[255:, :])




