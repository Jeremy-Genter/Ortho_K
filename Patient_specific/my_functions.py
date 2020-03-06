import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.odr as odr
import scipy.optimize as optimize
from sympy import solve, solveset, var
import sympy as sp
from scipy.io import loadmat
from copy import deepcopy
import time
import os
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def write_loadcurve(time, magnitude, file_name, id_numb, path=''):
    if not path == '':
        os.chdir(path)
    f = open(file_name, "w+")
    f.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n")
    f.write("<febio_spec version=\"2.5\">\n")
    f.write("\t<LoadData>\n")
    f.write("\t\t<loadcurve id=\"" + str(id_numb) + "\" type=\"linear\"extend=\"constant\">\n")
    for t, m in zip(time, magnitude):
        f.write("\t\t\t<loadpoint>" + str(t) + ", " + str(m) + "</loadpoint>\n")
    f.write("\t\t</loadcurve>\n")
    f.write("\t</LoadData>\n")
    f.write("</febio_spec>")
    f.close()


def read_data_thief(file_name, path=''):
    if not path == '':
        os.chdir(path)
    data = []
    with open(file_name, 'r') as fh:
        next(fh)
        for line in fh:
            data.append([float(x) for x in line.split(',')])
    data = np.asarray(data)
    return data


def write_parameters(parameters, parm_name, path=''):
    if not path == '':
        os.chdir(path)
    i = 0
    f = open("parameters.feb", "w+")
    f.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n")
    f.write("<febio_spec version=\"2.5\">\n")
    f.write("\t<Parameters>\n")
    for param in parameters:
        f.write("\t\t<param name=\"" + parm_name[i] + "\">" + str(param) + "</param>\n")
        i += 1
    f.write("\t</Parameters>\n")
    f.write("</febio_spec>")
    f.close()


def pre_stretch(ite_max, tol_error, path=''):
    if not path == '':
        os.chdir(path)
    error = np.inf  # [mm]
    i = 0
    # os.system('cp geometry_init.feb geometry_opt.feb')
    X_aim = np.asarray(load_feb_file_nodes('geometry_init.feb', '<Nodes name=\"Cornea\">', path=path))
    X_subopt = np.asarray(load_feb_file_nodes('geometry_opt.feb', '<Nodes name=\"Cornea\">', path=path))
    X_opt = deepcopy(X_subopt)
    #X_opt[:, 1:] = 0.875 * X_subopt[:, 1:]
    write_febio_geometry_file('geometry_opt.feb', X_opt, path=path)
    while (i < ite_max) and (error > tol_error):
        os.system('/home/ubelix/artorg/shared/software/FEBio2.8.5/bin/febio2.lnx64 -i pre_stretch.feb')
        X_subopt = np.asarray(load_feb_file_nodes('geometry_opt.feb', '<Nodes name=\"Cornea\">', path=path))
        t, x = load_output_dat_file('disp_pre_stretch.dat', path=path)
        x = np.asarray(x)
        X_def = x[np.where(x[:, 0] == 1)[0][-1]:np.where(x[:, 0] == X_aim.shape[0])[0][-1] + 1, :]
        X_error = X_aim[:, 1:] - X_def[:, 1:]
        error = np.max(np.abs(X_error))
        X_opt = deepcopy(X_def)
        X_opt[:, 1:] = X_error + X_subopt[:, 1:]
        write_febio_geometry_file('geometry_opt.feb', X_opt, path=path)
        print(i, error)
        i += 1


def write_febio_geometry_file(file_name, x, path=''):
    if not path == '':
        os.chdir(path)
    i = 0
    fh = open(file_name, 'r')
    with open('temp.feb', 'w+') as temp:
        for line in fh:
            if not line.find('<node id=\"' + str(int(x[i, 0])) + '\">') == -1:
                temp.write('\t\t\t<node id=\"' + str(int(x[i, 0])) + '\">  ' + str(x[i, 1]) + ',  ' + str(x[i, 2]) + ',  ' + str(x[i, 3]) + '</node>\n')
                i += 1
                i = int(np.min([i, x.shape[0]-1]))
            else:
                temp.write(line)
    os.system('mv temp.feb ' + file_name)



def load_feb_file_nodes(filename, section, path=''):
    if not path == '':
        os.chdir(path)
    nodes = []
    with open(filename) as fh:
        line = next(fh)
        while line.find(section) == -1:
            line = next(fh)
        for line in fh:
            if not line.find('</Nodes>') == -1:
                break
            id_1 = line.find("<node id=")
            id_2 = line.find("> ")
            id_3 = line.find("</node>")
            nodes.append([int(line[id_1 + 10:id_2 - 1])] + [float(x) for x in line[id_2+3:id_3].split(',')])
    return nodes


def load_feb_file_nodes_id(filename, section, path=''):
    if not path == '':
        os.chdir(path)
    nodes_index = []
    with open(filename) as fh:
        line = next(fh)
        while line.find(section) == -1:
            line = next(fh)
        for line in fh:
            if not line.find('</NodeSet>') == -1:
                break
            id_1 = line.find("<node id=")
            id_2 = line.find("/>")
            nodes_index.append(int(line[id_1 + 10:id_2 - 1]))
    return nodes_index


def load_output_dat_file(filename, path=''):
    if not path == '':
        os.chdir(path)
    nodes_disp = []
    t = []
    with open(filename) as fh:
        for line in fh:
            if line.find('*Step') == 0:
                line = next(fh)
                id_1 = line.find('=')
                t.append(float(line[id_1 + 1:-1]))
                line = next(fh)
                line = next(fh)
            nodes_disp.append([float(x) for x in line.split(',')])
    return t, nodes_disp

def biconic_fitting(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    z = np.reshape(data[:, 2], [len(data[:, 0]), 1])
    X = np.zeros([len(x), len(x)+3])
    # create Matrix for least square minimization
    for i in range(len(x)):
        X[i, 0:3] = [x[i, 0]**2, y[i, 0]**2, x[i, 0]*y[i, 0]]
        X[i, i+3] = z[i, 0]**2
    p_prime = np.linalg.lstsq(X, 2*z, rcond=-1)
    p_prime = p_prime[0]
    # X_inv = np.linalg.pinv(X)
    # p_prime = 2*np.dot(X_inv, z)
    term = np.zeros([len(x), 1])
    # create Matrix for least square minimization
    for i in range(len(x)):
        term[i, 0] = p_prime[i+3, 0]*(2*z[i, 0] - p_prime[i+3, 0]*z[i, 0]**2)
    p = -np.ones([3, 1])
    a_1 = 0.5*(-(-p_prime[0, 0]-p_prime[1, 0]) + np.sqrt((-p_prime[0, 0]-p_prime[1, 0])**2 - 4*(p_prime[0, 0]*p_prime[1, 0] - p_prime[2, 0]**2/4) + 0j))
    a_2 = 0.5*(-(-p_prime[0, 0]-p_prime[1, 0]) - np.sqrt((-p_prime[0, 0]-p_prime[1, 0])**2 - 4*(p_prime[0, 0]*p_prime[1, 0] - p_prime[2, 0]**2/4) + 0j))
    a_1 = np.round(a_1, decimals=5)
    a_2 = np.round(a_2, decimals=5)
    if a_1 > 0 and (p_prime[0, 0] - a_1)/(p_prime[0, 0]+p_prime[1, 0] - 2*a_1) >= 0:
        p[0] = np.real(a_1)
    elif a_2 > 0 and (p_prime[0, 0] - a_2)/(p_prime[0, 0]+p_prime[1, 0] - 2*a_2) >= 0:
        p[0] = np.real(a_2)
    else:
        p[0] = np.inf
    p[1] = -p[0] + (p_prime[0, 0] + p_prime[1, 0])
    if p[0] == p[1]:
        p[2] = 0
    else:
        p[2] = 0.5*(np.arcsin(p_prime[2, 0]/(p[1] - p[0])))
    p_prime_2 = np.linalg.lstsq(X[:, 0:3], term, rcond=-1)
    p_prime_2 = p_prime_2[0]
    # p_prime_2 = np.dot(np.linalg.pinv(X[:, 0:3]), term)
    R_x = 1/p[0]
    R_y = 1/p[1]
    Q_x = R_x**2*(p_prime_2[0] - 0.5*p_prime_2[2]*np.tan(p[2])) - 1
    Q_y = R_y**2*(p_prime_2[1] - 0.5*p_prime_2[2]*np.tan(p[2])) - 1
    phi = p[2]
    return R_x, R_y, phi, Q_x, Q_y

def f_biconic_model(init, *data):
    """biconical model; inital guess: init=[a',b',d',u',v',w'], data to fit to: data= [x_i,y_i,z_i]"""
    data = data[0]
    c = (init[3]*data[0, :]**2 + init[4]*data[1, :]**2 + init[5]*data[0, :]*data[1, :])/(init[0]*data[0, :]**2 + init[1]*data[1, :]**2 + init[2]*data[0, :]*data[1, :])
    return np.sum(( init[0]*data[0, :]**2 + init[1]*data[1, :]**2 + init[2]*data[0, :]*data[1, :] + c*(data[2, :])**2 - 2*(data[2, :]) )**2)


def f2_biconic_model(init, *data):
    data = data[0]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    return np.sum((-z + init[4] + (x**2/init[0] + y**2/init[1])/(1 + np.sqrt(1 - (1+init[2])*x**2/init[0]**2 - (1+init[3])*y**2/init[1]**2)))**2)

def nm_biconic_fit(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    z = np.reshape(data[:, 2], [len(data[:, 0]), 1])
    init = np.array([1/7.6, 1/7.6, 0, 0, 0, 0])
    res = optimize.minimize(f_biconic_model, init, np.array([x, y, z]), method='Nelder-Mead', options={'xtol': 1e-10})
    p_prime = res.x
    a_1 = 0.5 * (-(-p_prime[0] - p_prime[1]) + np.sqrt((-p_prime[0] - p_prime[1])**2 - 4*(p_prime[0]*p_prime[1] - p_prime[2]**2/4) + 0j))
    a_2 = 0.5 * (-(-p_prime[0] - p_prime[1]) - np.sqrt((-p_prime[0] - p_prime[1])**2 - 4*(p_prime[0]*p_prime[1] - p_prime[2]**2/4) + 0j))
    a_1 = np.round(a_1, decimals=5)
    a_2 = np.round(a_2, decimals=5)
    p = np.zeros([5,1])
    if a_1 > 0 and (p_prime[0] - a_1) / (p_prime[0] + p_prime[1] - 2 * a_1) >= 0:
        p[0] = np.real(a_1)
    elif a_2 > 0 and (p_prime[0] - a_2) / (p_prime[0] + p_prime[1] - 2 * a_2) >= 0:
        p[0] = np.real(a_2)
    else:
        p[0] = np.inf
    p[1] = -p[0] + (p_prime[0] + p_prime[1])
    if p[0] == p[1]:
        p[2] = 0
    else:
        p[2] = 0.5 * (np.arcsin(p_prime[2] / (p[1] - p[0])))
    R_x = 1 / p[0]
    R_y = 1 / p[1]
    Q_x = R_x**2*(p_prime[3] - 0.5*p_prime[5] * np.tan(p[2])) - 1
    Q_y = R_y**2*(p_prime[4] - 0.5*p_prime[5] * np.tan(p[2])) - 1
    phi = p[2]
    return R_x, R_y, phi, Q_x, Q_y


def f_sphere(init, *data):
    data = np.array(data[0:3])[:, :, 0]
    x = data[0, :]
    y = data[1, :]
    z = data[2, :]
    return (-init[0]**2 + (x-init[1])**2 + (y-init[2])**2 + (z-init[3])**2)**2  # (-init[0]**2 + x**2 + y**2 + (z-init[1])**2)**2


def sphere_fit(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    z = np.reshape(data[:, 2], [len(data[:, 0]), 1])
    init = np.array([7.6, 0, 0, 0])
    res = optimize.least_squares(f_sphere, init, args=np.array([x, y, z]))
    return res.x


def f_ellipsoid(init, *data):
    data = np.array(data[0:3])[:, :, 0]
    x = data[0, :]
    y = data[1, :]
    z = data[2, :]
    return (-1 + (x-init[3])**2/init[0]**2 + (y-init[4])**2/init[1]**2  + (z-init[5])**2)**2/init[2]**2


def ellipsoid_fit(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    z = np.reshape(data[:, 2], [len(data[:, 0]), 1])
    init = np.array([7.6, 7.6, 7.6, 0, 0, 0])
    res = optimize.least_squares(f_ellipsoid, init, args=np.array([x, y, z]))
    return res.x


def f_circ(init, *data):
    data = np.array(data[0:2])[:, :, 0]
    x = data[0, :]
    y = data[1, :]
    return (-init[0]**2 + x**2 + (y-init[1])**2)**2


def circ_fit(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    init = np.array([7.6, 0])
    res = optimize.least_squares(f_circ, init, args=np.array([x, y]))
    return res.x


def keratometry(self, mode='biconic'):
    # Coordinates of surface
    x = self[:, 0]
    y = self[:, 1]
    z = self[:, 2]

    # Least squares

    # Create X matrix based on measurements
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y
    z2 = z ** 2
    z2_diag = np.diag(z2)
    X = np.c_[x2, y2, xy, z2_diag]

    # Create target vector
    t = 2
    z_target = t * z

    # Solve least-squares
    Xinv = np.linalg.pinv(X)
    p = np.matmul(Xinv, z_target)

    # Obtain a', b', d'
    a_p = p[0]
    b_p = p[1]
    d_p = p[2]

    # Solve a and b to obtain Rx, Ry and Phi

    # Calculate a
    a = np.roots([1, -a_p - b_p, a_p * b_p - (d_p ** 2) / 4])
    print(a)
    aux = [np.real_if_close(a[0], tol=1e-5), np.real_if_close(a[1], tol=1e-5)]
    a = np.array(aux)

    # Avoid negative radii
    a = a[a > 0]
    print(a)

    # Avoid violating constrain on sin(phi)^2
    if np.abs(a_p - a[0]) < 1e-6:
        check = np.array([0, 0])
    else:
        check = (a_p - a) / ((a_p + b_p) - 2 * a)

    a = a[check >= 0]

    # Calculate b
    b = (a_p + b_p) - a

    if mode == 'biconic':

        # Calculate Radii and angle
        Rx = 1 / a
        Ry = 1 / b
        if (np.abs(d_p) < 1e-6) and (np.sum(np.abs(b - a)) < 1e-6):
            phi = np.array([0, 0])
        else:
            phi = 0.5 * np.arcsin(d_p / (b - a))  # Angle of flatter meridian

        # Double check the correct option if more than two options available
        if len(phi) == 2:
            if (phi[0] < 0) or (phi[0] >= np.pi/2):
                Rx = Rx[1]
                Ry = Ry[1]
                phi = phi[1]
            else:
                Rx = Rx[0]
                Ry = Ry[0]
                phi = phi[0]

        if Rx < Ry:
            phi = phi + np.pi / 2
            aux = Rx
            Rx = Ry
            Ry = aux

        phi_deg = phi * 180 / np.pi

        # Power
        Kmax = (1.3375 - 1) * 1000 / np.min(
            np.array([Rx, Ry]))  # Maximum curvature related to minimum radius (steeper meridian)
        Kmin = (1.3375 - 1) * 1000 / np.max(
            np.array([Rx, Ry]))  # Minimum curvature related to minimum radius (flatter meridian)
        Kmean = (Kmax + Kmin) / 2

    elif mode == 'sphere':
        Rx = 1 / np.real_if_close(a[0], tol=1e-6)
        Ry = Rx
        phi = 0
        phi_deg = 0

        # Power
        Kmax = (1.3375 - 1) * 1000 / Rx  # Maximum curvature related to minimum radius (steeper meridian)
        Kmin = (1.3375 - 1) * 1000 / Ry  # Minimum curvature related to minimum radius (flatter meridian)
        Kmean = (Kmax + Kmin) / 2

    else:
        raise ValueError('Unknown option (sphere or biconic)')

    # Solve u', v' and w' to determine conic constants Qx, Qy

    # c_target
    c_target = p[3:] * (t * z - p[3:] * z2)

    # X
    X = np.c_[x2, y2, xy]

    # Least squares
    p_u = np.matmul(np.linalg.pinv(X), c_target)

    u_p = p_u[0]
    v_p = p_u[1]
    w_p = p_u[2]

    # Conic values
    Qx = (Rx ** 2) * (u_p - w_p * np.tan(phi) / 2) - 1
    Qy = (Ry ** 2) * (v_p + w_p * np.tan(phi) / 2) - 1

    biconic = {'Rx': Rx, 'Ry': Ry, 'Qx': Qx, 'Qy': Qy, 'Phi': phi}

    # Fitting error
    a = 1 / Rx
    b = 1 / Ry
    u = (1 + Qx) / Rx ** 2
    v = (1 + Qy) / Ry ** 2
    t = 2

    # Reconstruct surface
    c_eq = (u * x ** 2 + v * y ** 2) / (a * x ** 2 + b * y ** 2)
    B = -t * np.ones(x.shape[0])
    C = a * x ** 2 + b * y ** 2

    # Predict sagitta
    z_pred = []
    for ix in range(B.shape[0]):
        z_pred.append(np.roots([c_eq[ix], B[ix], C[ix]]))

    z_pred = np.array(z_pred)

    # Select correct solution
    centroid_target = np.mean(z)
    centroids_pred = np.mean(z_pred, axis=0)
    diff = np.abs(centroids_pred - centroid_target)
    indx = int(np.where(diff == np.min(diff))[0])
    z_pred = z_pred[:, indx]

    # Calculate error
    MSE = np.sum(np.sqrt((z_pred - z) ** 2))

    # if self.verbose:
    #     print('MSE: %1.3f' % MSE)
    #
    #     print('Kmax: %1.2f D;' % Kmax, 'Kmin: %1.2f D;' % Kmin, 'Kmean: %1.2f D;' % Kmean,
    #           'Astigm: %1.2f D' % (Kmax - Kmin),
    #           r'Angle: %1.2f deg.' % phi_deg)

    return Kmax, Kmin, Kmean, biconic


def execute_simulation(cc):
    ite_max = 12  # [-]
    tol_error = 1e-3  # [mm]

    m_1 = 65.75
    c_1 = 0.0065
    k = 100
    k_epi = cc[1]
    gamma_stroma = 0  # 5.5
    tau_stroma = 0  # 38.1666

    E_epi = cc[0]  # Young's modulus [MPa]
    nu_epi = 0.075  # Poison ratio [-]
    k_stroma = cc[2]
    gamma_epi = 0
    tau_epi = 0
    eye_lid_pressure = cc[3]
    duration_initiating_contact = 10
    duration_load = 28790
    duration_unload = 3600 * 16
    time_prestretch = tau_stroma * 5 + 64
    time_initiating_contact = time_prestretch + duration_initiating_contact
    time_load_end = time_initiating_contact + duration_load
    time_unload_end = time_load_end + duration_unload
    parameter_name = ['m_1', 'c_1', 'k', 'k_stroma', 'gamma_stroma', 'tau_stroma', 'E_epi', 'nu_epi', 'k_epi',
                      'gamma_epi',
                      'tau_epi', 'time_prestretch', 'time_initiating_contact', 'time_load_end', 'time_unload_end',
                      'eye_lid_pressure']

    unload_disp = 0.0075 + k_epi / 1.995e-5 * 0.003 - 0.007 * \
                  ((0.00025 - eye_lid_pressure) / 0.0005 - 0.5 * (0.0015 - E_epi) / 0.0015)
    time_1 = [0, 64, time_prestretch]
    magnitude_1 = [0, 1, 1]
    time_2 = [0, 64 * 0.3, 64]
    magnitude_2 = [0.25, time_prestretch * 0.5, time_prestretch * 1.5]
    time_3 = [time_prestretch, time_initiating_contact, 3600 * 24 + time_prestretch,
              3600 * 24 + time_initiating_contact, 2 * 3600 * 24 + time_prestretch,
              2 * 3600 * 24 + time_initiating_contact, 3 * 3600 * 24 + time_prestretch,
              3 * 3600 * 24 + time_initiating_contact]
    magnitude_3 = [-2.5, 2, -2.5, 2.5, -2.5, 3, -2.5, 3.5]
    time_4 = [time_initiating_contact, time_initiating_contact + 50, 3600 * 24, 3600 * 24 + time_initiating_contact,
              3600 * 24 + time_initiating_contact + 50, 3600 * 24 * 2, 2 * 3600 * 24 + time_initiating_contact,
              2 * 3600 * 24 + time_initiating_contact + 50, 3 * 3600 * 24, 3 * 3600 * 24 + time_initiating_contact,
              3 * 3600 * 24 + time_initiating_contact + 50]
    magnitude_4 = [0.25, 1, 1, 0.25, 1, 1, 0.25, 1, 1, 0.25, 1]
    time_5 = [time_load_end, time_load_end + 50, time_load_end + 50.5, 24 * 3600, 24 * 3600 + time_load_end,
              24 * 3600 + time_load_end + 50, 24 * 3600 + time_load_end + 50.5, 2 * 24 * 3600,
              2 * 24 * 3600 + time_load_end, 2 * 24 * 3600 + time_load_end + 50, 2 * 24 * 3600 + time_load_end + 50.5,
              3 * 24 * 3600, 3 * 24 * 3600 + time_load_end, 3 * 24 * 3600 + time_load_end + 50,
              3 * 24 * 3600 + time_load_end + 50.5]
    magnitude_5 = [-unload_disp, 0.01, 1, 1, -(unload_disp + 0.001), 0.01, 1, 1, -(unload_disp + 0.0015), 0.01, 1,
                   1, -(unload_disp + 0.002), 0.01, 1]
    time_6 = [time_prestretch, time_prestretch + 60, time_prestretch + 500, time_prestretch + 2500, time_load_end,
              24 * 3600 + time_prestretch, 24 * 3600 + time_prestretch + 60, 24 * 3600 + time_prestretch + 500,
              24 * 3600 + time_prestretch + 2500, 24 * 3600 + time_load_end, 2 * 24 * 3600 + time_prestretch,
              2 * 24 * 3600 + time_prestretch + 60, 2 * 24 * 3600 + time_prestretch + 500,
              2 * 24 * 3600 + time_prestretch + 2500, 2 * 24 * 3600 + time_load_end, 3 * 24 * 3600 + time_prestretch,
              3 * 24 * 3600 + time_prestretch + 60, 3 * 24 * 3600 + time_prestretch + 500,
              3 * 24 * 3600 + time_prestretch + 2500, 3 * 24 * 3600 + time_load_end]
    magnitude_6 = [1, 50, 600, 1200, 2500, 1, 50, 600, 1200, 2500, 1, 50, 600, 1200, 2500, 1, 50, 600, 1200, 2500]
    time_7 = [time_load_end, time_load_end + 50, time_load_end + 500, time_load_end + 2500, time_unload_end,
              24 * 3600 + time_load_end, 24 * 3600 + time_load_end + 50, 24 * 3600 + time_load_end + 500,
              24 * 3600 + time_load_end + 2500, 24 * 3600 + time_unload_end, 2 * 24 * 3600 + time_load_end,
              2 * 24 * 3600 + time_load_end + 50, 2 * 24 * 3600 + time_load_end + 500,
              2 * 24 * 3600 + time_load_end + 2500, 2 * 24 * 3600 + time_unload_end, 3 * 24 * 3600 + time_load_end,
              3 * 24 * 3600 + time_load_end + 50, 3 * 24 * 3600 + time_load_end + 500,
              3 * 24 * 3600 + time_load_end + 2500, 3 * 24 * 3600 + time_unload_end]
    magnitude_7 = [0.1, 50, 600, 1200, 3500, 0.1, 50, 600, 1200, 3500, 0.1, 50, 600, 1200, 3500, 0.1, 50, 600, 1200,
                   3500]

    main_path = os.popen('pwd').read()[:-1]
    parameter = [m_1, c_1, k, k_stroma, gamma_stroma, tau_stroma, E_epi, nu_epi, k_epi, gamma_epi, tau_epi,
                 10 * time_prestretch, 10 * duration_initiating_contact, 10 * duration_load, 10 * duration_unload,
                 eye_lid_pressure]
    write_parameters(parameter, parameter_name, path=main_path)
    write_loadcurve(time_1, magnitude_1, 'pre_stretch_load_curve.feb', 1, path=main_path)
    write_loadcurve(time_2, magnitude_2, 'pre_stretch_must_point_curve.feb', 2, path=main_path)
    write_loadcurve(time_3, magnitude_3, 'initiating_contact_load_curve.feb', 3, path=main_path)
    write_loadcurve(time_4, magnitude_4, 'load_curve.feb', 4, path=main_path)
    write_loadcurve(time_5, magnitude_5, 'unload_curve.feb', 5, path=main_path)
    write_loadcurve(time_6, magnitude_6, 'must_point_curve_1.feb', 6, path=main_path)
    write_loadcurve(time_7, magnitude_7, 'must_point_curve_2.feb', 7, path=main_path)
    
    pre_stretch(ite_max, tol_error, path=main_path)
    os.system('/home/ubelix/artorg/shared/software/FEBio2.8.5/bin/febio2.lnx64 -i 4_day_with_prestretch.feb -o 4_day.log -p 4_day.xplt &>> 4_day-jg.log')

def check_success(path_log_name):
    # Sanity check
    if '.' not in path_log_name:
        raise ValueError('File must have the extension (%s)'%path_log_name)

    # Open log file from FEBio
    log = open(path_log_name, 'r')

    # Dumped all lines in list AND reverse the list
    log = log.readlines()[::-1]

    # Trim the list to keep only the part with interesting information (avoids long executions when failure)
    log = log[:20]

    # For all the lines in the list, check whether the Normal Termination is reached (returns 0). Otherwise, fails and returns 1
    for line in log:
        # Remove return carriage at the end of line and blank spaces at the beginning
        line = line.strip()

        # If the length of the line is 0, it is empty. Otherwise, check if it is normal termination
        if len(line) == 0:  #Skips empty line
            continue
        else:
            if line == 'N O R M A L   T E R M I N A T I O N':
                return 0
            elif line =='E R R O R   T E R M I N A T I O N':
                return 1

    # The simulation is running
    return 2


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

