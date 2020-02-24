import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.odr as odr
import scipy.optimize as optimize
from sympy import solve, solveset, var
import sympy as sp
from scipy.io import loadmat
from copy import deepcopy
import os


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
    X_aim = np.asarray(load_feb_file_nodes('geometry_init.feb', '<Nodes name=\"Cornea\">', path=path))
    X_subopt = np.asarray(load_feb_file_nodes('geometry_opt.feb', '<Nodes name=\"Cornea\">', path=path))
    X_opt = deepcopy(X_subopt)
    X_opt[:, 1:] = 0.92 * X_subopt[:, 1:]
    write_febio_geometry_file('geometry_opt.feb', X_opt, path=path)
    while (i < ite_max) and (error > tol_error):
        os.system('febio2 -i pre_stretch.feb')
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
            nodes.append([int(line[id_1 + 10:id_2 - 1])] + [float(x) for x in line[id_2+3:id_3].split(',  ')])
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
    """biconical model; inital guess: init=[a',b',d',u',v',w',mz], data to fit to: data= [x_i,y_i,z_i]"""
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
    res = optimize.minimize(f_biconic_model, init, np.array([x, y, z]), method='Nelder-Mead', options={'xtol': 1e-8})
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
    return (-init[0]**2 + x**2 + y**2 + (z-init[1])**2)**2

def sphere_fit(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    z = np.reshape(data[:, 2], [len(data[:, 0]), 1])
    init = np.array([7.6, 0])
    res = optimize.least_squares(f_sphere, init, args=np.array([x, y, z]))
    return res.x


def f_circ(init, *data):
    data = np.array(data[0:2])[:, :, 0]
    x = data[0, :]
    y = data[1, :]
    return (-init[0]**2 + x**2 + (y-init[1])**2)**2

def circ_fit(data):
    x = np.reshape(data[:, 0], [len(data[:, 0]), 1])
    y = np.reshape(data[:, 1], [len(data[:, 0]), 1])
    init = np.array([8.1, -8.1])
    res = optimize.least_squares(f_circ, init, args=np.array([x, y]))
    return res.x