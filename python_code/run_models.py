#!/usr/bin/env python

import copy
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree
import subprocess
import time
import pickle
import statistics
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.pyplot as plt
from datetime import datetime
from shutil import copyfile
from scipy.stats import norm

from variogram.variogram import ExponentialVariogram
from variogram.variogram import GeneralExponentialVariogram
from variogram.variogram import SphericalVariogram
from variogram.simulate import simulate_gaussian_field

def run_TRANE_simulations(n_simulations, model):
    path = "C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\\APSvsTPG\\TRANE_models"
    os.chdir(path)
    modelfile_path = os.path.join(path, "model" + model + ".xml")
    input_path = os.path.join(path, "input")
    et = xml.etree.ElementTree.parse(modelfile_path)
    
    out_z = []
    out_z_simbox = []
    parameters = []
    print("Start TRANE-simulations")
    for iteration in range(0, n_simulations):
        print("iteration = " + str(iteration))
        seed            = iteration
        seed_tag        = et.findall('.//seed')[0]
        seed_tag.text   = str(seed)
        output_tag      = et.findall('.//output-directory')[0]
        output_tag.text = "output" + model + "_edited"
        nx       = int(et.findall('.//nx')[0].text.strip())
        ny       = int(et.findall('.//ny')[0].text.strip())
        x_length = float(et.findall('.//x-length')[0].text.strip())
        y_length = float(et.findall('.//y-length')[0].text.strip())
        dx = x_length / nx
        dy = y_length / ny
        et.write('model' + model + '_edited.xml')
        modelfile_edited_path = os.path.join(path, "model" + model + "_edited.xml")
        output_path           = os.path.join(path, "output" + model + "_edited")
        results_path          = os.path.join(output_path, "result.roff")

        subprocess.call(["%tra%", modelfile_edited_path], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        lines = []
        with open(results_path) as f:
            lines = f.readlines()

        nx   = int(lines[13].split()[2])
        ny   = int(lines[14].split()[2])
        nz   = int(lines[15].split()[2])
        data = lines[20].split()

        x    = np.linspace(0.0, x_length, num=nx)
        y    = np.linspace(0.0, y_length, num=ny)
        X, Y = np.meshgrid(x, y)
        X2, Y2 = np.meshgrid(y, x)
        z = X2 ** 2 - Y2 ** 2
        z_simbox = X ** 2 - Y ** 2

        temp = np.zeros((nx,ny,nz))
        counter = 0
        for i in range(0, nx):
            for j in range(0, ny):
                for k in range(0, nz):
                    temp[i][j][k] = data[counter]
                    counter += 1
        for i in range(0, nx):
            for j in range(0, ny):
                for k in range(0, nz):
                    if k == 0:
                        z_simbox[ny - 1 - j][i] = temp[i][j][nz-k-1]
                        z[i][j] = temp[i][j][nz-k-1]

        out_z.append(z)
        out_z_simbox.append(z_simbox)
        if iteration == 0:
            parameters = [dx, dy, x_length, y_length]
    print("TRANE-simulations completed")
    return out_z, out_z_simbox, parameters

def run_APS_simulations(n_simulations, nx, ny, dx, dy, model):
    if model == "1" or model == "2" or model == "3":
        v1_variogram = "genexp"
        v1_range_x = 800.0
        v1_range_y = 500.0
        v1_range_z = 20.0
        v1_azimuth = 30.0 * 3.141592 / 180.0 # In radians, not degrees
        v1_genexp_power = 1.5
    elif model == "4":
        v1_variogram = "genexp"
        v1_range_x = 1600.0
        v1_range_y = 1000.0
        v1_range_z = 20.0
        v1_azimuth = 30.0 * 3.141592 / 180.0 # In radians, not degrees
        v1_genexp_power = 1.5

    if model == "2" or model == "3":
        v2_variogram = "genexp"
        v2_range_x = 400.0
        v2_range_y = 400.0
        v2_range_z = 20.0
        v2_azimuth = 0.0
        v2_genexp_power = 1.8
    elif model == "4":
        v2_variogram = "genexp"
        v2_range_x = 800.0
        v2_range_y = 800.0
        v2_range_z = 20.0
        v2_azimuth = 0.0
        v2_genexp_power = 1.8
 
    p_F1 = np.load("p1_from_TRANE.npy")
    p_F2 = np.load("p2_from_TRANE.npy")
    p_F3 = np.load("p3_from_TRANE.npy")
    
    # Calculate thresholds
    t1 = np.zeros((nx, ny))
    t2 = np.zeros((nx, ny))
    for i in range(0, nx):
        for j in range(0, ny):
            if model == "1":
                t1[i][j] = norm.ppf(p_F1[i][j])
                p1_p2 = min(1.0, p_F1[i][j] + p_F2[i][j])
                t2[i][j] = norm.ppf(p1_p2)
            elif model == "2" or model == "3" or model == "4":
                t1[i][j] = norm.ppf(p_F3[i][j])
                t2[i][j] = norm.ppf(min(1.0, p_F1[i][j] / (1.0 - p_F3[i][j])))

    v1 = GeneralExponentialVariogram(v1_range_x, v1_range_y, v1_range_z, azi=v1_azimuth, power=v1_genexp_power)
    if model == "2" or model == "3" or model == "4":
        v2 = GeneralExponentialVariogram(v2_range_x, v2_range_y, v2_range_z, azi=v2_azimuth, power=v2_genexp_power)

    out_z = []
    print("Start APS-simulations")
    for iteration in range(0, n_simulations):
        print("iteration = " + str(iteration))
        s1 = simulate_gaussian_field(v1, nx, dx, ny, dy, seed = iteration)
        if model == "2" or model == "3" or model == "4":
            s2 = simulate_gaussian_field(v2, nx, dx, ny, dy, seed = iteration)
        z = np.ndarray(s1.shape)
        for i in range(0, nx):
            for j in range(0, ny):
                if model == "1":
                    if s1[i][j] < t1[i][j]:
                        z[i][j] = 1
                    elif s1[i][j] < t2[i][j]:
                        z[i][j] = 2
                    else:
                        z[i][j] = 3
                elif model == "2" or model == "3" or model == "4":
                    if s1[i][j] < t1[i][j]:
                        z[i][j] = 3
                    elif s2[i][j] < t2[i][j]:
                        z[i][j] = 1
                    else:
                        z[i][j] = 2
        out_z.append(z)
    print("APS-simulations completed")
    return out_z

def save_facies_grids_as_png(facies_grids, parameters, prefix, indices_to_save="all"):
    print("save_facies_grids_as_png for " + prefix)
    F1 = ( 40.0/255.0, 118.0/255.0, 255.0/255.0) # Blue
    F2 = (242.0/255.0, 255.0/255.0,  57.0/255.0) # Yellow
    F3 = (138.0/255.0,  43.0/255.0, 226.0/255.0) # Purple
    nx = facies_grids[0].shape[0]
    ny = facies_grids[0].shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_length = parameters[2]
    y_length = parameters[3]
    x_min = 0.0
    x_max = dx * nx
    y_min = 0.0
    y_max = dy * ny
    x = [3000.0, 3500.0]
    y = [2000.0, 2000.0]
    extent = x_min, x_max, y_min, y_max

    folder = "facies_grids_" + prefix
    if not os.path.exists(folder):
        os.mkdir(folder)

    for iteration, z in enumerate(facies_grids):
        if indices_to_save == "all" or iteration in indices_to_save:
            # To plot the ndarray correctly:
            x_lin = np.linspace(0.0, x_length, num=nx)
            y_lin = np.linspace(0.0, y_length, num=ny)
            Y, X  = np.meshgrid(x_lin, y_lin)
            z_for_plotting  = (X ** 2 - Y ** 2) * 0.0
            for i in range(0, nx):
                for j in range(0, ny):
                    z_for_plotting[j][i] = z[i][ny - 1 - j]

            cmap = colors.ListedColormap([F1, F2, F3])
            fig = plt.figure(frameon=False)
            fig.set_size_inches(6,4)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # img = plt.imshow(z_simbox, cmap = cmap, alpha = 1.0, interpolation='none', extent = extent) # interpolation ='bilinear'
            img = plt.imshow(z_for_plotting, cmap = cmap, alpha = 1.0, interpolation='none', extent = extent) # interpolation ='bilinear'
            plt.scatter(x, y, facecolors='none', edgecolors='red', s = 150)
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S_%f")[:-2]
            plt.savefig(folder + "\\" + prefix + '_' + current_time + '_it' + str(iteration) + '.png', dpi=100)
            plt.close()
   
def count_connected_grid_nodes(facies_grids, parameters, x_observation, y_observation, extra_obs=None):
    nx = facies_grids[0].shape[0]
    ny = facies_grids[0].shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_length = parameters[2]
    y_length = parameters[3]
    x_1_ind = math.floor(x_observation / dx)
    y_1_ind = math.floor(y_observation / dy)
    x    = np.linspace(0.0, x_length, num=nx)
    y    = np.linspace(0.0, y_length, num=ny)
    X, Y = np.meshgrid(y, x)
    count_connected = []
    for iteration, z in enumerate(facies_grids):
        connected = X ** 2 - Y ** 2 # Dummy values
        for i in range(0, nx):
            for j in range(0, ny):
                connected[i][j] = False
        connected[x_1_ind][y_1_ind] = True
        
        need_to_check = [[x_1_ind, y_1_ind]]
        facies = z[x_1_ind][y_1_ind]
        while len(need_to_check) > 0:
            x = need_to_check[0][0]
            y = need_to_check[0][1]
            need_to_check.pop(0)
            if x + 1 < nx and z[x + 1][y] == facies and not connected[x + 1][y]:
                connected[x + 1][y] = True
                need_to_check.append([x + 1, y])
            if x - 1 >= 0 and z[x - 1][y] == facies and not connected[x - 1][y]:
                connected[x - 1][y] = True
                need_to_check.append([x - 1, y])
            if y + 1 < ny and z[x][y + 1] == facies and not connected[x][y + 1]:
                connected[x][y + 1] = True
                need_to_check.append([x, y + 1])
            if y - 1 >= 0 and z[x][y - 1] == facies and not connected[x][y - 1]:
                connected[x][y - 1] = True
                need_to_check.append([x, y - 1])

        if extra_obs == None:
            count_connected.append(np.count_nonzero(connected == True))
        else:
            x_2 = extra_obs[0]
            y_2 = extra_obs[1]
            x_2_ind = math.floor(x_2 / dx)
            y_2_ind = math.floor(y_2 / dy)
            if connected[x_2_ind][y_2_ind]:
                count_connected.append(np.count_nonzero(connected == True))
            else:
                count_connected.append(-1)
    return count_connected
  
def calculate_and_save_facies_prob_maps(facies_grids, parameters, prefix):
    print("calculate_and_save_facies_prob_maps for " + prefix)
    nx = facies_grids[0].shape[0]
    ny = facies_grids[0].shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_length = parameters[2]
    y_length = parameters[3]
    x_min = 0.0
    x_max = dx * nx
    y_min = 0.0
    y_max = dy * ny
    extent = x_min, x_max, y_min, y_max
    x    = np.linspace(0.0, x_length, num=nx)
    y    = np.linspace(0.0, y_length, num=ny)
    # X, Y = np.meshgrid(x, y)
    X, Y = np.meshgrid(y, x)
    p_F1 = (X ** 2 - Y ** 2) * 0.0
    p_F2 = (X ** 2 - Y ** 2) * 0.0
    p_F3 = (X ** 2 - Y ** 2) * 0.0
    for iteration, z in enumerate(facies_grids):
        for i in range(0, nx):
            for j in range(0, ny):
                a = 1 if z[i][j] == 1 else 0
                b = 1 if z[i][j] == 2 else 0
                c = 1 if z[i][j] == 3 else 0
                p_F1[i][j]  = (p_F1[i][j]  * iteration + a) / (iteration + 1)
                p_F2[i][j]  = (p_F2[i][j]  * iteration + b) / (iteration + 1)
                p_F3[i][j]  = (p_F3[i][j]  * iteration + c) / (iteration + 1)
    print("Done calculating facies probabilities")
    np.save("p1_from_" + prefix, p_F1)
    np.save("p2_from_" + prefix, p_F2)
    np.save("p3_from_" + prefix, p_F3)
    for i, p in enumerate([p_F1, p_F2, p_F3]):
        # To plot the ndarray correctly:
        x_lin = np.linspace(0.0, x_length, num=nx)
        y_lin = np.linspace(0.0, y_length, num=ny)
        Y, X  = np.meshgrid(x_lin, y_lin)
        p_for_plotting  = (X ** 2 - Y ** 2) * 0.0
        for ii in range(0, nx):
            for j in range(0, ny):
                p_for_plotting[j][ii] = p[ii][ny - 1 - j]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(6,4)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = plt.imshow(p_for_plotting, cmap = 'Blues', alpha = 1.0, interpolation='none', extent = extent, vmin = 0.0, vmax = 1.0)
        # fig.colorbar(img, ax=ax, shrink=0.8)
        fig.colorbar(img, ax=ax, shrink=0.5)
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S_%f")[:-2]
        plt.savefig(prefix + '_' + current_time + '_p' + str(i+1) + '_n' + str(len(facies_grids)) + '.png', dpi=100)
        plt.close()

def plot_histogram_of_connected_cells(sum_connected, prefix, xmin, xmax, ymin, ymax, n_bins):
    print("plot_histogram_of_connected_cells for " + prefix)
    fig = plt.figure(frameon=False)
    binwidth = xmax / n_bins
    density = False
    count = True
    plt.hist(sum_connected, density=density, bins=np.arange(xmin, xmax + binwidth, binwidth))
    if density:
        plt.ylabel('Probability')
    else:
        plt.ylabel('Count')
    if count:
        plt.xlabel('Connected grid nodes')
    else:
        plt.xlabel('Connected area')
    plt.xlim(xmin=xmin, xmax=xmax)
    if True:
        plt.ylim(ymin=ymin, ymax=ymax)
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S_%f")[:-2]
    plt.savefig(prefix + '_' + current_time + '_connectedvolume' + '_n' + str(len(sum_connected)) + '.png', dpi=100)
    # plt.show()
    plt.close()

def calculate_volume_fractions(facies_grids):
    n_simulations = len(facies_grids)
    nx = facies_grids[0].shape[0]
    ny = facies_grids[0].shape[1]
    v = {1: [], 2: [], 3: []}
    for z in facies_grids:
        unique, counts = np.unique(z, return_counts=True)
        unique = [int(facies) for facies in unique]
        percent = [count / (nx * ny) for count in counts]
        percent_dict = dict(zip(unique, percent))
        v[1].append(percent_dict[1])
        v[2].append(percent_dict[2])
        v[3].append(percent_dict[3])
    return v

def save_threshold_grids_as_png(parameters):
    p_F1 = np.load("p1_from_TRANE.npy")
    p_F2 = np.load("p2_from_TRANE.npy")
    p_F3 = np.load("p3_from_TRANE.npy")

    nx = p_F1.shape[0]
    ny = p_F1.shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_length = parameters[2]
    y_length = parameters[3]
    x_min = 0.0
    x_max = dx * nx
    y_min = 0.0
    y_max = dy * ny
    extent = x_min, x_max, y_min, y_max

    # Calculate thresholds
    t1 = np.zeros((nx, ny))
    t2 = np.zeros((nx, ny))
    for i in range(0, nx):
        for j in range(0, ny):
            t1[i][j] = norm.ppf(p_F1[i][j])
            p1_p2 = min(1.0, p_F1[i][j] + p_F2[i][j])
            t2[i][j] = norm.ppf(p1_p2)
    
    for i, t in enumerate([t1, t2]):
        # To plot the ndarray correctly:
        x_lin = np.linspace(0.0, x_length, num=nx)
        y_lin = np.linspace(0.0, y_length, num=ny)
        Y, X  = np.meshgrid(x_lin, y_lin)
        t_for_plotting  = (X ** 2 - Y ** 2) * 0.0
        for ii in range(0, nx):
            for j in range(0, ny):
                t_for_plotting[j][ii] = t[ii][ny - 1 - j]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(4,4)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = plt.imshow(t_for_plotting, cmap = 'Blues', alpha = 1.0, interpolation='none', extent = extent, vmin = -5.0, vmax = 5.0)
        fig.colorbar(img, ax=ax, shrink=0.5)
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S_%f")[:-2]
        plt.savefig('APS_' + current_time + '_t' + str(i+1) + '.png', dpi=100)
        plt.close()

n_simulations = 1000
# TRANE:
# ------
use_existing = True
model = "4"

if not use_existing:
    z_TRANE, z_simbox_TRANE, parameters = run_TRANE_simulations(n_simulations, model)
path = "C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\output1"
if not os.path.exists(path):
    os.mkdir(path)
else:
    i = 2
    while os.path.exists(path):
        path = path[:-1] + str(i)
        i += 1
    os.mkdir(path)
os.chdir(path)
if not use_existing:
    with open("z_TRANE", "wb") as fp:
        pickle.dump(z_TRANE, fp)
    with open("parameters", "wb") as fp:
        pickle.dump(parameters, fp)
else:
    with open("C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\data_model4\\z_TRANE", "rb") as fp:
        z_TRANE = pickle.load(fp)
    with open("C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\data_model4\\parameters", "rb") as fp:
        parameters = pickle.load(fp)
# save_facies_grids_as_png(z_TRANE, parameters, 'TRANE', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# calculate_and_save_facies_prob_maps(z_TRANE, parameters, 'TRANE')
count_connected_TRANE = count_connected_grid_nodes(z_TRANE, parameters, 3000.0, 2000.0, [3500, 2000])
dx = parameters[0]
dy = parameters[1]
# sum_connected_TRANE = [dx * dy * n for n in count_connected_TRANE]

# APS:
# ----
nx = z_TRANE[0].shape[0]
ny = z_TRANE[0].shape[1]
if not use_existing:
    z_APS = run_APS_simulations(n_simulations, nx, ny, dx, dy, model)
# save_threshold_grids_as_png(parameters)
if not use_existing:
    with open("z_APS", "wb") as fp:
        pickle.dump(z_APS, fp)
else:
    with open("C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\data_model4\\z_APS", "rb") as fp:
        z_APS = pickle.load(fp)
# save_facies_grids_as_png(z_APS, parameters, 'APS', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# calculate_and_save_facies_prob_maps(z_APS, parameters, 'APS')

# v_TRANE = calculate_volume_fractions(z_TRANE)
# v_APS = calculate_volume_fractions(z_APS)
# print(statistics.mean(v_TRANE[1]))
# print(statistics.mean(v_APS[1]))
# print(statistics.stdev(v_TRANE[1]))
# print(statistics.stdev(v_APS[1]))
# plot_histogram_of_connected_cells(v_TRANE[1], 'TRANE', 0.0, 0.2, 0.0, 120, 50)
# plot_histogram_of_connected_cells(v_APS[1], 'APS', 0.0, 0.2, 0.0, 120, 50)
count_connected_APS = count_connected_grid_nodes(z_APS, parameters, 3000.0, 2000.0, [3500, 2000])
# sum_connected_APS = [dx * dy * n for n in count_connected_APS]

count_connected_filtered_TRANE = []
count_connected_filtered_APS = []
for count in count_connected_TRANE:
    if count != -1:
        count_connected_filtered_TRANE.append(count)
# for i in range(0, len(count_connected_TRANE)):
#     if count_connected_TRANE[i] == -1:
#         print(i)
for count in count_connected_APS:
    if count != -1:
        count_connected_filtered_APS.append(count)
# for i in range(0, len(count_connected_APS)):
#     if count_connected_APS[i] == -1:
#         print(i)
print("Statistics connected grid nodes:")
print(len(count_connected_filtered_TRANE))
print(len(count_connected_filtered_APS))
print(statistics.mean(count_connected_filtered_TRANE))
print(statistics.mean(count_connected_filtered_APS))
print(statistics.stdev(count_connected_filtered_TRANE))
print(statistics.stdev(count_connected_filtered_APS))
max1 = max(count_connected_filtered_TRANE)
max2 = max(count_connected_filtered_APS)
max3 = max(max1, max2)
plot_histogram_of_connected_cells(count_connected_filtered_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 100, 50)
plot_histogram_of_connected_cells(count_connected_filtered_APS, 'APS', 0.0, max3*1.05, 0.0, 100, 50)
# save_facies_grids_as_png(z_TRANE, parameters, 'TRANE', [3, 10, 11, 12, 13, 14, 15])
# save_facies_grids_as_png(z_APS, parameters, 'APS', [6, 9, 11, 13, 14, 15, 16, 17])

# max1 = max(count_connected_TRANE)
# max2 = max(count_connected_APS)
# max3 = max(max1, max2)
# plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 250, 50)
# plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, max3*1.05, 0.0, 250, 50)
# plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, 100, 0.0, 130, 101)
# plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, 100, 0.0, 130, 101)
# plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, 20, 0.0, 130, 21)
# plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, 20, 0.0, 130, 21)

# print(count_connected_TRANE[0:5])
# print(count_connected_APS[0:5])

# indices_no_connection_APS = []
# for i, count in enumerate(count_connected_APS):
#     if count == 1:
#         indices_no_connection_APS.append(i)
# indices_no_connection_TRANE = []
# for i, count in enumerate(count_connected_TRANE):
#     if count == 1:
#         indices_no_connection_TRANE.append(i)
# save_facies_grids_as_png(z_APS, parameters, 'APS', indices_no_connection_APS)
# save_facies_grids_as_png(z_TRANE, parameters, 'TRANE', indices_no_connection_TRANE)













