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

def run_TRANE_simulations(n_simulations):
    path = "C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\\APSvsTPG\\TRANE_models"
    os.chdir(path)
    modelfile_path = os.path.join(path, "model.xml")
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
        output_tag.text = "output_edited"
        nx       = int(et.findall('.//nx')[0].text.strip())
        ny       = int(et.findall('.//ny')[0].text.strip())
        x_length = float(et.findall('.//x-length')[0].text.strip())
        y_length = float(et.findall('.//y-length')[0].text.strip())
        dx = x_length / nx
        dy = y_length / ny
        et.write('model_edited.xml')
        modelfile_edited_path = os.path.join(path, "model_edited.xml")
        output_path           = os.path.join(path, "output_edited")
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
        # for i in range(0, nx):
        #     for j in range(0, ny):
        #         test[j][i] = z[i][ny - 1 - j]

        out_z.append(z)
        out_z_simbox.append(z_simbox)
        if iteration == 0:
            parameters = [dx, dy, x_length, y_length]
    print("TRANE-simulations completed")
    return out_z, out_z_simbox, parameters

def run_APS_simulations(n_simulations, nx, ny, dx, dy):
    variogram = "genexp"
    range_x = 800.0
    range_y = 500.0
    range_z = 20.0
    azimuth = 30.0 * 3.141592 / 180.0 # In radians, not degrees
    genexp_power = 1.5
 
    p_F1 = np.load("p1_from_TRANE.npy")
    p_F2 = np.load("p2_from_TRANE.npy")
    p_F3 = np.load("p3_from_TRANE.npy")
    
    # Calculate thresholds
    t1 = np.zeros((nx, ny))
    t2 = np.zeros((nx, ny))
    for i in range(0, nx):
        for j in range(0, ny):
            t1[i][j] = norm.ppf(p_F1[i][j])
            p1_p2 = min(1.0, p_F1[i][j] + p_F2[i][j])
            t2[i][j] = norm.ppf(p1_p2)
            # if p_F1[i][j] > 0.7:
            #     print("\npF1 = " + str(p_F1[i][j]))
            #     print("pF2 = " + str(p_F2[i][j]))
            #     print("pF3 = " + str(p_F3[i][j]))
            #     print("1+2 = " + str(p_F1[i][j] + p_F2[i][j]))
            #     print("t1  = " + str(t1[i][j]))
            #     print("t2o = " + str(norm.ppf(p_F1[i][j] + p_F2[i][j])))
            #     print("t2n = " + str(t2[i][j]))

    if variogram == "exponential":
        v = ExponentialVariogram(range_x, range_y, range_z, azimuth)
    elif variogram == "genexp":
        v = GeneralExponentialVariogram(range_x, range_y, range_z, azi=azimuth, power=genexp_power)
    elif variogram == "spherical":
        v = SphericalVariogram(range_x, range_y, range_z, azimuth)

    out_z = []
    out_z_simbox = []
    print("Start APS-simulations")
    for iteration in range(0, n_simulations):
        print("iteration = " + str(iteration))
        # print("Start simulation of GRF")
        start = time.time()
        s = simulate_gaussian_field(v, nx, dx, ny, dy, seed = iteration)
        end = time.time()
        # print("Finished simulation of GRF in " + str(round(end-start, 2)))
        z = np.ndarray(s.shape)
        z_simbox = np.ndarray((s.shape[1], s.shape[0]))
        for i in range(0, nx):
            for j in range(0, ny):
                if s[i][j] < t1[i][j]:
                    z[i][j] = 1
                elif s[i][j] < t2[i][j]:
                    z[i][j] = 2
                else:
                    z[i][j] = 3
        for i in range(0, nx):
            for j in range(0, ny):
                z_simbox[j][i] = z[i][ny - 1 - j]
        out_z.append(z)
        out_z_simbox.append(z_simbox)
    print("APS-simulations completed")
    return out_z, out_z_simbox

def save_facies_grids_as_png(facies_grids, parameters, prefix, indices_to_save="all"):
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
    x = [3000.0]
    y = [2000.0]
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
   
def count_connected_grid_nodes(facies_grids, parameters, x_observation, y_observation):
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
        
        count_connected.append(np.count_nonzero(connected == True))
    return count_connected
  
def calculate_and_save_facies_prob_maps(facies_grids, parameters, prefix):
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

if not use_existing:
    z_TRANE, z_simbox_TRANE, parameters = run_TRANE_simulations(n_simulations)
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
    with open("C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\data1\\z_TRANE", "rb") as fp:
        z_TRANE = pickle.load(fp)
    with open("C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\data1\\parameters", "rb") as fp:
        parameters = pickle.load(fp)
# save_facies_grids_as_png(z_TRANE, parameters, 'TRANE')
# calculate_and_save_facies_prob_maps(z_TRANE, parameters, 'TRANE')
v_TRANE = calculate_volume_fractions(z_TRANE)
count_connected_TRANE = count_connected_grid_nodes(z_TRANE, parameters, 3000.0, 2000.0)
dx = parameters[0]
dy = parameters[1]
sum_connected_TRANE = [dx * dy * n for n in count_connected_TRANE]

# APS:
# ----
nx = z_TRANE[0].shape[0]
ny = z_TRANE[0].shape[1]
if not use_existing:
    z_APS, z_simbox_APS = run_APS_simulations(n_simulations, nx, ny, dx, dy)
# save_threshold_grids_as_png(parameters)
if not use_existing:
    with open("z_APS", "wb") as fp:
        pickle.dump(z_APS, fp)
else:
    with open("C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\data1\\z_APS", "rb") as fp:
        z_APS = pickle.load(fp)
# save_facies_grids_as_png(z_APS, parameters, 'APS')
# calculate_and_save_facies_prob_maps(z_APS, parameters, 'APS')
v_APS = calculate_volume_fractions(z_APS)
print(statistics.mean(v_TRANE[1]))
print(statistics.mean(v_APS[1]))
print(statistics.stdev(v_TRANE[1]))
print(statistics.stdev(v_APS[1]))
plot_histogram_of_connected_cells(v_TRANE[1], 'TRANE', 0.0, 0.2, 0.0, 100, 50)
plot_histogram_of_connected_cells(v_APS[1], 'APS', 0.0, 0.2, 0.0, 100, 50)
exit()
count_connected_APS = count_connected_grid_nodes(z_APS, parameters, 3000.0, 2000.0)
sum_connected_APS = [dx * dy * n for n in count_connected_APS]

indices_no_connection_APS = []
for i, count in enumerate(count_connected_APS):
    if count == 1:
        indices_no_connection_APS.append(i)
indices_no_connection_TRANE = []
for i, count in enumerate(count_connected_TRANE):
    if count == 1:
        indices_no_connection_TRANE.append(i)

save_facies_grids_as_png(z_APS, parameters, 'APS', [0,1,2,3,4,5])
save_facies_grids_as_png(z_TRANE, parameters, 'TRANE', [0,1,2,3,4,5])
# save_facies_grids_as_png(z_APS, parameters, 'APS', indices_no_connection_APS)
# save_facies_grids_as_png(z_TRANE, parameters, 'TRANE', indices_no_connection_TRANE)

max1 = max(count_connected_TRANE)
max2 = max(count_connected_APS)
max3 = max(max1, max2)

# plot_histogram_of_connected_cells(sum_connected_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 0.00001, 10)
# plot_histogram_of_connected_cells(sum_connected_APS, 'APS', 0.0, max3*1.05, 0.0, 0.00001, 10)
plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 250, 50)
plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, max3*1.05, 0.0, 250, 50)
plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, 100, 0.0, 130, 101)
plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, 100, 0.0, 130, 101)
plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, 20, 0.0, 130, 21)
plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, 20, 0.0, 130, 21)

print(count_connected_TRANE[0:5])
print(count_connected_APS[0:5])

# colormap_grf = 'bwr'
# colormap_facies = 'Set1'
# colormap_facies = cm.get_cmap('Set1', 3)
# F1 = ( 40.0/255.0, 118.0/255.0, 255.0/255.0) # Blue
# F2 = (242.0/255.0, 255.0/255.0,  57.0/255.0) # Yellow
# F3 = (138.0/255.0,  43.0/255.0, 226.0/255.0) # Purple

    # Plot and save facies grid
    # cmap = colors.ListedColormap([F1, F2, F3])
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(4, 4)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # img = plt.imshow(facies, cmap=cmap, alpha=1.0, interpolation='none', extent=extent)  # interpolation ='bilinear'
    # plt.scatter([x_1, 2000], [y_1, 2000], facecolors='none', edgecolors='red', s=150)
    # now = datetime.now()
    # current_time = now.strftime("%H_%M_%S_%f")[:-2]
    # plt.savefig('APS_' + current_time + '_it_' + str(iteration) + '.png', dpi=100)  
















