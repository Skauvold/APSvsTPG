#!/usr/bin/env python

import copy
import sys
import os
import math
import matplotlib.pyplot as plt
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

# use: "pip install git+https://code.nr.no/scm/git/variogram"
from variogram.variogram import ExponentialVariogram
from variogram.variogram import GeneralExponentialVariogram
from variogram.variogram import SphericalVariogram
from variogram.simulate import simulate_gaussian_field

from methods import run_TRANE_simulations


n_simulations = 10
# TRANE:
# ------
use_existing_results = False
model_number = "4"
path_trane_models = "C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\\APSvsTPG\\TRANE_models"
path_trane_results_to_save = "C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results"
path_trane_results_to_load = "C:\\Projects\\trane_work\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results_old"

if not use_existing_results:
    print("Start TRANE-simulations")
    z_TRANE, z_simbox_TRANE, parameters = run_TRANE_simulations(n_simulations, model_number, path_trane_models, True)
    print("TRANE-simulations completed")

# Make folder for results. If folder already exists, make a new one folder name to avoid writing over old results
if not os.path.exists(path_trane_results_to_save):
    os.mkdir(path_trane_results_to_save)
else:
    i = 2
    new_path = path_trane_results_to_save
    while os.path.exists(new_path):
        new_path = path_trane_results_to_save + "_" + str(i)
        i += 1
    os.mkdir(new_path)
    path_trane_results_to_save = new_path
os.chdir(path_trane_results_to_save)

if not use_existing_results:
    print("Save TRANE-simulations to file")
    with open("z_TRANE", "wb") as fp:
        pickle.dump(z_TRANE, fp)
    with open("parameters", "wb") as fp:
        pickle.dump(parameters, fp)
else:
    print("Load TRANE-simulations from file")
    with open(os.path.join(path_trane_results_to_load, "z_TRANE"), "rb") as fp:
        z_TRANE = pickle.load(fp)
    with open(os.path.join(path_trane_results_to_load, "parameters"), "rb") as fp:
        parameters = pickle.load(fp)
exit()


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
    z_APS = run_APS_simulations(n_simulations, nx, ny, dx, dy, model_number)
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













