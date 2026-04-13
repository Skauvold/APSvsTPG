#!/usr/bin/env python

import os
import sys

from methods import (run_TRANE_simulations, run_APS_simulations,
                     plot_histogram_of_connected_cells,
                     _analyse, _print_header, _save, _load)

# ============================================================
# Options
# ============================================================
MODEL = "4"
n_sim = 20
use_existing_results = False

RUN_TRANE = True
RUN_APS = True
verbose = True
verbose_trane = False
plot_histograms = True

# path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models"
path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models_autocreated"
path_trane_results_to_save = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results"
path_trane_results_to_load = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results_old"
path_trane_exe = "%tra%"

RED = "\033[31m"
BRIGHT_RED = "\033[91m"
RESET = "\033[0m"


# Make folder for results. If folder already exists, make a new one to avoid overwriting
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

if RUN_TRANE:
    resolved_trane_exe = os.path.expandvars(path_trane_exe)
    if not os.path.isfile(resolved_trane_exe):
        print(f"{RED}ERROR: TRANE executable not found: '{resolved_trane_exe}' (from '{path_trane_exe}'){RESET}")
        sys.exit(1)

    _print_header('TRANE simulations')
    if not use_existing_results:
        z_TRANE, parameters = run_TRANE_simulations(n_sim, MODEL, path_trane_models, path_trane_exe, True, verbose_trane)
        os.chdir(path_trane_results_to_save)
        _save("z_TRANE", z_TRANE)
        _save("parameters", parameters)
    else:
        print("  Loading from file...")
        z_TRANE    = _load(path_trane_results_to_load, "z_TRANE")
        parameters = _load(path_trane_results_to_load, "parameters")

    dx = parameters[0]
    dy = parameters[1]

    count_connected_filtered_TRANE = _analyse(z_TRANE, parameters, 'TRANE', dx, dy, verbose, save_thresholds=True)

if RUN_APS:
    _print_header('APS simulations')
    if not RUN_TRANE:
        parameters = _load(path_trane_results_to_load, "parameters")
        z_TRANE    = _load(path_trane_results_to_load, "z_TRANE")
        dx = parameters[0]
        dy = parameters[1]

    nx = z_TRANE[0].shape[0]
    ny = z_TRANE[0].shape[1]

    if not use_existing_results:
        z_APS = run_APS_simulations(n_sim, nx, ny, dx, dy, MODEL, True)
        _save("z_APS", z_APS)
    else:
        z_APS = _load(path_trane_results_to_load, "z_APS")

    count_connected_filtered_APS = _analyse(z_APS, parameters, 'APS', dx, dy, verbose, save_indices=[0, 1, 2])

# ============================================================
# Histograms
# ============================================================
if plot_histograms and RUN_TRANE and RUN_APS:
    max1 = max(count_connected_filtered_TRANE)
    max2 = max(count_connected_filtered_APS)
    max3 = max(max1, max2)
    plot_histogram_of_connected_cells(count_connected_filtered_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 100, 50)
    plot_histogram_of_connected_cells(count_connected_filtered_APS, 'APS', 0.0, max3*1.05, 0.0, 100, 50)











# ============================================================
# Old / alternative calls kept for reference
# ============================================================
# save_facies_grids_as_png(z_TRANE, parameters, 'TRANE', [3, 10, 11, 12, 13, 14, 15])
# save_facies_grids_as_png(z_APS, parameters, 'APS', [6, 9, 11, 13, 14, 15, 16, 17])
#
# plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 250, 50)
# plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, max3*1.05, 0.0, 250, 50)
# plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, 100, 0.0, 130, 101)
# plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, 100, 0.0, 130, 101)
# plot_histogram_of_connected_cells(count_connected_TRANE, 'TRANE', 0.0, 20, 0.0, 130, 21)
# plot_histogram_of_connected_cells(count_connected_APS, 'APS', 0.0, 20, 0.0, 130, 21)
#
# print(statistics.mean(v_TRANE[1]))
# print(statistics.mean(v_APS[1]))
# print(statistics.stdev(v_TRANE[1]))
# print(statistics.stdev(v_APS[1]))
# plot_histogram_of_connected_cells(v_TRANE[1], 'TRANE', 0.0, 0.2, 0.0, 120, 50)
# plot_histogram_of_connected_cells(v_APS[1], 'APS', 0.0, 0.2, 0.0, 120, 50)
#
# print(count_connected_TRANE[0:5])
# print(count_connected_APS[0:5])
#
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













