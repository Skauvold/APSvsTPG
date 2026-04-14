#!/usr/bin/env python

import os
import sys
from datetime import datetime

from methods import (run_TRANE_simulations, run_APS_simulations,
                     plot_histogram_of_connected_cells,
                     _analyse, _print_header, _save, _load)

# ============================================================
# Options
# ============================================================
MODEL = "0D"
n_sim = 100
use_existing_results = False

RUN_TRANE = True
RUN_APS = True
verbose = True
verbose_trane = False
plot_histograms = True

# path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models"
path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models_autocreated"
_path_results_base = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results"
path_trane_results_to_load = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results_old"
path_trane_exe = "%tra%"

RED = "\033[31m"
BRIGHT_RED = "\033[91m"
RESET = "\033[0m"


# Make results folder with timestamp-based name
_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
path_trane_results_to_save = _path_results_base + "_" + _timestamp
if os.path.exists(path_trane_results_to_save):
    _i = 2
    while os.path.exists(path_trane_results_to_save + "_" + str(_i)):
        _i += 1
    path_trane_results_to_save = path_trane_results_to_save + "_" + str(_i)
os.makedirs(path_trane_results_to_save)

path_output_trane = os.path.join(path_trane_results_to_save, "output_TRANE")
path_output_aps   = os.path.join(path_trane_results_to_save, "output_APS")
path_pickle_trane = os.path.join(path_output_trane, "pickle_backup")
path_pickle_aps   = os.path.join(path_output_aps,   "pickle_backup")
os.makedirs(path_pickle_trane)
os.makedirs(path_pickle_aps)
_log_file = os.path.join(path_trane_results_to_save, "run_log.txt")

with open(os.path.join(path_trane_results_to_save, "run_log.txt"), 'w') as _f:
    _f.write("Run log\n")
    _f.write(f"Date/time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _f.write("\n")
    _f.write(f"MODEL:                {MODEL}\n")
    _f.write(f"n_sim:                {n_sim}\n")
    _f.write(f"use_existing_results: {use_existing_results}\n")
    _f.write(f"RUN_TRANE:            {RUN_TRANE}\n")
    _f.write(f"RUN_APS:              {RUN_APS}\n")
    _f.write(f"verbose:              {verbose}\n")
    _f.write(f"verbose_trane:        {verbose_trane}\n")
    _f.write(f"plot_histograms:      {plot_histograms}\n")
    _f.write("\n")
    _f.write(f"path_trane_models:    {path_trane_models}\n")
    _f.write(f"path_trane_exe:       {path_trane_exe}\n")
    _f.write(f"Results saved to:     {path_trane_results_to_save}\n")
    if use_existing_results:
        _f.write(f"Loaded results from:  {path_trane_results_to_load}\n")

if RUN_TRANE:
    resolved_trane_exe = os.path.expandvars(path_trane_exe)
    if not os.path.isfile(resolved_trane_exe):
        print(f"{RED}ERROR: TRANE executable not found: '{resolved_trane_exe}' (from '{path_trane_exe}'){RESET}")
        sys.exit(1)

    _print_header('TRANE simulations')
    if not use_existing_results:
        z_TRANE, parameters = run_TRANE_simulations(n_sim, MODEL, path_trane_models, path_trane_exe, True, verbose_trane)
        _save(os.path.join(path_pickle_trane, "z_TRANE"), z_TRANE)
        _save(os.path.join(path_pickle_trane, "parameters"), parameters)
    else:
        print("  Loading from file...")
        _load_pickle_trane = os.path.join(path_trane_results_to_load, "output_TRANE", "pickle_backup")
        z_TRANE    = _load(_load_pickle_trane, "z_TRANE")
        parameters = _load(_load_pickle_trane, "parameters")

    dx = parameters[0]
    dy = parameters[1]

    count_connected_filtered_TRANE = _analyse(z_TRANE, parameters, 'TRANE', dx, dy, verbose, MODEL,
        save_thresholds=True, output_dir=path_output_trane, data_dir=path_pickle_trane, log_file=_log_file)

if RUN_APS:
    _print_header('APS simulations')
    if not RUN_TRANE:
        _load_pickle_trane = os.path.join(path_trane_results_to_load, "output_TRANE", "pickle_backup")
        parameters = _load(_load_pickle_trane, "parameters")
        z_TRANE    = _load(_load_pickle_trane, "z_TRANE")
        dx = parameters[0]
        dy = parameters[1]

    nx = z_TRANE[0].shape[0]
    ny = z_TRANE[0].shape[1]

    if not use_existing_results:
        _aps_data_dir = path_pickle_trane if RUN_TRANE else os.path.join(path_trane_results_to_load, "output_TRANE", "pickle_backup")
        z_APS = run_APS_simulations(n_sim, nx, ny, dx, dy, MODEL, True, data_dir=_aps_data_dir)
        _save(os.path.join(path_pickle_aps, "z_APS"), z_APS)
    else:
        _load_pickle_aps = os.path.join(path_trane_results_to_load, "output_APS", "pickle_backup")
        z_APS = _load(_load_pickle_aps, "z_APS")

    count_connected_filtered_APS = _analyse(z_APS, parameters, 'APS', dx, dy, verbose, MODEL,
        save_indices="all", output_dir=path_output_aps, data_dir=path_pickle_aps, log_file=_log_file)

# ============================================================
# Histograms
# ============================================================
if plot_histograms and RUN_TRANE and RUN_APS:
    max1 = max(count_connected_filtered_TRANE)
    max2 = max(count_connected_filtered_APS)
    max3 = max(max1, max2)
    plot_histogram_of_connected_cells(count_connected_filtered_TRANE, 'TRANE', 0.0, max3*1.05, 0.0, 100, 50,
        output_dir=path_output_trane)
    plot_histogram_of_connected_cells(count_connected_filtered_APS, 'APS', 0.0, max3*1.05, 0.0, 100, 50,
        output_dir=path_output_aps)











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













