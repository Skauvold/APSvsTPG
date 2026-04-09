#!/usr/bin/env python

import os
import pickle
import statistics

from methods import (run_TRANE_simulations, run_APS_simulations, save_facies_grids_as_png,
                     count_connected_grid_nodes, calculate_and_save_facies_prob_maps,
                     plot_histogram_of_connected_cells, calculate_volume_fractions,
                     save_threshold_grids_as_png)

# ============================================================
# Options
# ============================================================
MODEL = "4"
n_sim = 20
use_existing_results = False

RUN_TRANE = True
RUN_APS = True
verbose = True
plot_histograms = True

path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models"
# path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models_autocreated"
path_trane_results_to_save = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results"
path_trane_results_to_load = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results_old"
path_trane_exe = "%tra%"

RED = "\033[31m"
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

# ============================================================
# TRANE
# ============================================================
if RUN_TRANE:
    print(f"\n{RED}╔{'═' * 48}╗")
    print(f"║{'TRANE simulations':^48}║")
    print(f"╚{'═' * 48}╝{RESET}")
    if not use_existing_results:
        z_TRANE, parameters = run_TRANE_simulations(n_sim, MODEL, path_trane_models, path_trane_exe, True)
        os.chdir(path_trane_results_to_save)
        with open("z_TRANE", "wb") as fp:
            pickle.dump(z_TRANE, fp)
        with open("parameters", "wb") as fp:
            pickle.dump(parameters, fp)
    else:
        print("  Loading from file...")
        with open(os.path.join(path_trane_results_to_load, "z_TRANE"), "rb") as fp:
            z_TRANE = pickle.load(fp)
        with open(os.path.join(path_trane_results_to_load, "parameters"), "rb") as fp:
            parameters = pickle.load(fp)

    dx = parameters[0]
    dy = parameters[1]

    save_facies_grids_as_png(z_TRANE, parameters, 'TRANE')
    calculate_and_save_facies_prob_maps(z_TRANE, parameters, 'TRANE')
    save_threshold_grids_as_png(parameters)

    count_connected_TRANE = count_connected_grid_nodes(z_TRANE, parameters, 3000.0, 2000.0, [3500, 2000])
    sum_connected_TRANE = [dx * dy * n for n in count_connected_TRANE]
    count_connected_filtered_TRANE = [c for c in count_connected_TRANE if c != -1]
    v_TRANE = calculate_volume_fractions(z_TRANE)

    if verbose:
        print()
        print("=" * 50)
        print("Results")
        print("=" * 50)
        print("Sum connected area:")
        for i in range(0, len(sum_connected_TRANE), 5):
            row = ", ".join(f"{v:12.2f}" for v in sum_connected_TRANE[i:i+5])
            print(f"  {row}")
        # print(f"Volume fractions:          {v_TRANE}")
        print(f"Connected (filtered):      {len(count_connected_filtered_TRANE)} / {len(count_connected_TRANE)}")
        print(f"Mean connected nodes:      {statistics.mean(count_connected_filtered_TRANE):.2f}")
        print(f"Stdev connected nodes:     {statistics.stdev(count_connected_filtered_TRANE):.2f}")
        print(f"Max connected nodes:       {max(count_connected_filtered_TRANE)}")

# ============================================================
# APS
# ============================================================
if RUN_APS:
    print(f"\n\n{RED}╔{'═' * 48}╗")
    print(f"║{'APS simulations':^48}║")
    print(f"╚{'═' * 48}╝{RESET}")
    if not RUN_TRANE:
        # Load parameters from file if TRANE was not run
        with open(os.path.join(path_trane_results_to_load, "parameters"), "rb") as fp:
            parameters = pickle.load(fp)
        with open(os.path.join(path_trane_results_to_load, "z_TRANE"), "rb") as fp:
            z_TRANE = pickle.load(fp)
        dx = parameters[0]
        dy = parameters[1]

    nx = z_TRANE[0].shape[0]
    ny = z_TRANE[0].shape[1]

    if not use_existing_results:
        z_APS = run_APS_simulations(n_sim, nx, ny, dx, dy, MODEL, True)
        with open("z_APS", "wb") as fp:
            pickle.dump(z_APS, fp)
    else:
        with open(os.path.join(path_trane_results_to_load, "z_APS"), "rb") as fp:
            z_APS = pickle.load(fp)

    save_facies_grids_as_png(z_APS, parameters, 'APS', [0, 1, 2])
    calculate_and_save_facies_prob_maps(z_APS, parameters, 'APS')

    count_connected_APS = count_connected_grid_nodes(z_APS, parameters, 3000.0, 2000.0, [3500, 2000])
    sum_connected_APS = [dx * dy * n for n in count_connected_APS]
    count_connected_filtered_APS = [c for c in count_connected_APS if c != -1]
    v_APS = calculate_volume_fractions(z_APS)

    if verbose:
        print()
        print("=" * 50)
        print("Results")
        print("=" * 50)
        print("Sum connected area:")
        for i in range(0, len(sum_connected_APS), 5):
            row = ", ".join(f"{v:12.2f}" for v in sum_connected_APS[i:i+5])
            print(f"  {row}")
        # print(f"Volume fractions:          {v_APS}")
        print(f"Connected (filtered):      {len(count_connected_filtered_APS)} / {len(count_connected_APS)}")
        print(f"Mean connected nodes:      {statistics.mean(count_connected_filtered_APS):.2f}")
        print(f"Stdev connected nodes:     {statistics.stdev(count_connected_filtered_APS):.2f}")
        print(f"Max connected nodes:       {max(count_connected_filtered_APS)}")

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













