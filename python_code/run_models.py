#!/usr/bin/env python

import os
import shutil
import sys
import time
from datetime import datetime

from methods import (run_TRANE_simulations, run_APS_simulations,
                     plot_histogram_of_connected_cells,
                     _analyse, _print_header, _save, _load,
                     MODEL_CONFIGS, WELL_DATA)

# ============================================================
# Options
# ============================================================
MODELS = ["1Ib"]  # list of models to run sequentially; each gets its own results folder
n_sim = 200
use_existing_results = False

RUN_TRANE = True
RUN_APS = True
verbose = True
verbose_trane = False
plot_histograms = True
n_workers = 14  # parallel workers for TRANE and APS simulations
max_facies_grid_exports = 200  # max facies grid images saved per method (None = all)
save_pickles = False  # save z_TRANE / z_APS / parameters as pickle files

# path_trane_models = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models"
temp_project_dir = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\TRANE_models_autocreated"
_path_results_base = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results"
path_trane_results_to_load = "C:\\Projects\\trane\\trane_work\\2022\\2022_09_12_compare_pgs_blitzkriging\\APSvsTPG\\python_code\\results_old"
path_trane_exe = "%tra%"

RED = "\033[31m"
BRIGHT_RED = "\033[91m"
RESET = "\033[0m"


for MODEL in MODELS:
    # Make results folder with timestamp-based name
    _timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    path_trane_results_to_save = _path_results_base + "_" + _timestamp + "_M" + MODEL
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
        _f.write(f"temp_project_dir:     {temp_project_dir}\n")
        _f.write(f"path_trane_exe:       {path_trane_exe}\n")
        _f.write(f"Results saved to:     {path_trane_results_to_save}\n")
        if use_existing_results:
            _f.write(f"Loaded results from:  {path_trane_results_to_load}\n")
        _f.write("\n")
        _cfg = MODEL_CONFIGS[MODEL]
        _f.write(f"Model config ({MODEL}):\n")
        _f.write(f"    n_facies: {_cfg['n_facies']}\n")
        _f.write(f"    facies_models:\n")
        for _fm in _cfg["facies_models"]:
            _f.write(f"        parent={_fm['parent']}  names=\"{_fm['names']}\"  residual_ids={_fm['residual_ids'].strip()}  trend_ids={_fm['trend_ids'].strip()}\n")
        _f.write(f"    trends:\n")
        for _tid, _tval in _cfg["trends"]:
            _f.write(f"        id={_tid}  value={_tval}\n")
        _f.write(f"    residuals:\n")
        for _r in _cfg["residuals"]:
            _f.write(f"        id={_r['id']}  type={_r['type']}  range={_r['range']}  subrange={_r['subrange']}  power={_r['power']}  azimuth={_r['azimuth']}\n")
        _f.write(f"    wells:\n")
        for _wp in _cfg["wells"]:
            if _wp in WELL_DATA:
                _wd = WELL_DATA[_wp]
                _f.write(f"        {_wd['name']}  x={_wd['x']}  y={_wd['y']}  facies={_wd['facies']}\n")
            else:
                _f.write(f"        {_wp}\n")

    trane_well_data = []
    aps_well_data = []

    # Clear TRANE_models_autocreated before running
    if False and not use_existing_results and os.path.exists(temp_project_dir):
        shutil.rmtree(temp_project_dir)
    os.makedirs(temp_project_dir, exist_ok=True)

    if RUN_TRANE:
        resolved_trane_exe = os.path.expandvars(path_trane_exe)
        if not os.path.isfile(resolved_trane_exe):
            print(f"{RED}ERROR: TRANE executable not found: '{resolved_trane_exe}' (from '{path_trane_exe}'){RESET}")
            sys.exit(1)

        _print_header('TRANE simulations')
        if not use_existing_results:
            _t0 = time.time()
            z_TRANE, parameters = run_TRANE_simulations(n_sim, MODEL, temp_project_dir, path_trane_exe, True, verbose_trane, n_workers)
            print(f"\033[36m  [timing] {'run_TRANE_simulations:':<42} {time.time()-_t0:6.2f}s\033[0m")
            if save_pickles:
                _t0 = time.time()
                _save(os.path.join(path_pickle_trane, "z_TRANE"), z_TRANE)
                _save(os.path.join(path_pickle_trane, "parameters"), parameters)
                print(f"\033[36m  [timing] {'pickle save TRANE:':<42} {time.time()-_t0:6.2f}s\033[0m")
        else:
            print("  Loading from file...")
            _load_pickle_trane = os.path.join(path_trane_results_to_load, "output_TRANE", "pickle_backup")
            z_TRANE    = _load(_load_pickle_trane, "z_TRANE")
            parameters = _load(_load_pickle_trane, "parameters")

        dx = parameters[0]
        dy = parameters[1]

        _t0 = time.time()
        count_connected_filtered_TRANE, trane_well_data = _analyse(z_TRANE, parameters, 'TRANE', dx, dy, verbose, MODEL,
            max_facies_grid_exports=max_facies_grid_exports, save_thresholds=True, output_dir=path_output_trane, data_dir=path_pickle_trane, log_file=_log_file)

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
            _t0 = time.time()
            z_APS = run_APS_simulations(n_sim, nx, ny, dx, dy, MODEL, True, data_dir=_aps_data_dir, n_workers=n_workers)
            print(f"\033[36m  [timing] {'run_APS_simulations:':<42} {time.time()-_t0:6.2f}s\033[0m")
            if save_pickles:
                _t0 = time.time()
                _save(os.path.join(path_pickle_aps, "z_APS"), z_APS)
                print(f"\033[36m  [timing] {'pickle save APS:':<42} {time.time()-_t0:6.2f}s\033[0m")
        else:
            _load_pickle_aps = os.path.join(path_trane_results_to_load, "output_APS", "pickle_backup")
            z_APS = _load(_load_pickle_aps, "z_APS")

        _t0 = time.time()
        count_connected_filtered_APS, aps_well_data = _analyse(z_APS, parameters, 'APS', dx, dy, verbose, MODEL,
            max_facies_grid_exports=max_facies_grid_exports, output_dir=path_output_aps, data_dir=path_pickle_aps, log_file=_log_file)

    # ============================================================
    # Histograms
    # ============================================================
    # Per-well cluster size histograms — shared x/y axis across TRANE and APS
    _t0 = time.time()
    _all_well_counts = [c for _, counts in trane_well_data + aps_well_data for c in counts]
    _n_bins = max(10, n_sim // 25)
    if _all_well_counts:
        _xmax_well = max(_all_well_counts) * 1.1
        _n_bins = min(_n_bins, int(max(_all_well_counts)) + 1)  # bin width must be >= 1 (counts are integers)
        _binwidth = _xmax_well / _n_bins
        _bin_edges = [i * _binwidth for i in range(_n_bins + 1)]
        _ymax_well = 0
        for _, _counts in trane_well_data + aps_well_data:
            for _b in range(_n_bins):
                _bc = sum(1 for v in _counts if _bin_edges[_b] <= v < _bin_edges[_b + 1])
                if _bc > _ymax_well:
                    _ymax_well = _bc
        _ymax_well = _ymax_well * 1.1
        for _name, _counts in trane_well_data:
            plot_histogram_of_connected_cells(_counts, _name, 0.0, _xmax_well, 0.0, _ymax_well, _n_bins,
                output_dir=path_output_trane)
        for _name, _counts in aps_well_data:
            plot_histogram_of_connected_cells(_counts, _name, 0.0, _xmax_well, 0.0, _ymax_well, _n_bins,
                output_dir=path_output_aps)
    print(f"\033[36m  [timing] {'histogram plotting:':<42} {time.time()-_t0:6.2f}s\033[0m")

    # Two-point connection histograms (not applicable for model 0)
    if plot_histograms and RUN_TRANE and RUN_APS:
        all_connected = count_connected_filtered_TRANE + count_connected_filtered_APS
        if all_connected:
            max3 = max(all_connected) * 1.02
            _n_bins_tp = min(_n_bins, int(max(all_connected)) + 1)
            plot_histogram_of_connected_cells(count_connected_filtered_TRANE, 'TRANE', 0.0, max3, 0.0, n_sim, _n_bins_tp,
                output_dir=path_output_trane)
            plot_histogram_of_connected_cells(count_connected_filtered_APS, 'APS', 0.0, max3, 0.0, n_sim, _n_bins_tp,
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













