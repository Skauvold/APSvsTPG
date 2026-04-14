from datetime import datetime
import math
import os
import pickle
import subprocess
import statistics
import sys

import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from scipy.stats import norm

# use: "pip install git+https://code.nr.no/scm/git/variogram"
from variogram.variogram import ExponentialVariogram
from variogram.variogram import GeneralExponentialVariogram
from variogram.variogram import SphericalVariogram
from variogram.simulate import simulate_gaussian_field


MODEL_CONFIGS = {
    # ── Model 0: 2-facies (F1 F2), no trend, varying residual range ─────
    "0A": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 500.0, "subrange": 500.0, "power": 1.5, "azimuth": 0.0},
        ],
        "wells": ["wells/well2.rmswell"],
    },
    "0B": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 50.0, "subrange": 50.0, "power": 1.5, "azimuth": 0.0},
        ],
        "wells": ["wells/well2.rmswell"],
    },
    "0C": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 5000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 0.0},
        ],
        "wells": ["wells/well2.rmswell"],
    },
    "0D": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "-2.5")],  # => Mostly F1
        "residuals": [
            {"id": "1", "type": "genexp", "range": 500.0, "subrange": 500.0, "power": 1.5, "azimuth": 0.0},
        ],
        "wells": ["wells/well1.rmswell"],
    },
    # ── Model 1: 3-facies (F1 F2 F3), varying trends and wells ──────────
    "1A": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "1.282"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "wells": ["wells/well2.rmswell"],
    },
    "1B": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.5"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "wells": ["wells/well2B.rmswell"],
    },
    "1C": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.5"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "wells": ["wells/well2B.rmswell", "wells/well2C.rmswell", "wells/well2D.rmswell", "wells/well2E.rmswell", "wells/well2F.rmswell", "wells/well2G.rmswell", "wells/well2H.rmswell", "wells/well2I.rmswell", "wells/well2J.rmswell", "wells/well2K.rmswell", "wells/well2L.rmswell", "wells/well2M.rmswell", "wells/well2N.rmswell", "wells/well2O.rmswell", "wells/well2P.rmswell", "wells/well2Q.rmswell"],
    },
    "1D": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "1.0"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "wells": ["wells/well2B.rmswell"],
    },
    "1E": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "2.0"), ("2", "-2.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "wells": ["wells/well2B.rmswell"],
    },
    # ── Model 2: 3-facies hierarchical, 1 well ───────────────────────────
    "2A": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F3 F1F2", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2", "names": "F1 F2", "residual_ids": "2", "trend_ids": "2"},
        ],
        "trends": [("1", "0.0"), ("2", "0.8416")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 400.0, "subrange": 400.0, "power": 1.8, "azimuth": 0.0},
        ],
        "wells": ["wells/well2.rmswell"],
    },
    # ── Model 3: same as 2A but with 2 wells ─────────────────────────────
    "3A": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F3 F1F2", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2", "names": "F1 F2", "residual_ids": "2", "trend_ids": "2"},
        ],
        "trends": [("1", "0.0"), ("2", "0.8416")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 400.0, "subrange": 400.0, "power": 1.8, "azimuth": 0.0},
        ],
        "wells": ["wells/well2.rmswell", "wells/well3.rmswell"],
    },
    # ── Model 4: same as 3A but larger variogram ranges ──────────────────
    "4A": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F3 F1F2", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2", "names": "F1 F2", "residual_ids": "2", "trend_ids": "2"},
        ],
        "trends": [("1", "0.0"), ("2", "0.8416")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 1600.0, "subrange": 1000.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 800.0, "subrange": 800.0, "power": 1.8, "azimuth": 0.0},
        ],
        "wells": ["wells/well2.rmswell", "wells/well3.rmswell"],
    },
}

GRID_NX = 151
GRID_NY = 101
GRID_NZ = 2
X_LENGTH = 6000.0
Y_LENGTH = 4000.0
Z_LENGTH = 20.0

WELL_DATA = {
    "wells/well1.rmswell": {"name": "well1", "x": 3000.0, "y": 2000.0, "facies": 2},
    "wells/well2.rmswell": {"name": "well2", "x": 3000.0, "y": 2000.0, "facies": 1},
    "wells/well2B.rmswell": {"name": "well2B", "x": 3000.0, "y": 2000.0, "facies": 2},
    "wells/well2C.rmswell": {"name": "well2C", "x": 3040.0, "y": 2000.0, "facies": 2},
    "wells/well2D.rmswell": {"name": "well2D", "x": 3080.0, "y": 2000.0, "facies": 2},
    "wells/well2E.rmswell": {"name": "well2E", "x": 3120.0, "y": 2000.0, "facies": 2},
    "wells/well2F.rmswell": {"name": "well2F", "x": 3160.0, "y": 2000.0, "facies": 2},
    "wells/well2G.rmswell": {"name": "well2G", "x": 3200.0, "y": 2000.0, "facies": 2},
    "wells/well2H.rmswell": {"name": "well2H", "x": 3240.0, "y": 2000.0, "facies": 2},
    "wells/well2I.rmswell": {"name": "well2I", "x": 3280.0, "y": 2000.0, "facies": 3},
    "wells/well2J.rmswell": {"name": "well2J", "x": 3320.0, "y": 2000.0, "facies": 3},
    "wells/well2K.rmswell": {"name": "well2K", "x": 3360.0, "y": 2000.0, "facies": 3},
    "wells/well2L.rmswell": {"name": "well2L", "x": 3400.0, "y": 2000.0, "facies": 3},
    "wells/well2M.rmswell": {"name": "well2M", "x": 3440.0, "y": 2000.0, "facies": 3},
    "wells/well2N.rmswell": {"name": "well2N", "x": 3480.0, "y": 2000.0, "facies": 3},
    "wells/well2O.rmswell": {"name": "well2O", "x": 3520.0, "y": 2000.0, "facies": 3},
    "wells/well2P.rmswell": {"name": "well2P", "x": 3560.0, "y": 2000.0, "facies": 3},
    "wells/well2Q.rmswell": {"name": "well2Q", "x": 3600.0, "y": 2000.0, "facies": 3},
    "wells/well3.rmswell": {"name": "well3", "x": 3500.0, "y": 2000.0, "facies": 1},
}


def _build_well_file(well_name, x, y, facies_code):
    z_values = [i * 2.0 for i in range(int(Z_LENGTH / 2))]
    data_lines = [f"{x}   {y}   {z:6.1f}    {facies_code}" for z in z_values]
    return "\n".join([
        "1.0",
        "UNDEFINED",
        f"{well_name} {x} {y} 0.0",
        "1",
        "FACIES DISC 1 F1 2 F2 3 F3",
    ] + data_lines) + "\n"


def _xml_tag(indent, tag, value):
    prefix = f"{indent}<{tag}>"
    width = max(35, len(prefix) + 1)
    return f"{prefix:<{width}}{str(value):<33}</{tag}>"


def _build_model_xml(model_number, seed, output_dir):
    cfg = MODEL_CONFIGS[model_number]

    fm_parts = []
    for fm in cfg["facies_models"]:
        fm_parts += [
            "    <facies-model>",
            _xml_tag("      ", "parent-facies-name", fm["parent"]),
            _xml_tag("      ", "facies-names",       fm["names"]),
            _xml_tag("      ", "residual-ids",       fm["residual_ids"]),
            _xml_tag("      ", "trend-ids",          fm["trend_ids"]),
            "    </facies-model>",
        ]

    trend_parts = []
    for tid, value in cfg["trends"]:
        trend_parts += [
            "    <trend>",
            _xml_tag("      ", "trend-id", tid),
            _xml_tag("      ", "value",    value),
            "    </trend>",
        ]

    res_parts = []
    for r in cfg["residuals"]:
        res_parts += [
            "    <residual-field>",
            _xml_tag("      ", "residual-id", r["id"]),
            "      <variogram>",
            _xml_tag("        ", "standard-deviation", 1),
            _xml_tag("        ", "variogram-type",     r["type"]),
            _xml_tag("        ", "range",              r["range"]),
            _xml_tag("        ", "subrange",           r["subrange"]),
            _xml_tag("        ", "power",              r["power"]),
            _xml_tag("        ", "z-range",            Z_LENGTH),
            _xml_tag("        ", "azimuth",            r["azimuth"]),
            _xml_tag("        ", "dip",                "0.0"),
            "      </variogram>",
            "    </residual-field>",
        ]

    well_parts = []
    for well in cfg["wells"]:
        well_parts += [
            "    <well>",
            _xml_tag("      ", "file-name", well),
            "    </well>",
        ]

    hbelts = "\n".join(fm_parts + trend_parts + res_parts)
    wells  = "\n".join(well_parts)

    lines = [
        '<?xml version="1.0"?>',
        '<trane>',
        '  <project-settings>',
        _xml_tag("    ", "method",               "simulation"),
        _xml_tag("    ", "seed",                 seed),
        _xml_tag("    ", "n-threads",            4),
        _xml_tag("    ", "logging-level-screen", 4),
        '  </project-settings>',
        '  <grid-description>',
        '    <grid-resolution>',
        _xml_tag("      ", "nx", GRID_NX),
        _xml_tag("      ", "ny", GRID_NY),
        _xml_tag("      ", "nz", GRID_NZ),
        '    </grid-resolution>',
        '    <volume>',
        _xml_tag("      ", "x-length", X_LENGTH),
        _xml_tag("      ", "y-length", Y_LENGTH),
        _xml_tag("      ", "z-length", Z_LENGTH),
        _xml_tag("      ", "z-start",  "0.0"),
        '    </volume>',
        '  </grid-description>',
        '  <h-belts>',
        hbelts,
        '  </h-belts>',
        '  <well-data>',
        wells,
        '  </well-data>',
        '  <io-settings>',
        _xml_tag("    ", "input-directory",                      "input"),
        _xml_tag("    ", "output-directory",                     output_dir),
        _xml_tag("    ", "result-file-roff",                     "result.roff"),
        _xml_tag("    ", "facies-probability-file-prefix-roff",  "probabilities.roff"),
        _xml_tag("    ", "facies-probability-file-prefix-storm", "probabilities.storm"),
        _xml_tag("    ", "trend-file-prefix-storm",              "trend.storm"),
        _xml_tag("    ", "residual-field-file-prefix-storm",     "residual_field.storm"),
        _xml_tag("    ", "log-file",                             "trane.log"),
        '  </io-settings>',
        '</trane>',
    ]
    return "\n".join(lines) + "\n"


def run_TRANE_simulations(n_simulations, model_number, path_trane_models, path_trane_exe, print_info=False, verbose_trane=False):
    os.chdir(path_trane_models)
    dx = X_LENGTH / GRID_NX
    dy = Y_LENGTH / GRID_NY

    cfg = MODEL_CONFIGS[model_number]
    wells_dir = os.path.join(path_trane_models, "input", "wells")
    os.makedirs(wells_dir, exist_ok=True)
    for well_path in cfg["wells"]:
        wd = WELL_DATA[well_path]
        well_file_path = os.path.join(path_trane_models, "input", well_path)
        with open(well_file_path, 'w') as f:
            f.write(_build_well_file(wd["name"], wd["x"], wd["y"], wd["facies"]))

    out_z = []
    for iteration in range(0, n_simulations):
        if print_info:
            _print_progress_bar(iteration, n_simulations, prefix="Progress")
        output_dir            = "output" + model_number + "_edited"
        modelfile_edited_path = os.path.join(path_trane_models, "model" + model_number + "_edited.xml")
        output_path           = os.path.join(path_trane_models, output_dir)
        results_path          = os.path.join(output_path, "result.roff")
        with open(modelfile_edited_path, 'w') as f:
            f.write(_build_model_xml(model_number, seed=iteration, output_dir=output_dir))

        trane_output = None if verbose_trane else subprocess.DEVNULL
        result = subprocess.run([path_trane_exe, modelfile_edited_path], shell=True, stdout=trane_output, stderr=trane_output)
        if result.returncode != 0:
            print(f"\033[31m\nERROR: TRANE failed on iteration {iteration} with return code {result.returncode}\033[0m")
            sys.exit(1)

        lines = []
        with open(results_path) as f:
            lines = f.readlines()

        nx   = int(lines[13].split()[2])
        ny   = int(lines[14].split()[2])
        nz   = int(lines[15].split()[2])
        data = lines[20].split()

        x_lin = np.linspace(0.0, X_LENGTH, num=nx)
        y_lin = np.linspace(0.0, Y_LENGTH, num=ny)
        X2, Y2 = np.meshgrid(y_lin, x_lin)
        z = X2 ** 2 - Y2 ** 2

        temp = np.zeros((nx, ny, nz))
        counter = 0
        for i in range(0, nx):
            for j in range(0, ny):
                for k in range(0, nz):
                    temp[i][j][k] = data[counter]
                    counter += 1
        for i in range(0, nx):
            for j in range(0, ny):
                z[i][j] = temp[i][j][nz - 1]
                # Backup:
                # for k in range(0, nz):
                #     if k == 0:
                #         z[i][j] = temp[i][j][nz-k-1]

        out_z.append(z)

    parameters = [dx, dy, X_LENGTH, Y_LENGTH]
    if print_info:
        _print_progress_bar(n_simulations, n_simulations, prefix="Progress")
        print()
    return out_z, parameters

def run_APS_simulations(n_simulations, nx, ny, dx, dy, model_number, print_info=False, data_dir="."):
    cfg = MODEL_CONFIGS[model_number]
    r1 = cfg["residuals"][0]
    v1_range_x      = r1["range"]
    v1_range_y      = r1["subrange"]
    v1_range_z      = Z_LENGTH
    v1_azimuth      = r1["azimuth"] * 3.141592 / 180.0  # In radians, not degrees
    v1_genexp_power = r1["power"]

    if model_number[0] in ("2", "3", "4"):
        r2 = cfg["residuals"][1]
        v2_range_x      = r2["range"]
        v2_range_y      = r2["subrange"]
        v2_range_z      = Z_LENGTH
        v2_azimuth      = r2["azimuth"]  # Already 0.0, no conversion needed
        v2_genexp_power = r2["power"]
 
    n_facies = MODEL_CONFIGS[model_number]["n_facies"]
    p_F1 = np.load(os.path.join(data_dir, "p1_from_TRANE.npy"))
    if n_facies >= 3:
        p_F2 = np.load(os.path.join(data_dir, "p2_from_TRANE.npy"))
        p_F3 = np.load(os.path.join(data_dir, "p3_from_TRANE.npy"))
    
    # Calculate thresholds
    t1 = np.zeros((nx, ny))
    t2 = np.zeros((nx, ny))
    for i in range(0, nx):
        for j in range(0, ny):
            if model_number[0] == "0":
                t1[i][j] = norm.ppf(p_F1[i][j])
            elif model_number[0] == "1":
                t1[i][j] = norm.ppf(p_F1[i][j])
                p1_p2 = min(1.0, p_F1[i][j] + p_F2[i][j])
                t2[i][j] = norm.ppf(p1_p2)
            elif model_number[0] in ("2", "3", "4"):
                t1[i][j] = norm.ppf(p_F3[i][j])
                t2[i][j] = norm.ppf(min(1.0, p_F1[i][j] / (1.0 - p_F3[i][j])))

    v1 = GeneralExponentialVariogram(v1_range_x, v1_range_y, v1_range_z, azi=v1_azimuth, power=v1_genexp_power)
    if model_number[0] in ("2", "3", "4"):
        v2 = GeneralExponentialVariogram(v2_range_x, v2_range_y, v2_range_z, azi=v2_azimuth, power=v2_genexp_power)

    out_z = []
    for iteration in range(0, n_simulations):
        if print_info:
            _print_progress_bar(iteration, n_simulations, prefix="Progress")
        s1 = simulate_gaussian_field(v1, nx, dx, ny, dy, seed = iteration)
        if model_number[0] in ("2", "3", "4"):
            s2 = simulate_gaussian_field(v2, nx, dx, ny, dy, seed = iteration)
        z = np.ndarray(s1.shape)
        for i in range(0, nx):
            for j in range(0, ny):
                if model_number[0] == "0":
                    if s1[i][j] < t1[i][j]:
                        z[i][j] = 1
                    else:
                        z[i][j] = 2
                elif model_number[0] == "1":
                    if s1[i][j] < t1[i][j]:
                        z[i][j] = 1
                    elif s1[i][j] < t2[i][j]:
                        z[i][j] = 2
                    else:
                        z[i][j] = 3
                elif model_number[0] in ("2", "3", "4"):
                    if s1[i][j] < t1[i][j]:
                        z[i][j] = 3
                    elif s2[i][j] < t2[i][j]:
                        z[i][j] = 1
                    else:
                        z[i][j] = 2
        out_z.append(z)
    if print_info:
        _print_progress_bar(n_simulations, n_simulations, prefix="Progress")
        print()
    return out_z


def _print_progress_bar(current, total, prefix="", bar_length=80):
    BLUE = "\033[34m"
    GREY = "\033[90m"
    RESET = "\033[0m"
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = f"{BLUE}{'█' * filled}{GREY}{'░' * (bar_length - filled)}{RESET}"
    print(f"\r{prefix}: {BLUE}|{RESET}{bar}{BLUE}|{RESET} {current}/{total} ({fraction:.0%})", end="", flush=True)


def _print_results(sum_connected, count_connected, count_connected_filtered, dx, dy):
    RED   = "\033[31m"
    RESET = "\033[0m"
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print("Sum connected area:")
    for i in range(0, len(sum_connected), 5):
        row = ", ".join(f"{v:12.2f}" if v != -1 else f"{'—':>12}" for v in sum_connected[i:i+5])
        print(f"  {row}")
    print(f"#realizations with connection: {len(count_connected_filtered)} / {len(count_connected)}")
    if count_connected_filtered:
        print(f"Mean connected nodes:      {statistics.mean(count_connected_filtered):.2f}")
        print(f"Stdev connected nodes:     {statistics.stdev(count_connected_filtered):.2f}")
        print(f"Max connected nodes:       {max(count_connected_filtered)}")
    else:
        print(f"{RED}No connected realizations — both observation points never in same facies body{RESET}")


def _print_header(title):
    BRIGHT_RED = "\033[91m"
    RESET      = "\033[0m"
    print(f"\n{BRIGHT_RED}╔{'═' * 48}╗")
    print(f"║{title:^48}║")
    print(f"╚{'═' * 48}╝{RESET}")


def _cluster_size_stats_lines(counts, well_name, prefix):
    n = len(counts)
    lines = [f"Cluster size stats for {prefix}_{well_name} (n={n}):"]
    for size in range(1, 11):
        c = sum(1 for v in counts if v == size)
        pct = 100.0 * c / n if n > 0 else 0.0
        lines.append(f"  Size {size:2d}: {c:5d} / {n}  ({pct:5.1f}%)")
    return lines


def _save(name, obj):
    with open(name, "wb") as fp:
        pickle.dump(obj, fp)


def _load(path, name):
    with open(os.path.join(path, name), "rb") as fp:
        return pickle.load(fp)


def _analyse(z, parameters, prefix, dx, dy, verbose, model_number, save_indices="all", save_thresholds=False, output_dir=".", data_dir=".", log_file=None):
    save_facies_grids_as_png(z, parameters, prefix, model_number, save_indices, output_dir=output_dir)
    calculate_and_save_facies_prob_maps(z, parameters, prefix, model_number, output_dir=output_dir, data_dir=data_dir)
    if save_thresholds:
        save_threshold_grids_as_png(parameters, model_number, output_dir=output_dir, data_dir=data_dir, prefix=prefix)
    count_connected = count_connected_grid_nodes(z, parameters, 3000.0, 2000.0, [3500, 2000])
    sum_connected = [dx * dy * n if n != -1 else -1 for n in count_connected]
    count_connected_filtered = [c for c in count_connected if c != -1]
    if verbose:
        _print_results(sum_connected, count_connected, count_connected_filtered, dx, dy)
    for wp in MODEL_CONFIGS[model_number]["wells"]:
        wd = WELL_DATA[wp]
        per_well_counts = count_connected_nodes_from_point(z, parameters, wd["x"], wd["y"])
        xmax = max(per_well_counts) * 1.1
        plot_histogram_of_connected_cells(
            per_well_counts, prefix + "_" + wd["name"],
            0.0, xmax, 0.0, len(z), 50, output_dir=output_dir
        )
        stat_lines = _cluster_size_stats_lines(per_well_counts, wd["name"], prefix)
        if verbose:
            print()
            for line in stat_lines:
                print(line)
        if log_file:
            with open(log_file, 'a') as _lf:
                _lf.write("\n")
                _lf.write("\n".join(stat_lines) + "\n")
    return count_connected_filtered


def save_facies_grids_as_png(facies_grids, parameters, prefix, model_number, indices_to_save="all", output_dir="."):
    F1 = (255/255,  69/255,   0/255)  # Orange-Red
    F2 = ( 75/255,   0/255, 130/255)  # Indigo
    F3 = (  0/255, 206/255, 209/255)  # Dark Turquoise
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
    well_x = [WELL_DATA[wp]["x"] for wp in MODEL_CONFIGS[model_number]["wells"]]
    well_y = [WELL_DATA[wp]["y"] for wp in MODEL_CONFIGS[model_number]["wells"]]
    extent = x_min, x_max, y_min, y_max

    folder = os.path.join(output_dir, "facies_grids_" + prefix)
    if not os.path.exists(folder):
        os.mkdir(folder)

    for iteration, z in enumerate(facies_grids):
        if indices_to_save == "all" or iteration in indices_to_save:
            x_lin = np.linspace(0.0, x_length, num=nx)
            y_lin = np.linspace(0.0, y_length, num=ny)
            Y, X  = np.meshgrid(x_lin, y_lin)
            z_for_plotting  = (X ** 2 - Y ** 2) * 0.0
            for i in range(0, nx):
                for j in range(0, ny):
                    z_for_plotting[j][i] = z[i][ny - 1 - j]

            cmap = colors.ListedColormap([F1, F2, F3])
            fig = plt.figure(frameon=False)
            fig.set_size_inches(9,6)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # img = plt.imshow(z_simbox, cmap = cmap, alpha = 1.0, interpolation='none', extent = extent) # interpolation ='bilinear'
            img = plt.imshow(z_for_plotting, cmap = cmap, alpha = 1.0, interpolation='none', extent = extent) # interpolation ='bilinear'
            for wx, wy in zip(well_x, well_y):
                ax.add_patch(patches.Rectangle(
                    (wx - dx/2, wy - dy/2), dx, dy,
                    linewidth=0.5, edgecolor='black', facecolor='none'
                ))
            plt.savefig(os.path.join(folder, prefix + '_it' + str(iteration) + '.png'), dpi=100)
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


def count_connected_nodes_from_point(facies_grids, parameters, x_obs, y_obs):
    nx = facies_grids[0].shape[0]
    ny = facies_grids[0].shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_ind = math.floor(x_obs / dx)
    y_ind = math.floor(y_obs / dy)
    counts = []
    for z in facies_grids:
        connected = np.zeros((nx, ny), dtype=bool)
        connected[x_ind][y_ind] = True
        need_to_check = [[x_ind, y_ind]]
        facies = z[x_ind][y_ind]
        while need_to_check:
            xi, yi = need_to_check.pop(0)
            for xn, yn in [(xi+1,yi),(xi-1,yi),(xi,yi+1),(xi,yi-1)]:
                if 0 <= xn < nx and 0 <= yn < ny and z[xn][yn] == facies and not connected[xn][yn]:
                    connected[xn][yn] = True
                    need_to_check.append([xn, yn])
        counts.append(int(np.count_nonzero(connected)))
    return counts


def calculate_and_save_facies_prob_maps(facies_grids, parameters, prefix, model_number, output_dir=".", data_dir="."):
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
    n_facies = MODEL_CONFIGS[model_number]["n_facies"]
    p_F1 = (X ** 2 - Y ** 2) * 0.0
    p_F2 = (X ** 2 - Y ** 2) * 0.0
    p_F3 = (X ** 2 - Y ** 2) * 0.0 if n_facies >= 3 else None
    for iteration, z in enumerate(facies_grids):
        for i in range(0, nx):
            for j in range(0, ny):
                a = 1 if z[i][j] == 1 else 0
                b = 1 if z[i][j] == 2 else 0
                p_F1[i][j]  = (p_F1[i][j]  * iteration + a) / (iteration + 1)
                p_F2[i][j]  = (p_F2[i][j]  * iteration + b) / (iteration + 1)
                if n_facies >= 3:
                    c = 1 if z[i][j] == 3 else 0
                    p_F3[i][j]  = (p_F3[i][j]  * iteration + c) / (iteration + 1)
    np.save(os.path.join(data_dir, "p1_from_" + prefix), p_F1)
    np.save(os.path.join(data_dir, "p2_from_" + prefix), p_F2)
    if n_facies >= 3:
        np.save(os.path.join(data_dir, "p3_from_" + prefix), p_F3)
    facies_probs = [p_F1, p_F2] if n_facies == 2 else [p_F1, p_F2, p_F3]
    for i, p in enumerate(facies_probs):
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
        fig.colorbar(img, ax=ax, shrink=0.5)
        plt.savefig(os.path.join(output_dir, prefix + '_p' + str(i+1) + '_n' + str(len(facies_grids)) + '.png'), dpi=100)
        plt.close()

def plot_histogram_of_connected_cells(sum_connected, prefix, xmin, xmax, ymin, ymax, n_bins, output_dir="."):
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
    plt.savefig(os.path.join(output_dir, prefix + '_connectedvolume' + '_n' + str(len(sum_connected)) + '.png'), dpi=100)
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

def save_threshold_grids_as_png(parameters, model_number, output_dir=".", data_dir=".", prefix="TRANE"):
    n_facies = MODEL_CONFIGS[model_number]["n_facies"]
    p_F1 = np.load(os.path.join(data_dir, "p1_from_" + prefix + ".npy"))
    p_F2 = np.load(os.path.join(data_dir, "p2_from_" + prefix + ".npy")) if n_facies >= 3 else None

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
    t2 = np.zeros((nx, ny)) if n_facies >= 3 else None
    for i in range(0, nx):
        for j in range(0, ny):
            t1[i][j] = norm.ppf(p_F1[i][j])
            if n_facies >= 3:
                p1_p2 = min(1.0, p_F1[i][j] + p_F2[i][j])
                t2[i][j] = norm.ppf(p1_p2)
    
    thresholds = [t1] if n_facies == 2 else [t1, t2]
    for i, t in enumerate(thresholds):
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
        plt.savefig(os.path.join(output_dir, prefix + '_t' + str(i+1) + '.png'), dpi=100)
        plt.close()
