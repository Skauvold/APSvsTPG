from datetime import datetime
import math
import os
import pickle
import shutil
import subprocess
import statistics
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import colors, patches
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 1}],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 1}],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 1}],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "0E": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 500.0, "subrange": 500.0, "power": 1.5, "azimuth": 0.0},
        ],
        "observations": [
            {"x":   300.0, "y":  500.0, "facies": 1},
            {"x":   900.0, "y":  500.0, "facies": 1},
            {"x":  1500.0, "y":  500.0, "facies": 2},
            {"x":  2100.0, "y":  500.0, "facies": 1},
            {"x":  2700.0, "y":  500.0, "facies": 2},
            {"x":  3300.0, "y":  500.0, "facies": 2},
            {"x":  3900.0, "y":  500.0, "facies": 1},
            {"x":  4500.0, "y":  500.0, "facies": 2},
            {"x":  5100.0, "y":  500.0, "facies": 2},
            {"x":  5700.0, "y":  500.0, "facies": 1},
            {"x":   300.0, "y": 1300.0, "facies": 2},
            {"x":   900.0, "y": 1300.0, "facies": 1},
            {"x":  1500.0, "y": 1300.0, "facies": 1},
            {"x":  2100.0, "y": 1300.0, "facies": 2},
            {"x":  2700.0, "y": 1300.0, "facies": 1},
            {"x":  3300.0, "y": 1300.0, "facies": 1},
            {"x":  3900.0, "y": 1300.0, "facies": 2},
            {"x":  4500.0, "y": 1300.0, "facies": 1},
            {"x":  5100.0, "y": 1300.0, "facies": 2},
            {"x":  5700.0, "y": 1300.0, "facies": 2},
            {"x":   300.0, "y": 2000.0, "facies": 1},
            {"x":   900.0, "y": 2000.0, "facies": 2},
            {"x":  1500.0, "y": 2000.0, "facies": 2},
            {"x":  2100.0, "y": 2000.0, "facies": 1},
            {"x":  2700.0, "y": 2000.0, "facies": 2},
            {"x":  3300.0, "y": 2000.0, "facies": 1},
            {"x":  3900.0, "y": 2000.0, "facies": 1},
            {"x":  4500.0, "y": 2000.0, "facies": 2},
            {"x":  5100.0, "y": 2000.0, "facies": 1},
            {"x":  5700.0, "y": 2000.0, "facies": 2},
            {"x":   300.0, "y": 2700.0, "facies": 2},
            {"x":   900.0, "y": 2700.0, "facies": 2},
            {"x":  1500.0, "y": 2700.0, "facies": 1},
            {"x":  2100.0, "y": 2700.0, "facies": 2},
            {"x":  2700.0, "y": 2700.0, "facies": 1},
            {"x":  3300.0, "y": 2700.0, "facies": 1},
            {"x":  3900.0, "y": 2700.0, "facies": 2},
            {"x":  4500.0, "y": 2700.0, "facies": 1},
            {"x":  5100.0, "y": 2700.0, "facies": 1},
            {"x":  5700.0, "y": 2700.0, "facies": 2},
            {"x":   300.0, "y": 3500.0, "facies": 1},
            {"x":   900.0, "y": 3500.0, "facies": 2},
            {"x":  1500.0, "y": 3500.0, "facies": 1},
            {"x":  2100.0, "y": 3500.0, "facies": 1},
            {"x":  2700.0, "y": 3500.0, "facies": 2},
            {"x":  3300.0, "y": 3500.0, "facies": 2},
            {"x":  3900.0, "y": 3500.0, "facies": 1},
            {"x":  4500.0, "y": 3500.0, "facies": 2},
            {"x":  5100.0, "y": 3500.0, "facies": 2},
            {"x":  5700.0, "y": 3500.0, "facies": 1},
        ],
    },
    "0F": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "-2.5")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 500.0, "subrange": 500.0, "power": 1.9, "azimuth": 0.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "0G": {
        "n_facies": 2,
        "facies_models": [
            {"parent": "background", "names": "F1 F2", "residual_ids": "1", "trend_ids": "1"},
        ],
        "trends": [("1", "-2.5")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 500.0, "subrange": 500.0, "power": 1.0, "azimuth": 0.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 1}],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
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
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  3040.0, "y": 2000.0, "facies": 2},
            {"x":  3080.0, "y": 2000.0, "facies": 2},
            {"x":  3120.0, "y": 2000.0, "facies": 2},
            {"x":  3160.0, "y": 2000.0, "facies": 2},
            {"x":  3200.0, "y": 2000.0, "facies": 2},
            {"x":  3240.0, "y": 2000.0, "facies": 2},
            {"x":  3280.0, "y": 2000.0, "facies": 3},
            {"x":  3320.0, "y": 2000.0, "facies": 3},
            {"x":  3360.0, "y": 2000.0, "facies": 3},
            {"x":  3400.0, "y": 2000.0, "facies": 3},
            {"x":  3440.0, "y": 2000.0, "facies": 3},
            {"x":  3480.0, "y": 2000.0, "facies": 3},
            {"x":  3520.0, "y": 2000.0, "facies": 3},
            {"x":  3560.0, "y": 2000.0, "facies": 3},
            {"x":  3600.0, "y": 2000.0, "facies": 3},
        ],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1F": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.25"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1G": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.1"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1H": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1I": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 4000.0, "subrange": 2500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1Ib": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 4000.0, "subrange": 2500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1J": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 200.0, "subrange": 125.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1K": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1L": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 16000.0, "subrange": 10000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "1M": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.25"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  1000.0, "y": 1000.0, "facies": 2},
            {"x":  5000.0, "y": 1000.0, "facies": 2},
            {"x":  1000.0, "y": 3000.0, "facies": 2},
            {"x":  5000.0, "y": 3000.0, "facies": 2},
        ],
    },
    "1N": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1O": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.15"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1P": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.5"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1Q": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "2.5"), ("2", "2.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 1},
            {"x":  4000.0, "y": 2000.0, "facies": 1},
        ],
    },
    "1R": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "2.5"), ("2", "2.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 3},
            {"x":  4000.0, "y": 2000.0, "facies": 3},
        ],
    },
    # = 1N, but with higher power in variogram
    "1S": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.7, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1S2": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.2"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.7, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1S3": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.1"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.7, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    # = 1N, but with higher power in variogram
    "1T": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.3"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.9, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1T2": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.2"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.9, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    "1T3": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.1"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.9, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    # = 1N, but with lower power in variogram and higher probability of F2
    "1U": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.7"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.1, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
    },
    # = 1U, but with bigger range
    "1V": {
        "n_facies": 3,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3", "residual_ids": "1  1", "trend_ids": "1  2"},
        ],
        "trends": [("1", "0.7"), ("2", "0.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 12000.0, "subrange": 7500.0, "power": 1.1, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 2},
            {"x":  4000.0, "y": 2000.0, "facies": 2},
        ],
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
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 1}],
    },
    "2B": {
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
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 1},
            {"x":  3500.0, "y": 2000.0, "facies": 1},
        ],
    },
    "2C": {
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
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 1},
            {"x":  3500.0, "y": 2000.0, "facies": 1},
        ],
    },
    # ── Model 5: 8-facies, single Gaussian field, equal prior probability ──
    # Thresholds split standard normal into 8 equal intervals (each p=0.125).
    # norm.ppf(k/8) for k=1..7: -1.1503, -0.6745, -0.3186, 0.0, 0.3186, 0.6745, 1.1503
    "5A": {
        "n_facies": 8,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3 F4 F5 F6 F7 F8",
             "residual_ids": "1  1  1  1  1  1  1",
             "trend_ids":    "1  2  3  4  5  6  7"},
        ],
        # 7 trends that place equal-probability thresholds at norm.ppf(k/8) for k=1..7
        "trends": [
            ("1", "1.1503"),
            ("2", "0.6745"),
            ("3", "0.3186"),
            ("4",  "0.0000"),
            ("5",  "-0.3186"),
            ("6",  "-0.6745"),
            ("7",  "-1.1503"),
        ],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 5}],
    },
    "5A2": {
        "n_facies": 8,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3 F4 F5 F6 F7 F8",
             "residual_ids": "1  1  1  1  1  1  1",
             "trend_ids":    "1  2  3  4  5  6  7"},
        ],
        # 7 trends that place equal-probability thresholds at norm.ppf(k/8) for k=1..7
        "trends": [
            ("1", "1.1503"),
            ("2", "0.6745"),
            ("3", "0.3186"),
            ("4",  "0.0000"),
            ("5",  "-0.3186"),
            ("6",  "-0.6745"),
            ("7",  "-1.1503"),
        ],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 5}, {"x": 3500.0, "y": 2000.0, "facies": 5}],
    },
    "5B": {
        "n_facies": 8,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3 F4 F5 F6 F7 F8",
             "residual_ids": "1  1  1  1  1  1  1",
             "trend_ids":    "1  2  3  4  5  6  7"},
        ],
        "trends": [
            ("1", "1.1503"),
            ("2", "0.6745"),
            ("3", "0.3186"),
            ("4",  "0.0000"),
            ("5",  "-0.3186"),
            ("6",  "-0.6745"),
            ("7",  "-1.1503"),
        ],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 800.0, "subrange": 500.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 5},
            {"x":  1000.0, "y": 1000.0, "facies": 5},
            {"x":  5000.0, "y": 1000.0, "facies": 5},
            {"x":  1000.0, "y": 3000.0, "facies": 5},
            {"x":  5000.0, "y": 3000.0, "facies": 5},
        ],
    },
    "5C": {
        "n_facies": 8,
        "facies_models": [
            {"parent": "background", "names": "F1 F2 F3 F4 F5 F6 F7 F8",
             "residual_ids": "1  1  1  1  1  1  1",
             "trend_ids":    "1  2  3  4  5  6  7"},
        ],
        "trends": [
            ("1", "1.1503"),
            ("2", "0.6745"),
            ("3", "0.3186"),
            ("4",  "0.0000"),
            ("5",  "-0.3186"),
            ("6",  "-0.6745"),
            ("7",  "-1.1503"),
        ],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 8000.0, "subrange": 5000.0, "power": 1.5, "azimuth": 30.0},
        ],
        "observations": [
            {"x":  3000.0, "y": 2000.0, "facies": 5},
            {"x":  4000.0, "y": 2000.0, "facies": 5},
        ],
    },
    # ── Model 6: 4-facies, 3-level hierarchy, 3 independent GRFs ────────────
    # Level 1 (Background):  F4  | F1F2F3   (GRF 1, trend 1)
    # Level 2 (F1F2F3):      F3  | F1F2     (GRF 2, trend 2)
    # Level 3 (F1F2):        F1  | F2       (GRF 3, trend 3)
    # An F1 or F2 observation constrains all three GRFs.
    "6A": {
        "n_facies": 4,
        "facies_models": [
            {"parent": "background", "names": "F4 F1F2F3", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2F3",     "names": "F3 F1F2",   "residual_ids": "2", "trend_ids": "2"},
            {"parent": "F1F2",       "names": "F1 F2",     "residual_ids": "3", "trend_ids": "3"},
        ],
        "trends": [("1", "2.0"), ("2", "2.0"), ("3", "2.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 1600.0, "subrange": 1000.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 800.0,  "subrange": 800.0,  "power": 1.8, "azimuth": 0.0},
            {"id": "3", "type": "genexp", "range": 400.0,  "subrange": 400.0,  "power": 1.8, "azimuth": 0.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 4}],
    },
    "6B": {
        "n_facies": 4,
        "facies_models": [
            {"parent": "background", "names": "F4 F1F2F3", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2F3",     "names": "F3 F1F2",   "residual_ids": "2", "trend_ids": "2"},
            {"parent": "F1F2",       "names": "F1 F2",     "residual_ids": "3", "trend_ids": "3"},
        ],
        "trends": [("1", "-2.0"), ("2", "-2.0"), ("3", "-2.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 1600.0, "subrange": 1000.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 800.0,  "subrange": 800.0,  "power": 1.8, "azimuth": 0.0},
            {"id": "3", "type": "genexp", "range": 400.0,  "subrange": 400.0,  "power": 1.8, "azimuth": 0.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "6C": {
        "n_facies": 4,
        "facies_models": [
            {"parent": "background", "names": "F4 F1F2F3", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2F3",     "names": "F3 F1F2",   "residual_ids": "2", "trend_ids": "2"},
            {"parent": "F1F2",       "names": "F1 F2",     "residual_ids": "3", "trend_ids": "3"},
        ],
        "trends": [("1", "-1.0"), ("2", "-1.0"), ("3", "-1.0")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 1600.0, "subrange": 1000.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 800.0,  "subrange": 800.0,  "power": 1.8, "azimuth": 0.0},
            {"id": "3", "type": "genexp", "range": 400.0,  "subrange": 400.0,  "power": 1.8, "azimuth": 0.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
    "6D": {
        "n_facies": 4,
        "facies_models": [
            {"parent": "background", "names": "F4 F1F2F3", "residual_ids": "1", "trend_ids": "1"},
            {"parent": "F1F2F3",     "names": "F3 F1F2",   "residual_ids": "2", "trend_ids": "2"},
            {"parent": "F1F2",       "names": "F1 F2",     "residual_ids": "3", "trend_ids": "3"},
        ],
        "trends": [("1", "-2.5"), ("2", "-2.5"), ("3", "-2.5")],
        "residuals": [
            {"id": "1", "type": "genexp", "range": 1600.0, "subrange": 1000.0, "power": 1.5, "azimuth": 30.0},
            {"id": "2", "type": "genexp", "range": 800.0,  "subrange": 800.0,  "power": 1.8, "azimuth": 0.0},
            {"id": "3", "type": "genexp", "range": 400.0,  "subrange": 400.0,  "power": 1.8, "azimuth": 0.0},
        ],
        "observations": [{"x": 3000.0, "y": 2000.0, "facies": 2}],
    },
}

GRID_NX = 151
GRID_NY = 101
GRID_NZ = 2
X_LENGTH = 6000.0
Y_LENGTH = 4000.0
Z_LENGTH = 20.0


def _build_well_file(well_name, x, y, facies_code, n_facies=3):
    z_values = [i * 2.0 for i in range(int(Z_LENGTH / 2))]
    data_lines = [f"{x}   {y}   {z:6.1f}    {facies_code}" for z in z_values]
    facies_disc = " ".join(f"{i} F{i}" for i in range(1, n_facies + 1))
    return "\n".join([
        "1.0",
        "UNDEFINED",
        f"{well_name} {x} {y} 0.0",
        "1",
        f"FACIES DISC {facies_disc}",
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
    for i in range(len(cfg["observations"])):
        well_parts += [
            "    <well>",
            _xml_tag("      ", "file-name", f"wells/obs_{i}.rmswell"),
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


def _run_trane_single(iteration, model_number, path_trane_models, path_trane_exe, verbose_trane):
    output_dir            = f"output{model_number}_edited_{iteration}"
    modelfile_edited_path = os.path.join(path_trane_models, f"model{model_number}_edited_{iteration}.xml")
    results_path          = os.path.join(path_trane_models, output_dir, "result.roff")
    with open(modelfile_edited_path, 'w') as f:
        f.write(_build_model_xml(model_number, seed=iteration, output_dir=output_dir))
    trane_output = None if verbose_trane else subprocess.DEVNULL
    result = subprocess.run([path_trane_exe, modelfile_edited_path], shell=True, stdout=trane_output, stderr=trane_output)
    if result.returncode != 0:
        raise RuntimeError(f"TRANE failed on iteration {iteration} with return code {result.returncode}")
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
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                temp[i][j][k] = data[counter]
                counter += 1
    for i in range(nx):
        for j in range(ny):
            z[i][j] = temp[i][j][nz - 1]
    return iteration, z


def run_TRANE_simulations(n_simulations, model_number, path_trane_models, path_trane_exe, print_info=False, verbose_trane=False, n_workers=4):
    os.chdir(path_trane_models)
    dx = X_LENGTH / GRID_NX
    dy = Y_LENGTH / GRID_NY

    cfg = MODEL_CONFIGS[model_number]
    wells_dir = os.path.join(path_trane_models, "input", "wells")
    os.makedirs(wells_dir, exist_ok=True)
    for i, obs in enumerate(cfg["observations"]):
        well_file_path = os.path.join(wells_dir, f"obs_{i}.rmswell")
        with open(well_file_path, 'w') as f:
            f.write(_build_well_file(f"obs_{i}", obs["x"], obs["y"], obs["facies"], n_facies=cfg["n_facies"]))

    out_z = [None] * n_simulations
    completed = [0]
    if print_info:
        _print_progress_bar(0, n_simulations, prefix="Progress")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_trane_single, i, model_number, path_trane_models, path_trane_exe, verbose_trane): i
            for i in range(n_simulations)
        }
        for future in as_completed(futures):
            try:
                iteration, z = future.result()
            except RuntimeError as e:
                print(f"\033[31m\nERROR: {e}\033[0m")
                sys.exit(1)
            out_z[iteration] = z
            completed[0] += 1
            if print_info:
                _print_progress_bar(completed[0], n_simulations, prefix="Progress")
            # Delete per-iteration files immediately after data is read (keep iteration 0)
            if iteration != 0:
                xml_path = os.path.join(path_trane_models, f"model{model_number}_edited_{iteration}.xml")
                out_dir  = os.path.join(path_trane_models, f"output{model_number}_edited_{iteration}")
                if os.path.exists(xml_path):
                    os.remove(xml_path)
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)

    parameters = [dx, dy, X_LENGTH, Y_LENGTH]
    if print_info:
        print()
    return out_z, parameters

def _run_aps_single(iteration, v1, v2, nx, ny, dx, dy, t1, t2, model_family, thresholds=None, v3=None, t3=None):
    _t_single = time.time()
    s1 = simulate_gaussian_field(v1, nx, dx, ny, dy, seed=iteration)
    s2 = simulate_gaussian_field(v2, nx, dx, ny, dy, seed=iteration) if v2 is not None else None
    s3 = simulate_gaussian_field(v3, nx, dx, ny, dy, seed=iteration) if v3 is not None else None
    if model_family == "0":
        z = np.where(s1 < t1, 1, 2)
    elif model_family == "1":
        z = np.where(s1 < t1, 1, np.where(s1 < t2, 2, 3))
    elif model_family == "5":
        # Apply thresholds from last to first so earlier ones win
        z = np.full(s1.shape, len(thresholds) + 1, dtype=np.int8)
        for k in range(len(thresholds) - 1, -1, -1):
            z = np.where(s1 < thresholds[k], k + 1, z)
    elif model_family == "6":
        # 3-level hierarchy: Background→F4|F1F2F3 → F3|F1F2 → F1|F2
        z = np.where(s1 < t1, 4, np.where(s2 < t2, 3, np.where(s3 < t3, 1, 2)))
    else:
        z = np.where(s1 < t1, 3, np.where(s2 < t2, 1, 2))
    return iteration, z, time.time() - _t_single


def run_APS_simulations(n_simulations, nx, ny, dx, dy, model_number, print_info=False, data_dir=".", n_workers=4):
    cfg = MODEL_CONFIGS[model_number]
    r1 = cfg["residuals"][0]
    v1_range_x      = r1["range"]
    v1_range_y      = r1["subrange"]
    v1_range_z      = Z_LENGTH
    v1_azimuth      = r1["azimuth"] * 3.141592 / 180.0  # In radians, not degrees
    v1_genexp_power = r1["power"]

    if model_number[0] in ("2", "3", "4", "6"):
        r2 = cfg["residuals"][1]
        v2_range_x      = r2["range"]
        v2_range_y      = r2["subrange"]
        v2_range_z      = Z_LENGTH
        v2_azimuth      = r2["azimuth"]  # Already 0.0, no conversion needed
        v2_genexp_power = r2["power"]
    if model_number[0] == "6":
        r3 = cfg["residuals"][2]
        v3_range_x      = r3["range"]
        v3_range_y      = r3["subrange"]
        v3_range_z      = Z_LENGTH
        v3_azimuth      = r3["azimuth"]
        v3_genexp_power = r3["power"]

    n_facies = MODEL_CONFIGS[model_number]["n_facies"]
    p_F1 = np.load(os.path.join(data_dir, "p1_from_TRANE.npy"))
    if n_facies >= 3:
        p_F2 = np.load(os.path.join(data_dir, "p2_from_TRANE.npy"))
        p_F3 = np.load(os.path.join(data_dir, "p3_from_TRANE.npy"))
    if n_facies > 3:
        p_all = [np.load(os.path.join(data_dir, f"p{k}_from_TRANE.npy")) for k in range(1, n_facies + 1)]

    # Calculate thresholds (vectorized)
    thresholds_5 = None
    if model_number[0] == "0":
        t1 = norm.ppf(np.clip(p_F1, 1e-9, 1.0 - 1e-9))
        t2 = t3 = np.zeros((nx, ny))
    elif model_number[0] == "1":
        t1 = norm.ppf(np.clip(p_F1, 1e-9, 1.0 - 1e-9))
        t2 = norm.ppf(np.clip(p_F1 + p_F2, 1e-9, 1.0 - 1e-9))
        t3 = np.zeros((nx, ny))
    elif model_number[0] in ("2", "3", "4"):
        t1 = norm.ppf(np.clip(p_F3, 1e-9, 1.0 - 1e-9))
        t2 = norm.ppf(np.clip(p_F1 / np.maximum(1e-9, 1.0 - p_F3), 1e-9, 1.0 - 1e-9))
        t3 = np.zeros((nx, ny))
    elif model_number[0] == "5":
        t1 = t2 = t3 = np.zeros((nx, ny))
        thresholds_5 = []
        for k in range(1, n_facies):
            cum_p = np.clip(sum(p_all[:k]), 1e-9, 1.0 - 1e-9)
            thresholds_5.append(norm.ppf(cum_p))
    elif model_number[0] == "6":
        p_not_f4 = np.maximum(1e-9, 1.0 - p_all[3])
        p_f1f2   = np.maximum(1e-9, p_all[0] + p_all[1])
        t1 = norm.ppf(np.clip(p_all[3],              1e-9, 1.0 - 1e-9))
        t2 = norm.ppf(np.clip(p_all[2] / p_not_f4,  1e-9, 1.0 - 1e-9))
        t3 = norm.ppf(np.clip(p_all[0] / p_f1f2,    1e-9, 1.0 - 1e-9))
    else:
        t1 = t2 = t3 = np.zeros((nx, ny))

    v1 = GeneralExponentialVariogram(v1_range_x, v1_range_y, v1_range_z, azi=v1_azimuth, power=v1_genexp_power)
    v2 = None
    if model_number[0] in ("2", "3", "4", "6"):
        v2 = GeneralExponentialVariogram(v2_range_x, v2_range_y, v2_range_z, azi=v2_azimuth, power=v2_genexp_power)
    v3 = None
    if model_number[0] == "6":
        v3 = GeneralExponentialVariogram(v3_range_x, v3_range_y, v3_range_z, azi=v3_azimuth, power=v3_genexp_power)

    out_z = [None] * n_simulations
    completed = [0]
    iter_times = []
    if print_info:
        _print_progress_bar(0, n_simulations, prefix="Progress")

    _t_aps_start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_aps_single, i, v1, v2, nx, ny, dx, dy, t1, t2, model_number[0], thresholds_5, v3, t3): i
            for i in range(n_simulations)
        }
        for future in as_completed(futures):
            iteration, z, _elapsed = future.result()
            iter_times.append(_elapsed)
            out_z[iteration] = z
            completed[0] += 1
            if print_info:
                _print_progress_bar(completed[0], n_simulations, prefix="Progress")

    _t_aps_total = time.time() - _t_aps_start
    if print_info:
        print()
    CYAN  = "\033[36m"
    RESET = "\033[0m"
    print(f"{CYAN}  [APS timing] total: {_t_aps_total:.2f}s  per-realization (median): {statistics.median(iter_times):.3f}s  mean: {statistics.mean(iter_times):.3f}s{RESET}")
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
        if len(count_connected_filtered) >= 2:
            print(f"Stdev connected nodes:     {statistics.stdev(count_connected_filtered):.2f}")
        else:
            print(f"Stdev connected nodes:     n/a (only 1 realization)")
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


def _cluster_summary_stats_lines(counts, well_name, prefix):
    n = len(counts)
    lines = [f"Cluster size summary for {prefix}_{well_name} (n={n}):"]
    if n > 0:
        lines.append(f"  Mean:   {statistics.mean(counts):.2f}")
        if n >= 2:
            lines.append(f"  Stdev:  {statistics.stdev(counts):.2f}")
        lines.append(f"  Median: {statistics.median(counts):.1f}")
        lines.append(f"  Min:    {min(counts)}")
        lines.append(f"  Max:    {max(counts)}")
    return lines


def _save(name, obj):
    with open(name, "wb") as fp:
        pickle.dump(obj, fp)


def _load(path, name):
    with open(os.path.join(path, name), "rb") as fp:
        return pickle.load(fp)


def _analyse(z, parameters, prefix, dx, dy, verbose, model_number, max_facies_grid_exports=100, save_thresholds=False, output_dir=".", data_dir=".", log_file=None):
    CYAN  = "\033[36m"
    RESET = "\033[0m"
    _t0 = time.time()

    n_sim = len(z)
    if max_facies_grid_exports is None or max_facies_grid_exports >= n_sim:
        save_indices = "all"
    else:
        save_indices = set(range(max_facies_grid_exports))

    # Compute shortest connection paths (saved iterations for plot; all iterations for length stats)
    _connection_paths = None
    path_lengths = []
    _obs_cfg = MODEL_CONFIGS[model_number]["observations"]
    if len(_obs_cfg) >= 2:
        _obs0 = _obs_cfg[0]
        _obs1 = _obs_cfg[1]
        _nx = z[0].shape[0]
        _ny = z[0].shape[1]
        _x1_ind = math.floor(_obs0["x"] / dx)
        _y1_ind = math.floor(_obs0["y"] / dy)
        _x2_ind = math.floor(_obs1["x"] / dx)
        _y2_ind = math.floor(_obs1["y"] / dy)
        _t = time.time()
        _connection_paths = [None] * n_sim
        for _it, z_i in enumerate(z):
            _p = _find_shortest_path(z_i, _nx, _ny, dx, dy, _x1_ind, _y1_ind, _x2_ind, _y2_ind)
            if save_indices == "all" or _it in save_indices:
                _connection_paths[_it] = _p
            if _p is not None:
                path_lengths.append(sum(
                    math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    for p1, p2 in zip(_p, _p[1:])
                ))
        print(f"{CYAN}  [timing] {prefix + ' find_shortest_paths:':<42} {time.time()-_t:6.2f}s{RESET}")

    _t = time.time()
    save_facies_grids_as_png(z, parameters, prefix, model_number, save_indices, output_dir=output_dir, connection_paths=_connection_paths)
    print(f"{CYAN}  [timing] {prefix + ' save_facies_grids_as_png:':<42} {time.time()-_t:6.2f}s{RESET}")

    _t = time.time()
    calculate_and_save_facies_prob_maps(z, parameters, prefix, model_number, output_dir=output_dir, data_dir=data_dir)
    print(f"{CYAN}  [timing] {prefix + ' calculate_and_save_facies_prob:':<42} {time.time()-_t:6.2f}s{RESET}")

    if save_thresholds:
        _t = time.time()
        save_threshold_grids_as_png(parameters, model_number, output_dir=output_dir, data_dir=data_dir, prefix=prefix)
        print(f"{CYAN}  [timing] {prefix + ' save_threshold_grids_as_png:':<42} {time.time()-_t:6.2f}s{RESET}")

    obs_cfg = MODEL_CONFIGS[model_number]["observations"]
    if len(obs_cfg) >= 2:
        obs0 = obs_cfg[0]
        obs1 = obs_cfg[1]
        _t = time.time()
        count_connected = count_connected_grid_nodes(
            z, parameters, obs0["x"], obs0["y"], [obs1["x"], obs1["y"]]
        )
        print(f"{CYAN}  [timing] {prefix + ' count_connected_grid_nodes:':<42} {time.time()-_t:6.2f}s{RESET}")
        count_connected_filtered = [c for c in count_connected if c != -1]
        n_total  = len(count_connected)
        n_conn   = len(count_connected_filtered)
        pct_conn = 100.0 * n_conn / n_total if n_total > 0 else 0.0
        conn_mean = statistics.mean(count_connected_filtered) if count_connected_filtered else None
        conn_min  = min(count_connected_filtered) if count_connected_filtered else None
        conn_max  = max(count_connected_filtered) if count_connected_filtered else None
        conn_lines = [
            f"Connections for {prefix} [obs_0 -> obs_1] (n={n_total}):",
            f"  Connected: {n_conn:5d} / {n_total}  ({pct_conn:5.1f}%)",
        ]
        if conn_mean is not None:
            conn_lines += [
                f"  Mean:   {conn_mean:.2f}",
                f"  Min:    {conn_min}",
                f"  Max:    {conn_max}",
            ]
        if verbose:
            sum_connected = [dx * dy * n if n != -1 else -1 for n in count_connected]
            _print_results(sum_connected, count_connected, count_connected_filtered, dx, dy)
        print()
        for line in conn_lines:
            print(line)
        if log_file:
            with open(log_file, 'a') as _lf:
                _lf.write("\n")
                _lf.write("\n".join(conn_lines) + "\n")
        # Path length stats
        if path_lengths:
            pl_mean  = statistics.mean(path_lengths)
            pl_stdev = statistics.stdev(path_lengths) if len(path_lengths) >= 2 else None
            pl_min   = min(path_lengths)
            pl_max   = max(path_lengths)
            pl_lines = [
                f"Path lengths for {prefix} [obs_0 -> obs_1] (n_connected={len(path_lengths)}):",
                f"  Mean:   {pl_mean:.2f}",
            ]
            if pl_stdev is not None:
                pl_lines.append(f"  Stdev:  {pl_stdev:.2f}")
            pl_lines += [f"  Min:    {pl_min:.2f}", f"  Max:    {pl_max:.2f}"]
            print()
            for line in pl_lines:
                print(line)
            if log_file:
                with open(log_file, 'a') as _lf:
                    _lf.write("\n")
                    _lf.write("\n".join(pl_lines) + "\n")
    else:
        count_connected_filtered = []

    well_plot_data = []
    for i, obs in enumerate(MODEL_CONFIGS[model_number]["observations"]):
        obs_name = f"obs_{i}"
        _t = time.time()
        per_well_counts = count_connected_nodes_from_point(z, parameters, obs["x"], obs["y"])
        print(f"{CYAN}  [timing] {(prefix + ' count_connected_nodes [' + obs_name + ']:'):<42} {time.time()-_t:6.2f}s{RESET}")
        well_plot_data.append((prefix + "_" + obs_name, per_well_counts))
        stat_lines = _cluster_size_stats_lines(per_well_counts, obs_name, prefix)
        summary_lines = _cluster_summary_stats_lines(per_well_counts, obs_name, prefix)
        all_lines = stat_lines + summary_lines
        if verbose:
            print()
            for line in all_lines:
                print(line)
        if log_file:
            with open(log_file, 'a') as _lf:
                _lf.write("\n")
                _lf.write("\n".join(all_lines) + "\n")

    print(f"{CYAN}  [timing] {prefix + ' _analyse total:':<42} {time.time()-_t0:6.2f}s{RESET}")
    return count_connected_filtered, well_plot_data, path_lengths


def save_facies_grids_as_png(facies_grids, parameters, prefix, model_number, indices_to_save="all", output_dir=".", connection_paths=None):
    _FACIES_COLORS = [
        (255/255,  69/255,   0/255),   # F1  Orange-Red
        ( 75/255,   0/255, 130/255),   # F2  Indigo
        (  0/255, 206/255, 209/255),   # F3  Dark Turquoise
        ( 34/255, 139/255,  34/255),   # F4  Forest Green
        (255/255, 215/255,   0/255),   # F5  Gold
        (220/255,  20/255,  60/255),   # F6  Crimson
        (  0/255, 191/255, 255/255),   # F7  Deep Sky Blue
        (148/255,   0/255, 211/255),   # F8  Dark Violet
        (255/255, 140/255,   0/255),   # F9  Dark Orange
        ( 64/255, 224/255, 208/255),   # F10 Turquoise
    ]
    n_facies = MODEL_CONFIGS[model_number]["n_facies"]
    cmap = colors.ListedColormap(_FACIES_COLORS[:n_facies])
    nx = facies_grids[0].shape[0]
    ny = facies_grids[0].shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_length = parameters[2]
    y_length = parameters[3]
    well_x = [obs["x"] for obs in MODEL_CONFIGS[model_number]["observations"]]
    well_y = [obs["y"] for obs in MODEL_CONFIGS[model_number]["observations"]]
    x_min = 0.0
    x_max = x_length
    y_min = 0.0
    y_max = y_length
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

            cmap = colors.ListedColormap(_FACIES_COLORS[:n_facies])
            fig = plt.figure(frameon=False)
            fig.set_size_inches(9,6)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # img = plt.imshow(z_simbox, cmap = cmap, alpha = 1.0, interpolation='none', extent = extent) # interpolation ='bilinear'
            img = plt.imshow(z_for_plotting, cmap = cmap, alpha = 1.0, interpolation='none', extent = extent, vmin=0.5, vmax=n_facies + 0.5) # interpolation ='bilinear'
            x_grid = np.linspace(0, x_length, nx + 1)
            y_grid = np.linspace(0, y_length, ny + 1)
            ax.vlines(x_grid, 0, y_length, colors='black', linewidths=0.2, alpha=0.4)
            ax.hlines(y_grid, 0, x_length, colors='black', linewidths=0.2, alpha=0.4)
            for wx, wy in zip(well_x, well_y):
                cell_x = math.floor(wx / dx) * dx
                cell_y = math.floor(wy / dy) * dy
                ax.add_patch(patches.Rectangle(
                    (cell_x, cell_y), dx, dy,
                    linewidth=0.5, edgecolor='black', facecolor='none'
                ))
                ax.plot(wx, wy, marker='.', color='black', markersize=0.3, linewidth=0.0)
            if connection_paths is not None and connection_paths[iteration] is not None:
                path = connection_paths[iteration]
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax.plot(path_x, path_y, color='white', linewidth=0.75, alpha=0.6,
                        solid_capstyle='round', solid_joinstyle='round')
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


def _find_shortest_path(z, nx, ny, dx, dy, x1_ind, y1_ind, x2_ind, y2_ind):
    """BFS shortest path through same-facies cells from (x1_ind,y1_ind) to (x2_ind,y2_ind).
    Returns list of world (x,y) coordinates (cell centres) or None if not connected."""
    if z[x1_ind][y1_ind] != z[x2_ind][y2_ind]:
        return None
    facies = z[x1_ind][y1_ind]
    parent = {(x1_ind, y1_ind): None}
    queue = [(x1_ind, y1_ind)]
    head = 0
    found = False
    while head < len(queue):
        xi, yi = queue[head]; head += 1
        if xi == x2_ind and yi == y2_ind:
            found = True
            break
        for xn, yn in ((xi+1,yi),(xi-1,yi),(xi,yi+1),(xi,yi-1)):
            if 0 <= xn < nx and 0 <= yn < ny and z[xn][yn] == facies and (xn,yn) not in parent:
                parent[(xn,yn)] = (xi,yi)
                queue.append((xn,yn))
    if not found:
        return None
    path = []
    cur = (x2_ind, y2_ind)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return [(xi * dx + dx * 0.5, yi * dy + dy * 0.5) for xi, yi in path]


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
    x_max = x_length
    y_min = 0.0
    y_max = y_length
    extent = x_min, x_max, y_min, y_max

    n_facies = MODEL_CONFIGS[model_number]["n_facies"]

    z_stack = np.stack(facies_grids, axis=0)  # shape: (n_sim, nx, ny)
    p_F1 = np.mean(z_stack == 1, axis=0)
    p_F2 = np.mean(z_stack == 2, axis=0)
    p_F3 = np.mean(z_stack == 3, axis=0) if n_facies >= 3 else None

    np.save(os.path.join(data_dir, "p1_from_" + prefix), p_F1)
    np.save(os.path.join(data_dir, "p2_from_" + prefix), p_F2)
    if n_facies >= 3:
        np.save(os.path.join(data_dir, "p3_from_" + prefix), p_F3)
    if n_facies > 3:
        for k in range(4, n_facies + 1):
            pk = np.mean(z_stack == k, axis=0)
            np.save(os.path.join(data_dir, f"p{k}_from_" + prefix), pk)

    facies_probs = [np.mean(z_stack == k, axis=0) for k in range(1, n_facies + 1)]
    for i, p in enumerate(facies_probs):
        p_for_plotting = np.flipud(p.T)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(6,4)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = plt.imshow(p_for_plotting, cmap = 'Blues', alpha = 1.0, interpolation='none', extent = extent, vmin = 0.0, vmax = 1.0)
        fig.colorbar(img, ax=ax, shrink=0.5)
        plt.savefig(os.path.join(output_dir, prefix + '_p' + str(i+1) + '_n' + str(len(facies_grids)) + '.png'), dpi=100)
        plt.close()

def compare_prob_maps(model_number, trane_data_dir, aps_data_dir, output_dir=None, log_file=None):
    """Compare TRANE vs APS probability maps. Prints and logs max|diff| and mean|diff| per facies.
    If output_dir is given, saves a single PNG with signed-difference subplots (one per facies)."""
    CYAN  = "\033[36m"
    RESET = "\033[0m"
    n_facies = MODEL_CONFIGS[model_number]["n_facies"]
    header = "Probability map sanity check (TRANE vs APS):"
    col_header = f"  {'Facies':<8} {'max|diff|':>12} {'mean|diff|':>12}"
    lines = [header, col_header]
    print(f"\n{CYAN}{header}")
    print(col_header)
    diffs_signed = []
    for k in range(1, n_facies + 1):
        p_trane = np.load(os.path.join(trane_data_dir, f"p{k}_from_TRANE.npy"))
        p_aps   = np.load(os.path.join(aps_data_dir,   f"p{k}_from_APS.npy"))
        signed    = p_trane - p_aps
        diff      = np.abs(signed)
        max_diff  = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        line = f"  {'F' + str(k):<8} {max_diff:>12.5f} {mean_diff:>12.5f}"
        lines.append(line)
        print(line)
        diffs_signed.append(signed)
    print(RESET, end="")
    if log_file:
        with open(log_file, 'a') as _f:
            _f.write("\n")
            _f.write("\n".join(lines) + "\n")
    if output_dir is not None:
        abs_max = max(float(np.max(np.abs(d))) for d in diffs_signed)
        abs_max = max(abs_max, 1e-6)  # avoid zero range if maps are identical
        ncols = min(n_facies, 4)
        nrows = math.ceil(n_facies / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for k, (d, ax) in enumerate(zip(diffs_signed, axes.flat)):
            img = ax.imshow(np.flipud(d.T), cmap='RdBu_r', interpolation='none',
                            vmin=-abs_max, vmax=abs_max)
            ax.set_title(f"F{k+1}", fontsize=9)
            ax.axis('off')
            fig.colorbar(img, ax=ax, shrink=0.7)
        for ax in list(axes.flat)[n_facies:]:
            ax.set_visible(False)
        fig.suptitle(f"Probability difference TRANE − APS  (model {model_number})", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prob_diff_TRANE_minus_APS.png"), dpi=100, bbox_inches='tight')
        plt.close()


def plot_histogram_of_connected_cells(sum_connected, prefix, xmin, xmax, ymin, ymax, n_bins, output_dir=".", xlabel='Connected grid nodes', filename_tag='connectedvolume', log_file=None):
    fig, ax1 = plt.subplots(figsize=(15, 10))
    binwidth = max(1, int(round((xmax - xmin) / n_bins)))
    _bin_msg = f"  [histogram] {prefix} {filename_tag}: bin_size={binwidth} (n_bins={n_bins}, xrange=[{xmin},{xmax}])"
    print(_bin_msg)
    if log_file:
        with open(log_file, "a") as _lf:
            _lf.write(_bin_msg + "\n")
    weights = np.ones(len(sum_connected)) / len(sum_connected) if sum_connected else np.array([])
    ax1.hist(sum_connected, bins=np.arange(xmin, xmax + binwidth, binwidth), color='steelblue', weights=weights)
    ax1.set_ylabel('Proportion')
    ax1.set_xlabel(xlabel)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    if sum_connected:
        ax2 = ax1.twinx()
        sorted_vals = sorted(sum_connected)
        n = len(sorted_vals)
        cdf_x = [xmin] + sorted_vals + [xmax]
        cdf_y = [0.0] + [(i + 1) / n for i in range(n)] + [1.0]
        ax2.step(cdf_x, cdf_y, where='post', color='darkorange', linewidth=1.5)
        ax2.set_ylabel('CDF')
        ax2.set_ylim(0.0, 1.0)

    plt.savefig(os.path.join(output_dir, prefix + '_' + filename_tag + '_n' + str(len(sum_connected)) + '.png'), dpi=100, bbox_inches='tight', pad_inches=0.3)
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
    p_all = [np.load(os.path.join(data_dir, f"p{k}_from_" + prefix + ".npy")) for k in range(1, n_facies + 1)]

    nx = p_all[0].shape[0]
    ny = p_all[0].shape[1]
    dx = parameters[0]
    dy = parameters[1]
    x_length = parameters[2]
    y_length = parameters[3]
    x_min = 0.0
    x_max = x_length
    y_min = 0.0
    y_max = y_length
    extent = x_min, x_max, y_min, y_max

    # Build n_facies-1 cumulative threshold arrays
    thresholds = []
    if n_facies == 2:
        t1 = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                t1[i][j] = norm.ppf(np.clip(p_all[0][i][j], 1e-9, 1.0 - 1e-9))
        thresholds = [t1]
    elif n_facies == 3:
        t1 = np.zeros((nx, ny))
        t2 = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                t1[i][j] = norm.ppf(np.clip(p_all[0][i][j], 1e-9, 1.0 - 1e-9))
                p1_p2 = min(1.0 - 1e-9, p_all[0][i][j] + p_all[1][i][j])
                t2[i][j] = norm.ppf(p1_p2)
        thresholds = [t1, t2]
    else:
        # General case: n_facies-1 cumulative thresholds
        for k in range(1, n_facies):
            cum_p = np.clip(sum(p_all[:k]), 0.0, 1.0 - 1e-9)
            thresholds.append(norm.ppf(cum_p))

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
