#!/usr/bin/env python
"""
Summarize all results_* run folders found next to this script.
Reads run_log.txt from each and prints a table with key metrics.
T = TRANE, A = APS.  Uses the first cluster-stats/summary block found
if a log contains multiple wells.  Writes 'x' for any missing value.
"""

import os
import re
import argparse


def _parse_log(path):
    result = {
        "model":          "x",
        "n_sim":          "x",
        "trane_size1":    "x",
        "trane_mean":     "x",
        "trane_stdev":    "x",
        "trane_median":   "x",
        "trane_min":      "x",
        "trane_max":      "x",
        "aps_size1":      "x",
        "aps_mean":       "x",
        "aps_stdev":      "x",
        "aps_median":     "x",
        "aps_min":        "x",
        "aps_max":        "x",
        "model_config":   None,
        "n_obs":          "x",
        "trane_conn":     "x",
        "trane_cmean":    "x",
        "trane_cmin":     "x",
        "trane_cmax":     "x",
        "aps_conn":       "x",
        "aps_cmean":      "x",
        "aps_cmin":       "x",
        "aps_cmax":       "x",
    }
    try:
        with open(path, "r") as f:
            text = f.read()
    except Exception:
        return result

    m = re.search(r"^MODEL:\s+(\S+)", text, re.MULTILINE)
    if m:
        result["model"] = m.group(1)

    m = re.search(r"^n_sim:\s+(\d+)", text, re.MULTILINE)
    if m:
        result["n_sim"] = m.group(1)

    # TRANE: first "Size  1" percentage from first TRANE stats block
    m = re.search(
        r"Cluster size stats for TRANE_[^\n]+\n"
        r"\s+Size\s+1:\s+\d+ / \d+\s+\(\s*([\d.]+)%\)",
        text,
    )
    if m:
        result["trane_size1"] = m.group(1) + " %"

    # TRANE: first summary block
    m = re.search(
        r"Cluster size summary for TRANE_[^\n]+\n"
        r"\s+Mean:\s+([\d.]+)\n"
        r"\s+Stdev:\s+([\d.]+)\n"
        r"\s+Median:\s+([\d.]+)\n"
        r"\s+Min:\s+([\d.]+)\n"
        r"\s+Max:\s+([\d.]+)",
        text,
    )
    if m:
        result["trane_mean"]   = m.group(1)
        result["trane_stdev"]  = m.group(2)
        result["trane_median"] = m.group(3)
        result["trane_min"]    = m.group(4)
        result["trane_max"]    = m.group(5)

    # APS: first "Size  1" percentage
    m = re.search(
        r"Cluster size stats for APS_[^\n]+\n"
        r"\s+Size\s+1:\s+\d+ / \d+\s+\(\s*([\d.]+)%\)",
        text,
    )
    if m:
        result["aps_size1"] = m.group(1) + " %"

    # APS: first summary block
    m = re.search(
        r"Cluster size summary for APS_[^\n]+\n"
        r"\s+Mean:\s+([\d.]+)\n"
        r"\s+Stdev:\s+([\d.]+)\n"
        r"\s+Median:\s+([\d.]+)\n"
        r"\s+Min:\s+([\d.]+)\n"
        r"\s+Max:\s+([\d.]+)",
        text,
    )
    if m:
        result["aps_mean"]   = m.group(1)
        result["aps_stdev"]  = m.group(2)
        result["aps_median"] = m.group(3)
        result["aps_min"]    = m.group(4)
        result["aps_max"]    = m.group(5)

    # Extract model config block (lines between "Model config (..):'' and next blank line)
    config_lines = []
    in_config = False
    for line in text.splitlines():
        if line.startswith("Model config ("):
            in_config = True
            config_lines.append(line)
        elif in_config:
            if line.strip() == "":
                break
            config_lines.append(line)
    if config_lines:
        result["model_config"] = "\n".join(config_lines)

    # Count well observations: lines indented 8 spaces with x= (well entries in config block)
    well_entries = re.findall(r"^        \S.*x=", text, re.MULTILINE)
    if well_entries:
        result["n_obs"] = str(len(well_entries))

    # TRANE connection stats
    m = re.search(
        r"Connections for TRANE \[[^\]]+\] \(n=(\d+)\):\n"
        r"\s+Connected:\s+(\d+)\s*/",
        text,
    )
    if m:
        result["trane_conn"] = f"{m.group(2)}/{m.group(1)}"
    m = re.search(
        r"Connections for TRANE \[[^\]]+\] \(n=\d+\):\n"
        r"\s+Connected:.*\n"
        r"\s+Mean:\s+([\d.]+)\n"
        r"\s+Min:\s+(\d+)\n"
        r"\s+Max:\s+(\d+)",
        text,
    )
    if m:
        result["trane_cmean"] = m.group(1)
        result["trane_cmin"]  = m.group(2)
        result["trane_cmax"]  = m.group(3)

    # APS connection stats
    m = re.search(
        r"Connections for APS \[[^\]]+\] \(n=(\d+)\):\n"
        r"\s+Connected:\s+(\d+)\s*/",
        text,
    )
    if m:
        result["aps_conn"] = f"{m.group(2)}/{m.group(1)}"
    m = re.search(
        r"Connections for APS \[[^\]]+\] \(n=\d+\):\n"
        r"\s+Connected:.*\n"
        r"\s+Mean:\s+([\d.]+)\n"
        r"\s+Min:\s+(\d+)\n"
        r"\s+Max:\s+(\d+)",
        text,
    )
    if m:
        result["aps_cmean"] = m.group(1)
        result["aps_cmin"]  = m.group(2)
        result["aps_cmax"]  = m.group(3)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print model config details below the table")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dirs = sorted(
        d for d in os.listdir(script_dir)
        if d.startswith("results_") and os.path.isdir(os.path.join(script_dir, d))
    )

    if not results_dirs:
        print("No results_* folders found.")
        return

    rows = []
    for d in results_dirs:
        run_name = d[len("results_"):]
        log_path = os.path.join(script_dir, d, "run_log.txt")
        data = _parse_log(log_path)
        rows.append((run_name, data))

    # (header, data-key or None for run name, left-justify?)
    base_col_defs = [
        ("Run",       None,            True),
        ("Model",     "model",         False),
        ("n_obs",     "n_obs",         False),
        ("n_sim",     "n_sim",         False),
        ("T size1%",  "trane_size1",   False),
        ("A size1%",  "aps_size1",     False),
        ("T mean",    "trane_mean",    False),
        ("A mean",    "aps_mean",      False),
        ("T stdev",   "trane_stdev",   False),
        ("A stdev",   "aps_stdev",     False),
        ("T median",  "trane_median",  False),
        ("A median",  "aps_median",    False),
        ("T min",     "trane_min",     False),
        ("A min",     "aps_min",       False),
        ("T max",     "trane_max",     False),
        ("A max",     "aps_max",       False),
    ]
    conn_col_candidates = [
        ("T conn",    "trane_conn",    False),
        ("A conn",    "aps_conn",      False),
        ("T cmean",   "trane_cmean",   False),
        ("A cmean",   "aps_cmean",     False),
        ("T cmin",    "trane_cmin",    False),
        ("A cmin",    "aps_cmin",      False),
        ("T cmax",    "trane_cmax",    False),
        ("A cmax",    "aps_cmax",      False),
    ]
    col_defs = base_col_defs + conn_col_candidates

    # Compute column widths from headers and all row values
    def cell_value(run_name, data, key):
        return run_name if key is None else data[key]

    widths = []
    for header_text, key, _ in col_defs:
        w = len(header_text)
        for run_name, data in rows:
            w = max(w, len(cell_value(run_name, data, key)))
        widths.append(w)

    GREY    = "\033[90m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[96m"
    GREEN   = "\033[94m"
    RESET   = "\033[0m"

    # 3 newest runs by timestamp prefix (YYMMDD_HHMMSS = first 13 chars of run_name)
    newest_names = set(
        n for n, _ in sorted(rows, key=lambda x: x[0][:13], reverse=True)[:1]
    )

    size1_keys = {"trane_size1", "aps_size1", "trane_conn", "aps_conn"}

    sep = "  "

    def fmt(value, width, left):
        return value.ljust(width) if left else value.rjust(width)

    def colorize(value, raw, key, run_name):
        if raw == "x":
            return GREY + value + RESET
        if key in size1_keys:
            return YELLOW + value + RESET
        if key is None and run_name in newest_names:
            return GREEN + value + RESET
        if key is None and "keep" in run_name.lower():
            return CYAN + value + RESET
        return value

    header  = sep.join(fmt(h, w, lj) for (h, _, lj), w in zip(col_defs, widths))
    divider = sep.join("-" * w for w in widths)
    def _sort_key(item):
        run_name, data = item
        model = data["model"]
        mm = re.match(r"^(\d+)([A-Za-z]+)$", model)
        model_key = (int(mm.group(1)), mm.group(2)) if mm else (999, model)
        return (model_key, run_name)

    keep_rows  = sorted([(n, d) for n, d in rows if "keep"  in n.lower()], key=_sort_key)
    other_rows = sorted([(n, d) for n, d in rows if "keep" not in n.lower()], key=_sort_key)

    def print_row(run_name, data):
        cells = []
        for (_, key, lj), w in zip(col_defs, widths):
            raw = cell_value(run_name, data, key)
            cells.append(colorize(fmt(raw, w, lj), raw, key, run_name))
        print(sep.join(cells))

    print()
    print(header)
    print(divider)
    for run_name, data in keep_rows:
        print_row(run_name, data)
    if keep_rows and other_rows:
        pass
    for run_name, data in other_rows:
        print_row(run_name, data)
    print()

    # Per-run model config detail blocks
    if args.verbose:
        configs = [(name, data["model_config"]) for name, data in rows if data["model_config"]]
        if configs:
            BOLD_YELLOW = "\033[1;33m"
            print()
            print()
            for run_name, cfg_text in configs:
                title = f"{run_name}:"
                print(BOLD_YELLOW + title + RESET)
                print(BOLD_YELLOW + "=" * len(title) + RESET)
                print(cfg_text)
                print()


if __name__ == "__main__":
    main()
