"""Summarize all fixed cells and evaluate the pre-specified network stopping rule."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows_from_file(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            document = json.load(stream)
    else:
        document = json.loads(path.read_text())
    if "rows" in document:
        return document["rows"]
    return [row for batch in document["batches"] for row in batch["result"]["rows"]]


def bootstrap_interval(values: np.ndarray, indices: np.ndarray) -> list[float]:
    resampled = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(resampled, [0.025, 0.975])]


def summarize(config: dict, population: list[dict], sampled: list[dict]) -> dict:
    seeds = config["seeds"]
    generator = np.random.default_rng(config["bootstrap_seed"])
    indices = generator.integers(0, len(seeds), size=(config["bootstrap_replicates"], len(seeds)))
    grouped = defaultdict(list)
    for row in sampled:
        grouped[(row["case"], row["method"], row["schedule"])].append(row)
    expected_sampled = {
        (case["name"], method, schedule)
        for case in config["cases"]
        for method in config["sampled_methods"]
        for schedule in config["schedules"]
    }
    if set(grouped) != expected_sampled:
        raise ValueError("Sampled cells do not match the fixed configuration")
    scalar = {}
    for key, rows in grouped.items():
        by_seed = {row["seed"]: row for row in rows}
        if len(rows) != len(seeds) or set(by_seed) != set(seeds):
            raise ValueError(f"Missing or duplicate seeds in {key}")
        if key[1] == "scalar":
            scalar[(key[0], key[2])] = np.array([by_seed[seed]["endpoint_normalized_regret"] for seed in seeds])
    cells = []
    for (case, method, schedule), rows in sorted(grouped.items()):
        by_seed = {row["seed"]: row for row in rows}
        normalized = np.array([by_seed[seed]["endpoint_normalized_regret"] for seed in seeds])
        absolute = np.array([by_seed[seed]["endpoint_absolute_regret"] for seed in seeds])
        excess = normalized - scalar[(case, schedule)]
        cells.append({
            "case": case,
            "method": method,
            "schedule": schedule,
            "seeds": len(seeds),
            "normalized_regret_mean": float(normalized.mean()),
            "normalized_regret_ci95": bootstrap_interval(normalized, indices),
            "absolute_regret_mean": float(absolute.mean()),
            "absolute_regret_ci95": bootstrap_interval(absolute, indices),
            "paired_normalized_excess_mean": float(excess.mean()),
            "paired_normalized_excess_ci95": bootstrap_interval(excess, indices),
            "switching_rate_mean": float(np.mean([by_seed[seed]["endpoint_switching_rate"] for seed in seeds])),
            "seed_normalized_regrets": [float(value) for value in normalized],
            "seed_paired_normalized_excess": [float(value) for value in excess],
        })
    population_by_key = {(row["case"], row["method"], row["schedule"]): row for row in population}
    expected_population = {
        (case["name"], method, schedule)
        for case in config["cases"]
        for method in config["population_methods"]
        for schedule in (["none"] if method == "hard" else config["schedules"])
    }
    if set(population_by_key) != expected_population or len(population_by_key) != len(population):
        raise ValueError("Population cells do not match the fixed configuration")
    gate = config["network_gate"]
    threshold = gate["minimum_normalized_excess_regret"]
    gate_cells = []
    for capacity in gate["family_critic_k"]:
        for schedule in gate["schedules"]:
            case = f"family-{capacity}"
            row = next(row for row in cells if (row["case"], row["method"], row["schedule"]) == (case, gate["loss"], schedule))
            population_excess = population_by_key[(case, gate["loss"], schedule)]["endpoint_normalized_regret"] - population_by_key[(case, "scalar", schedule)]["endpoint_normalized_regret"]
            population_pass = not gate["require_population_material_effect"] or population_excess >= threshold
            sampled_pass = row["paired_normalized_excess_mean"] >= threshold and (not gate["require_positive_paired_interval_lower_endpoint"] or row["paired_normalized_excess_ci95"][0] > 0)
            gate_cells.append({
                "case": case,
                "schedule": schedule,
                "population_normalized_excess": population_excess,
                "sampled_normalized_excess": row["paired_normalized_excess_mean"],
                "sampled_excess_ci95": row["paired_normalized_excess_ci95"],
                "population_pass": population_pass,
                "sampled_pass": sampled_pass,
                "pass": population_pass and sampled_pass,
            })
    uncertainty = []
    for capacity in (2, 8, 32):
        margin = 1 / (4 * capacity**2)
        tau = (capacity - 1) / (2 * capacity)
        standard_error = math.sqrt(tau * (1 - tau) / config["batch_size"])
        uncertainty.append({
            "family_k": capacity,
            "hard_cycle_cdf_margin": margin,
            "central_midpoint_tau": tau,
            "batch_size": config["batch_size"],
            "batch_cdf_standard_error": standard_error,
            "margin_over_standard_error": margin / standard_error,
            "independent_fixed_target_samples_for_1_96_se_margin": math.ceil(1.96**2 * tau * (1 - tau) / margin**2),
            "note": "Normal-approximation scale diagnostic, not a guarantee or an effective sample count for changing SGD targets.",
        })
    return {
        "endpoint": "Stationary-policy regret averaged over post-update policies at t=80001..100000, normalized by the always-B loss",
        "bootstrap": {"replicates": config["bootstrap_replicates"], "seed": config["bootstrap_seed"], "unit": "paired seed", "interval": "percentile 95%"},
        "population_rows": population,
        "sampled_cells": cells,
        "network_gate": {"pass": all(row["pass"] for row in gate_cells), "cells": gate_cells, "rule": gate},
        "cdf_sampling_scale": uncertainty,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "sampled-config.json")
    parser.add_argument("--population", type=Path, default=ROOT / "results/population.json.gz")
    parser.add_argument("--sampled", type=Path, default=ROOT / "results/sampled-raw.json.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/sampled-summary.json")
    parser.add_argument("--check", action="store_true", help="Check the committed summary without rewriting it")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = summarize(config, rows_from_file(args.population), rows_from_file(args.sampled))
    result["inputs_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("config", args.config), ("population", args.population), ("sampled", args.sampled))}
    if args.check:
        if json.loads(args.output.read_text()) != result:
            raise SystemExit("Stored sampled summary differs from the raw results")
    else:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["network_gate"], indent=2))


if __name__ == "__main__":
    main()
