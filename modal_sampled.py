"""Run the fixed tabular study in ephemeral Modal CPU containers.

Examples:
    modal run modal_sampled.py --stage pilot --output results/sampled-pilot.json
    modal run modal_sampled.py --stage population --output results/population.json.gz
    modal run modal_sampled.py --stage sampled --output results/sampled-raw.json.gz
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
TIMEOUT_SECONDS = int(os.environ.get("QUANTILE_TIMEOUT_SECONDS", "1200"))
MAX_CONTAINERS = int(os.environ.get("QUANTILE_MAX_CONTAINERS", "4"))
app = modal.App("quantile-cycles-tabular")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "numba==0.61.2")
    .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"})
    .add_local_dir(
        ROOT,
        remote_path="/root/study",
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "results", "historical", "lean"],
    )
)


@app.function(image=image, cpu=2, memory=1024, timeout=TIMEOUT_SECONDS, max_containers=MAX_CONTAINERS)
def run_batch(stage: str, config: dict, seeds: list[int]) -> dict:
    import os
    import platform
    import sys
    import time

    sys.path.insert(0, "/root/study")
    import llvmlite
    import numba
    import numpy as np
    import sampled

    started = time.monotonic()
    result = (
        sampled.run_population(config)
        if stage == "population"
        else sampled.run_sampled(config, seeds)
    )
    # Keep all saved critics so the atom-wise diagnostics can be regenerated,
    # but summarize CDF rows before transferring the result to the client.
    for row in result["rows"]:
        for point in row["checkpoints"]:
            point["cdf_resolution_by_action"] = [
                {
                    "minimum_left_margin": min(atom["left_margin"] for atom in action),
                    "minimum_right_margin": min(atom["right_margin"] for atom in action),
                    "maximum_cdf_standard_error": max(max(atom["cdf_left_standard_error"], atom["cdf_right_standard_error"]) for atom in action),
                    "maximum_empirical_crossing_miss_probability": max(atom["empirical_crossing_miss_probability"] for atom in action),
                }
                for action in point.pop("cdf_crossings_by_action")
            ]
    return {
        "result": result,
        "hardware": {
            "provider": "Modal",
            "gpu": "none",
            "cpu_cores_requested": 2,
            "memory_mib_requested": 1024,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "llvmlite": llvmlite.__version__,
            "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()),
            "threads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "remote_wall_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main(stage: str = "pilot", output: str = "results/sampled-pilot.json", seed_batch: int = 8):
    if stage not in {"pilot", "population-pilot", "population", "sampled"}:
        raise ValueError("stage must be pilot, population-pilot, population or sampled")
    if seed_batch < 1:
        raise ValueError("seed_batch must be positive")
    config_path = ROOT / "sampled-config.json"
    config = json.loads(config_path.read_text())
    inputs = []
    if stage in {"pilot", "population-pilot"}:
        pilot = copy.deepcopy(config)
        pilot["updates"] = 2000
        pilot["endpoint_start"] = 1601
        pilot["checkpoints"] = [0, 1, 10, 100, 1000, 2000]
        pilot["cases"] = [case for case in config["cases"] if case["name"] == "family-32"]
        inputs.append(("population", pilot, []) if stage == "population-pilot" else ("sampled", pilot, [0, 1]))
    else:
        for case in config["cases"]:
            subset = copy.deepcopy(config)
            subset["cases"] = [case]
            if stage == "population":
                inputs.append((stage, subset, []))
            else:
                seeds = config["seeds"]
                for start in range(0, len(seeds), seed_batch):
                    inputs.append((stage, subset, seeds[start : start + seed_batch]))
    batches = list(run_batch.starmap(inputs, order_outputs=True))
    result = {
        "stage": stage,
        "protocol_commit": "beb7921",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sampled_source_sha256": hashlib.sha256((ROOT / "sampled.py").read_bytes()).hexdigest(),
        "config": config,
        "batches": batches,
        "billing": {
            "gpu": "none",
            "maximum_containers": MAX_CONTAINERS,
            "container_timeout_seconds": TIMEOUT_SECONDS,
            "cpu_core_hour_usd": 0.047160,
            "memory_gib_hour_usd": 0.007992,
            "estimated_active_container_cost_usd": sum(batch["remote_wall_seconds"] for batch in batches) / 3600 * (2 * 0.047160 + 0.007992),
            "note": "Active-function estimate excludes image build and startup. The separately recorded client wall-time upper bound includes these costs.",
        },
    }
    destination = ROOT / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".gz":
        with gzip.open(destination, "wt", compresslevel=6) as stream:
            json.dump(result, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    else:
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(batches)} batches to {output}")
    print(json.dumps(result["billing"], indent=2))
