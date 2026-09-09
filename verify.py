"""Re-execute exact checkers and compare their source-bound certificates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_json(script):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # The preserved proof checkers use assertions; never disable them in children.
    environment.pop("PYTHONOPTIMIZE", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def compare(script, certificate, ignored=()):
    actual = run_json(script)
    expected = json.loads((ROOT / certificate).read_text())
    for key in ignored:
        actual.pop(key, None)
        expected.pop(key, None)
    if actual != expected:
        raise RuntimeError(f"Certificate mismatch: {script} versus {certificate}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Also recheck preserved certificates and sampled artifacts (needs NumPy).",
    )
    args = parser.parse_args()
    current = compare(
        "certify.py", "results/certificates.json", ignored=("wall_seconds", "python")
    )
    zero = compare(
        "zero_initialization.py", "results/zero-initialization-certificate.json"
    )
    report = {
        "status": "verified",
        "one_state_instances": len(current["one_state"]),
        "fixed_discount_instances": len(current["fixed_discount"]),
        "zero_start_instances": len(zero["checks"]),
        "fixed_discount_periods": [row["exact_period"] for row in current["fixed_discount"]],
        "comparison": "Exact JSON equality, excluding certificate runtime and Python version.",
        "source_hashes_matched": True,
        "scope": "Finite certificate reproduction, not a machine-checked universal proof.",
    }
    if args.historical:
        historical = "historical/quantile-continuation/"
        original = compare(
            "historical/certify_quantile_cycle.py",
            "historical/quantile-cycle-certificate.json",
        )
        damping = compare(historical + "certify_damped.py", historical + "damped-certificate.json")
        robustness = compare(
            historical + "certify_robustness.py", historical + "robustness-certificate.json"
        )
        report["historical"] = {
            "original_period": original["exact_period"],
            "damped_period": damping["period"],
            "robustness_inequalities_verified": robustness["all_exact_inequalities_verified"],
            "sampled_artifacts": run_json(historical + "verify.py"),
            "scope": "Rechecks stored runs and recomputes endpoints; does not rerun training.",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
