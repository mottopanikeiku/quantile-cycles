# Earlier three-state example

Before the one-state arbitrary-K construction, I found a three-state, K=3 MDP with an exact six-cycle. Damping the backup by 1/2 gives an exact 26-cycle. Sampled learning at larger critic capacities did not reproduce the material excess-regret effect being tested.

## Results and files

- [Original example](quantile-control-thesis.md): reward/transition table, exact six-cycle, local attraction and scalar baseline.
- [Search protocol](quantile-cycle-protocol.md), `quantile_cycle.py` and `quantile-cycle-result.json`: the 512-MDP search and selected witness.
- `certify_quantile_cycle.py` and `quantile-cycle-certificate.json`: exact cycle, unique quantiles, strict action gaps and bad-policy loss.
- `quantile_control_baseline.py` and `quantile-control-baseline-result.json`: scalar-control comparison.
- [Continuation report](quantile-continuation/REPORT.md): larger critics, damping, sampled learning and reward/probability perturbations.
- `quantile-continuation/protocol.md` and `run.py`: continuation settings and experiment implementation.
- `quantile-continuation/results-v1/` and `reproduction-v1/`: two runs with the same seeds, including policy traces, endpoints and manifests.
- `quantile-continuation/certify_damped.py` and `certify_robustness.py`: exact period-26 and open-neighborhood checks.

The sampled experiment used 16 seeds, 20,000 updates per setting, paired transition draws, pinball and Huber losses, and constant/decaying schedules. The criterion was mean excess regret at least 0.25 with a positive lower 95% paired-bootstrap endpoint, under both schedules for the same loss at K≥16. No setting passed. K=3 pinball retained substantial excess regret, while Huber smoothing reduced it and larger critics usually matched or improved on scalar control.

The later [arbitrary-K theorem](../THEOREM.md) changes the MDP with K. The capacity comparison here holds this older MDP fixed.

## Check the stored runs

From the repository root, with NumPy installed:

```sh
python -m pip install -r requirements-historical.txt
python -B verify.py --historical
```

The wrapper compares exact certificates, checks input/output hashes, compares stored arrays, solves policy-value equations and recomputes sampled endpoints. It does not retrain. The continuation protocol remains unchanged because both run manifests check its SHA256.

## Rerun the experiment

Use a new output directory; the runner refuses to overwrite an existing one:

```sh
python -B historical/quantile-continuation/run.py --output runs/historical-rerun-v1
```

This runs the exact-backup and sampled-learning experiments and writes a new manifest. Compare arrays and JSON values: NumPy ZIP archive metadata can differ between runs. The historical `verify.py` compares the two stored directories rather than an arbitrary new run.

Keep the directory layout intact; the runner and certifiers use relative paths to the original witness. Local `runs/` output is ignored by Git.
