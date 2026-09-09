# Preserved predecessor artifacts

These artifacts predate the arbitrary-K family in the repository root. They are retained to expose the full research trajectory, including a failed practical extension. Historical claims such as “arbitrary-K unestablished” describe the state at that stage, not the final theorem. Bound source files, protocols, outputs, and reports remain unchanged.

## Contents

- `quantile_control_baseline.py`, `quantile-control-baseline-result.json`, `quantile-control-thesis.md`: initial baseline probe.
- `quantile_cycle.py`, `quantile-cycle-protocol.md`, `quantile-cycle-result.json`: bounded search and discovered three-state, K=3 witness.
- `certify_quantile_cycle.py`, `quantile-cycle-certificate.json`: exact six-cycle, unique quantiles, strict action gaps, and genuine bad-policy loss.
- `quantile-continuation/protocol.md`: frozen capacity/damping and sampled-learning continuation.
- `quantile-continuation/run.py`: exact-backup sweeps and actual sampled tabular training.
- `quantile-continuation/results-v1/` and `reproduction-v1/`: original run and independent deterministic reproduction, including policy traces, endpoints, and manifests.
- `quantile-continuation/certify_damped.py`: exact period-26 certificate for alpha=1/2.
- `quantile-continuation/certify_robustness.py`: exact open-neighborhood inequalities for the original six-cycle.
- `quantile-continuation/verify.py`: bound input/output hashes, semantic reproduction, eight exact policy-value systems, and recomputed sampled endpoints.
- [`quantile-continuation/REPORT.md`](quantile-continuation/REPORT.md): results and frozen decision.

## The failed gate is part of the result

The sampled experiment used 16 seeds, 20,000 updates per setting, paired transition draws, pinball and Huber losses, and constant/decaying schedules. The practical gate required mean excess regret at least 0.25, with a positive lower 95% paired-bootstrap endpoint, under both schedules for the same loss at K>=16.

**No cell passed.** K=3 pinball retained substantial excess regret, but Huber smoothing greatly reduced it and larger critics did not support the predeclared claim. Finite-run checkpoint switching is not a certified periodic orbit. These data do not establish deep QR-DQN nonconvergence.

The new theorem changes the MDP with K. It does not overturn this negative result on the older fixed witness.

## Recheck stored evidence

From the repository root, using the optional NumPy environment described in the main README:

```sh
.venv/bin/python -B verify.py --historical
```

The wrapper compares exact certificate JSON, rechecks manifests, compares stored arrays semantically, independently solves policy values, and recomputes sampled endpoints. It does not retrain.

## Rerun the actual sampled experiment

Use a new output directory; the historical runner refuses to overwrite an existing one:

```sh
.venv/bin/python -B historical/quantile-continuation/run.py --output runs/historical-rerun-v1
```

This runs the full frozen exact-backup and sampled-learning protocol and writes a new manifest. It does not replace the preserved v1 runs. Compare arrays and JSON semantically: NumPy ZIP archive container metadata need not be byte-identical. The existing historical `verify.py` intentionally compares the two preserved directories, not an arbitrary new run.

The retained protocol and runner bind their predecessor witness by relative path, so keep this directory layout intact. Generated local reruns are ignored by Git.
