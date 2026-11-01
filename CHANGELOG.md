# Changelog

## 2026-11-01 — Clarify the theorem and learning result

- Put the no-fixed-point construction and sampled Huber comparison first in the README.
- State the K=2 Lean proof boundary alongside the general theorem.
- Shorten the method, related-work and historical reports; remove research-planning and AI-assistance prose.
- Keep proof code, figures, result files and hash-bound derivations unchanged.

## 2026-10-08 — Tighten checks

- Pin every GitHub Action to a full commit SHA.
- Add standard-library tests for unique quantiles, strict greedy gaps, parameter ranges, the K=2 values, delay length and certificate mismatch detection.
- Recheck historical certificates and stored runs in CI.
- Correct the Modal examples to write the compressed files the summary reads, and update the citation abstract.

## 2026-10-07 — Formalize the smallest exact cycle

- Add a pinned Lean 4.19.0/Mathlib project for the rational K=2, beta=1/4 one-state MDP.
- Define exact reward/continuation target laws and midpoint generalized-inverse projection; prove strict choices and distinct two-cycle closure.
- Prove reward-law validity, bounded rewards, true Bellman optimality, stationary randomized-policy values, finite-policy bounds and all eight cycle quantiles.
- Save axiom output using only Lean's standard logical axioms and add proof CI.

## 2026-10-07 — Compare sampling and Huber smoothing

- Compare hard projection, damping, population pinball/Huber and paired sampled control.
- Run every sampled setting with 32 seeds, 100,000 updates and batch 32, including fixed-MDP capacity comparisons and paired scalar control.
- Find no material Huber excess regret in the four larger-capacity comparisons. At family K=32 with constant steps, Huber has less final-window regret than scalar control. Stop before the conditional replay/network experiment.
- Include seed endpoints, checkpoint critics, losses, Bellman residuals, paired bootstrap intervals and CPU costs.
- Add gradient, projection, sample-pairing and endpoint tests, plus CI that checks saved results without retraining.
- Support direct local JSON and batched cloud results in the summary reader, both plain and compressed.

## 2026-10-06 — Illustrate the construction

- Add a generated SVG of the smallest cycle, including projected atoms, greedy means, optimal-continuation action values and stationary-policy values.
- Add the fixed-discount states/period table from the certificates.
- Add CPU verification and a generated-figure consistency check in GitHub Actions.

## 2026-09-09 — Initial construction

- Add the explicit arbitrary-K≥2 one-state two-cycle, global attraction modulo phase and no-fixed-point consequence.
- Add fixed-discount delay embedding and the K=1 discounted contraction boundary.
- Include exact rational checks at 18 one-state capacities, five fixed-discount embeddings and 18 zero-start instances.
- Include the initial derivation as a certificate input and the final proof with the discounted K=1 assumption and minimal delay length.
- Include the earlier K=3 search, exact and damped cycle certificates, open-neighborhood bounds and two sampled-learning runs. The larger-critic experiment found no material excess regret.
- Add a standard-library verifier and NumPy-backed historical checks.
- Add related-work comparisons, citation metadata and the MIT license.
