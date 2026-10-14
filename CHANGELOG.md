# Changelog

## 2026-10-08 — Tighten checks

- Pinned every GitHub Action to a full commit SHA.
- Added standard-library tests for the exact checker's guards (unique
  quantiles, strict greedy gaps, parameter range), the README's K=2 numbers,
  the delay-length formula, and the verifier's mismatch detection.
- CI now also rechecks the preserved historical certificates and stored runs.
- Corrected the Modal examples to write the compressed files the summary reads,
  and updated the citation abstract. No certificate, result or claim changed.

## 2026-10-07 — Formalize the smallest exact cycle

- Added a pinned Lean 4.19.0/Mathlib Lake project for the rational K=2,
  beta=1/4 one-state/two-action MDP.
- Defined the exact reward-plus-continuation target laws and midpoint
  generalized-inverse projection, then proved strict choices and distinct
  two-cycle closure without assuming either update identity.
- Checked normalized positive reward probabilities, reward bounds, true
  Bellman optimality, stationary randomized-policy values, finite-policy
  bounds and all cycle quantiles. Saved actual axiom output for every proven
  statement, using only Lean's standard logical axioms, and added proof CI.
- Distinguished this finite formal result from the still-unchecked general
  real, attraction, no-fixed-point and delay-embedding claims.
- Preserved the exact Python checker and all historical certificate bindings.

## 2026-10-07 — Test sampling separately from Huber smoothing

- I fixed the settings and stopping rules before the first run, then compared
  hard projection, damping, population pinball/Huber and paired sampled control.
- I ran every tabular cell with 32 seeds, 100,000 updates and batch 32, including
  a separately labeled fixed-MDP capacity comparison and paired scalar control.
- All four required Huber excess-regret comparisons failed. At family K=32 with
  constant steps, Huber had less final-window regret than scalar control.
  I stopped before the conditional replay/network stage.
- I saved every seed, checkpoint critic, loss, Bellman residual and paired
  bootstrap interval, plus CPU cloud costs and reproducible analysis.
- I added independent small gradient, projection, sample-pairing and endpoint
  tests, and CI that checks the committed results without repeating training.
- I made the summary reader accept both direct local-run JSON and batched
  cloud results, and tested both formats with plain and compressed files.
- I keep the specific Lean proof separate from the general formal attraction
  and no-fixed-point claims; combining these results does not expand its scope.

## 2026-10-06 — Make the construction easier to inspect

- Rewrote the README around the exact result and the dependency-free verifier.
- Added a reproducible SVG of the smallest cycle, including projected atoms,
  greedy means, optimal-continuation action values and stationary-policy values.
- Added the fixed-discount states/period table from the existing certificates.
- Added CPU GitHub Actions verification and a generated-figure consistency check.
- Moved the earlier handoff record to `docs/HANDOFF.md`; preserved certificates,
  checker sources and historical experiments unchanged.
- Added `docs/NEXT.md` with a Lean 4 lemma plan and a bounded sampled-learning
  experiment. Neither formalization nor new training was attempted.

## 2026-09-09 — Initial research artifact

- Added the explicit arbitrary-K>=2 one-state strict two-cycle construction, global attraction modulo phase, and no-fixed-point consequence.
- Added the fixed-discount delay embedding and the K=1 discounted contraction boundary.
- Included exact rational checks at 18 one-state capacities, five fixed-discount embeddings, and 18 zero-start instances.
- Incorporated review clarifications for the discounted K=1 assumption and minimal delay choice; retained the source-bound initial derivation separately from the final theorem.
- Preserved the earlier K=3 search, exact and damped cycle certificates, robustness result, and both sampled-learning runs, including the failed larger-critic practical gate.
- Added a dependency-free current verifier and optional NumPy-backed historical verification, with successful output recorded.
- Added primary-source comparisons, claim limits, citation metadata, MIT license, and a durable research graph/handoff.
- No new learning algorithm, practical QR-DQN failure, theoretical priority, or external peer-review acceptance is claimed.
