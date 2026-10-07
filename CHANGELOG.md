# Changelog

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
