# Quantile Cycles

An exact counterexample to convergence of mean-greedy, hard quantile-projected Bellman control in distributional reinforcement learning.

**Question:** can a finite quantile critic keep changing its greedy action forever, even when the MDP has a unique optimal policy?

I constructed a one-state, two-action MDP family. [`THEOREM.md`](THEOREM.md) gives the general mathematical argument; [`certify.py`](certify.py) checks finite instances exactly. I formalized the smallest rational counterexample in [`lean/QuantileCycles.lean`](lean/QuantileCycles.lean), including target laws and generalized-inverse midpoint projection.

**Result:** yes, for every quantile count **K ≥ 2**, with a corresponding MDP. One cycle phase strictly chooses the genuinely suboptimal action; no greedy tie is needed on the cycle. This is a theorem about an exact operator, not an observed QR-DQN training failure.

## Reproduce

From the repository root, the first check is:

```sh
python -B verify.py
```

The existing verifier needs only Python 3.11+ and its standard library. It reproduces [18 one-state instances, five fixed-discount instances and 18 zero-start trajectories](results/verification.json), comparing exact JSON apart from runtime/version metadata. Finite checks do not prove the universal theorem.

The separate Lean check uses the [pinned Lean 4.19.0 and Mathlib revision](lean/lakefile.toml):

```sh
cd lean && lake exe cache get Mathlib.Data.Rat.Defs Mathlib.Tactic.NormNum Mathlib.Tactic.Linarith && lake build
```

CI also checks the SVG and compares actual `#print axioms` output with [`results/lean-axioms.txt`](results/lean-axioms.txt). I checked the [pinned proof](lean/modal_build.py) on two CPU cores and 2 GiB; [cloud budget estimate](results/lean-compute.json): **$0.02**.

## The smallest cycle

![Two exact critic phases: quantile atoms and their means make the greedy action flip from A to B and back, although true expected values favor A.](docs/two-phase-cycle.svg)

The figure is generated directly from [`family(2, Fraction(1, 4))`](certify.py), with both Bellman backups checked by the [drawing script](tools/render_cycle.py). Circles are projected quantiles; diamonds are their arithmetic means. These are exact constructed values, not sampled measurements.

For **K = 2**, discount **1/4**, action A always pays **91/128**. B pays **0, 1/2, 1** with probabilities **3/16, 1/2, 5/16** ([construction](THEOREM.md#6-a-smallest-explicit-example)). Both projected greedy gaps are **3/128**, but A's true mean-reward advantage is **19/128**. Always taking B loses **19/96** in discounted value ([certificate](results/certificates.json)). The figure distinguishes optimal-continuation action values from the values of always taking each action.

## General statement

For every integer **K ≥ 2** and **0 < β ≤ 1/(2K)**, the constructed one-state MDP has rewards in **[0,1]**, unique quantiles on its exact two-cycle, and strict alternating greedy choices. Every finite real initial atom table converges to that cycle **modulo phase**, regardless of transient greedy tie choices. Thus the operator has **no fixed point**.

For any prescribed **0 < γ < 1**, a deterministic delay chain gives **L states** and a locally attracting cycle of primitive period **2L**, where **L = ceil(log(2K)/(−log γ))**. Its operator also has no fixed point. Unlike the one-state result, this does not assert global convergence to one common phase alignment.

At **γ = 9/10**, the [stored fixed-discount certificates](results/certificates.json) check every primitive state backup:

| Quantiles K | States L | Exact period 2L |
|---:|---:|---:|
| 2 | 14 | 28 |
| 3 | 18 | 36 |
| 8 | 27 | 54 |
| 16 | 33 | 66 |
| 32 | 40 | 80 |

## What is machine-checked

The Lean scope is **K = 2, β = 1/4 over rationals**: normalized positive probabilities, reward bounds, true Bellman optimality, randomized-policy values and finite-policy bounds, strict greedy choices, all eight generalized-inverse quantiles, and distinct exact two-cycle closure. The update constructs reward-plus-continuation laws, not a phase lookup.

This finite proof is checked by Lean's kernel with no sorry and no axioms beyond Lean's standard three (propext, Classical.choice, Quot.sound). [Actual `#print axioms` output](results/lean-axioms.txt) covers every public theorem. The arbitrary-K **real** theorem, attraction, no-fixed-point conclusion, delay embedding and K=1 boundary are **not Lean-checked**. A checked cycle alone does not exclude other fixed points.

## Limitations and next steps

- The MDP and reward law depend on K; the suboptimal-policy loss shrinks with K. This is not one MDP failing at every capacity.
- Hard midpoint projection is not pinball/Huber SGD, replay, target networks or deep QR-DQN.
- Earlier sampled experiments on a **different** witness failed their larger-critic practical criterion; [the historical record](historical/README.md) is retained unchanged.
- At **K = 1**, the discounted single-atom operator is a contraction, though its fixed point need not optimize expected return ([proof](THEOREM.md#8-why-k1-is-different)).
- The general proof is not machine-checked or externally peer-reviewed; priority is unestablished. [Next steps](docs/NEXT.md) separate the completed rational example from general formalization and sampled learning. [Supporting records](docs/HANDOFF.md) preserve earlier decisions.

## Prior work

This builds on [Dabney et al.'s quantile projection](https://arxiv.org/abs/1710.10044), [Rowland et al.'s finite-quantile suboptimality examples](https://proceedings.mlr.press/v97/rowland19a.html), and [distributional control nonconvergence examples](https://www.distributional-rl.org/contents/chapter7). The proposed distinction is strict suboptimal cycling with one-state global attraction, not distributional nonconvergence itself. [`PRIOR_ART.md`](PRIOR_ART.md) gives the comparison and attribution.

Written with AI coding assistance.
