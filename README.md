# Quantile Cycles

Quantile-projected Bellman control can have no fixed point, even in a one-state MDP with a unique optimal action. I construct a strict two-cycle for every finite quantile count K≥2: the greedy policy alternates between the optimal action and a worse one.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/hero-dark.svg"><img alt="Exact quantile-projected Bellman updates alternate between the optimal and a worse action." src="docs/hero-light.svg" width="600"></picture>

The sampled result goes the other way. Across 32 seeds and 100,000 updates per setting, quantile-Huber learning matched or beat scalar Q-learning on final-window regret. At family K=32 with constant steps, Huber chose the worse action on 0.43% of updates, versus 7.83% for scalar control. [Full results](docs/SAMPLED_RESULTS.md).

## Check the result

The exact certificates need only Python's standard library:

```sh
python -B verify.py
```

With Python 3.11, check the sampled analysis without retraining:

```sh
python -m pip install -r requirements-sampled.txt
python -B tools/summarize_sampled.py --check
```

The Lean proof uses the pinned toolchain in `lean/` and requires [elan](https://github.com/leanprover/elan):

```sh
cd lean
lake exe cache get Mathlib.Data.Rat.Defs Mathlib.Tactic.NormNum Mathlib.Tactic.Linarith
lake build
```

## The construction

![Two exact critic phases make the greedy action flip, although expected rewards favor A.](docs/two-phase-cycle.svg)

The smallest example has K=2 and discount 1/4. Action A pays 91/128. Action B pays 0, 1/2 and 1 with probabilities 3/16, 1/2 and 5/16. Both actions return to the same state.

Exact midpoint-quantile backups swap the two critic tables shown above. Both greedy choices are strict, with gap 3/128. A's true reward advantage is 19/128; always choosing B loses 19/96 in discounted value. These fractions are checked in [the exact certificate](results/certificates.json).

The [general proof](THEOREM.md) constructs a corresponding MDP for each K≥2 and discount 0<β≤1/(2K). Every finite real initial table approaches its two-cycle modulo phase, including arbitrary transient tie choices. This rules out fixed points. For a prescribed 0<γ<1, a delay chain gives L=ceil(log(2K)/(−log γ)) states and exact period 2L, with local attraction. At γ=9/10, the certificates check periods 28, 36, 54, 66 and 80.

The [Lean formalization](lean/QuantileCycles.lean) proves the rational K=2, β=1/4 instance: valid bounded reward laws, true Bellman values, all eight generalized-inverse quantiles, strict greedy choices and a distinct two-cycle. The [saved build](results/lean-build.json) and [axiom output](results/lean-axioms.txt) use only Lean's standard logical axioms. The arbitrary-K theorem, attraction, absence of other fixed points, delay embedding and K=1 boundary are proved in the research note, outside the formalization.

## Learning on the same family

I compared hard projection, damped projection, population pinball/Huber gradients and sampled learning. Each sampled run used 32 target pairs per action per update, zero initialization and a paired scalar baseline. The [study protocol](docs/SAMPLED_PROTOCOL.md) gives the updates and stopping rule.

The table shows the fraction of greedy choices that were B during updates 80,001–100,000, averaged over seeds. This is stationary-policy regret normalized by the always-B loss. Each entry is constant / decaying step size; each family row uses a different K-dependent MDP.

| Family K | Sampled pinball | Sampled Huber κ=1 | Paired scalar |
|---:|---:|---:|---:|
| 2 | 0.37519 / 0.39939 | 0 / 0 | 0 / 0 |
| 8 | 0.07916 / 0.11637 | 0 / 0 | 0 / 0 |
| 32 | 0.10325 / 0.02846 | 0.00429 / 0 | 0.07826 / 0.00229 |

Population Huber had zero final-window regret in every setting. At K=32 with constant steps, Huber picked the worse action about 7.4 percentage points less often than scalar Q-learning in the final window (95% interval 7.1–7.7 points). None of the four Huber comparisons met the excess-regret threshold for running the replay/network experiment, so I stopped at tabular learning.

Holding the K=2 MDP fixed and increasing critic capacity to 8 or 32 gave zero observed final-window regret for every sampled loss. [All seeds and intervals](results/sampled-summary.json), [raw checkpoints](results/sampled-raw.json.gz) and [training commands](docs/SAMPLED_RESULTS.md#reproduce) are available.

## Interpretation

- The reward law changes with K, and the constructed loss shrinks with K.
- Hard quantile projection, pinball SGD and Huber SGD are different operators. Finite noisy switches do not establish an asymptotic periodic orbit.
- The experiments use a tabular generative sampler, without exploration, replay or target networks. Zero observed regret across 32 seeds does not exclude rare failures.

An [earlier three-state example](historical/README.md) also cycles under exact backups, while its sampled larger-critic experiment found no material excess regret.

## Prior work

I build on [Dabney et al.'s quantile projection and Huber loss](https://arxiv.org/abs/1710.10044), [Rowland et al.'s finite-quantile suboptimality examples](https://proceedings.mlr.press/v97/rowland19a.html), and [distributional control nonconvergence examples](https://www.distributional-rl.org/contents/chapter7). [PRIOR_ART.md](PRIOR_ART.md) compares their assumptions with this construction.
